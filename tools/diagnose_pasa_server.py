"""Read-only AutoDL diagnostics of an existing recent-alignment campaign."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import train_semantic_alignment as base
import train_recent_alignment as recent
from experiment_dataset import _bbox_from_mask, _z_indices, _resize_mask_stack
from dataset import find_modality_file, find_segmentation_file
from recent_alignment import RecentAlignmentModel
from semantic_evaluation import validate_patient_splits
from json_results import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_base_helpers(expected):
    """Allow the known JSON-writer patch, not silent changes to scientific helpers."""
    frozen = subprocess.check_output(['git', 'show', 'b4cfc9c:train_semantic_alignment.py'], cwd=ROOT)
    if hashlib.sha256(frozen).hexdigest() != expected:
        raise ValueError('Unsupported historical base source; audit before using this diagnostic')
    def helpers(source):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name not in ('main', 'save_json')}
    if helpers(frozen) != helpers((ROOT/'train_semantic_alignment.py').read_bytes()):
        raise ValueError('Scientific base helper functions changed since frozen training')


def region_counts(seg, roi_size, z_slices):
    """Count both sampled native voxels and resized network mask voxels."""
    y0, y1, x0, x1, z = _bbox_from_mask(seg)
    indices = _z_indices(z, seg.shape[2], z_slices)
    counts = []
    for label in (1, 2, 4):
        stack = np.stack([(seg[y0:y1+1, x0:x1+1, i] == label).astype(np.float32) for i in indices])
        counts.append({'label': label, 'sampled_native_voxels': int(stack.sum()),
            'network_mask_voxels': int(_resize_mask_stack(stack, roi_size).sum())})
    return counts


def scan_regions(groups, args, out):
    import nibabel as nib
    records = []
    for split in ('train', 'val'):
        for i, case in enumerate(groups[split], 1):
            first = find_modality_file(case['patient_dir'], 't1', prefer_registered=args.prefer_registered)
            shape = tuple(nib.load(first).shape)
            seg = nib.load(find_segmentation_file(case['patient_dir'], image_shape=shape)).get_fdata(dtype=np.float32)
            records.append({'split': split, 'subject_id': str(case['subject_id']),
                'regions': region_counts(seg, args.roi_size, args.z_slices)})
            if i % 50 == 0:
                print(f'region scan {split}: {i}/{len(groups[split])}', flush=True)
        write_json(out / 'region_records.json', records)
    summary = {}
    for split in ('train', 'val'):
        rows = [r for r in records if r['split'] == split]
        summary[split] = {'patients': len(rows), 'empty_network_regions_by_label': {
            str(label): sum(r['regions'][j]['network_mask_voxels'] == 0 for r in rows)
            for j, label in enumerate((1, 2, 4))}}
    write_json(out / 'region_summary.json', summary)


def validation_pass(model, loader, device, lookup, vocab, keys):
    model.eval()
    scores, subjects = [], []
    with torch.no_grad():
        for batch in loader:
            result = model(batch['images'].to(device), region_masks=batch['region_masks'].to(device))
            scores.extend(result['scores'].cpu().tolist())
            subjects.extend(str(s) for s in batch['subject_id'])
    metrics, records = recent.score_patients(np.asarray(scores), subjects, lookup, vocab, keys)
    return metrics, records, np.asarray(scores), subjects


def gradient_probe(model, loader, device, lookup, vocab, keys, args, epoch, batches, out):
    """Probe saved checkpoint in train mode; restore all buffers after every batch."""
    rows = []
    params = tuple(model.encoder.shared_heads.parameters())
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.train()
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        base.set_seed(20261019+i)
        result = model(batch['images'].to(device), region_masks=batch['region_masks'].to(device),
            freeze_graph=epoch <= args.graph_warmup_epochs)
        pos, known, ids = recent.label_masks(batch['subject_id'], lookup, vocab, keys, device)
        local = result['local']
        weighted = {'alignment': model.alignment_loss(result, pos, known)}
        node_ids = [target for target in ids for _ in range(local.shape[1])]
        weighted['anchor'] = args.lambda_anchor * base.anchor_center_loss(
            local.reshape(-1, local.shape[-1]), node_ids, result['candidates'])
        for name in ('cons', 'decouple', 'leak', 'diff', 'diff_norm', 'gate_entropy', 'load_balance'):
            weight = getattr(args, 'lambda_'+name)
            if name == 'cons':
                weight *= base.graph_cons_scale(epoch, args.graph_warmup_epochs)
            weighted[name] = weight * result['encoder_output']['losses'][name]
        details = {}
        for name, loss in weighted.items():
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite {name} loss')
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            details[name] = {'weighted_loss': float(loss.detach()), 'shared_head_gradient_norm':
                float(sum((g.detach().square().sum() for g in grads if g is not None),
                          torch.zeros((), device=device)).sqrt())}
        node_grad = torch.autograd.grad(weighted['alignment'], result['node_scores'], retain_graph=False)[0].detach()
        node_pos = pos[:, None, :].expand_as(node_grad)
        anchors = []
        for a, anchor in enumerate(vocab):
            values = node_grad[:, :, a][node_pos[:, :, a]]
            if values.numel():
                anchors.append({'key': anchor['key'], 'field': anchor['field'], 'positive_nodes': values.numel(),
                    'mean_absolute_positive_score_gradient': float(values.abs().mean())})
        rows.append({'batch': i, 'subject_ids': [str(s) for s in batch['subject_id']],
            'checkpoint_context_epoch': epoch, 'components': details, 'positive_gradients': anchors})
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        write_json(out / 'gradient_probe.json', {'scope': 'saved checkpoint, train-mode local sensitivity; no optimizer steps; not historical training gradients',
            'gradient_parameters': 'encoder.shared_heads', 'batches': rows})
    model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', default='output/sota_alignmentv2_20261009')
    p.add_argument('--data-root', default='/root/autodl-tmp/dataset/UTSW-Glioma')
    p.add_argument('--metadata', default='/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv')
    p.add_argument('--out', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--gradient-batches', type=int, default=3)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu', action='store_true')
    cli = p.parse_args()
    if cli.repeats < 2 or cli.gradient_batches < 1 or cli.workers < 0:
        p.error('repeats >= 2, gradient-batches >= 1, workers >= 0 required')
    campaign = Path(cli.campaign).resolve()
    identity = read(campaign / 'prepared.json')['identity']
    frozen = read(campaign / 'splits.json')
    if frozen != identity['splits']:
        raise ValueError('Frozen splits mismatch')
    if digest(cli.metadata) != identity['metadata_sha256']:
        raise ValueError('Metadata differs from frozen campaign')
    # Serialization patches may differ; model/data/metric implementation must match.
    core = ('recent_alignment.py', 'experiment_model.py', 'experiment_dataset.py', 'dataset.py',
            'semantic_evaluation.py', 'train_recent_alignment.py')
    for name in core:
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != identity['source_sha256'][name]:
            raise ValueError(f'Scientific source differs from frozen campaign: {name}')
    verify_base_helpers(identity['source_sha256']['train_semantic_alignment.py'])
    device = 'cpu' if cli.cpu else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --cpu explicitly if intended')
    args = base.build_parser().parse_args(['--data_root', cli.data_root])
    for name in ('roi_size', 'z_slices', 'batch_size', 'sample_seed', 'split_seed'):
        setattr(args, name, identity[name])
    args.metadata_tsv, args.num_workers, args.augment, args.cache = cli.metadata, cli.workers, False, False
    cases = base.discover_semantic_cases(cli.data_root, metadata_tsv=cli.metadata, seed=args.sample_seed)
    lookup = {str(c['subject_id']): c for c in cases}
    groups = {name: [lookup[str(s)] for s in ids] for name, ids in frozen.items()}
    validate_patient_splits(groups)
    vocab, keys = base.build_anchor_vocab(groups['train'])
    if [a['key'] for a in vocab] != identity['vocab_keys']:
        raise ValueError('Train vocabulary changed')
    out = Path(cli.out or 'output/pasa_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    # Refuse all campaign ancestors/descendants to protect original evidence.
    if out == campaign or campaign in out.parents or out in campaign.parents:
        raise ValueError('Diagnostic output must be outside original campaign')
    out.mkdir(parents=True, exist_ok=False)
    source_names = set(identity['source_sha256']) | {'tools/diagnose_pasa_server.py', 'json_results.py'}
    write_json(out/'diagnostic_config.json', {'version': 'pasa_diagnostics_v1', 'arguments': vars(cli),
        'scope': 'train and validation only; no test loader or optimizer', 'frozen_identity': identity,
        'current_source_sha256': {n: digest(ROOT/n) for n in sorted(source_names)},
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'torch_version': torch.__version__})
    scan_regions(groups, args, out)
    loaders = {name: base.make_loader(groups[name], args, 'val') for name in ('train', 'val')}
    # Both loaders are unaugmented and ordered; train membership used only for gradients.
    all_summaries=[]
    for seed in cli.seeds:
        folder = campaign/f'runs/sota/pasa_full_s{seed}/attempt_001'
        config = read(folder/'config.json')
        if config['method'] != 'pasa_full' or config['settings'] != identity:
            raise ValueError('Checkpoint config does not match frozen campaign')
        if read(folder/'anchor_vocab.json') != vocab:
            raise ValueError('Checkpoint vocabulary differs')
        h = read(folder/'history.json')
        epoch = int(h[int(np.argmax([r['validation']['map'] for r in h]))]['epoch'])
        base.set_seed(seed)
        model = RecentAlignmentModel('pasa_full', len(vocab), z_slices=args.z_slices).to(device)
        checkpoint=folder/'best.pt'
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        run_out=out/f'pasa_full_s{seed}'
        run_out.mkdir()
        write_json(run_out/'checkpoint.json',{'path':str(checkpoint),'sha256':digest(checkpoint),'selected_epoch_from_history':epoch})
        first=None
        repeats=[]
        for repeat in range(cli.repeats+1):
            replay = repeat == cli.repeats
            base.set_seed(20261019 if replay else 20261019+repeat)
            metrics, records, scores, subjects = validation_pass(model,loaders['val'],device,lookup,vocab,keys)
            if first is None:
                first, first_ids = scores.copy(), subjects
            if subjects != first_ids:
                raise ValueError('Validation patient order changed')
            delta=float(np.abs(scores-first).max())
            label='fixed_seed_replay' if replay else f'repeat_{repeat}'
            write_json(run_out/(label+'.json'),{'metrics':metrics,'records':records,'max_score_difference_from_first':delta})
            repeats.append({'pass':label,'map':metrics['map'],'biomarker_macro_auc':metrics['biomarker_macro_auc'],
                'max_score_difference_from_first':delta})
            print(f'seed={seed} {label} val_mAP={metrics["map"]:.6f} val_AUROC={metrics["biomarker_macro_auc"]:.6f} score_delta={delta:.6g}',flush=True)
        gradient_probe(model,loaders['train'],device,lookup,vocab,keys,args,epoch,cli.gradient_batches,run_out)
        all_summaries.append({'seed':seed,'passes':repeats})
        del model
        write_json(out/'summary.json',all_summaries)
    write_json(out/'complete.json',{'status':'complete','seeds':cli.seeds,'test_evaluated':False,'optimizer_steps':0})
    print('COMPLETE: '+str(out),flush=True)


if __name__ == '__main__':
    main()
