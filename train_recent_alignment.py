"""Controlled recent-method MRI/metadata benchmark; actual scores, no fabricated SOTA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch

import train_semantic_alignment as base
from recent_alignment import BENCHMARK_VERSION, METHODS, RecentAlignmentModel
from semantic_evaluation import binary_auc, score_retrieval_metrics, supervision_from_keys, validate_patient_splits
from tools.run_pasa_server import digest, write_json


def label_masks(subject_ids, lookup, vocab, keys, device):
    specs = [supervision_from_keys([anchor['key'] for anchor in base.semantic_anchors(lookup[str(sid)]['metadata'])], vocab, keys)
             for sid in subject_ids]
    positives = torch.zeros((len(subject_ids), len(vocab)), dtype=torch.bool, device=device)
    known = torch.zeros_like(positives)
    for row, spec in enumerate(specs):
        positives[row, spec['positive_ids']] = True
        known[row, spec['valid_ids']] = True
    return positives, known, [spec['positive_ids'] for spec in specs]


def score_patients(scores, subjects, lookup, vocab, keys):
    rows, targets, valid, counts, patient_ids, fields = [], [], [], [], [], []
    endpoint = argparse.Namespace(evaluation_fields='molecular')
    for row, sid in enumerate(subjects):
        for field, spec in base.evaluation_specs(lookup[str(sid)]['metadata'], vocab, keys, endpoint):
            rows.append(scores[row])
            targets.append(spec['positive_ids'])
            valid.append(spec['valid_ids'])
            counts.append(spec['positive_count'])
            patient_ids.append(str(sid))
            fields.append(field)
    metrics = score_retrieval_metrics(np.asarray(rows), targets, subject_ids=patient_ids,
        valid_ids=valid, positive_counts=counts, field_names=fields)
    metrics.update(protocol_version=BENCHMARK_VERSION, aggregation='patient_score_then_field_then_patient',
                   score_type='alignment_logits', case_count=len(subjects))
    field_aucs = {}
    for field in base.MOLECULAR_FIELDS:
        candidates = [i for i, anchor in enumerate(vocab) if anchor['field'] == field]
        observed, labels = [], []
        for row, sid in enumerate(subjects):
            anchors = {anchor['field']: anchor['key'] for anchor in base.semantic_anchors(lookup[str(sid)]['metadata'])}
            if field in anchors and anchors[field] in keys:
                observed.append(row)
                labels.append(keys[anchors[field]])
        aucs = {}
        if observed and len(candidates) > 1:
            logits = np.asarray(scores)[observed][:, candidates]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            for column, anchor_id in enumerate(candidates):
                aucs[vocab[anchor_id]['key']] = binary_auc([int(label == anchor_id) for label in labels], probabilities[:, column])
        estimable = [auc for auc in aucs.values() if np.isfinite(auc)]
        field_aucs[field] = {'macro_one_vs_rest_auc': float(np.mean(estimable)) if estimable else float('nan'),
                             'per_class_auc': aucs, 'in_vocabulary_patients': len(observed),
                             'total_patients': len(subjects), 'estimable_classes': len(estimable)}
    estimable_fields = [item['macro_one_vs_rest_auc'] for item in field_aucs.values() if np.isfinite(item['macro_one_vs_rest_auc'])]
    metrics['biomarker_macro_auc'] = float(np.mean(estimable_fields)) if estimable_fields else float('nan')
    metrics['biomarker_auc_by_field'] = field_aucs
    return metrics, {'scores': np.asarray(rows).tolist(), 'target_ids': targets, 'valid_ids': valid,
                     'positive_counts': counts, 'subject_ids': patient_ids, 'field_names': fields}


def evaluate(model, loader, device, lookup, vocab, keys):
    model.eval()
    scores, subjects = [], []
    with torch.no_grad():
        for batch in loader:
            masks = batch.get('region_masks')
            result = model(batch['images'].to(device), region_masks=masks.to(device) if masks is not None else None)
            scores.extend(result['scores'].cpu().numpy())
            subjects.extend(str(sid) for sid in batch['subject_id'])
    return score_patients(np.asarray(scores), subjects, lookup, vocab, keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign_root', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--out_dir', required=True)
    cli = parser.parse_args()
    root, out = Path(os.path.abspath(cli.campaign_root)), Path(os.path.abspath(cli.out_dir))
    if (out / 'config.json').exists():
        raise ValueError('Existing run; use a new attempt')
    prepared = json.loads((root / 'prepared.json').read_text(encoding='utf-8'))['identity']
    for name, expected in prepared['source_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if digest(prepared['metadata_tsv']) != prepared['metadata_sha256']:
        raise ValueError('Metadata changed after protocol freeze')
    frozen = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    if frozen != prepared['splits']:
        raise ValueError('Frozen splits changed')
    args = base.build_parser().parse_args(['--data_root', prepared['data_root']])
    for name in ('metadata_tsv', 'roi_size', 'z_slices', 'batch_size', 'cpu', 'sample_seed', 'split_seed'):
        setattr(args, name, prepared[name])
    args.augment = True
    args.num_workers = prepared['benchmark_settings']['num_workers']
    device = 'cpu' if args.cpu else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    base.set_seed(cli.seed)
    cases = base.discover_semantic_cases(args.data_root, metadata_tsv=args.metadata_tsv, seed=args.sample_seed)
    lookup = {str(case['subject_id']): case for case in cases}
    splits = {name: [lookup[str(sid)] for sid in ids] for name, ids in frozen.items()}
    validate_patient_splits(splits)
    vocab, keys = base.build_anchor_vocab(splits['train'])
    if [anchor['key'] for anchor in vocab] != prepared['vocab_keys']:
        raise ValueError('Train vocabulary changed')
    text_metadata = json.loads((root / 'text_cache.json').read_text(encoding='utf-8'))
    if digest(root / 'text_cache.pt') != text_metadata['sha256']:
        raise ValueError('Text cache changed')
    text = torch.load(root / 'text_cache.pt', map_location='cpu', weights_only=True)
    if text['keys'] != prepared['vocab_keys']:
        raise ValueError('Text cache vocabulary mismatch')
    model = RecentAlignmentModel(cli.method, len(vocab), text, z_slices=args.z_slices).to(device)
    loaders = {name: base.make_loader(group, args, name) for name, group in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.epochs)
    out.mkdir(parents=True, exist_ok=True)
    settings = {'method': cli.method, 'experiment_name': cli.method, 'seed': cli.seed, 'epochs': cli.epochs, 'protocol_version': BENCHMARK_VERSION,
                'status': 'metadata_adapted_not_original_checkpoint', 'settings': prepared,
                'text_cache_sha256': text_metadata['sha256'],
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'learning_rate': args.lr, 'weight_decay': args.weight_decay,
                'upstream': json.loads((Path(__file__).parent / 'third_party' / 'ALIGNMENT_SOURCES.json').read_text())}
    write_json(out / 'config.json', settings)
    write_json(out / 'anchor_vocab.json', vocab)
    history, best, steps = [], -float('inf'), 0
    started = time.time()
    for epoch in range(1, cli.epochs + 1):
        model.train()
        totals = []
        for batch in loaders['train']:
            masks = batch.get('region_masks')
            result = model(batch['images'].to(device), region_masks=masks.to(device) if masks is not None else None,
                           freeze_graph=epoch <= args.graph_warmup_epochs)
            positives, known, ids = label_masks(batch['subject_id'], lookup, vocab, keys, device)
            loss = model.alignment_loss(result, positives, known)
            if cli.method in ('pasa_full', 'pasa_text_control'):
                local = result['local']
                node_ids = [target for target in ids for _ in range(local.shape[1])]
                loss = loss + args.lambda_anchor * base.anchor_center_loss(local.reshape(-1, local.shape[-1]), node_ids, result['candidates'])
                auxiliary = result['encoder_output']['losses']
                for name in ('cons', 'decouple', 'leak', 'diff', 'diff_norm', 'gate_entropy', 'load_balance'):
                    weight = getattr(args, 'lambda_' + name)
                    if name == 'cons':
                        weight *= base.graph_cons_scale(epoch, args.graph_warmup_epochs)
                    loss = loss + weight * auxiliary[name]
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            steps += 1
            totals.append(float(loss.detach()))
        scheduler.step()
        validation, _ = evaluate(model, loaders['val'], device, lookup, vocab, keys)
        if not np.isfinite(validation['map']):
            raise ValueError('Validation endpoint cannot be scored')
        history.append({'epoch': epoch, 'loss': float(np.mean(totals)), 'validation': validation})
        write_json(out / 'history.json', history)
        if validation['map'] > best:
            best = validation['map']
            torch.save(model.state_dict(), out / 'best.pt')
        print(f'{cli.method} seed={cli.seed} epoch={epoch} loss={np.mean(totals):.4f} val_mAP={validation["map"]:.4f}', flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    metrics, records = evaluate(model, loaders['test'], device, lookup, vocab, keys)
    metrics.update(method=cli.method, seed=cli.seed, training_steps=steps, wall_seconds=time.time() - started,
                   trainable_parameters=settings['trainable_parameters'], peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
    write_json(out / 'patient_score_records.json', {'protocol_version': BENCHMARK_VERSION, **records})
    write_json(out / 'test_metrics.json', metrics)
    frequencies = np.ones(len(vocab), dtype=float)
    for case in splits['train']:
        for anchor in base.semantic_anchors(case['metadata']):
            if anchor['key'] in keys:
                frequencies[keys[anchor['key']]] += 1
    test_subjects = [str(case['subject_id']) for case in splits['test']]
    prior, prior_records = score_patients(np.tile(np.log(frequencies), (len(test_subjects), 1)),
                                         test_subjects, lookup, vocab, keys)
    prior.update(method='train_frequency', seed=None, wall_seconds=0., trainable_parameters=0)
    write_json(out / 'train_prior_metrics.json', prior)
    write_json(out / 'train_prior_score_records.json', {'protocol_version': BENCHMARK_VERSION, **prior_records})
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
