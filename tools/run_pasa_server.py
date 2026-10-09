"""AutoDL runner for frozen PASA v2 comparisons; no shell interpolation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.path.abspath(__file__)).parent.parent
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(ROOT))
SOURCES = ['train_semantic_alignment.py', 'semantic_evaluation.py', 'experiment_model.py',
           'experiment_dataset.py', 'dataset.py', 'semantic_graph_visualize.py',
           'tools/run_pasa_server.py', 'utils/bootstrap_semantic_5seed.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def parser(stages=('prepare', 'smoke', 'core', 'ablation', 'all')):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data_root', default='/root/autodl-tmp/dataset/UTSW-Glioma')
    result.add_argument('--metadata_tsv', default='/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv')
    result.add_argument('--output_root', required=True)
    result.add_argument('--stage', choices=stages, default='smoke')
    result.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    result.add_argument('--epochs', type=int, default=30)
    result.add_argument('--batch_size', type=int, default=2)
    result.add_argument('--roi_size', type=int, default=96)
    result.add_argument('--z_slices', type=int, default=7)
    result.add_argument('--num_workers', type=int, default=2)
    result.add_argument('--split_seed', type=int, default=42)
    result.add_argument('--sample_seed', type=int, default=42)
    result.add_argument('--train_ratio', type=float, default=.7)
    result.add_argument('--val_ratio', type=float, default=.1)
    result.add_argument('--cpu', action='store_true')
    result.add_argument('--dry_run', action='store_true')
    return result


def prepare(args):
    import torch
    import train_semantic_alignment as training
    from experiment_dataset import stratified_split
    from semantic_evaluation import PROTOCOL_VERSION, validate_patient_splits
    data_root = Path(os.path.abspath(args.data_root))
    metadata = Path(os.path.abspath(args.metadata_tsv))
    if not data_root.is_dir() or not metadata.is_file():
        raise ValueError('Image directory or metadata TSV does not exist')
    if not args.cpu and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; check the server PyTorch installation before running')
    cases = training.discover_semantic_cases(str(data_root), metadata_tsv=str(metadata), seed=args.sample_seed)
    if getattr(args, 'reference_splits', None):
        frozen = json.loads(Path(args.reference_splits).read_text(encoding='utf-8'))
        lookup = {str(case['subject_id']): case for case in cases}
        splits = {name: [lookup[str(sid)] for sid in frozen[name]] for name in ('train', 'val', 'test')}
        if {str(case['subject_id']) for group in splits.values() for case in group} != set(lookup):
            raise ValueError('Reference splits do not cover exactly this cohort')
    else:
        splits = stratified_split(cases, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.split_seed)
    validate_patient_splits(splits)
    vocab, key_to_id = training.build_anchor_vocab(splits['train'])
    endpoint = argparse.Namespace(evaluation_fields='molecular')
    coverage = {}
    for name, group in splits.items():
        specs = [spec for case in group for _, spec in training.evaluation_specs(case['metadata'], vocab, key_to_id, endpoint)]
        known = sum(spec['positive_count'] for spec in specs)
        if not known:
            raise ValueError(f'{name} has no molecular labels')
        coverage[name] = {'patients': len(group), 'known_fields': known,
                          'missing_fields': sum(spec['positive_count'] == 0 for spec in specs),
                          'oov_positives': sum(len(spec['oov_positive_keys']) for spec in specs)}
    identity = {'protocol_version': PROTOCOL_VERSION, 'data_root': str(data_root),
                'metadata_tsv': str(metadata), 'metadata_sha256': digest(metadata),
                'source_sha256': {name: digest(ROOT / name) for name in SOURCES + getattr(args, 'extra_source_files', [])},
                'benchmark_settings': getattr(args, 'benchmark_settings', None),
                'split_seed': args.split_seed, 'sample_seed': args.sample_seed,
                'train_ratio': args.train_ratio, 'val_ratio': args.val_ratio,
                'roi_size': args.roi_size, 'z_slices': args.z_slices,
                'epochs': args.epochs, 'batch_size': args.batch_size, 'cpu': args.cpu,
                'splits': {name: [str(case['subject_id']) for case in group] for name, group in splits.items()},
                'vocab_keys': [anchor['key'] for anchor in vocab]}
    root = Path(os.path.abspath(args.output_root))
    manifest = root / 'prepared.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['identity'] != identity:
            raise ValueError('Prepared protocol differs in code/data/settings. Use a new output_root, preserve this campaign.')
        saved_splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
        if saved_splits != identity['splits']:
            raise ValueError('Frozen split file was modified')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('output_root is not empty and has no prepared manifest; choose a fresh directory')
        write_json(root / 'splits.json', identity['splits'])
        write_json(manifest, {'identity': identity, 'coverage': coverage})
        write_json(root / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (root / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    print(json.dumps({'prepared': str(root), 'coverage': coverage}, ensure_ascii=False, indent=2))


def jobs(args):
    core = [('unit_retrieval', 'clip', ['--lambda_anchor', '0']),
            ('pooled_global', 'global_clip', ['--lambda_anchor', '0']),
            ('unit_multilabel', 'multilabel', ['--lambda_anchor', '0']),
            ('unit_single_positive', 'clip', ['--lambda_anchor', '0', '--alignment_objective', 'single_positive'])]
    ablations = [('full', 'full', []), ('without_pathology_supervision', 'no_anchor', []),
                 ('without_center_loss', 'no_anchor_loss', []), ('without_graph', 'no_graph', []),
                 ('without_diffusion', 'graph_only', [])]
    if args.stage == 'smoke':
        groups = [('smoke', [core[0], ablations[0]], [args.seeds[0]], 1)]
    else:
        groups = []
        if args.stage in ('core', 'all'):
            groups.append(('core', core, args.seeds, args.epochs))
        if args.stage in ('ablation', 'all'):
            groups.append(('ablation', ablations, args.seeds, args.epochs))
    for group, definitions, seeds, epochs in groups:
        for name, variant, extra in definitions:
            for seed in seeds:
                command = [sys.executable, '-u', '-B', str(ROOT / 'train_semantic_alignment.py'),
                    '--data_root', os.path.abspath(args.data_root), '--metadata_tsv', os.path.abspath(args.metadata_tsv),
                    '--variant', variant, '--experiment_name', name, '--evaluation_fields', 'molecular',
                    '--target_policy', 'all_patient_anchors', '--splits_file', str(Path(os.path.abspath(args.output_root)) / 'splits.json'),
                    '--epochs', str(epochs), '--batch_size', str(args.batch_size), '--roi_size', str(args.roi_size),
                    '--z_slices', str(args.z_slices), '--num_workers', str(args.num_workers),
                    '--seed', str(seed), '--split_seed', str(args.split_seed), '--sample_seed', str(args.sample_seed), '--augment']
                if args.cpu:
                    command.append('--cpu')
                yield group, name, seed, command + extra


def run_job(root, group, name, seed, command, expected_protocol='pasa_patient_metadata_v2'):
    job_root = root / 'runs' / group / f'{name}_s{seed}'
    attempts = sorted(job_root.glob('attempt_*')) if job_root.exists() else []
    for attempt in attempts:
        complete = attempt / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text(encoding='utf-8'))
            if saved['base_command'] != command:
                raise ValueError('Completed job command changed; use a new campaign')
            print(f'SKIP completed {name} seed={seed}', flush=True)
            return
    attempt = job_root / f'attempt_{len(attempts) + 1:03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    actual = command + ['--out_dir', str(attempt)]
    write_json(attempt / 'launch.json', {'command': actual})
    started = time.time()
    print(f'RUN {group}/{name} seed={seed} -> {attempt}', flush=True)
    with (attempt / 'train.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(actual, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=dict(os.environ, MPLBACKEND='Agg'))
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
    if code != 0:
        raise RuntimeError(f'Job failed ({code}). See {attempt / "train.log"}. Rerunning creates a fresh attempt.')
    metrics = json.loads((attempt / 'test_metrics.json').read_text(encoding='utf-8'))
    if metrics.get('protocol_version') != expected_protocol or not metrics.get('scored_queries'):
        raise ValueError('Job finished without scorable v2 test output')
    write_json(attempt / 'complete.json', {'base_command': command, 'duration_seconds': time.time() - started})


def summarize(root):
    rows = []
    for complete in sorted((root / 'runs').glob('*/*/attempt_*/complete.json')):
        directory = complete.parent
        metrics = json.loads((directory / 'test_metrics.json').read_text(encoding='utf-8'))
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        rows.append({'stage': directory.parent.parent.name, 'experiment': config['experiment_name'],
                     'seed': config['seed'], 'run_dir': str(directory),
                     **{key: metrics.get(key) for key in ['hit@1', 'recall@1', 'map', 'mrr', 'pair_auc',
                                                        'positive_negative_similarity_gap', 'scored_queries']}})
    if rows:
        with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parser().parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Seeds must be unique')
    if min(args.epochs, args.batch_size, args.roi_size, args.z_slices) <= 0 or args.num_workers < 0:
        raise ValueError('Invalid resource settings')
    if args.dry_run:
        for group, name, seed, command in jobs(args):
            print(json.dumps({'stage': group, 'experiment': name, 'seed': seed, 'command': command}, ensure_ascii=False))
        return
    prepare(args)
    root = Path(os.path.abspath(args.output_root))
    for group, name, seed, command in jobs(args):
        try:
            run_job(root, group, name, seed, command)
        finally:
            summarize(root)


if __name__ == '__main__':
    main()
