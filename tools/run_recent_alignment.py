"""One-command recent fine-grained alignment comparison on AutoDL."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))
from tools import run_pasa_server as server

METHODS = ['pasa_full', 'pasa_minimal', 'pasa_text_control', 'text_cosine', 'carzero_metadata', 'radzero_metadata']


def main():
    parser = server.parser(stages=('prepare', 'smoke', 'sota'))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--text_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--text_revision', default=None)
    parser.add_argument('--reference_splits', default=None)
    args = parser.parse_args()
    if not args.methods or len(args.methods) != len(set(args.methods)) or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Methods and seeds must be unique')
    if min(args.epochs, args.batch_size, args.roi_size, args.z_slices) <= 0 or args.num_workers < 0:
        raise ValueError('Invalid resource settings')
    root = Path(os.path.abspath(args.output_root))
    default_reference = Path('/root/autodl-tmp/pasa_runs/pasa_v2_20261009/splits.json')
    if args.reference_splits is None and default_reference.is_file():
        args.reference_splits = str(default_reference)
        print(f'Reusing previous frozen patient IDs: {default_reference}', flush=True)
    args.extra_source_files = ['recent_alignment.py', 'train_recent_alignment.py',
        'tools/run_recent_alignment.py', 'tools/build_alignment_text_cache.py',
        'tools/summarize_recent_alignment.py', 'third_party/carzero/dqn.py',
        'third_party/carzero/transformer_decoder.py', 'third_party/radzero/similarity.py',
        'third_party/radzero/dinov2_blocks.py', 'third_party/ALIGNMENT_SOURCES.json']
    args.benchmark_settings = {'text_model': args.text_model, 'text_revision': args.text_revision,
        'num_workers': args.num_workers, 'version': 'pasa_recent_alignment_v1',
        'primary_endpoint': 'patient_molecular_macro_map', 'raw_report_available': False}
    seeds = args.seeds[:1] if args.stage == 'smoke' else args.seeds
    epochs = 1 if args.stage == 'smoke' else args.epochs
    tasks = [(method, seed, [sys.executable, '-u', '-B', str(ROOT / 'train_recent_alignment.py'),
        '--campaign_root', str(root), '--method', method, '--seed', str(seed), '--epochs', str(epochs)])
        for method in args.methods for seed in seeds]
    if args.stage == 'prepare':
        tasks = []
    if args.dry_run:
        print(json.dumps({'planned_runs': len(tasks), 'stage': args.stage, 'methods': args.methods,
                          'seeds': seeds, 'epochs': epochs, 'frozen_splits': str(root / 'splits.json')}, indent=2))
        return
    server.prepare(args)
    from tools.build_alignment_text_cache import build
    build(root, args.text_model, args.text_revision)
    if args.stage == 'prepare':
        return
    for method, seed, command in tasks:
        try:
            server.run_job(root, args.stage, method, seed, command, expected_protocol='pasa_recent_alignment_v1')
        finally:
            server.summarize(root)
    if args.stage == 'sota':
        from tools.summarize_recent_alignment import summarize
        summarize(root, args.methods, args.seeds)


if __name__ == '__main__':
    main()
