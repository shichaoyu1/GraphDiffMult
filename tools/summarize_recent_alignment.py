"""Matched patient-level bootstrap of recent-method comparisons, fixed trained seeds."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))


def patient_aps(records):
    import numpy as np
    from semantic_evaluation import score_retrieval_metrics
    patients = sorted(set(records['subject_ids']))
    results = {}
    scores = np.asarray(records['scores'])
    for patient in patients:
        indices = [i for i, sid in enumerate(records['subject_ids']) if sid == patient]
        result = score_retrieval_metrics(scores[indices], [records['target_ids'][i] for i in indices],
            subject_ids=[patient] * len(indices), valid_ids=[records['valid_ids'][i] for i in indices],
            positive_counts=[records['positive_counts'][i] for i in indices],
            field_names=[records['field_names'][i] for i in indices])
        if np.isfinite(result['map']):
            results[patient] = result['map']
    return results


def summarize(root, methods, seeds, n_bootstrap=2000):
    import numpy as np
    from tools.run_pasa_server import write_json
    runs, vectors = {}, {}
    common_signature = None
    common_epochs = None
    prior_metrics, prior_vector = None, None
    for method in methods:
        vectors[method], runs[method] = [], []
        for seed in seeds:
            attempts = sorted((root / 'runs' / 'sota' / f'{method}_s{seed}').glob('attempt_*/complete.json'))
            if len(attempts) != 1:
                raise ValueError(f'Need exactly one completed run: {method} seed={seed}')
            directory = attempts[0].parent
            if prior_metrics is None:
                prior_metrics = json.loads((directory / 'train_prior_metrics.json').read_text(encoding='utf-8'))
                prior_vector = patient_aps(json.loads((directory / 'train_prior_score_records.json').read_text(encoding='utf-8')))
            config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            if config['method'] != method or config['seed'] != seed:
                raise ValueError('Run directory and recorded method/seed do not match')
            if common_epochs is not None and common_epochs != config['epochs']:
                raise ValueError('Cannot compare different training epoch budgets')
            common_epochs = config['epochs']
            signature = json.dumps({'settings': config['settings'], 'text_cache': config['text_cache_sha256']}, sort_keys=True)
            if common_signature is not None and signature != common_signature:
                raise ValueError('Cannot compare different protocol/text/cohort signatures')
            common_signature = signature
            metrics = json.loads((directory / 'test_metrics.json').read_text(encoding='utf-8'))
            records = json.loads((directory / 'patient_score_records.json').read_text(encoding='utf-8'))
            patient_results = patient_aps(records)
            if not patient_results or not np.isclose(np.mean(list(patient_results.values())), metrics['map']):
                raise ValueError('Saved scores do not reproduce the reported primary metric')
            vectors[method].append(patient_results)
            runs[method].append(metrics)
    methods = list(methods) + ['train_frequency']
    vectors['train_frequency'] = [prior_vector] * len(seeds)
    runs['train_frequency'] = [prior_metrics] * len(seeds)
    patients = sorted(vectors[methods[0]][0])
    if any(sorted(mapping) != patients for series in vectors.values() for mapping in series):
        raise ValueError('Methods/seeds have different scorable test patients')
    if not patients:
        raise ValueError('No scorable test patients')
    arrays = {method: np.asarray([[row[patient] for patient in patients] for row in series])
              for method, series in vectors.items()}
    paired = {}
    for reference in ('pasa_full', 'pasa_text_control'):
        if reference not in arrays:
            continue
        for method in methods:
            if method == reference:
                continue
            delta = arrays[method] - arrays[reference]
            patient_delta = delta.mean(axis=0)
            rng = np.random.default_rng(20261009)
            samples = [float(patient_delta[rng.integers(len(patients), size=len(patients))].mean()) for _ in range(n_bootstrap)]
            paired[f'{method} - {reference}'] = {'macro_map_difference': float(delta.mean()),
                'per_seed_differences': delta.mean(axis=1).tolist(),
                'patient_bootstrap_ci95': np.percentile(samples, [2.5, 97.5]).tolist()}
    write_json(root / 'paired_comparisons.json', {'protocol_version': 'pasa_recent_alignment_v1',
        'status': 'metadata_adapted_comparison_not_original_benchmark', 'seeds': seeds,
        'patients': len(patients), 'primary_endpoint': 'patient_molecular_macro_map',
        'ci_scope': 'patient sampling with all trained seeds fixed; exploratory multiple comparisons',
        'n_bootstrap': n_bootstrap, 'comparisons': paired})
    rows = []
    for method in methods:
        row = {'method': method, 'seeds': 'train-only deterministic prior' if method == 'train_frequency' else ' '.join(map(str, seeds)), 'n_patients': len(patients)}
        for metric in ('map', 'mrr', 'hit@1', 'recall@1', 'pair_auc', 'biomarker_macro_auc', 'wall_seconds', 'trainable_parameters'):
            values = np.asarray([r[metric] for r in runs[method]], dtype=float)
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    with (root / 'sota_mean.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {root / "sota_mean.csv"} and paired_comparisons.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign_root', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    summarize(Path(os.path.abspath(args.campaign_root)), args.methods, args.seeds)
