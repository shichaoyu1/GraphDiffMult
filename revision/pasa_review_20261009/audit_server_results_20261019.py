"""Read-only audit of returned server results. No training or result rewriting."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semantic_evaluation import score_retrieval_metrics, retrieval_metrics, binary_auc

DATA = ROOT / 'output/20261019_AAAI_review/sota_baline_output'
OUT = Path(__file__).parent / 'server_audit_20261019.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def patient_hits(records):
    grouped = collections.defaultdict(list)
    for sid, scores, targets, valid in zip(records['subject_ids'], records['scores'], records['target_ids'], records['valid_ids']):
        if targets:
            grouped[sid].append(float(max(valid, key=lambda i: scores[i]) in targets))
    return {sid: float(np.mean(vals)) for sid, vals in grouped.items()}


result = {'campaigns': {}, 'recent_runs': [], 'recent_means': {}, 'paired_patient_bootstrap': {}}
for folder in sorted(DATA.iterdir()):
    if not (folder / 'prepared.json').exists():
        continue
    splits = read(folder / 'splits.json')
    groups = collections.defaultdict(list)
    for p in sorted(folder.glob('runs/*/*/attempt_*/test_metrics.json')):
        m = read(p)
        old_records = p.parent / 'patient_level_records.json'
        if old_records.exists():
            x = read(old_records)
            rescored = retrieval_metrics(x['query_vectors'], x['query_targets'], x['prototypes'],
                subject_ids=x['subject_ids'], valid_ids=x['query_valid_ids'],
                positive_counts=x['query_positive_counts'], field_names=x['query_fields'])
            assert all(np.isclose(rescored[k], m[k]) for k in ['hit@1', 'map', 'mrr', 'pair_auc'])
        groups[p.parts[-4] + '/' + p.parts[-3].rsplit('_s', 1)[0]].append({
            'run': str(p.relative_to(DATA)), 'complete': (p.parent / 'complete.json').exists(),
            'rescored_from_vectors': old_records.exists(),
            **{k: m.get(k) for k in ['hit@1', 'map', 'biomarker_macro_auc', 'wall_seconds']}})
    result['campaigns'][folder.name] = {
        'split_sizes': {k: len(v) for k, v in splits.items()},
        'split_overlaps': {a+'_'+b: len(set(splits[a]) & set(splits[b])) for a in splits for b in splits if a < b},
        'groups': dict(groups)}

campaign = DATA / 'sota_alignmentv2_20261009'
identity = read(campaign / 'prepared.json')['identity']
result['recent_sources_match_b4cfc9c'] = {}
for name, expected in identity['source_sha256'].items():
    blob = subprocess.run(['git', 'show', 'b4cfc9c:'+name], cwd=ROOT, capture_output=True, check=True).stdout
    result['recent_sources_match_b4cfc9c'][name] = hashlib.sha256(blob).hexdigest() == expected

hits = collections.defaultdict(list)
for p in sorted(campaign.glob('runs/sota/*/attempt_*/test_metrics.json')):
    m, x = read(p), read(p.parent / 'patient_score_records.json')
    prior = read(p.parent / 'train_prior_score_records.json')
    recomputed = score_retrieval_metrics(x['scores'], x['target_ids'], subject_ids=x['subject_ids'],
        valid_ids=x['valid_ids'], positive_counts=x['positive_counts'], field_names=x['field_names'])
    assert all(np.isclose(recomputed[k], m[k]) for k in ['hit@1', 'map', 'mrr', 'pair_auc'])
    predictions, labels = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    agree = count = 0
    auc_input = collections.defaultdict(list)
    for i, (scores, target, valid, field) in enumerate(zip(x['scores'], x['target_ids'], x['valid_ids'], x['field_names'])):
        if not target:
            continue
        pred = max(valid, key=lambda j: scores[j])
        agree += pred == max(valid, key=lambda j: prior['scores'][i][j])
        count += 1
        predictions[field][pred] += 1
        labels[field][target[0]] += 1
        assert len(valid) == 2 and len(target) == 1
        auc_input[field].append((int(target[0] == valid[0]), scores[valid[0]]-scores[valid[1]]))
    for field, pairs in auc_input.items():
        auc = binary_auc(*zip(*pairs))
        assert np.isclose(auc, m['biomarker_auc_by_field'][field]['macro_one_vs_rest_auc'])
    h = read(p.parent / 'history.json')
    vals = [row['validation']['map'] for row in h]
    hits[m['method']].append(patient_hits(x))
    row = {'method': m['method'], 'seed': m['seed'], 'hit@1': m['hit@1'], 'map': m['map'],
        'biomarker_macro_auc': m['biomarker_macro_auc'],
        'field_auc': {k: v['macro_one_vs_rest_auc'] for k, v in m['biomarker_auc_by_field'].items()},
        'prior_top1_agreement': [agree, count], 'prediction_counts': dict(predictions), 'label_counts': dict(labels),
        'selected_epoch_from_history': vals.index(max(vals))+1, 'epochs': len(h),
        'unique_validation_map': len(set(vals)), 'metrics_recomputed': True}
    result['recent_runs'].append(row)

for method in hits:
    rows = [r for r in result['recent_runs'] if r['method'] == method]
    result['recent_means'][method] = {k: {'mean': float(np.mean([r[k] for r in rows])),
        'seed_sd': float(np.std([r[k] for r in rows], ddof=1))} for k in ['hit@1', 'map', 'biomarker_macro_auc']}

# Exploratory intervals: resample patients after averaging the three trained seeds.
# Conditional on this split and these fitted models; not independent external confirmation.
ids = sorted(hits['pasa_full'][0])
rng = np.random.default_rng(20261019)
draws = rng.integers(0, len(ids), size=(10000, len(ids)))
base = np.mean([[h[s] for s in ids] for h in hits['pasa_full']], axis=0)
for method, repeats in hits.items():
    delta = np.mean([[h[s] for s in ids] for h in repeats], axis=0) - base
    bounds = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    result['paired_patient_bootstrap'][method + '_minus_pasa_full'] = {
        'hit@1_difference': float(delta.mean()), 'percentile_95_interval': bounds.tolist(),
        'patients': len(ids), 'resamples': 10000,
        'scope': 'exploratory; conditional on fitted seeds and fixed test split; no multiplicity correction'}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'means': result['recent_means'], 'paired': result['paired_patient_bootstrap']}, indent=2))
