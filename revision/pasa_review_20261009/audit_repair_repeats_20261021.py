"""Recompute the returned three-seed pilot; never load MRI or modify raw results."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semantic_evaluation import binary_auc, score_retrieval_metrics

BASE = ROOT / 'output/20261019_AAAI_review'
REFERENCE = BASE / 'pasa_repair_pilot_20261020_095259_432520'
REPEATS = BASE / 'pasa_repair_repeats_20261020_171817_445198'
OUT = Path(__file__).parent / 'repair_repeats_audit_20261021'
METHODS = ('whole_roi_mass', 'whole_roi_field', 'region_pool_mass', 'region_pool_field')
FIELDS = ('IDH', 'MGMT', '1p19Q CODEL')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def statistics(values):
    return {'mean': float(np.mean(values)), 'training_seed_sd': float(np.std(values, ddof=1)),
            'min': float(min(values)), 'max': float(max(values))}


def label_signature(records):
    return {(s, f): (tuple(t), tuple(v), n) for s, f, t, v, n in zip(
        records['subject_ids'], records['field_names'], records['target_ids'],
        records['valid_ids'], records['positive_counts'])}


def rescore(payload, subjects, vocab, signature=None):
    records, metrics = payload['records'], payload['metrics']
    assert len(records['subject_ids']) == len(subjects) * len(FIELDS)
    assert set(records['subject_ids']) == set(subjects)
    assert set(Counter(zip(records['subject_ids'], records['field_names'])).values()) == {1}
    assert set(records['field_names']) == set(FIELDS)
    assert label_signature(records) == (signature or label_signature(records))
    scores = np.asarray(records['scores'])
    assert scores.shape == (len(subjects) * len(FIELDS), len(vocab))
    assert np.isfinite(scores).all()
    # Every field uses the same patient's score vector, not a different input.
    by_subject = defaultdict(list)
    for subject, score in zip(records['subject_ids'], records['scores']):
        by_subject[subject].append(score)
    assert all(all(row == rows[0] for row in rows) for rows in by_subject.values())
    calculated = score_retrieval_metrics(scores, records['target_ids'],
        subject_ids=records['subject_ids'], valid_ids=records['valid_ids'],
        positive_counts=records['positive_counts'], field_names=records['field_names'])
    for key, value in calculated.items():
        if key in ('protocol_version', 'aggregation'):
            continue
        expected = metrics[key]
        if isinstance(value, float) and not np.isfinite(value):
            assert expected is None, key
        else:
            assert expected is not None and np.isclose(value, expected, rtol=0, atol=1e-12), key
    results = {}
    for field in FIELDS:
        ids = [i for i, anchor in enumerate(vocab) if anchor['field'] == field]
        assert len(ids) == 2
        rows = []
        for score, target, valid, count, name in zip(records['scores'], records['target_ids'],
                records['valid_ids'], records['positive_counts'], records['field_names']):
            if name != field:
                continue
            if not target:
                assert count == 0 and not valid
                continue
            assert len(target) == count == 1 and valid == ids
            margin = score[ids[0]] - score[ids[1]]
            rows.append((int(target[0] == ids[0]), int(margin >= 0), margin))
        y, prediction, margin = np.asarray(rows).T
        auc = binary_auc(y, margin)
        assert np.isclose(auc, metrics['biomarker_auc_by_field'][field]['macro_one_vs_rest_auc'], atol=1e-12)
        # Independent direct pair formula checks the rank-based AUC implementation.
        pair_difference = margin[y == 1, None] - margin[y == 0][None, :]
        direct_auc = np.mean((pair_difference > 0) + .5 * (pair_difference == 0))
        assert np.isclose(auc, direct_auc, rtol=0, atol=1e-12)
        recalls = [float(np.mean(prediction[y == label] == label)) for label in (1, 0)]
        results[field] = {'patients': len(rows), 'class_a': vocab[ids[0]]['key'],
            'class_b': vocab[ids[1]]['key'], 'class_a_count': int(y.sum()),
            'class_b_count': int((1 - y).sum()), 'predicted_a': int(prediction.sum()),
            'true_positive_a': int(np.sum((prediction == 1) & (y == 1))),
            'false_positive_a': int(np.sum((prediction == 1) & (y == 0))),
            'recall_a': recalls[0], 'recall_b': recalls[1],
            'balanced_accuracy': float(np.mean(recalls)), 'auc': auc,
            'exact_score_ties': int(np.sum(margin == 0))}
    assert np.isclose(np.mean([r['auc'] for r in results.values()]), metrics['biomarker_macro_auc'], atol=1e-12)
    assert metrics['total_queries'] == 189 and metrics['scored_queries'] == 120
    assert metrics['skipped_no_positive'] == 69 and metrics['unavailable_positive_count'] == 0
    # The binary candidate task makes these dependent endpoints.
    assert np.isclose(metrics['map'], (1 + metrics['hit@1']) / 2)
    assert np.isclose(metrics['map'], metrics['mrr'])
    return results


def main():
    protocol = read(REFERENCE / 'protocol.json')
    vocab = read(REFERENCE / 'anchor_vocab.json')
    manifests = read(REFERENCE / 'inputs.json')
    assert len(protocol['splits']['train']) == 435 and len(protocol['splits']['val']) == 63
    assert not set(protocol['splits']['train']) & set(protocol['splits']['val'])
    assert set(manifests) == set(protocol['splits']['train']) | set(protocol['splits']['val'])
    assert all(any(row['region_present']) for row in manifests.values())
    identity = read(REPEATS / 'repeat_config.json')['identity']
    for name, digest in identity['reference_sha256'].items():
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == digest
    result = {'version': 'pasa_repair_repeats_audit_20261021',
        'scope': 'Three training seeds on one reused development validation split; selected validation, not test.',
        'protocol_version': protocol['version'], 'seeds': [42, 43, 44],
        'split_sizes': {key: len(value) for key, value in protocol['splits'].items()},
        'empty_retained_inputs': 0, 'input_coverage': protocol['input_coverage'],
        'test_evaluated': False, 'runs': [], 'method_summary': {}, 'paired_comparisons': {}}
    signature = None
    roots = {42: REFERENCE, **{s: REPEATS / f'seed_{s}/attempt_001' for s in (43, 44)}}
    audit_files = {}
    for seed, folder in roots.items():
        child = read(folder / 'protocol.json')
        child_base = {k: v for k, v in child.items() if k != 'repeat_reference'}
        assert child_base == {**protocol, 'seed': seed}
        for name in ('inputs.json', 'anchor_vocab.json'):
            assert (folder / name).read_bytes() == (REFERENCE / name).read_bytes()
        for method in METHODS:
            run = folder / method
            config = read(run / 'config.json')
            complete = read(run / 'complete.json')
            assert config['protocol'] == child and config['parameter_count'] == 409056
            assert complete['training_steps'] == 2180 and complete['validation_replay_equal']
            assert complete['test_evaluated'] is False
            history = read(run / 'history.json')
            assert [r['epoch'] for r in history] == list(range(1, 11))
            for epoch in history:
                payload = read(run / f'validation_epoch_{epoch["epoch"]:03d}.json')
                if signature is None:
                    signature = label_signature(payload['records'])
                rescore(payload, child['splits']['val'], vocab, signature)
                assert epoch['validation'] == payload['metrics']
            selected = int(np.argmax([r['validation']['biomarker_macro_auc'] for r in history])) + 1
            best = read(run / 'validation_best_auc.json')
            fields = rescore(best, child['splits']['val'], vocab, signature)
            assert selected == best['selected_epoch'] == complete['selected_epoch']
            assert best['records'] == read(run / f'validation_epoch_{selected:03d}.json')['records']
            result['runs'].append({'seed': seed, 'method': method, 'selected_epoch': selected,
                'parameters': config['parameter_count'], 'training_steps': complete['training_steps'],
                'auc': best['metrics']['biomarker_macro_auc'], 'map': best['metrics']['map'],
                'hit@1': best['metrics']['hit@1'], 'fields': fields,
                'last_auc': history[-1]['validation']['biomarker_macro_auc'],
                'auc_trajectory': [r['validation']['biomarker_macro_auc'] for r in history],
                'map_trajectory': [r['validation']['map'] for r in history],
                'loss_trajectory': [r['loss'] for r in history],
                'encoded_training_views_per_epoch': [r['encoded_training_views'] for r in history],
                'seconds': complete['seconds']})
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '.ipynb_checkpoints' not in path.parts:
                audit_files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [*REPEATS.glob('*.json'), BASE / (REFERENCE.name + '.log'), BASE / (REPEATS.name + '.log')]:
        if path.is_file():
            audit_files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for method in METHODS:
        rows = [r for r in result['runs'] if r['method'] == method]
        result['method_summary'][method] = {'auc': statistics([r['auc'] for r in rows]),
            'last_auc': statistics([r['last_auc'] for r in rows]),
            'map': statistics([r['map'] for r in rows]),
            'fields': {f: {'auc': statistics([r['fields'][f]['auc'] for r in rows]),
                'recall_a': statistics([r['fields'][f]['recall_a'] for r in rows]),
                'predicted_a_by_seed': [r['fields'][f]['predicted_a'] for r in rows]}
                for f in FIELDS}}
    indexed = {(r['seed'], r['method']): r for r in result['runs']}
    pairs = {'field_minus_mass_whole_roi': ('whole_roi_field', 'whole_roi_mass'),
        'field_minus_mass_region_pool': ('region_pool_field', 'region_pool_mass'),
        'region_minus_whole_field': ('region_pool_field', 'whole_roi_field'),
        'region_minus_whole_mass': ('region_pool_mass', 'whole_roi_mass')}
    for name, (a, b) in pairs.items():
        deltas = [indexed[(s, a)]['auc'] - indexed[(s, b)]['auc'] for s in (42, 43, 44)]
        result['paired_comparisons'][name] = {'by_seed': dict(zip(('42', '43', '44'), deltas)),
            **statistics(deltas), 'positive_directions': sum(x > 0 for x in deltas),
            'field_differences_by_seed': {str(s): {f: indexed[(s, a)]['fields'][f]['auc'] -
                indexed[(s, b)]['fields'][f]['auc'] for f in FIELDS} for s in (42, 43, 44)}}
    result['checks'] = {'rescored_epoch_files': 120, 'rescored_selected_files': 12,
        'unique_patient_field_rows_per_epoch': 189, 'known_patient_field_rows_per_epoch': 120,
        'missing_patient_field_rows_per_epoch': 69, 'identical_label_and_mask_signatures': True,
        'direct_pairwise_auc_crosscheck': True, 'finite_raw_scores': True,
        'strict_first_maximum_selection': True}
    result['total_training_hours'] = sum(r['seconds'] for r in result['runs']) / 3600
    result['new_training_hours'] = sum(r['seconds'] for r in result['runs'] if r['seed'] != 42) / 3600
    OUT.mkdir(exist_ok=True)
    save(OUT / 'summary.json', result)
    save(OUT / 'returned_file_sha256.json', audit_files)
    # A small companion notebook reruns the exact audit and exposes the full aggregate.
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'markdown', 'metadata': {}, 'id': 'scope', 'source': [
            '# PASA three-seed returned-results audit\n',
            'Run from the repository or this directory. Reads returned files only; no MRI, test evaluation or training.\n',
            'Three seeds share the same 63 development validation patients. Seed SD is not a patient confidence interval.']},
            {'cell_type': 'code', 'metadata': {}, 'id': 'audit', 'execution_count': None, 'outputs': [],
             'source': ["from pathlib import Path\n", "import runpy, json\n",
                "root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'PROJECT_STATE.md').exists())\n",
                "runpy.run_path(str(root/'revision/pasa_review_20261009/audit_repair_repeats_20261021.py'), run_name='__main__')\n",
                "summary = json.loads((root/'revision/pasa_review_20261009/repair_repeats_audit_20261021/summary.json').read_text(encoding='utf-8'))\n",
                "summary['method_summary']"]}]}
    save(OUT / 'audit.ipynb', notebook)
    print(json.dumps({k: result[k] for k in ('checks', 'method_summary', 'paired_comparisons',
        'total_training_hours', 'new_training_hours')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
