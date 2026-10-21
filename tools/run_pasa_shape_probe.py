"""Frozen CPU segmentation-count probe; original metadata, no MRI/test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from json_results import write_json
from semantic_evaluation import binary_auc

VERSION = 'pasa_shape_probe_v1'
LABELS = ('1', '2', '3', '4', '5')
FIELDS = ('IDH', 'MGMT', '1p19Q CODEL')
CLASS_KEYS = {
    'IDH': ('idh::mutated', 'idh::wild type'),
    'MGMT': ('mgmt::methylated', 'mgmt::unmethylated'),
    '1p19Q CODEL': ('1p19q_codel::co-deleted', '1p19q_codel::non co-deleted'),
}
FEATURE_NAMES = (['log1p_total_resized_foreground_count']
    + [f'raw_label_{label}_fraction_of_foreground' for label in LABELS]
    + [f'raw_label_{label}_present' for label in LABELS])
MODEL_PARAMS = dict(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000,
    tol=1e-8, class_weight=None, fit_intercept=True, random_state=None)
PROTOCOL_DOCUMENT = ROOT / 'revision/pasa_review_20261009/SHAPE_PROBE_PROTOCOL_20261021.md'
REFERENCE_FILES = ('protocol.json', 'inputs.json', 'anchor_vocab.json', 'complete.json', 'summary.json')
LABEL_SOURCES = ('dataset.py', 'train_semantic_alignment.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_row(record, z_slices, roi_size):
    """Only the five resized-slab counts enter the eleven fixed features."""
    counts = record['audit']['slab_voxel_counts_after_resize']
    if set(counts) != set(LABELS):
        raise ValueError('Expected all five raw-label counts, with no extra labels')
    values = np.asarray([counts[label] for label in LABELS], dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('Counts must be finite nonnegative integers')
    total = values.sum()
    if not 0 < total <= z_slices * roi_size * roi_size:
        raise ValueError('Foreground count outside the resized-slab bounds')
    present = np.asarray(record['region_present'])
    if present.dtype != np.bool_ or present.shape != (5,) or not np.array_equal(present, values > 0):
        raise ValueError('Presence differs from resized counts')
    return np.concatenate(([np.log1p(total)], values / total, (values > 0).astype(float)))


def build_features(manifest, subject_ids, protocol):
    return np.stack([feature_row(manifest[sid], protocol['z_slices'], protocol['roi_size'])
        for sid in subject_ids])


def recover_labels(records, vocab, expected_subjects):
    """Recover only observed categorical labels; missing is None, never zero."""
    expected = set(expected_subjects)
    if len(expected) != len(expected_subjects):
        raise ValueError('Duplicated expected patients')
    keys = [a['key'] for a in vocab]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicated vocabulary keys')
    for field, candidates in CLASS_KEYS.items():
        if {a['key'] for a in vocab if a['field'] == field} != set(candidates):
            raise ValueError('Canonical molecular candidate set differs: ' + field)
    columns = [records[name] for name in
        ('subject_ids', 'field_names', 'target_ids', 'valid_ids', 'positive_counts')]
    if len({len(column) for column in columns}) != 1:
        raise ValueError('Record columns have different lengths')
    labels = {sid: {} for sid in expected_subjects}
    for sid, field, target, valid, count in zip(*columns):
        if sid not in expected or field not in FIELDS or field in labels[sid]:
            raise ValueError('Unexpected or duplicated patient/field record')
        if not target:
            if valid or count != 0:
                raise ValueError('Missing targets must also have empty candidates and zero positives')
            labels[sid][field] = None
            continue
        if (len(target) != 1 or len(valid) != 2 or count != 1
                or any(not isinstance(i, int) or i < 0 or i >= len(vocab) for i in valid + target)
                or len(set(valid)) != 2 or target[0] not in valid):
            raise ValueError('Expected one observed class among two molecular candidates')
        if {keys[i] for i in valid} != set(CLASS_KEYS[field]):
            raise ValueError('Candidate keys differ from the canonical molecular classes')
        labels[sid][field] = int(keys[target[0]] == CLASS_KEYS[field][0])
    if any(set(row) != set(FIELDS) for row in labels.values()):
        raise ValueError('Each patient must have exactly three field records, including missing fields')
    return labels


def validate_reference_binding(reference, binding):
    """Bind the seed-42 reference to the five hashes retained by the completed repeats."""
    identity = read(binding)['identity']
    if identity['version'] != 'pasa_repair_repeats_v1' or set(identity['reference_sha256']) != set(REFERENCE_FILES):
        raise ValueError('Incompatible reference binding')
    for name, expected in identity['reference_sha256'].items():
        if digest(reference / name) != expected:
            raise ValueError('Frozen reference artifact changed: ' + name)


def labels_from_metadata(metadata, subject_ids):
    """Use the alignment's existing canonical keys, without synonym relabeling."""
    from train_semantic_alignment import semantic_anchors
    labels = {}
    for sid in subject_ids:
        if sid not in metadata:
            raise ValueError('Frozen patient missing from original metadata: ' + sid)
        anchors = {a['field']: a['key'] for a in semantic_anchors(metadata[sid], include_pathology=False)}
        labels[sid] = {}
        for field in FIELDS:
            key = anchors.get(field)
            if key is not None and key not in CLASS_KEYS[field]:
                raise ValueError('Observed molecular key outside frozen candidates: ' + field + ' ' + key)
            labels[sid][field] = None if key is None else int(key == CLASS_KEYS[field][0])
    return labels


def load_training_bundle(reference, metadata_path):
    from dataset import load_utsw_metadata
    from train_semantic_alignment import build_anchor_vocab
    protocol = read(reference / 'protocol.json')
    if read(reference / 'complete.json') != {
            'version': 'pasa_repair_pilot_v1', 'stage': 'pilot', 'runs': 4, 'test_evaluated': False}:
        raise ValueError('Reference is not a completed four-arm repair pilot')
    if (protocol['version'] != 'pasa_repair_pilot_v1'
            or protocol['stage'] != 'pilot' or protocol['seed'] != 42
            or protocol['roi_size'] != 96 or protocol['z_slices'] != 7
            or protocol['raw_region_labels'] != [1, 2, 3, 4, 5]):
        raise ValueError('Expected the completed frozen seed-42 7x96x96 repair pilot')
    train_ids, val_ids = protocol['splits']['train'], protocol['splits']['val']
    if (len(train_ids) != 435 or len(val_ids) != 63 or len(set(train_ids)) != 435
            or len(set(val_ids)) != 63 or set(train_ids) & set(val_ids)):
        raise ValueError('Expected the common frozen 435/63 disjoint patients')
    manifest = read(reference / 'inputs.json')
    if set(manifest) != set(train_ids) | set(val_ids):
        raise ValueError('Manifest membership differs from the frozen split')
    for split, ids in protocol['splits'].items():
        if any(manifest[sid]['split'] != split for sid in ids):
            raise ValueError('Manifest split membership differs')
    if digest(metadata_path) != protocol['metadata_sha256']:
        raise ValueError('Original metadata SHA256 differs from the frozen pilot')
    for name in LABEL_SOURCES:
        if digest(ROOT / name) != protocol['source_sha256'][name]:
            raise ValueError('Frozen label-construction source changed: ' + name)
    metadata = load_utsw_metadata(str(metadata_path))
    if any(sid not in metadata for sid in train_ids):
        raise ValueError('Some frozen training patients are absent from original metadata')
    # No discovery function is called: these cases contain labels only, no MRI paths.
    train_cases = [{'subject_id': sid, 'metadata': metadata[sid]} for sid in train_ids]
    rebuilt_vocab, _ = build_anchor_vocab(train_cases)
    if rebuilt_vocab != read(reference / 'anchor_vocab.json'):
        raise ValueError('Training-only rebuilt vocabulary differs from the frozen pilot')
    return protocol, manifest, labels_from_metadata(metadata, train_ids), metadata


def fit_field_models(x_train, train_ids, labels):
    """Fit scaler on training patients only, classifiers on each observed training field."""
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2 or len(x_train) != len(train_ids) or not np.isfinite(x_train).all():
        raise ValueError('Invalid training features')
    scaler = StandardScaler().fit(x_train)
    scaled = scaler.transform(x_train)
    models, fitted = {}, {}
    for field in FIELDS:
        indices = [i for i, sid in enumerate(train_ids) if labels[sid][field] is not None]
        y = np.asarray([labels[train_ids[i]][field] for i in indices], dtype=int)
        if set(y.tolist()) != {0, 1}:
            raise ValueError('Each training field must have both observed classes: ' + field)
        model = LogisticRegression(**MODEL_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(scaled[indices], y)
        if model.classes_.tolist() != [0, 1]:
            raise ValueError('Unexpected fitted class order')
        models[field] = model
        fitted[field] = {'patients': len(indices), 'missing': len(train_ids) - len(indices),
            'positive': int(y.sum()), 'negative': int((y == 0).sum()),
            'fitted_subject_ids': [train_ids[i] for i in indices],
            'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
            'classes': model.classes_.tolist(), 'n_iter': model.n_iter_.tolist()}
    return scaler, models, fitted


def classification_metrics(targets, scores, predictions):
    y, s, p = np.asarray(targets), np.asarray(scores), np.asarray(predictions)
    if len(y) == 0 or set(y.tolist()) != {0, 1}:
        raise ValueError('Validation field needs both observed classes')
    positive, negative = y == 1, y == 0
    sensitivity = float((p[positive] == 1).mean())
    specificity = float((p[negative] == 0).mean())
    return {'patients': int(len(y)), 'positive': int(positive.sum()), 'negative': int(negative.sum()),
        'auroc': float(binary_auc(y, s)), 'predicted_positive': int((p == 1).sum()),
        'true_positive': int(((p == 1) & positive).sum()),
        'positive_recall': sensitivity, 'negative_recall': specificity,
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'accuracy': float((y == p).mean())}


def artifact_paths(reference, binding):
    paths = [reference / name for name in REFERENCE_FILES]
    paths += [reference / 'whole_roi_field/validation_best_auc.json']
    paths += [binding]
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=ROOT / 'output/pasa_repair_pilot_20261020_095259_432520')
    p.add_argument('--binding', type=Path,
        default=ROOT / 'output/pasa_repair_repeats_20261020_171817_445198/repeat_config.json')
    p.add_argument('--metadata', type=Path, required=True, help='Original metadata TSV with the frozen pilot SHA256')
    p.add_argument('--out', type=Path, default=ROOT / 'output/pasa_shape_probe_20261021')
    p.add_argument('--summary-out', type=Path, default=None, help='Optional additional aggregate-only summary')
    args = p.parse_args()
    for name in ('reference', 'binding'):
        path = getattr(args, name).resolve()
        if ROOT not in path.parents:
            raise ValueError('Source evidence must be inside the workspace: ' + str(path))
        setattr(args, name, path)
    args.metadata = args.metadata.resolve()
    out = args.out.resolve()
    if out.exists() or (args.summary_out is not None and args.summary_out.exists()):
        raise ValueError('Output exists; preserve the result and choose a new output and summary path')
    validate_reference_binding(args.reference, args.binding)
    if digest(args.metadata) != read(args.reference / 'protocol.json')['metadata_sha256']:
        raise ValueError('Original metadata SHA256 differs from the frozen pilot')
    artifacts = artifact_paths(args.reference, args.binding)
    frozen = {'version': VERSION, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'post-development exploratory alternative-explanation probe on the reused validation patients',
        'feature_names': FEATURE_NAMES, 'fraction_denominator': 'sum of the five resized-slab foreground counts',
        'feature_source': 'only audit.slab_voxel_counts_after_resize; presence checked against counts',
        'classifier_params': MODEL_PARAMS, 'scaler': 'StandardScaler; fit all 435 training patients only',
        'fit_policy': 'three independent binary molecular classifiers; each fit only observed training labels',
        'selection': 'no tuning, seed selection, epoch selection, threshold fitting or feature selection',
        'default_boundary': 'sklearn predict: decision_function > 0 gives positive; exact zero gives negative',
        'label_source': 'original metadata TSV, alignment semantic_anchors canonical keys',
        'metadata_sha256': digest(args.metadata),
        'test_policy': 'original TSV contains the cohort; extract only frozen train435 and val63 labels; no historical test artifacts or test evaluation',
        'comparison_limit': 'not a matched architecture ablation; CNN trains five tasks, probe trains three independently',
        'forbidden_features': ['MRI intensity', 'patient ID', 'path', 'mtime', 'segmentation source', 'original 3D volume'],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'artifact_sha256': {str(path.relative_to(ROOT)): digest(path) for path in artifacts},
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), ROOT / 'semantic_evaluation.py', ROOT / 'json_results.py',
             ROOT / 'dataset.py', ROOT / 'train_semantic_alignment.py', PROTOCOL_DOCUMENT)}}
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', frozen)
    # Model fitting completes before the validation labels or scores are decoded.
    protocol, manifest, train_labels, metadata = load_training_bundle(args.reference, args.metadata)
    train_ids, val_ids = protocol['splits']['train'], protocol['splits']['val']
    x_train = build_features(manifest, train_ids, protocol)
    scaler, models, fitted = fit_field_models(x_train, train_ids, train_labels)
    train_fit = {'version': VERSION, 'train_subject_ids': train_ids, 'feature_names': FEATURE_NAMES,
        'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'var': scaler.var_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_)},
        'fields': fitted, 'validation_used_for_fitting': False}
    write_json(out / 'train_fit.json', train_fit)
    validation = read(args.reference / 'whole_roi_field/validation_best_auc.json')['records']
    val_labels = recover_labels(validation, read(args.reference / 'anchor_vocab.json'), val_ids)
    if val_labels != labels_from_metadata(metadata, val_ids):
        raise ValueError('Validation targets differ from the original metadata/canonical-key conversion')
    x_val = scaler.transform(build_features(manifest, val_ids, protocol))
    rows, metrics = [], {}
    for field in FIELDS:
        margins = models[field].decision_function(x_val)
        probabilities = models[field].predict_proba(x_val)[:, 1]
        predictions = models[field].predict(x_val)
        for i, sid in enumerate(val_ids):
            rows.append({'subject_id': sid, 'field': field, 'target': val_labels[sid][field],
                'positive_class_key': CLASS_KEYS[field][0], 'negative_class_key': CLASS_KEYS[field][1],
                'decision_function': float(margins[i]), 'positive_probability': float(probabilities[i]),
                'prediction': int(predictions[i])})
        known = [i for i, sid in enumerate(val_ids) if val_labels[sid][field] is not None]
        metrics[field] = classification_metrics([val_labels[val_ids[i]][field] for i in known],
            margins[known], predictions[known])
        metrics[field]['missing'] = len(val_ids) - len(known)
    write_json(out / 'validation_patient_scores.json', {'version': VERSION, 'records': rows})
    summary = {'version': VERSION, 'scope': frozen['scope'], 'train_patients': len(train_ids),
        'validation_patients': len(val_ids), 'features': FEATURE_NAMES,
        'training_counts': {field: {key: fitted[field][key] for key in ('patients', 'missing', 'positive', 'negative')}
            for field in FIELDS},
        'validation_metrics': metrics,
        'validation_molecular_macro_auroc': float(np.mean([metrics[field]['auroc'] for field in FIELDS])),
        'test_evaluated': False, 'validation_used_for_fitting': False,
        'classifier_params': MODEL_PARAMS, 'versions': frozen['versions'],
        'metadata_sha256': frozen['metadata_sha256'],
        'protocol_sha256': digest(out / 'protocol.json'), 'artifact_sha256': frozen['artifact_sha256'],
        'source_sha256': frozen['source_sha256'],
        'comparison_limit': frozen['comparison_limit'],
        'interpretation_limit': 'A shape-only result does not quantify the incremental contribution of MRI intensities; no regional pathology is measured'}
    write_json(out / 'summary.json', summary)
    if args.summary_out is not None:
        write_json(args.summary_out, summary)
    write_json(out / 'complete.json', {'version': VERSION, 'fitted_fields': list(FIELDS),
        'test_evaluated': False, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'validation_molecular_macro_auroc': summary['validation_molecular_macro_auroc'],
        'validation_metrics': metrics, 'output': str(out)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
