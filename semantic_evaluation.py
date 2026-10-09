"""Versioned retrieval scoring with explicit known-label masks."""
from collections import defaultdict

import numpy as np


PROTOCOL_VERSION = 'pasa_patient_metadata_v2'


def validate_patient_splits(splits):
    seen = set()
    for name in ('train', 'val', 'test'):
        cases = splits[name]
        if not cases:
            raise ValueError(f'{name} split is empty; choose a valid split, never reuse patients')
        ids = [str(case['subject_id']) for case in cases]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Patient identifiers must be unique and disjoint across splits')
        seen.update(ids)


def supervision_from_keys(keys, anchor_vocab, key_to_id):
    """Only mutually exclusive categorical fields have confirmed negatives."""
    categorical = {'Tumor Grade', 'Tumor Type', 'IDH', 'MGMT', '1p19Q CODEL', 'Gender'}
    key_fields = {anchor['key']: anchor['field'] for anchor in anchor_vocab}
    # OOV field names are recoverable from canonical keys, without inventing a prototype.
    field_keys = {field.lower().replace(' ', '_').replace('/', '').replace('-', '_'): field
                  for field in categorical | {'Age at Histological Diagnosis'}}
    fields = {key_fields.get(key, field_keys.get(key.split('::', 1)[0])) for key in keys}
    positives = [key_to_id[key] for key in keys if key in key_to_id]
    valid = [idx for idx, anchor in enumerate(anchor_vocab)
             if anchor['key'] in keys or (anchor['field'] in fields and anchor['field'] in categorical)]
    return {'positive_ids': positives, 'valid_ids': valid,
            'positive_count': len(set(keys)), 'positive_keys': list(keys),
            'oov_positive_keys': [key for key in keys if key not in key_to_id]}


def binary_auc(labels, scores):
    """Mann-Whitney AUC with half credit for ties."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if not n_pos or not n_neg:
        return float('nan')
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def retrieval_metrics(query_vectors, target_ids, prototypes, gallery_ids=None,
                      ks=(1, 5, 10), subject_ids=None, valid_ids=None,
                      positive_counts=None, field_names=None, precomputed_scores=None):
    """Field-region-patient macro scoring; missing candidates stay in recall denominators."""
    q = np.asarray(query_vectors, dtype=np.float32)
    p = np.asarray(prototypes, dtype=np.float32)
    n = len(target_ids)
    if precomputed_scores is not None:
        precomputed_scores = np.asarray(precomputed_scores, dtype=float)
        if precomputed_scores.shape != (n, len(p)) or not np.all(np.isfinite(precomputed_scores)):
            raise ValueError('Precomputed scores must be finite query-by-gallery logits')
    if len(q) != n or (n and (q.ndim != 2 or p.ndim != 2 or q.shape[1] != p.shape[1])):
        raise ValueError('Query, target and prototype shapes do not agree')
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
        raise ValueError('Nonfinite retrieval vectors must be diagnosed, not silently sanitized')
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError('Retrieval cutoffs must be positive integers')
    gallery = list(range(len(p))) if gallery_ids is None else list(gallery_ids)
    if len(set(gallery)) != len(gallery) or any(i < 0 or i >= len(p) for i in gallery):
        raise ValueError('Gallery identifiers must be unique valid prototype indices')
    valid_ids = [gallery] * n if valid_ids is None else valid_ids
    positive_counts = [len(set(ids)) for ids in target_ids] if positive_counts is None else positive_counts
    subject_ids = list(range(n)) if subject_ids is None else subject_ids
    field_names = ['all'] * n if field_names is None else field_names
    if any(len(items) != n for items in (valid_ids, positive_counts, subject_ids, field_names)):
        raise ValueError('Per-query metadata lengths do not agree')
    names = [f'{kind}@{k}' for kind in ('hit', 'recall') for k in ks]
    names += ['map', 'mrr', 'pair_auc', 'positive_negative_similarity_gap',
              'positive_negative_distance_gap', 'average_positive_similarity', 'average_negative_similarity']
    rows = []
    skipped, oov, no_negatives = 0, 0, 0
    if n:
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    for i, ids in enumerate(target_ids):
        expected = positive_counts[i]
        if expected < len(set(ids)) or expected < 0:
            raise ValueError('Positive count cannot be smaller than labelled positives')
        if expected == 0:
            skipped += 1
            continue
        candidates = [idx for idx in gallery if idx in set(valid_ids[i])]
        positives = set(ids).intersection(candidates)
        oov += expected - len(positives)
        scores = q[i] @ p[candidates].T if precomputed_scores is None else precomputed_scores[i, candidates]
        order = np.argsort(-scores, kind='stable')
        ranking = [candidates[idx] for idx in order]
        row = {}
        for k in ks:
            hits = len(positives.intersection(ranking[:k]))
            row[f'hit@{k}'] = float(hits > 0)
            row[f'recall@{k}'] = hits / expected
        ranks = [rank for rank, idx in enumerate(ranking, 1) if idx in positives]
        row['mrr'] = 1 / ranks[0] if ranks else 0.0
        row['map'] = sum(hit / rank for hit, rank in enumerate(ranks, 1)) / expected
        labels = [int(idx in positives) for idx in candidates]
        row['pair_auc'] = binary_auc(labels, scores)
        pos = [j for j, idx in enumerate(candidates) if idx in positives]
        neg = [j for j, idx in enumerate(candidates) if idx not in positives]
        no_negatives += int(not neg)
        row['average_positive_similarity'] = float(np.mean(scores[pos])) if pos and precomputed_scores is None else float('nan')
        row['average_negative_similarity'] = float(np.mean(scores[neg])) if neg and precomputed_scores is None else float('nan')
        row['positive_negative_similarity_gap'] = (row['average_positive_similarity'] - row['average_negative_similarity'])
        distances = np.linalg.norm(q[i] - p[candidates], axis=1)
        row['positive_negative_distance_gap'] = (float(np.mean(distances[neg]) - np.mean(distances[pos]))
                                               if pos and neg and precomputed_scores is None else float('nan'))
        rows.append((str(subject_ids[i]), str(field_names[i]), row))
    metrics = {}
    for name in names:
        grouped = defaultdict(list)
        for patient, field, row in rows:
            if np.isfinite(row[name]):
                grouped[(patient, field)].append(row[name])
        patients = defaultdict(list)
        for (patient, field), values in grouped.items():
            patients[patient].append(float(np.mean(values)))
        metrics[name] = float(np.mean([np.mean(v) for v in patients.values()])) if patients else float('nan')
        metrics[f'{name}_scored_queries'] = sum(len(v) for v in grouped.values())
    metrics.update(protocol_version=PROTOCOL_VERSION, aggregation='region_then_field_then_patient',
                   total_queries=n, scored_queries=len(rows), skipped_no_positive=skipped,
                   unavailable_positive_count=oov, queries_without_known_negatives=no_negatives,
                   scored_patients=len({patient for patient, _, _ in rows}))
    return metrics


def score_retrieval_metrics(scores, target_ids, **kwargs):
    """Evaluate arbitrary alignment logits without calling them cosine similarities."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError('Expected a query-by-gallery score matrix')
    return retrieval_metrics(np.zeros((len(scores), 1)), target_ids,
                             np.zeros((scores.shape[1], 1)), precomputed_scores=scores, **kwargs)
