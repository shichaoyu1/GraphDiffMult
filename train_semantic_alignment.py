"""
Train a pathology-anchored multimodal semantic-unit alignment experiment.

This entry point treats glioma grading as a downstream sanity check, not the
main objective.  The main objective is cross-modal semantic-unit retrieval:
MRI lesion-region nodes are pulled toward pathology/molecular semantic anchors
derived from patient metadata.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_utsw_cases, load_utsw_metadata, find_utsw_metadata
from experiment_dataset import UTSWROIPatientDataset, describe_cases, parse_utsw_label, stratified_split
from experiment_model import GliomaGraphDiffusionNet
from semantic_graph_visualize import (
    adjacency_to_laplacian,
    node_names_for_mode,
    plot_matrix,
    plot_semantic_graph,
)


PATHOLOGY_FIELDS = ('Tumor Grade', 'Tumor Type')
MOLECULAR_FIELDS = ('IDH', 'MGMT', '1p19Q CODEL')
CLINICAL_FIELDS = ('Age at Histological Diagnosis', 'Gender')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_value(value):
    if value is None:
        return ''
    return str(value).strip()


def canonical_field(field):
    return field.lower().replace(' ', '_').replace('/', '').replace('-', '_')


def anchor_source(field):
    if field in PATHOLOGY_FIELDS:
        return 'Pathology'
    if field in MOLECULAR_FIELDS:
        return 'Gene'
    return 'Clinical'


def anchor_type(field):
    if field == 'Tumor Grade':
        return 'pathology-grade'
    if field == 'Tumor Type':
        return 'pathology-diagnosis'
    if field in MOLECULAR_FIELDS:
        return 'molecular-marker'
    return 'clinical-context'


def make_anchor(field, value):
    value = clean_value(value)
    key = f'{canonical_field(field)}::{value.lower()}'
    if field == 'Tumor Grade':
        label = f'Pathology grade {value}'
    elif field == 'Tumor Type':
        label = f'Pathology {value}'
    else:
        label = f'{field} {value}'
    return {
        'key': key,
        'label': label,
        'field': field,
        'value': value,
        'source': anchor_source(field),
        'node_type': anchor_type(field),
    }


def semantic_anchors(metadata, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = []
    fields = []
    if include_pathology:
        fields.extend(PATHOLOGY_FIELDS)
    if include_molecular:
        fields.extend(MOLECULAR_FIELDS)
    if include_clinical:
        fields.extend(CLINICAL_FIELDS)

    for field in fields:
        value = clean_value(metadata.get(field))
        if value and value.lower() not in {'na', 'n/a', 'nan', 'none', 'unknown'}:
            anchors.append(make_anchor(field, value))
    return anchors


def grade_or_fallback_label(metadata):
    for task in ('grade', 'idh', 'mgmt', '1p19q'):
        try:
            label = parse_utsw_label(metadata, task)
        except ValueError:
            label = None
        if label is not None:
            return int(label)
    return 0


def discover_semantic_cases(root_dir, metadata_tsv=None, max_cases=None, seed=42, include_clinical=False):
    metadata_path = metadata_tsv or find_utsw_metadata(root_dir)
    metadata = load_utsw_metadata(metadata_path) if metadata_path else {}
    cases = []
    for case in get_utsw_cases(root_dir, metadata_tsv=metadata_path):
        info = metadata.get(case['subject_id'], case.get('metadata', {}))
        anchors = semantic_anchors(info, include_clinical=include_clinical)
        if not anchors:
            continue
        case = dict(case)
        case['metadata'] = info
        case['label'] = grade_or_fallback_label(info)
        cases.append(case)

    if max_cases and len(cases) > max_cases:
        rng = np.random.default_rng(seed)
        grouped = defaultdict(list)
        for case in cases:
            grouped[case['label']].append(case)
        sampled = []
        for group in grouped.values():
            rng.shuffle(group)
        while len(sampled) < max_cases and any(grouped.values()):
            for label in sorted(grouped):
                if grouped[label] and len(sampled) < max_cases:
                    sampled.append(grouped[label].pop())
        cases = sorted(sampled, key=lambda item: item['subject_id'])
    return cases


def build_anchor_vocab(cases, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = {}
    for case in cases:
        for anchor in semantic_anchors(
            case.get('metadata', {}),
            include_pathology=include_pathology,
            include_molecular=include_molecular,
            include_clinical=include_clinical,
        ):
            anchors[anchor['key']] = anchor
    ordered = [anchors[key] for key in sorted(anchors)]
    key_to_id = {anchor['key']: idx for idx, anchor in enumerate(ordered)}
    return ordered, key_to_id


def target_anchor_keys(metadata, node_name, policy, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = {
        anchor['field']: anchor['key']
        for anchor in semantic_anchors(
            metadata,
            include_pathology=include_pathology,
            include_molecular=include_molecular,
            include_clinical=include_clinical,
        )
    }
    if policy == 'all_patient_anchors':
        return list(anchors.values())

    node = node_name.lower()
    fields = []
    if 'enhancing' in node or 't1ce' in node:
        fields = ['Tumor Grade', 'MGMT', 'Tumor Type']
    elif 'edema' in node or 'flair' in node or 't2' in node:
        fields = ['IDH', 'Tumor Type', 'Tumor Grade']
    elif 'necrotic' in node or 'core' in node or 't1' in node:
        fields = ['Tumor Grade', '1p19Q CODEL', 'Tumor Type']
    else:
        fields = ['Tumor Grade', 'IDH', 'MGMT', '1p19Q CODEL', 'Tumor Type']

    keys = [anchors[field] for field in fields if field in anchors]
    return keys or list(anchors.values())


class SemanticPrototypeBank(nn.Module):
    def __init__(self, num_anchors, dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)

    def forward(self):
        return self.prototypes


def multi_positive_contrastive_loss(queries, target_ids, prototypes, temperature=0.07):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    logits = queries @ prototypes.t() / temperature

    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, ids in enumerate(target_ids):
        if ids:
            mask[row, ids] = True
    if not mask.any():
        return logits.sum() * 0

    masked_logits = logits.masked_fill(~mask, -1e9)
    return -(torch.logsumexp(masked_logits, dim=-1) - torch.logsumexp(logits, dim=-1)).mean()


def medclip_multi_positive_loss(queries, target_ids, prototypes, ignore_ids_by_anchor, temperature=0.07):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    logits = queries @ prototypes.t() / temperature

    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        positive_mask[row, ids] = True
        for anchor_id in ids:
            for ignore_id in ignore_ids_by_anchor[anchor_id]:
                valid_mask[row, ignore_id] = False
        valid_mask[row, ids] = True
    if not positive_mask.any():
        return logits.sum() * 0

    masked_pos_logits = logits.masked_fill(~positive_mask, -1e9)
    masked_all_logits = logits.masked_fill(~valid_mask, -1e9)
    return -(torch.logsumexp(masked_pos_logits, dim=-1) - torch.logsumexp(masked_all_logits, dim=-1)).mean()


def dcca_alignment_loss(queries, target_ids, prototypes, reg=1e-3):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)

    x_rows = []
    y_rows = []
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        x_rows.append(queries[row])
        y_rows.append(prototypes[ids].mean(dim=0))
    if len(x_rows) < 2:
        return queries.sum() * 0

    x = torch.stack(x_rows, dim=0)
    y = torch.stack(y_rows, dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    n = x.shape[0]
    dim_x = x.shape[1]
    dim_y = y.shape[1]
    eye_x = torch.eye(dim_x, device=x.device, dtype=x.dtype)
    eye_y = torch.eye(dim_y, device=y.device, dtype=y.dtype)

    c_xx = (x.T @ x) / max(n - 1, 1) + reg * eye_x
    c_yy = (y.T @ y) / max(n - 1, 1) + reg * eye_y
    c_xx = 0.5 * (c_xx + c_xx.T)
    c_yy = 0.5 * (c_yy + c_yy.T)
    c_xy = (x.T @ y) / max(n - 1, 1)

    eval_x, evec_x = torch.linalg.eigh(c_xx)
    eval_y, evec_y = torch.linalg.eigh(c_yy)
    invsqrt_x = evec_x @ torch.diag(torch.rsqrt(torch.clamp(eval_x, min=1e-6))) @ evec_x.T
    invsqrt_y = evec_y @ torch.diag(torch.rsqrt(torch.clamp(eval_y, min=1e-6))) @ evec_y.T
    t_mat = invsqrt_x @ c_xy @ invsqrt_y
    corr = torch.linalg.svdvals(t_mat).sum()
    return -(corr / float(min(dim_x, dim_y)))


def anchor_center_loss(queries, target_ids, prototypes):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    losses = []
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        center = prototypes[ids].mean(dim=0, keepdim=True)
        center = F.normalize(center, dim=-1).squeeze(0)
        losses.append(1.0 - torch.sum(queries[row] * center))
    if not losses:
        return queries.sum() * 0
    return torch.stack(losses).mean()


def build_medclip_ignore_ids(anchor_vocab):
    buckets = defaultdict(list)
    for idx, anchor in enumerate(anchor_vocab):
        key = (anchor.get('source', ''), anchor.get('field', ''))
        buckets[key].append(idx)
    ignore_ids = []
    for anchor in anchor_vocab:
        key = (anchor.get('source', ''), anchor.get('field', ''))
        ignore_ids.append(buckets.get(key, []))
    return ignore_ids


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[pos].sum()
    return float((pos_ranks - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def retrieval_metrics(query_vectors, target_ids, prototypes, gallery_ids=None, ks=(1, 5, 10), subject_ids=None):
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if gallery_ids is None:
        gallery_ids = list(range(len(prototypes)))
    gallery_ids = list(gallery_ids)
    if len(query_vectors) == 0 or len(gallery_ids) == 0:
        return {}

    query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ proto_norm[np.asarray(gallery_ids)].T

    recalls = {k: [] for k in ks}
    reciprocal_ranks = []
    average_precisions = []
    pos_scores = []
    neg_scores = []
    pos_dists = []
    neg_dists = []
    edge_labels = []
    edge_scores = []

    for row, positives in enumerate(target_ids):
        positives = set(positives).intersection(gallery_ids)
        if not positives:
            continue
        ranking = np.argsort(-scores[row])
        ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
        for k in ks:
            top_k = ranked_anchor_ids[:min(k, len(ranked_anchor_ids))]
            recalls[k].append(float(any(anchor_id in positives for anchor_id in top_k)))
        first_rank = next((rank + 1 for rank, anchor_id in enumerate(ranked_anchor_ids) if anchor_id in positives), None)
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        precision_hits = []
        hit_count = 0
        for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
            if anchor_id in positives:
                hit_count += 1
                precision_hits.append(hit_count / rank_idx)
        if precision_hits:
            average_precisions.append(float(np.mean(precision_hits)))

        for col, anchor_id in enumerate(gallery_ids):
            score = float(scores[row, col])
            distance = float(np.linalg.norm(query_norm[row] - proto_norm[anchor_id]))
            is_positive = anchor_id in positives
            edge_scores.append(score)
            edge_labels.append(1 if is_positive else 0)
            if is_positive:
                pos_scores.append(score)
                pos_dists.append(distance)
            else:
                neg_scores.append(score)
                neg_dists.append(distance)

    metrics = {f'recall@{k}': float(np.mean(values)) if values else float('nan') for k, values in recalls.items()}
    map_query = float(np.mean(average_precisions)) if average_precisions else float('nan')
    metrics['map_query'] = map_query
    if subject_ids is not None and len(subject_ids) == len(target_ids):
        patient_ap = defaultdict(list)
        for row, positives in enumerate(target_ids):
            positives = set(positives).intersection(gallery_ids)
            if not positives:
                continue
            ranking = np.argsort(-scores[row])
            ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
            precision_hits = []
            hit_count = 0
            for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
                if anchor_id in positives:
                    hit_count += 1
                    precision_hits.append(hit_count / rank_idx)
            if precision_hits:
                patient_ap[str(subject_ids[row])].append(float(np.mean(precision_hits)))
        patient_means = [float(np.mean(values)) for values in patient_ap.values() if values]
        metrics['map'] = float(np.mean(patient_means)) if patient_means else map_query
    else:
        metrics['map'] = map_query
    metrics['mrr'] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float('nan')
    metrics['pair_auc'] = binary_auc(edge_labels, edge_scores)
    metrics['average_positive_similarity'] = float(np.mean(pos_scores)) if pos_scores else float('nan')
    metrics['average_negative_similarity'] = float(np.mean(neg_scores)) if neg_scores else float('nan')
    metrics['positive_negative_distance_gap'] = (
        float(np.mean(neg_dists) - np.mean(pos_dists)) if pos_dists and neg_dists else float('nan')
    )
    metrics['anchor_consistency'] = (
        float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else float('nan')
    )

    if edge_scores:
        order = np.argsort(-np.asarray(edge_scores))
        for k in (10, 25, 50):
            top = order[:min(k, len(order))]
            metrics[f'edge_precision@{k}'] = float(np.mean(np.asarray(edge_labels)[top])) if len(top) else float('nan')
    return metrics


def bootstrap_ci(values, seed=42, n_bootstrap=1000):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def make_loader(cases, args, split_name):
    dataset = UTSWROIPatientDataset(
        cases,
        roi_size=args.roi_size,
        z_slices=args.z_slices,
        prefer_registered=args.prefer_registered,
        augment=split_name == 'train' and args.augment,
        cache=args.cache,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split_name == 'train',
        num_workers=args.num_workers,
        drop_last=False,
    )


def build_query_targets(subject_ids, node_names, case_lookup, key_to_id, args):
    target_ids = []
    for subject_id in subject_ids:
        metadata = case_lookup[str(subject_id)]['metadata']
        for node_name in node_names:
            keys = target_anchor_keys(
                metadata,
                node_name,
                args.target_policy,
                include_pathology=not args.exclude_pathology_anchors,
                include_molecular=not args.exclude_molecular_anchors,
                include_clinical=args.include_clinical_anchors,
            )
            target_ids.append([key_to_id[key] for key in keys if key in key_to_id])
    return target_ids


def graph_cons_scale(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def run_epoch(model, bank, loader, optimizer, device, args, case_lookup, key_to_id, epoch, loss_context):
    training = optimizer is not None
    model.train(training)
    bank.train(training)
    node_names = node_names_for_mode(args.node_mode)
    totals = Counter()
    cons_scale = graph_cons_scale(epoch, args.graph_warmup_epochs)
    freeze_graph = training and (epoch <= args.graph_warmup_epochs)

    for batch in loader:
        images = batch['images'].to(device)
        region_masks = batch.get('region_masks')
        if region_masks is not None:
            region_masks = region_masks.to(device)
        subject_ids = batch['subject_id']

        with torch.set_grad_enabled(training):
            output = model(
                images,
                region_masks=region_masks,
                return_extras=True,
                freeze_graph=freeze_graph,
            )
            shared = output['extras']['shared']
            queries = shared.reshape(-1, shared.shape[-1])
            target_ids = build_query_targets(subject_ids, node_names, case_lookup, key_to_id, args)
            prototypes = bank()
            dcca_fallback = 0.0
            if args.alignment_objective == 'medclip':
                alignment_loss = medclip_multi_positive_loss(
                    queries,
                    target_ids,
                    prototypes,
                    ignore_ids_by_anchor=loss_context['medclip_ignore_ids'],
                    temperature=args.temperature,
                )
            elif args.alignment_objective == 'dcca':
                clip_loss = multi_positive_contrastive_loss(
                    queries,
                    target_ids,
                    prototypes,
                    temperature=args.temperature,
                )
                try:
                    dcca_loss = dcca_alignment_loss(
                        queries,
                        target_ids,
                        prototypes,
                        reg=args.dcca_reg,
                    )
                    dcca_fallback = 0.0
                    alignment_loss = dcca_loss + args.dcca_clip_weight * clip_loss
                except RuntimeError:
                    dcca_fallback = 1.0
                    alignment_loss = clip_loss
            else:
                alignment_loss = multi_positive_contrastive_loss(
                    queries,
                    target_ids,
                    prototypes,
                    temperature=args.temperature,
                )
                dcca_fallback = 0.0
            anchor_loss = anchor_center_loss(queries, target_ids, prototypes)
            losses = output['losses']
            total = (
                alignment_loss
                + args.lambda_anchor * anchor_loss
                + args.lambda_cons * cons_scale * losses['cons']
                + args.lambda_decouple * losses['decouple']
                + args.lambda_leak * losses['leak']
                + args.lambda_diff * losses['diff']
                + args.lambda_diff_norm * losses['diff_norm']
            )

            if training:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(bank.parameters()), args.grad_clip)
                optimizer.step()

        batch_size = images.shape[0]
        totals['n'] += batch_size
        totals['total'] += float(total.detach().cpu()) * batch_size
        totals['alignment'] += float(alignment_loss.detach().cpu()) * batch_size
        totals['anchor'] += float(anchor_loss.detach().cpu()) * batch_size
        totals['dcca_fallback'] += float(dcca_fallback) * batch_size
        for name in ['cons', 'decouple', 'leak', 'diff', 'diff_norm', 'graph_energy']:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
        totals['shared_norm'] += float(output['extras']['shared_norm'].detach().cpu()) * batch_size
        totals['private_norm'] += float(output['extras']['private_norm'].detach().cpu()) * batch_size
        totals['diffusion_residual_norm'] += float(output['extras']['diffusion_residual_norm'].detach().cpu()) * batch_size

    n_samples = max(totals['n'], 1)
    return {
        name: totals[name] / n_samples
        for name in [
            'total',
            'alignment',
            'anchor',
            'cons',
            'decouple',
            'leak',
            'diff',
            'diff_norm',
            'graph_energy',
            'dcca_fallback',
            'shared_norm',
            'private_norm',
            'diffusion_residual_norm',
        ]
    }


def collect_alignment_records(model, bank, loader, device, args, case_lookup, key_to_id, anchor_vocab, max_cases=None):
    model.eval()
    bank.eval()
    node_names = node_names_for_mode(args.node_mode)
    query_vectors = []
    query_targets = []
    query_records = []
    adjacency_mats = []
    seen_cases = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            region_masks = batch.get('region_masks')
            if region_masks is not None:
                region_masks = region_masks.to(device)
            subject_ids = batch['subject_id']
            output = model(images, region_masks=region_masks, return_extras=True)
            shared = output['extras']['shared'].detach().cpu().numpy()
            adjacency_mats.append(output['extras']['adjacency'].detach().cpu().numpy())

            for sample_idx, subject_id in enumerate(subject_ids):
                if max_cases is not None and seen_cases >= max_cases:
                    break
                metadata = case_lookup[str(subject_id)]['metadata']
                for node_idx, node_name in enumerate(node_names):
                    keys = target_anchor_keys(
                        metadata,
                        node_name,
                        args.target_policy,
                        include_pathology=not args.exclude_pathology_anchors,
                        include_molecular=not args.exclude_molecular_anchors,
                        include_clinical=args.include_clinical_anchors,
                    )
                    ids = [key_to_id[key] for key in keys if key in key_to_id]
                    if not ids:
                        continue
                    query_vectors.append(shared[sample_idx, node_idx])
                    query_targets.append(ids)
                    query_records.append({
                        'subject_id': str(subject_id),
                        'node_name': node_name,
                        'source': 'MRI',
                        'target_labels': [anchor_vocab[idx]['label'] for idx in ids],
                    })
                seen_cases += 1
            if max_cases is not None and seen_cases >= max_cases:
                break

    prototypes = bank().detach().cpu().numpy()
    adjacency = np.concatenate(adjacency_mats, axis=0).mean(axis=0) if adjacency_mats else None
    return {
        'query_vectors': np.asarray(query_vectors, dtype=np.float32),
        'query_targets': query_targets,
        'query_subject_ids': [record['subject_id'] for record in query_records],
        'query_records': query_records,
        'prototypes': prototypes,
        'adjacency': adjacency,
        'case_count': seen_cases,
    }


def save_alignment_space_plot(records, anchor_vocab, out_dir, max_edges=160):
    query_vectors = records['query_vectors']
    prototypes = records['prototypes']
    if len(query_vectors) == 0:
        return

    points = np.concatenate([query_vectors, prototypes], axis=0)
    # Guard visualization against NaN/Inf and SVD convergence failures.
    points = np.asarray(points, dtype=np.float32)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    centered = points - points.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    def project_to_2d(matrix):
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if matrix.shape[1] == 0:
            return np.zeros((matrix.shape[0], 2), dtype=np.float32)
        try:
            _, _, vt = np.linalg.svd(matrix, full_matrices=False)
            if vt.shape[0] >= 2:
                return matrix @ vt[:2].T
            if vt.shape[0] == 1:
                return np.pad(matrix @ vt[:1].T, ((0, 0), (0, 1)))
        except np.linalg.LinAlgError:
            pass
        # Fallback: direct feature projection to avoid blocking eval pipeline.
        if matrix.shape[1] >= 2:
            return matrix[:, :2]
        return np.pad(matrix[:, :1], ((0, 0), (0, 1)))

    coords = project_to_2d(centered)
    query_coords = coords[:len(query_vectors)]
    anchor_coords = coords[len(query_vectors):]

    colors = {'MRI': '#2E86DE', 'Pathology': '#E67E22', 'Gene': '#27AE60', 'Clinical': '#9B59B6'}
    markers = {
        'Necrotic/Core': '^',
        'Edema': 'o',
        'Enhancing': '*',
        'T1': '^',
        'T1ce': '*',
        'T2': 's',
        'FLAIR': 'o',
        'pathology-grade': 'D',
        'pathology-diagnosis': 'P',
        'molecular-marker': 'X',
        'clinical-context': 'h',
    }

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#101725')

    edge_budget = min(max_edges, len(records['query_targets']))
    edge_ids = np.linspace(0, len(records['query_targets']) - 1, edge_budget, dtype=int) if edge_budget else []
    for query_idx in edge_ids:
        for anchor_id in records['query_targets'][query_idx][:1]:
            ax.plot(
                [query_coords[query_idx, 0], anchor_coords[anchor_id, 0]],
                [query_coords[query_idx, 1], anchor_coords[anchor_id, 1]],
                color='#d0d7de',
                alpha=0.12,
                linewidth=0.7,
            )

    seen = set()
    for idx, record in enumerate(records['query_records']):
        node_name = record['node_name']
        label = f'MRI: {node_name}'
        ax.scatter(
            query_coords[idx, 0],
            query_coords[idx, 1],
            s=150 if node_name == 'Enhancing' else 72,
            c=colors['MRI'],
            marker=markers.get(node_name, 'o'),
            edgecolors='white',
            linewidths=0.5,
            alpha=0.78,
            label=label if label not in seen else None,
        )
        seen.add(label)

    for idx, anchor in enumerate(anchor_vocab):
        label = f"{anchor['source']}: {anchor['label']}"
        ax.scatter(
            anchor_coords[idx, 0],
            anchor_coords[idx, 1],
            s=160,
            c=colors.get(anchor['source'], '#95A5A6'),
            marker=markers.get(anchor['node_type'], 'D'),
            edgecolors='white',
            linewidths=0.8,
            alpha=0.94,
            label=label if label not in seen else None,
        )
        seen.add(label)

    ax.set_title('Semantic Unit Alignment Space', color='white', fontsize=15, fontweight='bold')
    ax.set_xlabel('PC1 of shared semantic space', color='white')
    ax.set_ylabel('PC2 of shared semantic space', color='white')
    ax.tick_params(colors='#bdc3c7')
    ax.grid(color='white', alpha=0.10)
    for spine in ax.spines.values():
        spine.set_color('#34495e')
    ax.legend(loc='best', fontsize=7, facecolor='#17202a', edgecolor='#566573', labelcolor='white')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'semantic_unit_alignment_space.png'), dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_semantic_unit_graph(records, anchor_vocab, args, out_dir):
    node_names = node_names_for_mode(args.node_mode)
    graph_nodes = list(node_names) + [anchor['label'] for anchor in anchor_vocab]
    groups = ['MRI'] * len(node_names) + [anchor['source'] for anchor in anchor_vocab]
    adjacency = np.zeros((len(graph_nodes), len(graph_nodes)), dtype=np.float32)
    if len(records['query_vectors']) == 0:
        return

    query_vec = np.nan_to_num(records['query_vectors'], nan=0.0, posinf=0.0, neginf=0.0)
    proto_vec = np.nan_to_num(records['prototypes'], nan=0.0, posinf=0.0, neginf=0.0)
    query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_vec / (np.linalg.norm(proto_vec, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ proto_norm.T
    score_buckets = defaultdict(list)
    for query_idx, record in enumerate(records['query_records']):
        node_idx = node_names.index(record['node_name'])
        for anchor_idx in range(len(anchor_vocab)):
            score_buckets[(node_idx, len(node_names) + anchor_idx)].append(float(scores[query_idx, anchor_idx]))

    for (source, target), values in score_buckets.items():
        weight = float(np.mean(values))
        adjacency[source, target] = max(weight, 0.0)
        adjacency[target, source] = max(weight, 0.0)

    positive_edges = []
    for source in range(len(node_names)):
        candidates = [(target, adjacency[source, target]) for target in range(len(node_names), len(graph_nodes))]
        candidates.sort(key=lambda item: item[1], reverse=True)
        positive_edges.extend((source, target, weight) for target, weight in candidates[:args.graph_top_k] if weight > 0)

    sparse = np.zeros_like(adjacency)
    for source, target, weight in positive_edges:
        sparse[source, target] = weight
        sparse[target, source] = weight

    plot_semantic_graph(
        graph_nodes,
        sparse,
        groups,
        save_path=os.path.join(out_dir, 'semantic_unit_graph_50patients.png'),
        title=f'Multi-patient Semantic Unit Graph (n={records["case_count"]})',
    )
    plot_matrix(
        sparse,
        graph_nodes,
        save_path=os.path.join(out_dir, 'semantic_unit_adjacency.png'),
        title='Semantic Unit Adjacency',
        cmap='magma',
    )
    plot_matrix(
        adjacency_to_laplacian(sparse),
        graph_nodes,
        save_path=os.path.join(out_dir, 'semantic_unit_laplacian.png'),
        title='Semantic Unit Laplacian',
    )


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_patient_level_records(records, out_dir):
    payload = {
        'subject_ids': [record['subject_id'] for record in records['query_records']],
        'node_names': [record['node_name'] for record in records['query_records']],
        'query_targets': records['query_targets'],
        'query_vectors': records['query_vectors'].tolist(),
        'prototypes': records['prototypes'].tolist(),
    }
    save_json(os.path.join(out_dir, 'patient_level_records.json'), payload)


def anchor_gallery(anchor_vocab, excluded_sources=None):
    excluded_sources = set(excluded_sources or [])
    return [idx for idx, anchor in enumerate(anchor_vocab) if anchor['source'] not in excluded_sources]


def evaluate_and_save(model, bank, loader, device, args, case_lookup, key_to_id, anchor_vocab, out_dir):
    records = collect_alignment_records(
        model,
        bank,
        loader,
        device,
        args,
        case_lookup,
        key_to_id,
        anchor_vocab,
        max_cases=args.align_max_cases,
    )
    metrics = retrieval_metrics(
        records['query_vectors'],
        records['query_targets'],
        records['prototypes'],
        subject_ids=records['query_subject_ids'],
    )
    metrics['case_count'] = records['case_count']
    metrics['query_count'] = int(len(records['query_vectors']))
    metrics['anchor_count'] = int(len(anchor_vocab))

    no_pathology_gallery = anchor_gallery(anchor_vocab, excluded_sources={'Pathology'})
    if no_pathology_gallery:
        missing_pathology = retrieval_metrics(
            records['query_vectors'],
            records['query_targets'],
            records['prototypes'],
            gallery_ids=no_pathology_gallery,
            subject_ids=records['query_subject_ids'],
        )
        metrics['pathology_unavailable'] = missing_pathology
        if 'map' in metrics and 'map' in missing_pathology:
            metrics['pathology_unavailable_map_drop'] = float(metrics['map'] - missing_pathology['map'])

    no_gene_gallery = anchor_gallery(anchor_vocab, excluded_sources={'Gene'})
    if no_gene_gallery:
        metrics['molecular_unavailable'] = retrieval_metrics(
            records['query_vectors'],
            records['query_targets'],
            records['prototypes'],
            gallery_ids=no_gene_gallery,
            subject_ids=records['query_subject_ids'],
        )

    save_json(os.path.join(out_dir, 'semantic_alignment_metrics.json'), metrics)
    save_patient_level_records(records, out_dir)
    save_alignment_space_plot(records, anchor_vocab, out_dir)
    save_semantic_unit_graph(records, anchor_vocab, args, out_dir)
    return metrics


def apply_variant(args):
    if args.variant == 'full':
        return args
    if args.variant == 'clip':
        args.graph_type = 'no_graph'
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = 'clip'
    elif args.variant == 'medclip_style':
        args.graph_type = 'no_graph'
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = 'medclip'
    elif args.variant == 'dcca':
        args.graph_type = 'no_graph'
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = 'dcca'
    elif args.variant in {'hgt', 'graph_shared_only'}:
        args.graph_type = 'learnable'
        args.no_private = True
        args.no_diffusion = True
        args.alignment_objective = 'clip'
        args.variant = 'graph_shared_only'
    elif args.variant == 'no_anchor':
        args.exclude_pathology_anchors = True
    elif args.variant == 'graph_only':
        args.no_diffusion = True
    elif args.variant == 'modality_vector':
        args.node_mode = 'modalities'
    elif args.variant == 'no_private':
        args.no_private = True
        args.no_diffusion = True
    elif args.variant == 'no_graph':
        args.graph_type = 'no_graph'
    else:
        raise ValueError(f'Unsupported variant: {args.variant}')
    return args


def main(args):
    args = apply_variant(args)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    cases = discover_semantic_cases(
        args.data_root,
        metadata_tsv=args.metadata_tsv,
        max_cases=args.max_cases,
        seed=args.seed,
        include_clinical=args.include_clinical_anchors,
    )
    if len(cases) < 2:
        raise ValueError(f'Need at least 2 semantic cases; found {len(cases)}')

    splits = stratified_split(cases, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    if not splits['val']:
        splits['val'] = list(splits['test'])
    if not splits['test']:
        splits['test'] = list(splits['val'] or splits['train'])

    anchor_vocab, key_to_id = build_anchor_vocab(
        splits['train'],
        include_pathology=not args.exclude_pathology_anchors,
        include_molecular=not args.exclude_molecular_anchors,
        include_clinical=args.include_clinical_anchors,
    )
    if len(anchor_vocab) < 2:
        raise ValueError('Need at least two train-set semantic anchors for contrastive alignment')

    case_lookup = {case['subject_id']: case for case in cases}
    loaders = {split: make_loader(split_cases, args, split) for split, split_cases in splits.items()}
    loss_context = {
        'medclip_ignore_ids': build_medclip_ignore_ids(anchor_vocab),
    }

    model = GliomaGraphDiffusionNet(
        num_classes=1,
        z_slices=args.z_slices,
        node_mode=args.node_mode,
        feat_dim=args.feat_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        graph_type=args.graph_type,
        diffusion_T=args.diffusion_T,
        graph_ema_momentum=args.graph_ema_momentum,
        graph_ema_blend=args.graph_ema_blend,
        diffusion_init_alpha=args.diffusion_init_alpha,
        shared_private_mix_init=args.shared_private_mix_init,
        classifier_private_scale_init=args.classifier_private_scale_init,
        diffusion_max_ratio=args.diffusion_max_ratio,
        use_anchor=False,
        use_private=not args.no_private,
        use_diffusion=not args.no_diffusion,
    ).to(device)
    bank = SemanticPrototypeBank(len(anchor_vocab), args.shared_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(bank.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f'Semantic alignment cases: {len(cases)} | split labels: {describe_cases(cases)}')
    for split_name in ['train', 'val', 'test']:
        print(f'{split_name}: {len(splits[split_name])} | distribution: {describe_cases(splits[split_name])}')
    print(
        f'Variant={args.variant} node_mode={args.node_mode} graph={args.graph_type} '
        f'pathology_anchors={not args.exclude_pathology_anchors} molecular_anchors={not args.exclude_molecular_anchors} '
        f'private={not args.no_private} diffusion={not args.no_diffusion} objective={args.alignment_objective}'
    )
    print(f'Anchor vocabulary ({len(anchor_vocab)}): {[anchor["label"] for anchor in anchor_vocab]}')

    save_json(os.path.join(args.out_dir, 'config.json'), vars(args))
    save_json(os.path.join(args.out_dir, 'anchor_vocab.json'), anchor_vocab)
    save_json(
        os.path.join(args.out_dir, 'splits.json'),
        {name: [case['subject_id'] for case in split_cases] for name, split_cases in splits.items()},
    )

    history = []
    best_score = -float('inf')
    best_path = os.path.join(args.out_dir, 'best_semantic_alignment.pt')

    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(model, bank, loaders['train'], optimizer, device, args, case_lookup, key_to_id, epoch, loss_context)
        val_losses = run_epoch(model, bank, loaders['val'], None, device, args, case_lookup, key_to_id, epoch, loss_context)
        val_records = collect_alignment_records(model, bank, loaders['val'], device, args, case_lookup, key_to_id, anchor_vocab)
        val_metrics = retrieval_metrics(
            val_records['query_vectors'],
            val_records['query_targets'],
            val_records['prototypes'],
            subject_ids=val_records['query_subject_ids'],
        )
        scheduler.step()

        epoch_record = {'epoch': epoch, 'train': train_losses, 'val': {**val_losses, **val_metrics}}
        history.append(epoch_record)
        save_json(os.path.join(args.out_dir, 'history.json'), history)

        score = val_metrics.get('map', float('nan'))
        if np.isnan(score):
            score = val_metrics.get('mrr', float('nan'))
        if np.isnan(score):
            score = -val_losses['alignment']
        if score > best_score:
            best_score = score
            torch.save({'model': model.state_dict(), 'bank': bank.state_dict(), 'args': vars(args), 'anchor_vocab': anchor_vocab}, best_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={train_losses['total']:.4f} align={train_losses['alignment']:.4f} anchor={train_losses['anchor']:.4f} "
            f"shared_norm={train_losses['shared_norm']:.4f} private_norm={train_losses['private_norm']:.4f} "
            f"diff_res_norm={train_losses['diffusion_residual_norm']:.4f} graphE={train_losses['graph_energy']:.4f} "
            f"dcca_fb={train_losses['dcca_fallback']:.4f} "
            f"val_mAP={val_metrics.get('map', float('nan')):.4f} "
            f"val_mrr={val_metrics.get('mrr', float('nan')):.4f} "
            f"val_r@1={val_metrics.get('recall@1', float('nan')):.4f} "
            f"val_pairAUC={val_metrics.get('pair_auc', float('nan')):.4f}"
        )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        bank.load_state_dict(checkpoint['bank'])

    test_metrics = evaluate_and_save(
        model,
        bank,
        loaders['test'],
        device,
        args,
        case_lookup,
        key_to_id,
        anchor_vocab,
        args.out_dir,
    )
    save_json(os.path.join(args.out_dir, 'test_metrics.json'), test_metrics)
    print(
        f"Test semantic alignment: mAP={test_metrics.get('map', float('nan')):.4f} "
        f"MRR={test_metrics.get('mrr', float('nan')):.4f} "
        f"R@1={test_metrics.get('recall@1', float('nan')):.4f} "
        f"pairAUC={test_metrics.get('pair_auc', float('nan')):.4f}"
    )
    print(f'Output directory: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pathology-anchored semantic-unit alignment experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_root', type=str, required=True, help='UTSW-Glioma root directory')
    parser.add_argument('--metadata_tsv', type=str, default=None, help='UTSW metadata TSV path')
    parser.add_argument('--variant', type=str, default='full', choices=['full', 'clip', 'medclip_style', 'dcca', 'graph_shared_only', 'hgt', 'no_anchor', 'graph_only', 'modality_vector', 'no_private', 'no_graph'])
    parser.add_argument('--out_dir', type=str, default='output/semantic_alignment_experiment')

    parser.add_argument('--roi_size', type=int, default=96)
    parser.add_argument('--z_slices', type=int, default=7)
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--prefer_registered', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--node_mode', type=str, default='regions', choices=['regions', 'modalities'])
    parser.add_argument('--graph_type', type=str, default='learnable', choices=['no_graph', 'fixed', 'similarity', 'learnable', 'random'])
    parser.add_argument('--shared_dim', type=int, default=128)
    parser.add_argument('--private_dim', type=int, default=128)
    parser.add_argument('--diffusion_T', type=int, default=20)
    parser.add_argument('--graph_warmup_epochs', type=int, default=5)
    parser.add_argument('--graph_ema_momentum', type=float, default=0.95)
    parser.add_argument('--graph_ema_blend', type=float, default=0.5)
    parser.add_argument('--diffusion_init_alpha', type=float, default=0.05)
    parser.add_argument('--shared_private_mix_init', type=float, default=0.05)
    parser.add_argument('--classifier_private_scale_init', type=float, default=0.05)
    parser.add_argument('--diffusion_max_ratio', type=float, default=0.5)
    parser.add_argument('--no_private', action='store_true')
    parser.add_argument('--no_diffusion', action='store_true')

    parser.add_argument('--target_policy', type=str, default='region_rules', choices=['region_rules', 'all_patient_anchors'])
    parser.add_argument('--exclude_pathology_anchors', action='store_true')
    parser.add_argument('--exclude_molecular_anchors', action='store_true')
    parser.add_argument('--include_clinical_anchors', action='store_true')
    parser.add_argument('--align_max_cases', type=int, default=50)
    parser.add_argument('--graph_top_k', type=int, default=3)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--alignment_objective', type=str, default='clip', choices=['clip', 'medclip', 'dcca'])
    parser.add_argument('--dcca_reg', type=float, default=1e-3)
    parser.add_argument('--dcca_clip_weight', type=float, default=0.2)
    parser.add_argument('--lambda_anchor', type=float, default=0.05)
    parser.add_argument('--lambda_cons', type=float, default=0.05)
    parser.add_argument('--lambda_decouple', type=float, default=0.01)
    parser.add_argument('--lambda_leak', type=float, default=0.02)
    parser.add_argument('--lambda_diff', type=float, default=0.05)
    parser.add_argument('--lambda_diff_norm', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
