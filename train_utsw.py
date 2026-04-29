"""
Train the pathology-guided graph/diffusion experiment on UTSW-Glioma.

This is the main patient-level experiment entry point:
IDH, MGMT, 1p/19q, or WHO grade prediction from 2.5D tumor ROI crops.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment_dataset import (
    UTSWROIPatientDataset,
    describe_cases,
    discover_utsw_labeled_cases,
    label_names_for_task,
    stratified_split,
)
from experiment_model import GliomaGraphDiffusionNet
from semantic_graph_visualize import (
    save_alignment_visualization,
    save_initial_semantic_visuals,
    save_trained_semantic_visuals,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_variant(args):
    if args.variant == 'shared_private':
        args.graph_type = 'no_graph'
        args.no_anchor = True
        args.no_diffusion = True
    elif args.variant == 'graph':
        args.no_anchor = True
        args.no_diffusion = True
    elif args.variant == 'anchor':
        args.no_diffusion = True
    elif args.variant == 'full':
        pass
    else:
        raise ValueError(f'Unsupported variant: {args.variant}')
    return args


def class_weights(cases, num_classes, device):
    counts = Counter(case['label'] for case in cases)
    total = sum(counts.values())
    weights = []
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        weights.append(0.0 if count == 0 else total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def binary_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float('nan')

    try:
        from scipy.stats import rankdata

        ranks = rankdata(scores)
    except Exception:
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)

    pos_ranks = ranks[pos].sum()
    return float((pos_ranks - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def macro_auc(labels, probs, num_classes):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if num_classes == 2:
        return binary_auc(labels, probs[:, 1])

    aucs = []
    for class_idx in range(num_classes):
        binary_labels = (labels == class_idx).astype(np.int64)
        aucs.append(binary_auc(binary_labels, probs[:, class_idx]))
    aucs = [auc for auc in aucs if not np.isnan(auc)]
    return float(np.mean(aucs)) if aucs else float('nan')


def classification_metrics(labels, probs, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(axis=1)

    accuracy = float((preds == labels).mean()) if len(labels) else float('nan')
    f1_scores = []
    recalls = []
    specificities = []
    for class_idx in range(num_classes):
        tp = int(((preds == class_idx) & (labels == class_idx)).sum())
        fp = int(((preds == class_idx) & (labels != class_idx)).sum())
        fn = int(((preds != class_idx) & (labels == class_idx)).sum())
        tn = int(((preds != class_idx) & (labels != class_idx)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        f1_scores.append(f1)
        recalls.append(recall)
        specificities.append(specificity)

    return {
        'auc': macro_auc(labels, probs, num_classes),
        'accuracy': accuracy,
        'f1_macro': float(np.mean(f1_scores)),
        'balanced_accuracy': float(np.mean(recalls)),
        'sensitivity_macro': float(np.mean(recalls)),
        'specificity_macro': float(np.mean(specificities)),
    }


def run_epoch(model, loader, optimizer, device, args, ce_weight=None):
    training = optimizer is not None
    model.train(training)

    totals = Counter()
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch['images'].to(device)
        region_masks = batch.get('region_masks')
        if region_masks is not None:
            region_masks = region_masks.to(device)
        labels = batch['label'].to(device)

        with torch.set_grad_enabled(training):
            output = model(images, labels=labels, region_masks=region_masks)
            logits = output['logits']
            losses = output['losses']
            task_loss = F.cross_entropy(logits, labels, weight=ce_weight)
            total = (
                task_loss
                + args.lambda_cons * losses['cons']
                + args.lambda_anchor * losses['anchor']
                + args.lambda_decouple * losses['decouple']
                + args.lambda_diff * losses['diff']
            )

            if training:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = labels.shape[0]
        totals['total'] += float(total.detach().cpu()) * batch_size
        totals['task'] += float(task_loss.detach().cpu()) * batch_size
        for name in ['cons', 'anchor', 'decouple', 'diff', 'graph_energy']:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
        totals['n'] += batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(torch.softmax(logits.detach().cpu(), dim=-1).numpy().tolist())

    n_samples = max(totals['n'], 1)
    loss_summary = {name: totals[name] / n_samples for name in ['total', 'task', 'cons', 'anchor', 'decouple', 'diff', 'graph_energy']}
    metrics = classification_metrics(all_labels, all_probs, model.classifier[-1].out_features)
    return {**loss_summary, **metrics}


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


def alignment_cases_from_splits(splits, split_name):
    if split_name == 'all':
        merged = []
        seen = set()
        for name in ['train', 'val', 'test']:
            for case in splits[name]:
                subject_id = case['subject_id']
                if subject_id not in seen:
                    merged.append(case)
                    seen.add(subject_id)
        return merged
    return list(splits[split_name])


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def collect_mean_adjacency(model, loader, device):
    model.eval()
    matrices = []
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            region_masks = batch.get('region_masks')
            if region_masks is not None:
                region_masks = region_masks.to(device)
            output = model(images, region_masks=region_masks, return_extras=True)
            matrices.append(output['extras']['adjacency'].detach().cpu())
    if not matrices:
        return None
    adjacency = torch.cat(matrices, dim=0).mean(dim=0).numpy()
    return adjacency


def main(args):
    args = apply_variant(args)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    label_names = label_names_for_task(args.task)
    num_classes = len(label_names)

    cases = discover_utsw_labeled_cases(
        args.data_root,
        task=args.task,
        metadata_tsv=args.metadata_tsv,
        max_cases=args.max_cases,
        seed=args.seed,
    )
    if len(cases) < 2:
        raise ValueError(f'Need at least 2 labeled cases for task={args.task}; found {len(cases)}')

    splits = stratified_split(
        cases,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not splits['val']:
        splits['val'] = list(splits['test'])
    if not splits['test']:
        splits['test'] = list(splits['val'] or splits['train'])

    print(f'Task: {args.task} | labels: {label_names}')
    print(f'Total cases: {len(cases)} | distribution: {describe_cases(cases)}')
    for split_name in ['train', 'val', 'test']:
        print(f'{split_name}: {len(splits[split_name])} | distribution: {describe_cases(splits[split_name])}')
    print(f'Variant: {args.variant} | graph={args.graph_type} | anchor={not args.no_anchor} | private={not args.no_private} | diffusion={not args.no_diffusion}')
    save_initial_semantic_visuals(splits['train'][0], args.out_dir)

    loaders = {split: make_loader(split_cases, args, split) for split, split_cases in splits.items()}
    align_cases = alignment_cases_from_splits(splits, args.align_split)
    align_loader = make_loader(align_cases, args, 'align')

    model = GliomaGraphDiffusionNet(
        num_classes=num_classes,
        z_slices=args.z_slices,
        node_mode=args.node_mode,
        feat_dim=args.feat_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        graph_type=args.graph_type,
        diffusion_T=args.diffusion_T,
        use_anchor=not args.no_anchor,
        use_private=not args.no_private,
        use_diffusion=not args.no_diffusion,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ce_weight = class_weights(splits['train'], num_classes, device) if args.class_weight else None

    history = []
    best_score = -float('inf')
    best_path = os.path.join(args.out_dir, 'best_model.pt')

    save_json(os.path.join(args.out_dir, 'config.json'), vars(args))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders['train'], optimizer, device, args, ce_weight=ce_weight)
        val_metrics = run_epoch(model, loaders['val'], None, device, args, ce_weight=None)
        scheduler.step()

        epoch_record = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        }
        history.append(epoch_record)
        save_json(os.path.join(args.out_dir, 'history.json'), history)

        score = val_metrics['auc']
        if np.isnan(score):
            score = val_metrics['balanced_accuracy']
        if score > best_score:
            best_score = score
            torch.save({'model': model.state_dict(), 'args': vars(args), 'label_names': label_names}, best_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['total']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"graphE={val_metrics['graph_energy']:.4f}"
        )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    trained_adjacency = collect_mean_adjacency(model, loaders['test'], device)
    if trained_adjacency is not None:
        save_trained_semantic_visuals(trained_adjacency, args.node_mode, args.out_dir)
    save_alignment_visualization(
        model,
        align_loader,
        device,
        args.task,
        label_names,
        args.node_mode,
        args.out_dir,
        max_cases=args.align_max_cases,
    )

    test_metrics = run_epoch(model, loaders['test'], None, device, args, ce_weight=None)
    save_json(os.path.join(args.out_dir, 'test_metrics.json'), test_metrics)
    print(f"Test: auc={test_metrics['auc']:.4f} acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1_macro']:.4f} bacc={test_metrics['balanced_accuracy']:.4f}")
    print(f'Output directory: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UTSW-Glioma pathology-guided graph/diffusion experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--data_root', type=str, required=True, help='UTSW-Glioma root directory')
    parser.add_argument('--metadata_tsv', type=str, default=None, help='UTSW metadata TSV path')
    parser.add_argument('--task', type=str, default='idh', choices=['idh', 'mgmt', '1p19q', 'grade'])
    parser.add_argument('--variant', type=str, default='full', choices=['shared_private', 'graph', 'anchor', 'full'])
    parser.add_argument('--graph_type', type=str, default='learnable', choices=['no_graph', 'fixed', 'similarity', 'learnable', 'random'])
    parser.add_argument('--out_dir', type=str, default='output/utsw_experiment')

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
    parser.add_argument('--align_split', type=str, default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--align_max_cases', type=int, default=50)
    parser.add_argument('--shared_dim', type=int, default=128)
    parser.add_argument('--private_dim', type=int, default=128)
    parser.add_argument('--diffusion_T', type=int, default=20)
    parser.add_argument('--no_anchor', action='store_true')
    parser.add_argument('--no_private', action='store_true')
    parser.add_argument('--no_diffusion', action='store_true')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_cons', type=float, default=0.05)
    parser.add_argument('--lambda_anchor', type=float, default=0.05)
    parser.add_argument('--lambda_decouple', type=float, default=0.01)
    parser.add_argument('--lambda_diff', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--class_weight', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
