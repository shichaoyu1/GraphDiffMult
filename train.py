"""
train.py - BraTS / UTSW-Glioma demo entry point.

Examples:
    python train.py --patient_dir datasetDemo/BraTS2021_00060
    python train.py --patient_dir D:/dataset/.../UTSW-Glioma --patient_id BT0001
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BraTSPatchDataset, collate_fn, load_brats_patient, resolve_patient_dir
from model import MultimodalFusionNet
from visualize import (
    plot_diffusion_process,
    plot_graph_topology,
    plot_modality_slices,
    plot_training_results,
)


def train(
    model,
    loader,
    epochs: int,
    lr: float,
    lambda_diff: float,
    lambda_graph: float,
    device: str = 'cpu',
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {key: [] for key in ['total', 'task', 'diff', 'graph']}

    model.to(device)
    model.train()

    print('\n' + '-' * 60)
    print(f'  Train config: epochs={epochs}, lr={lr}, lambda_diff={lambda_diff}, lambda_graph={lambda_graph}')
    print(f'  Device: {device.upper()}')
    print('-' * 60)

    for epoch in range(1, epochs + 1):
        epoch_losses = {key: [] for key in history}

        for imgs, labels in loader:
            imgs = [image.to(device) for image in imgs]
            labels = labels.to(device)

            logits, diff_loss, graph_loss = model(imgs)
            task_loss = F.cross_entropy(logits, labels)
            total = task_loss + lambda_diff * diff_loss + lambda_graph * graph_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key, value in zip(
                ['total', 'task', 'diff', 'graph'],
                [total, task_loss, diff_loss, graph_loss],
            ):
                epoch_losses[key].append(value.item())

        scheduler.step()
        for key in history:
            history[key].append(float(np.mean(epoch_losses[key])))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"total={history['total'][-1]:.4f}  "
                f"task={history['task'][-1]:.4f}  "
                f"diff={history['diff'][-1]:.4f}  "
                f"graph={history['graph'][-1]:.4f}"
            )

    print('-' * 60)
    return history


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    patient_dir = resolve_patient_dir(args.patient_dir, args.patient_id)
    print(f'\nUsing patient folder: {patient_dir}')

    print('\n[Step 1] Loading volumes for visualization...')
    vols, seg, _ = load_brats_patient(
        patient_dir,
        prefer_registered=args.prefer_registered,
        metadata_tsv=args.metadata_tsv,
    )

    print('\n[Step 1b] Saving multimodal slice preview...')
    plot_modality_slices(
        vols,
        seg,
        save_path=os.path.join(args.out_dir, 'brats_slices.png'),
    )

    print('\n[Step 2] Building patch dataset...')
    ds = BraTSPatchDataset(
        patient_dir=patient_dir,
        patch_size=args.patch_size,
        n_patches=args.n_patches,
        prefer_registered=args.prefer_registered,
        metadata_tsv=args.metadata_tsv,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )

    print('\n[Step 3] Building multimodal fusion model...')
    model = MultimodalFusionNet(
        num_classes=3,
        feat_dim=args.feat_dim,
        num_modalities=3,
        num_heads=4,
        diffusion_T=100,
    )
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'  Trainable parameters: {n_params:,}')

    print('\n[Step 4] Training...')
    history = train(
        model,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        lambda_diff=args.lambda_diff,
        lambda_graph=args.lambda_graph,
        device=device,
    )

    print('\n[Step 5] Preparing inference patches...')
    model.eval().cpu()

    all_imgs = [[], [], []]
    all_labels = []
    all_zs = []

    for cy, cx, z in ds.samples:
        patches, label = ds[len(all_labels)]
        for modality_idx in range(3):
            all_imgs[modality_idx].append(patches[modality_idx])
        all_labels.append(label)
        all_zs.append(z)

    imgs3 = [torch.stack(all_imgs[modality_idx]) for modality_idx in range(3)]
    labels = torch.stack(all_labels)
    patch_zs = all_zs

    print('\n[Step 6] Saving visualizations...')
    with torch.no_grad():
        _, _, _, _, raw_feats, _ = model(imgs3, return_extras=True)

    plot_training_results(
        model,
        history,
        imgs3,
        labels,
        vols=vols,
        seg=seg,
        patch_zs=patch_zs,
        mod_names=['T1', 'FLAIR', 'T1ce'],
        save_path=os.path.join(args.out_dir, 'brats_results.png'),
    )

    plot_diffusion_process(
        model,
        raw_feats,
        save_path=os.path.join(args.out_dir, 'diffusion_process.png'),
    )

    plot_graph_topology(
        model,
        imgs3,
        labels,
        mod_names=['T1', 'FLAIR', 'T1ce'],
        save_path=os.path.join(args.out_dir, 'graph_topology.png'),
    )

    print('\n' + '=' * 60)
    print(f'  Done. Output directory: {os.path.abspath(args.out_dir)}')
    print('    brats_slices.png')
    print('    brats_results.png')
    print('    diffusion_process.png')
    print('    graph_topology.png')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BraTS / UTSW-Glioma multimodal diffusion + graph fusion demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--patient_dir',
        type=str,
        required=True,
        help='Patient folder, or dataset root such as UTSW-Glioma',
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        default=None,
        help='Case ID under a dataset root, e.g. BT0001',
    )
    parser.add_argument(
        '--metadata_tsv',
        type=str,
        default=None,
        help='Path to UTSW_Glioma_Metadata-2-1.tsv',
    )
    parser.add_argument(
        '--prefer_registered',
        action='store_true',
        help='Prefer *_ants.nii.gz registered modality files when available',
    )
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')

    parser.add_argument('--patch_size', type=int, default=32, help='Patch side length in pixels')
    parser.add_argument('--n_patches', type=int, default=200, help='Number of sampled patches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--feat_dim', type=int, default=64, help='Feature dimension')

    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lambda_diff', type=float, default=0.1, help='Diffusion loss weight')
    parser.add_argument('--lambda_graph', type=float, default=0.05, help='Graph loss weight')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    main(parser.parse_args())
