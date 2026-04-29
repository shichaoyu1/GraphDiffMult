"""
Semantic graph visualization for glioma experiments.

The graph shown here is intentionally semantic-unit based:
MRI lesion regions are nodes, and pathology/molecular findings are visual
anchor nodes. Molecular/pathology nodes are used for visualization only.
"""

import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch


MRI_REGION_NAMES = ['Necrotic/Core', 'Edema', 'Enhancing']
MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']


def adjacency_to_laplacian(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    adjacency_sym = 0.5 * (adjacency + adjacency.T)
    degree = np.diag(adjacency_sym.sum(axis=1))
    return degree - adjacency_sym


def node_names_for_mode(node_mode):
    if node_mode == 'regions':
        return MRI_REGION_NAMES
    return MODALITY_NAMES


def metadata_anchor_nodes(metadata):
    grade = metadata.get('Tumor Grade') or 'NA'
    idh = metadata.get('IDH') or 'NA'
    mgmt = metadata.get('MGMT') or 'NA'
    codeletion = metadata.get('1p19Q CODEL') or 'NA'
    return [
        f'Pathology\nWHO grade {grade}',
        f'Gene\nIDH {idh}',
        f'Gene\nMGMT {mgmt}',
        f'Gene\n1p/19q {codeletion}',
    ]


def build_initial_semantic_adjacency(metadata=None):
    metadata = metadata or {}
    node_names = MRI_REGION_NAMES + metadata_anchor_nodes(metadata)
    groups = ['MRI', 'MRI', 'MRI', 'Pathology', 'Gene', 'Gene', 'Gene']
    n_nodes = len(node_names)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    # MRI region relations are conceptual priors, not learned weights.
    region_edges = {
        (0, 1): 0.35,
        (0, 2): 0.65,
        (1, 2): 0.45,
    }
    for (source, target), weight in region_edges.items():
        adjacency[source, target] = weight
        adjacency[target, source] = weight

    grade_text = str(metadata.get('Tumor Grade') or '').strip()
    try:
        grade_strength = min(max((float(grade_text) - 1.0) / 3.0, 0.2), 1.0)
    except ValueError:
        grade_strength = 0.55

    # Visual semantic anchors. These are not model inputs during prediction.
    adjacency[0, 3] = adjacency[3, 0] = 0.55 * grade_strength
    adjacency[2, 3] = adjacency[3, 2] = 0.75 * grade_strength
    adjacency[1, 3] = adjacency[3, 1] = 0.35 * grade_strength

    adjacency[4, 3] = adjacency[3, 4] = 0.55
    adjacency[5, 3] = adjacency[3, 5] = 0.35
    adjacency[6, 3] = adjacency[3, 6] = 0.45
    adjacency[4, 2] = adjacency[2, 4] = 0.25
    adjacency[5, 2] = adjacency[2, 5] = 0.20
    adjacency[6, 0] = adjacency[0, 6] = 0.25

    return node_names, groups, adjacency


def _positions(groups):
    positions = {}
    mri_ids = [idx for idx, group in enumerate(groups) if group == 'MRI']
    anchor_ids = [idx for idx, group in enumerate(groups) if group != 'MRI']

    for order, idx in enumerate(mri_ids):
        if len(mri_ids) <= 1:
            y = 0.5
        else:
            y = 0.86 - order * (0.72 / max(len(mri_ids) - 1, 1))
        positions[idx] = (0.20, y)

    for order, idx in enumerate(anchor_ids):
        if len(anchor_ids) <= 1:
            y = 0.5
        else:
            y = 0.90 - order * (0.80 / max(len(anchor_ids) - 1, 1))
        positions[idx] = (0.76, y)

    return positions


def plot_semantic_graph(node_names, adjacency, groups=None, save_path='semantic_graph.png', title='Semantic Unit Graph'):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    groups = groups or ['MRI'] * len(node_names)
    positions = _positions(groups)
    colors = {
        'MRI': '#5DADE2',
        'Pathology': '#E67E22',
        'Gene': '#58D68D',
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_weight = float(adjacency.max()) if adjacency.size else 1.0
    max_weight = max(max_weight, 1e-6)
    for source in range(len(node_names)):
        for target in range(len(node_names)):
            weight = float(adjacency[source, target])
            if source == target or weight <= 1e-6:
                continue
            if source > target and abs(weight - adjacency[target, source]) < 1e-6:
                continue

            x0, y0 = positions[source]
            x1, y1 = positions[target]
            alpha = 0.20 + 0.70 * weight / max_weight
            line_width = 0.8 + 4.0 * weight / max_weight
            arrow = '->' if abs(weight - adjacency[target, source]) > 1e-6 else '-'
            ax.annotate(
                '',
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    'arrowstyle': arrow,
                    'color': '#c8d6e5',
                    'lw': line_width,
                    'alpha': alpha,
                    'connectionstyle': 'arc3,rad=0.08',
                    'shrinkA': 28,
                    'shrinkB': 28,
                },
            )
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.025,
                f'{weight:.2f}',
                color='#dfe6e9',
                fontsize=8,
                ha='center',
                va='center',
                alpha=alpha,
            )

    for idx, name in enumerate(node_names):
        x, y = positions[idx]
        group = groups[idx]
        color = colors.get(group, '#a29bfe')
        ax.scatter([x], [y], s=2200, color=color, alpha=0.25, edgecolor=color, linewidth=2)
        ax.text(x, y, name, color='white', fontsize=10, ha='center', va='center')

    ax.text(0.5, 0.97, title, color='white', fontsize=16, ha='center', va='top', fontweight='bold')
    ax.text(
        0.5,
        0.03,
        'MRI region nodes are trainable semantic units; pathology/gene nodes are visual anchors only.',
        color='#95a5a6',
        fontsize=9,
        ha='center',
    )
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_matrix(matrix, node_names, save_path, title, cmap='coolwarm'):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(node_names)))
    ax.set_yticks(range(len(node_names)))
    ax.set_xticklabels(node_names, rotation=45, ha='right', color='white', fontsize=8)
    ax.set_yticklabels(node_names, color='white', fontsize=8)
    ax.set_title(title, color='white', fontsize=14, pad=14)
    ax.tick_params(colors='white')

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f'{matrix[row, col]:.2f}', ha='center', va='center', color='white', fontsize=7)

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_initial_semantic_visuals(case, out_dir):
    node_names, groups, adjacency = build_initial_semantic_adjacency(case.get('metadata', {}))
    laplacian = adjacency_to_laplacian(adjacency)
    plot_semantic_graph(
        node_names,
        adjacency,
        groups,
        save_path=os.path.join(out_dir, 'semantic_graph_initial.png'),
        title=f"Initial Semantic Unit Graph - {case.get('subject_id', '')}",
    )
    plot_matrix(
        laplacian,
        node_names,
        save_path=os.path.join(out_dir, 'semantic_laplacian_initial.png'),
        title='Initial Semantic Laplacian',
    )


def save_trained_semantic_visuals(adjacency, node_mode, out_dir):
    node_names = node_names_for_mode(node_mode)
    groups = ['MRI'] * len(node_names)
    laplacian = adjacency_to_laplacian(adjacency)
    plot_semantic_graph(
        node_names,
        adjacency,
        groups,
        save_path=os.path.join(out_dir, 'semantic_graph_trained.png'),
        title='Trained Learned Semantic Graph',
    )
    plot_matrix(
        laplacian,
        node_names,
        save_path=os.path.join(out_dir, 'semantic_laplacian_trained.png'),
        title='Trained Laplacian Matrix',
    )
    plot_matrix(
        adjacency,
        node_names,
        save_path=os.path.join(out_dir, 'semantic_adjacency_trained.png'),
        title='Trained Adjacency Matrix',
        cmap='magma',
    )


def _pca2d(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] == 1:
        projected = centered @ vt[:1].T
        return np.concatenate([projected, np.zeros_like(projected)], axis=1)
    return centered @ vt[:2].T


def _anchor_source_for_task(task):
    task = task.lower()
    if task == 'grade':
        return 'Pathology'
    return 'Gene'


def _anchor_label_for_task(task, label_name):
    task = task.lower()
    if task == 'grade':
        return f'Pathology\n{label_name}'
    return f'Gene report\n{label_name}'


def save_alignment_visualization(
    model,
    loader,
    device,
    task,
    label_names,
    node_mode,
    out_dir,
    max_cases=50,
):
    model.eval()
    node_names = node_names_for_mode(node_mode)
    records = []
    edges = []
    distance_summary = {name: [] for name in node_names}
    seen_cases = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            region_masks = batch.get('region_masks')
            if region_masks is not None:
                region_masks = region_masks.to(device)

            output = model(images, labels=labels, region_masks=region_masks, return_extras=True)
            shared = output['extras']['shared'].detach().cpu().numpy()
            anchors = model.anchor_prototypes[labels].detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            subject_ids = batch.get('subject_id', [f'case_{idx}' for idx in range(shared.shape[0])])

            for sample_idx in range(shared.shape[0]):
                if seen_cases >= max_cases:
                    break
                subject_id = subject_ids[sample_idx]
                class_name = label_names[int(labels_np[sample_idx])]
                anchor_source = _anchor_source_for_task(task)
                anchor_id = len(records)
                records.append({
                    'vector': anchors[sample_idx],
                    'source': anchor_source,
                    'semantic': _anchor_label_for_task(task, class_name),
                    'subject_id': subject_id,
                    'label': class_name,
                    'node_name': anchor_source,
                })

                for node_idx, node_name in enumerate(node_names):
                    point_id = len(records)
                    records.append({
                        'vector': shared[sample_idx, node_idx],
                        'source': 'MRI',
                        'semantic': node_name,
                        'subject_id': subject_id,
                        'label': class_name,
                        'node_name': node_name,
                    })
                    edges.append((point_id, anchor_id))
                    distance = float(np.linalg.norm(shared[sample_idx, node_idx] - anchors[sample_idx]))
                    distance_summary[node_name].append(distance)
                seen_cases += 1
            if seen_cases >= max_cases:
                break

    if not records:
        return

    vectors = np.stack([record['vector'] for record in records], axis=0)
    coords = _pca2d(vectors)

    colors = {
        'MRI': '#2E86DE',
        'Pathology': '#E67E22',
        'Gene': '#27AE60',
    }
    markers = {
        'Necrotic/Core': '^',
        'Edema': 'o',
        'Enhancing': '*',
        'T1': '^',
        'T1ce': '*',
        'T2': 's',
        'FLAIR': 'o',
        'Pathology': 'D',
        'Gene': 'X',
    }

    fig, ax = plt.subplots(figsize=(10, 8.8))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#101725')

    for source_idx, anchor_idx in edges:
        x0, y0 = coords[source_idx]
        x1, y1 = coords[anchor_idx]
        ax.plot([x0, x1], [y0, y1], color='#d0d7de', alpha=0.10, linewidth=0.7)

    legend_keys = set()
    for idx, record in enumerate(records):
        source = record['source']
        semantic = record['semantic']
        node_name = record['node_name']
        color = colors.get(source, '#a29bfe')
        marker = markers.get(node_name, 'o')
        size = 160 if marker == '*' else 72
        semantic_label = semantic.replace('\n', ' ')
        label = f'{source}: {semantic_label}' if source != 'MRI' else f'{source}: {node_name}'
        show_label = label not in legend_keys
        legend_keys.add(label)
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=size,
            c=color,
            marker=marker,
            edgecolors='white',
            linewidths=0.6,
            alpha=0.78 if source == 'MRI' else 0.92,
            label=label if show_label else None,
        )

    ax.set_title(
        f'Cross-source Semantic Unit Alignment (n={seen_cases})',
        color='white',
        fontsize=15,
        pad=14,
        fontweight='bold',
    )
    ax.set_xlabel('PC1 of shared semantic space', color='white')
    ax.set_ylabel('PC2 of shared semantic space', color='white')
    ax.tick_params(colors='#bdc3c7')
    for spine in ax.spines.values():
        spine.set_color('#34495e')
    ax.grid(color='white', alpha=0.10)
    ax.legend(loc='best', fontsize=8, facecolor='#17202a', edgecolor='#566573', labelcolor='white')
    ax.text(
        0.5,
        -0.12,
        'Lines connect MRI lesion semantic units to the supervised pathology/gene prototype anchor. '
        'Anchors regularize training; they are not test-time inputs.',
        transform=ax.transAxes,
        ha='center',
        va='top',
        color='#95a5a6',
        fontsize=8,
    )

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'semantic_alignment_trained.png'), dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    metrics = {
        'task': task,
        'node_mode': node_mode,
        'max_cases': max_cases,
        'case_count': seen_cases,
        'mean_distance_to_anchor': {
            node_name: float(np.mean(values)) if values else None
            for node_name, values in distance_summary.items()
        },
    }
    with open(os.path.join(out_dir, 'semantic_alignment_metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
