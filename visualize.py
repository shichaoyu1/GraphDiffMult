"""
visualize.py — 全套可视化函数

函数列表:
  plot_modality_slices   : 四模态三视图切片 + 分割叠加
  plot_training_results  : 训练损失 + 分类概率 + GAT 注意力 + 扩散轨迹 + PCA 投影
  plot_diffusion_process : 扩散过程三面板（轨迹/分布演变/去噪）
  plot_graph_topology    : 图拓扑可视化（有向图 + 多头热力图 + PCA）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from numpy.linalg import svd

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 配色常量
# ─────────────────────────────────────────────────────────
BG_DARK   = '#0a0c14'
BG_PANEL  = '#1a1d27'
BG_PANEL2 = '#111520'

MOD_COLS  = ['#b39ddb', '#f4c775', '#e07b54']   # T1=紫, FLAIR=黄, T1ce=橙
LOSS_COLS = {
    'total': '#e07b54', 'task': '#b39ddb',
    'diff':  '#f4c775', 'graph': '#80cbc4',
}
SEG_COLORS = ['#888888', '#3399ff', '#ffdd00']   # 背景/水肿/增强

SEG_CMAP = ListedColormap(['none', '#3399ff', '#ff4444', 'none', '#ffdd00'])


def _rc():
    plt.rcParams.update({
        'text.color': 'white', 'axes.labelcolor': 'white',
        'xtick.color': '#aaaaaa', 'ytick.color': '#aaaaaa',
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def _panel(ax, title=None, color='white'):
    ax.set_facecolor(BG_PANEL)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(alpha=0.12, color='white')
    if title:
        ax.set_title(title, color=color, fontsize=10, pad=6)


def _norm(v):
    fg = v[v > 0]
    if len(fg) == 0:
        return v
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)


def _pca2d(X):
    X = X - X.mean(0)
    _, _, Vt = svd(X, full_matrices=False)
    return X @ Vt[:2].T


def _seg_rgba(seg_sl):
    rgba = np.zeros((*seg_sl.shape, 4))
    rgba[seg_sl == 2] = [0.2, 0.6, 1.0, 0.65]   # edema  → 蓝
    rgba[seg_sl == 1] = [1.0, 0.27, 0.27, 0.65]  # necrotic → 红
    rgba[seg_sl == 4] = [1.0, 0.87, 0.0,  0.65]  # enhancing → 黄
    return rgba


# ─────────────────────────────────────────────────────────
# 1. 四模态切片展示
# ─────────────────────────────────────────────────────────
def plot_modality_slices(vols: dict, seg: np.ndarray, save_path: str = 'brats_slices.png'):
    """
    vols : {'t1':ndarray, 't2':ndarray, 'flair':ndarray, 't1ce':ndarray}  shape=(H,W,D)
    seg  : ndarray (H,W,D)
    """
    _rc()
    tumor_mask   = seg > 0
    tumor_per_z  = tumor_mask.sum(axis=(0, 1))
    best_z  = int(tumor_per_z.argmax())
    best_y  = int(tumor_mask.any(axis=(0, 2)).nonzero()[0].mean())
    best_x  = int(tumor_mask.any(axis=(1, 2)).nonzero()[0].mean())

    rows_cfg = [
        # (modality_key, title, title_color)
        ('t1',    'T1',         '#b39ddb'),
        ('t2',    'T2',         '#80cbc4'),
        ('flair', 'FLAIR',      '#f4c775'),
        ('t1ce',  'T1ce',       '#e07b54'),
    ]
    views = [
        (best_z, 'axial',    lambda v, k: _norm(v[k][:, :, best_z]).T),
        (best_y, 'coronal',  lambda v, k: _norm(v[k][:, best_y, :]).T),
        (best_x, 'sagittal', lambda v, k: _norm(v[k][best_x, :, :]).T),
    ]
    seg_slices = {
        'axial':    seg[:, :, best_z].T,
        'coronal':  seg[:, best_y, :].T,
        'sagittal': seg[best_x, :, :].T,
    }

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(BG_DARK)
    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.06)

    for ri, (mod_key, mod_name, title_col) in enumerate(rows_cfg):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[ri], wspace=0.04)
        for ci, (_, view_name, get_sl) in enumerate(views):
            ax = fig.add_subplot(inner[ci])
            ax.imshow(get_sl(vols, mod_key), cmap='gray', origin='lower', aspect='auto')
            ax.imshow(_seg_rgba(seg_slices[view_name]), origin='lower', aspect='auto')
            if ri == 0:
                ax.set_title(view_name.capitalize(), color='#aaa', fontsize=10, pad=4)
            ax.axis('off')
        # Legend panel (4th column)
        ax_leg = fig.add_subplot(inner[3])
        ax_leg.set_facecolor('#0d0f1a')
        ax_leg.axis('off')
        ax_leg.text(0.05, 0.88, mod_name, color=title_col,
                    fontsize=14, fontweight='500', transform=ax_leg.transAxes)
        for li, (col, txt) in enumerate([
            ('#3399ff', 'Edema'),
            ('#ff4444', 'Necrotic'),
            ('#ffdd00', 'Enhancing'),
        ]):
            y = 0.60 - li * 0.20
            ax_leg.add_patch(plt.Rectangle(
                (0.05, y - 0.06), 0.15, 0.13, color=col, alpha=0.85,
                transform=ax_leg.transAxes))
            ax_leg.text(0.26, y, txt, color='white', fontsize=9,
                        va='center', transform=ax_leg.transAxes)

    fig.text(0.5, 0.995,
             f'BraTS patient — axial z={best_z} | coronal y={best_y} | sagittal x={best_x}',
             ha='center', color='white', fontsize=13, fontweight='500')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[图1] 切片展示已保存: {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────
# 2. 训练结果综合图
# ─────────────────────────────────────────────────────────
def plot_training_results(
    model, history: dict, imgs3: list, labels: torch.Tensor,
    vols: dict, seg: np.ndarray, patch_zs: list,
    mod_names: list = None, save_path: str = 'brats_results.png',
):
    _rc()
    mod_names = mod_names or ['T1', 'FLAIR', 'T1ce']
    best_z    = patch_zs[len(patch_zs) // 2]

    model.eval()
    with torch.no_grad():
        logits, _, _, attn_w, raw_feats, graph_feats = model(imgs3, return_extras=True)
        probs    = F.softmax(logits, -1).numpy()
        cond     = torch.cat([raw_feats[:, 0], raw_feats[:, 2]], -1)
        gen_feat = model.diffusion.sample(cond, modality_idx=1, steps=20)

    attn_mean = attn_w.mean(0).cpu().numpy()      # [N,N,H]
    raw_np    = raw_feats.cpu().numpy()
    graph_np  = graph_feats.cpu().numpy()
    diff_mod  = model.diffusion
    B         = raw_feats.shape[0]
    z0_ref    = raw_feats[:, 1]

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.42, wspace=0.26)

    # ── 行0: 三模态轴状切片 + Seg ──
    for ci, (mk, col, ttl) in enumerate(zip(
            ['t1', 'flair', 't1ce', None],
            ['#b39ddb', '#f4c775', '#e07b54', '#80cbc4'],
            ['T1', 'FLAIR', 'T1ce', 'Seg overlay'])):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor('#0d0f1a')
        ref = _norm(vols['t1'][:, :, best_z]).T
        if mk:
            ax.imshow(_norm(vols[mk][:, :, best_z]).T, cmap='gray',
                      origin='lower', aspect='auto')
        else:
            ax.imshow(ref, cmap='gray', origin='lower', aspect='auto')
        ax.imshow(_seg_rgba(seg[:, :, best_z].T), origin='lower', aspect='auto')
        ax.set_title(ttl, color=col, fontsize=11, fontweight='500', pad=5)
        ax.axis('off')

    # ── 行1: 训练损失 + 分类概率 ──
    ax_l = fig.add_subplot(gs[1, :2])
    _panel(ax_l, 'Training loss on real BraTS patches')
    ep = range(1, len(history['total']) + 1)
    for key, label in [('total','Total'),('task','Task (CE)'),
                        ('diff','Diffusion'),('graph','Graph contrastive')]:
        ax_l.plot(ep, history[key], color=LOSS_COLS[key], lw=2 if key=='total' else 1.5, label=label)
    ax_l.legend(fontsize=8, facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')
    ax_l.set_xlabel('Epoch')

    ax_p = fig.add_subplot(gs[1, 2:])
    _panel(ax_p, 'Predicted probabilities per patch')
    x = np.arange(3)
    for i in range(min(B, 20)):
        ax_p.plot(x, probs[i], 'o-', lw=1, ms=4,
                  color=SEG_COLORS[labels[i].item()], alpha=0.55)
    for lc, ln in zip(SEG_COLORS, ['Background', 'Edema/Necrotic', 'Enhancing']):
        ax_p.plot([], [], '-', color=lc, label=ln)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(['Background', 'Edema', 'Enhancing'], fontsize=9)
    ax_p.legend(fontsize=8, facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel('Probability')

    # ── 行2: GAT 注意力 均值 + 4 heads ──
    ax_a = fig.add_subplot(gs[2, :2])
    _panel(ax_a, 'GAT mean attention – real BraTS features')
    aw = attn_mean.mean(-1)
    im = ax_a.imshow(aw, cmap='magma', vmin=0, vmax=1, aspect='auto')
    ax_a.set_xticks(range(3)); ax_a.set_yticks(range(3))
    ax_a.set_xticklabels(mod_names, fontsize=10)
    ax_a.set_yticklabels(mod_names, fontsize=10)
    for ii in range(3):
        for jj in range(3):
            ax_a.text(jj, ii, f'{aw[ii,jj]:.3f}', ha='center', va='center',
                      color='white' if aw[ii,jj] < 0.6 else '#111',
                      fontsize=10, fontweight='500')
    plt.colorbar(im, ax=ax_a, fraction=0.046)

    igs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2, 2:],
                                            hspace=0.40, wspace=0.34)
    for h in range(4):
        axi = fig.add_subplot(igs[h // 2, h % 2])
        axi.set_facecolor(BG_PANEL2)
        awh = attn_mean[:, :, h]
        axi.imshow(awh, cmap='magma', vmin=0, vmax=1, aspect='auto')
        axi.set_xticks(range(3)); axi.set_yticks(range(3))
        axi.set_xticklabels(mod_names, fontsize=7, color='#aaa')
        axi.set_yticklabels(mod_names, fontsize=7, color='#aaa')
        axi.set_title(f'Head {h}', color='#aaa', fontsize=8)
        for ii in range(3):
            for jj in range(3):
                axi.text(jj, ii, f'{awh[ii,jj]:.2f}', ha='center', va='center',
                         color='white', fontsize=7)

    # ── 行3: 扩散轨迹 + 分布 ──
    ax_t  = fig.add_subplot(gs[3, :2])
    _panel(ax_t, 'Forward diffusion – real FLAIR features')
    ax_t2 = ax_t.twinx()
    ax_t2.set_facecolor(BG_PANEL)
    T_steps = list(range(0, diff_mod.T, 5))
    means2, stds2, ab_vals = [], [], []
    with torch.no_grad():
        for tv in T_steps:
            tt     = torch.tensor([tv] * B)
            zn, _  = diff_mod.q_sample(z0_ref, tt)
            means2.append(zn.mean().item())
            stds2.append(zn.std().item())
            ab_vals.append(diff_mod.alpha_bar[tv].item())
    ax_t.fill_between(T_steps,
                      np.array(means2) - np.array(stds2),
                      np.array(means2) + np.array(stds2),
                      color='#b39ddb', alpha=0.2)
    ax_t.plot(T_steps, means2, color='#b39ddb', lw=2, label='FLAIR mean±std')
    ax_t2.plot(T_steps, ab_vals, color='#80cbc4', lw=2, ls='--', label='ᾱₜ')
    ax_t.set_xlabel('Timestep t')
    ax_t.set_ylabel('Feature value', color='#b39ddb')
    ax_t2.set_ylabel('ᾱₜ', color='#80cbc4')
    l1, lb1 = ax_t.get_legend_handles_labels()
    l2, lb2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(l1 + l2, lb1 + lb2, fontsize=8,
                facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')

    ax_d = fig.add_subplot(gs[3, 2:])
    _panel(ax_d, 'Diffusion: real vs generated FLAIR features')
    rf   = raw_feats[:, 1].detach().numpy().flatten()
    gf2  = gen_feat.numpy().flatten()
    bins = np.linspace(min(rf.min(), gf2.min()) - .5,
                       max(rf.max(), gf2.max()) + .5, 50)
    ax_d.hist(rf,  bins=bins, color='#80cbc4', alpha=0.65, density=True, label='Real FLAIR')
    ax_d.hist(gf2, bins=bins, color='#e07b54', alpha=0.65, density=True,
              label='Generated FLAIR\n(cond. T1+T1ce)')
    ax_d.legend(fontsize=8, facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')
    ax_d.set_xlabel('Feature value')
    ax_d.set_ylabel('Density')

    # ── 行4: PCA 投影 ──
    for ci, (feats_np, title) in enumerate([
        (raw_np,   'Feature space before GNN'),
        (graph_np, 'Feature space after GNN'),
    ]):
        ax = fig.add_subplot(gs[4, ci * 2:(ci + 1) * 2])
        _panel(ax, title)
        mkrs = ['o', '^', 's']
        for m in range(3):
            pts = _pca2d(feats_np[:, m, :])
            for li, (lc, lname) in enumerate(
                    zip(SEG_COLORS, ['BG', 'Edema', 'Enh'])):
                mask = labels.numpy() == li
                if mask.sum() > 0:
                    ax.scatter(pts[mask, 0], pts[mask, 1],
                               color=lc, marker=mkrs[m], s=55, alpha=0.78,
                               label=(f'{mod_names[m]}/{lname}' if ci == 0 else None),
                               edgecolors=MOD_COLS[m], linewidths=0.8)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(colors='#888', labelsize=7)
        if ci == 0:
            ax.legend(fontsize=6.5, facecolor=BG_PANEL, edgecolor='#444',
                      labelcolor='white', ncol=3, loc='best')

    fig.text(0.5, 0.995,
             'BraTS Patient  ·  Diffusion + Graph Fusion Pipeline  ·  Training Results',
             ha='center', color='white', fontsize=14, fontweight='500')

    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[图2] 训练结果综合图已保存: {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────
# 3. 扩散过程专项可视化
# ─────────────────────────────────────────────────────────
def plot_diffusion_process(model, raw_feats: torch.Tensor,
                            save_path: str = 'diffusion_process.png'):
    _rc()
    diff = model.diffusion
    T    = diff.T
    B    = raw_feats.shape[0]
    z0   = raw_feats[:, 1]    # FLAIR 特征

    # 前向轨迹
    timesteps  = list(range(0, T, 5))
    means, stds, ab_vals = [], [], []
    with torch.no_grad():
        for tv in timesteps:
            tt    = torch.tensor([tv] * B)
            zn, _ = diff.q_sample(z0, tt)
            means.append(zn.mean().item())
            stds.append(zn.std().item())
            ab_vals.append(diff.alpha_bar[tv].item())

    # 不同时刻分布快照
    snap_ts    = [0, 20, 50, 80, 99]
    snap_feats = {}
    with torch.no_grad():
        for tv in snap_ts:
            tt       = torch.tensor([tv] * B)
            zn, _    = diff.q_sample(z0, tt)
            snap_feats[tv] = zn.numpy().flatten()

    # 逆向采样轨迹
    cond_dummy = torch.zeros(B, diff.feat_dim * 2)
    z_rev      = torch.randn(B, diff.feat_dim)
    rev_means, rev_stds = [z_rev.mean().item()], [z_rev.std().item()]
    stride = T // 20
    with torch.no_grad():
        for tv in reversed(range(0, T, stride)):
            t    = torch.full((B,), tv, dtype=torch.float)
            pred = diff.denoisers[1](z_rev, cond_dummy, t)
            ab   = diff.alpha_bar[tv]
            z_rev = (z_rev - (1 - ab).sqrt() * pred) / ab.sqrt()
            rev_means.append(z_rev.mean().item())
            rev_stds.append(z_rev.std().item())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG_DARK)
    PURPLE, CORAL, TEAL, AMBER = '#b39ddb', '#e07b54', '#80cbc4', '#f4c775'

    # 面板1：轨迹
    ax = axes[0]; _panel(ax, 'Forward diffusion trajectory')
    ax2 = ax.twinx(); ax2.set_facecolor(BG_PANEL)
    ax.fill_between(timesteps,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=PURPLE, alpha=0.2)
    ax.plot(timesteps, means, color=PURPLE, lw=2, label='FLAIR feat mean±std')
    ax2.plot(timesteps, ab_vals, color=TEAL, lw=2, ls='--', label='ᾱₜ (SNR)')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Feature value', color=PURPLE)
    ax2.set_ylabel('ᾱₜ', color=TEAL)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8,
              facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')

    # 面板2：分布演变
    ax = axes[1]; _panel(ax, 'Feature distribution at each t')
    snap_colors = [TEAL, '#5dcaa5', AMBER, CORAL, '#ff6b6b']
    for tv, col in zip(snap_ts, snap_colors):
        feats = snap_feats[tv]
        bins  = np.linspace(-4, 4, 60)
        hist, edges = np.histogram(feats, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, hist, alpha=0.35, color=col)
        ax.plot(centers, hist, color=col, lw=1.2, label=f't={tv}')
    ax.set_xlabel('Feature value'); ax.set_ylabel('Density')
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')

    # 面板3：逆向去噪
    ax = axes[2]; _panel(ax, 'Reverse denoising trajectory')
    rev_steps = list(range(len(rev_means)))
    rm, rs = np.array(rev_means), np.array(rev_stds)
    ax.fill_between(rev_steps, rm - rs, rm + rs, color=TEAL, alpha=0.2, label='±1 std')
    ax.plot(rev_steps, rm, color=TEAL, lw=2, label='mean')
    ax.plot(rev_steps, rs, color=AMBER, lw=1.5, ls='--', label='std')
    ax.axhline(0, color='#444', lw=0.8, ls=':')
    ax.set_xlabel('Denoising step (T → 0)')
    ax.set_ylabel('Feature value')
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor='#444', labelcolor='white')

    plt.suptitle('Diffusion Process Visualization', color='white',
                 fontsize=13, fontweight='500')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[图3] 扩散过程图已保存: {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────
# 4. 图拓扑可视化
# ─────────────────────────────────────────────────────────
def plot_graph_topology(model, imgs3: list, labels: torch.Tensor,
                         mod_names: list = None,
                         save_path: str = 'graph_topology.png'):
    _rc()
    import matplotlib.patches as mpatches
    mod_names = mod_names or ['T1', 'FLAIR', 'T1ce']

    model.eval()
    with torch.no_grad():
        _, _, _, attn_w, raw_feats, graph_feats = model(imgs3, return_extras=True)

    attn_mean = attn_w.mean(0).cpu().numpy()    # [N, N, H]
    raw_np    = raw_feats.cpu().numpy()
    graph_np  = graph_feats.cpu().numpy()

    node_cols = ['#b39ddb', '#f4c775', '#e07b54']
    node_pos  = np.array([[0.25, 0.75], [0.75, 0.75], [0.50, 0.22]])

    def draw_modality_graph(ax, A_mat, title):
        ax.set_facecolor(BG_PANEL2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        max_w = A_mat.max() + 1e-8
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                w = A_mat[i, j] / max_w
                if w < 0.05:
                    continue
                x0, y0 = node_pos[i]; x1, y1 = node_pos[j]
                dx, dy = x1 - x0, y1 - y0
                perp   = np.array([-dy, dx]) / (np.sqrt(dx**2 + dy**2) + 1e-8)
                mx     = (x0 + x1) / 2 + perp[0] * 0.04
                my     = (y0 + y1) / 2 + perp[1] * 0.04
                ax.annotate('', xy=(x1 - dx * 0.18, y1 - dy * 0.18),
                            xytext=(x0 + dx * 0.18, y0 + dy * 0.18),
                            arrowprops=dict(arrowstyle='->', color=node_cols[i],
                                           lw=0.5 + w * 4,
                                           connectionstyle='arc3,rad=0.18',
                                           alpha=0.5 + w * 0.5))
                ax.text(mx, my, f'{A_mat[i,j]:.2f}', fontsize=7,
                        ha='center', va='center', color=node_cols[i], alpha=0.9)
        for k, (name, pos, col) in enumerate(zip(mod_names, node_pos, node_cols)):
            ax.add_patch(plt.Circle(pos, 0.10, color=col, alpha=0.2))
            ax.add_patch(plt.Circle(pos, 0.10, fill=False, edgecolor=col, lw=1.5))
            ax.text(pos[0], pos[1] + 0.01, name,
                    ha='center', va='center', fontsize=10, fontweight='500', color=col)
            sw = A_mat[k, k]
            ax.text(pos[0], pos[1] - 0.065, f'self={sw:.2f}',
                    ha='center', va='center', fontsize=7, color=col, alpha=0.7)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.36)

    # 面板A: 均值图
    ax_a = fig.add_subplot(gs[0, 0])
    draw_modality_graph(ax_a, attn_mean.mean(-1), 'Mean GAT attention (all heads)')

    # 面板B: 4 heads 热力图
    igs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1],
                                            hspace=0.40, wspace=0.34)
    for h in range(4):
        axi = fig.add_subplot(igs[h // 2, h % 2])
        axi.set_facecolor(BG_PANEL2)
        awh = attn_mean[:, :, h]
        im  = axi.imshow(awh, cmap='magma', vmin=0, vmax=1, aspect='auto')
        axi.set_xticks(range(3)); axi.set_yticks(range(3))
        axi.set_xticklabels(mod_names, fontsize=7, color='#aaa')
        axi.set_yticklabels(mod_names, fontsize=7, color='#aaa')
        axi.set_title(f'Head {h}', color='#aaa', fontsize=8)
        for ii in range(3):
            for jj in range(3):
                axi.text(jj, ii, f'{awh[ii,jj]:.2f}',
                         ha='center', va='center', color='white', fontsize=7)

    # 面板C/D: PCA 前后
    for ci, (feats_np, title) in enumerate([
        (raw_np,   'Feature PCA before GNN'),
        (graph_np, 'Feature PCA after GNN'),
    ]):
        ax = fig.add_subplot(gs[1, ci])
        _panel(ax, title)
        mkrs = ['o', '^', 's']
        for m in range(3):
            pts = _pca2d(feats_np[:, m, :])
            for li, (lc, ln) in enumerate(
                    zip(SEG_COLORS, ['BG', 'Edema', 'Enh'])):
                mask = labels.numpy() == li
                if mask.sum() > 0:
                    ax.scatter(pts[mask, 0], pts[mask, 1],
                               color=lc, marker=mkrs[m], s=55, alpha=0.78,
                               label=(f'{mod_names[m]}/{ln}' if ci == 0 else None),
                               edgecolors=node_cols[m], linewidths=0.8)
        ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(colors='#888', labelsize=7)
        if ci == 0:
            handles = [mpatches.Patch(color=lc, label=f'{mn}')
                       for lc, mn in zip(node_cols, mod_names)]
            handles += [mpatches.Patch(color=lc, label=ln)
                        for lc, ln in zip(SEG_COLORS, ['BG', 'Edema', 'Enh'])]
            ax.legend(handles=handles, fontsize=7, facecolor=BG_PANEL,
                      edgecolor='#444', labelcolor='white', ncol=2)

    fig.text(0.5, 0.995, 'Graph Topology Visualization — GAT Attention & Feature Space',
             ha='center', color='white', fontsize=13, fontweight='500')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[图4] 图拓扑已保存: {save_path}')
    plt.close()
