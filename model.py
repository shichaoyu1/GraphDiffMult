"""
model.py — 多模态融合网络
  · ModalityEncoder     : 轻量 CNN，提取单模态特征
  · GATLayer            : 图注意力层，建模模态间拓扑关系
  · GraphModalityEncoder: 两层 GAT，输出图增强特征 + 注意力权重
  · Diffusion           : 条件 DDPM，捕捉模态间互补生成关系
  · MultimodalFusionNet : 整体网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 1. 模态编码器
# ─────────────────────────────────────────────────────────
class ModalityEncoder(nn.Module):
    """
    输入: [B, 1, H, W] 单通道 patch（任意分辨率）
    输出: [B, feat_dim] 特征向量
    """
    def __init__(self, in_ch: int = 1, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────
# 2. 图注意力层 (GAT)
# ─────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """
    多头图注意力（节点 = 模态，全连接图）
    输入 : x  [B, N, in_dim]   N = 模态数
    输出 : (h [B, N, out_dim], attn [B, N, N, num_heads])
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads
        self.d = out_dim // num_heads

        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.d, 1, bias=False)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.H, self.d)          # [B,N,H,d]

        # 计算所有节点对的注意力得分
        hi = h.unsqueeze(2).expand(-1, -1, N, -1, -1)     # [B,N,N,H,d]
        hj = h.unsqueeze(1).expand(-1, N, -1, -1, -1)     # [B,N,N,H,d]
        e  = self.leaky(self.attn(torch.cat([hi, hj], -1)))  # [B,N,N,H,1]
        a  = F.softmax(e, dim=2)                            # softmax over neighbors

        # 聚合邻居特征
        out = (a * hj).sum(dim=2).view(B, N, -1)           # [B,N,out_dim]
        return F.elu(out), a.squeeze(-1)                    # [B,N,out_dim], [B,N,N,H]


class GraphModalityEncoder(nn.Module):
    """
    两层 GAT + 残差连接
    输出: (graph_feats [B,N,feat_dim], attn_weights [B,N,N,H])
    """
    def __init__(self, feat_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.gat1 = GATLayer(feat_dim, feat_dim, num_heads)
        self.gat2 = GATLayer(feat_dim, feat_dim, num_heads)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor):
        # 第一层 GAT + 残差
        h1, _     = self.gat1(x)
        h1        = self.norm1(h1 + x)
        # 第二层 GAT + 残差
        h2, attn  = self.gat2(h1)
        h2        = self.norm2(h2 + h1)
        return h2, attn     # [B,N,D],  [B,N,N,H]


# ─────────────────────────────────────────────────────────
# 3. 条件扩散模型 (DDPM)
# ─────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    """时间步 t 的正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.log(torch.tensor(10000.0)) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs)
        args  = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class ConditionalDenoiser(nn.Module):
    """
    条件去噪 MLP
    输入: 噪声特征 z [B,D] + 条件特征 cond [B,D*2] + 时间步 t [B]
    输出: 预测噪声 [B,D]
    """
    def __init__(self, feat_dim: int = 64, cond_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        inp = feat_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([z, cond, t_emb], dim=-1))


class MultimodalDiffusion(nn.Module):
    """
    每个模态配一个条件去噪器：
      - 训练: 用其余 N-1 个模态的拼接特征作为条件，预测当前模态加噪噪声
      - 推理: 条件采样缺失模态（模态补全）
    线性 beta schedule: beta_1=1e-4 → beta_T=0.02
    """
    def __init__(self, feat_dim: int = 64, num_modalities: int = 3, T: int = 100):
        super().__init__()
        self.T          = T
        self.feat_dim   = feat_dim
        self.N          = num_modalities

        self.denoisers = nn.ModuleList([
            ConditionalDenoiser(feat_dim, feat_dim * (num_modalities - 1))
            for _ in range(num_modalities)
        ])

        # 预计算 alpha_bar
        betas     = torch.linspace(1e-4, 0.02, T)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)

    # ── 前向加噪 ──────────────────────────────────
    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise=None):
        """返回 (加噪后特征, 实际噪声)"""
        if noise is None:
            noise = torch.randn_like(z0)
        ab  = self.alpha_bar[t][:, None]          # [B,1]
        z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * noise
        return z_t, noise

    # ── 扩散损失 ──────────────────────────────────
    def diffusion_loss(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, N, D]
        对每个模态 i，用其余模态为条件预测噪声，返回平均 MSE
        """
        B, N, D = feats.shape
        total   = torch.tensor(0.0, device=feats.device)
        for i in range(N):
            z0      = feats[:, i]
            cond    = torch.cat([feats[:, j] for j in range(N) if j != i], dim=-1)
            t       = torch.randint(0, self.T, (B,), device=feats.device)
            z_noisy, noise = self.q_sample(z0, t)
            pred    = self.denoisers[i](z_noisy, cond, t.float())
            total   = total + F.mse_loss(pred, noise)
        return total / N

    # ── 逆向采样（推理时用，可补全缺失模态）──────
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, modality_idx: int, steps: int = 20) -> torch.Tensor:
        """
        cond: [B, D*(N-1)]  其他模态特征拼接
        返回生成的目标模态特征 [B, D]
        """
        B      = cond.shape[0]
        z      = torch.randn(B, self.feat_dim, device=cond.device)
        stride = self.T // steps
        for t_val in reversed(range(0, self.T, stride)):
            t       = torch.full((B,), t_val, device=cond.device, dtype=torch.float)
            pred    = self.denoisers[modality_idx](z, cond, t)
            ab      = self.alpha_bar[t_val]
            z       = (z - (1 - ab).sqrt() * pred) / ab.sqrt()
        return z


# ─────────────────────────────────────────────────────────
# 4. 图对比损失
# ─────────────────────────────────────────────────────────
def graph_contrastive_loss(
    feats_before: torch.Tensor,
    feats_after:  torch.Tensor,
    temperature:  float = 0.07,
) -> torch.Tensor:
    """
    feats_before / after: [B, N, D]
    同一样本不同模态为正对，不同样本为负对
    """
    B, N, D = feats_before.shape
    zb  = F.normalize(feats_before.reshape(B * N, D), dim=-1)
    za  = F.normalize(feats_after.reshape(B * N, D),  dim=-1)
    sim = torch.mm(zb, za.T) / temperature          # [BN, BN]
    lbl = torch.arange(B * N, device=zb.device)
    return (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl)) / 2


# ─────────────────────────────────────────────────────────
# 5. 完整融合网络
# ─────────────────────────────────────────────────────────
class MultimodalFusionNet(nn.Module):
    """
    三模态（T1 / FLAIR / T1ce）融合分类网络

    前向输出:
      正常模式: (logits, diff_loss, graph_loss)
      详细模式: (logits, diff_loss, graph_loss, attn_weights, raw_feats, graph_feats)
    """
    def __init__(
        self,
        num_classes:    int = 3,
        feat_dim:       int = 64,
        num_modalities: int = 3,
        num_heads:      int = 4,
        diffusion_T:    int = 100,
    ):
        super().__init__()
        self.num_modalities = num_modalities

        # 每个模态独立编码器
        self.encoders = nn.ModuleList([
            ModalityEncoder(in_ch=1, feat_dim=feat_dim)
            for _ in range(num_modalities)
        ])

        # 图学习：建模模态间拓扑关系
        self.graph_enc = GraphModalityEncoder(feat_dim, num_heads)

        # 扩散模型：捕捉模态间互补生成关系
        self.diffusion = MultimodalDiffusion(feat_dim, num_modalities, diffusion_T)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * num_modalities, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: list, return_extras: bool = False):
        """
        images: list of Tensor [B,1,H,W]，长度 = num_modalities
        """
        # ① 各模态独立编码
        raw_feats = torch.stack(
            [enc(img) for enc, img in zip(self.encoders, images)], dim=1
        )  # [B, N, D]

        # ② 图学习：GAT 聚合模态间信息
        graph_feats, attn_weights = self.graph_enc(raw_feats)   # [B,N,D], [B,N,N,H]

        # ③ 扩散损失（训练时有梯度）
        diff_loss  = self.diffusion.diffusion_loss(raw_feats)

        # ④ 图对比损失
        graph_loss = graph_contrastive_loss(raw_feats, graph_feats)

        # ⑤ 融合分类
        fused  = graph_feats.reshape(graph_feats.shape[0], -1)  # [B, N*D]
        logits = self.classifier(fused)

        if return_extras:
            return logits, diff_loss, graph_loss, attn_weights, raw_feats, graph_feats
        return logits, diff_loss, graph_loss
