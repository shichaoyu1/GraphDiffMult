"""
Pathology-guided graph-constrained shared/private model.

The graph is used as a Laplacian consistency constraint over modality-level
shared representations with a lightweight single-step message passing update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROI2DEncoder(nn.Module):
    def __init__(self, in_ch: int = 7, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class SemanticGraphBuilder(nn.Module):
    def __init__(self, shared_dim: int, graph_type: str = 'learnable', tau: float = 0.2):
        super().__init__()
        self.graph_type = graph_type
        self.tau = tau
        self.edge_mlp = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim),
            nn.SiLU(),
            nn.Linear(shared_dim, 1),
        )

    def _mask_self(self, scores):
        n_nodes = scores.shape[1]
        eye = torch.eye(n_nodes, dtype=torch.bool, device=scores.device).unsqueeze(0)
        return scores.masked_fill(eye, -1e9)

    def forward(self, shared):
        batch_size, n_nodes, dim = shared.shape
        device = shared.device

        if self.graph_type == 'no_graph':
            return torch.zeros(batch_size, n_nodes, n_nodes, device=device)

        if self.graph_type == 'fixed':
            adjacency = torch.ones(batch_size, n_nodes, n_nodes, device=device)
            adjacency = adjacency - torch.eye(n_nodes, device=device).unsqueeze(0)
            return adjacency / max(n_nodes - 1, 1)

        if self.graph_type == 'random':
            scores = torch.rand(batch_size, n_nodes, n_nodes, device=device)
            return F.softmax(self._mask_self(scores), dim=-1)

        if self.graph_type == 'similarity':
            normalized = F.normalize(shared, dim=-1)
            scores = torch.matmul(normalized, normalized.transpose(1, 2)) / self.tau
            return F.softmax(self._mask_self(scores), dim=-1)

        if self.graph_type == 'learnable':
            source = shared.unsqueeze(2).expand(-1, -1, n_nodes, -1)
            target = shared.unsqueeze(1).expand(-1, n_nodes, -1, -1)
            pair = torch.cat([source, target, (source - target).abs(), source * target], dim=-1)
            scores = self.edge_mlp(pair).squeeze(-1)
            return F.softmax(self._mask_self(scores), dim=-1)

        raise ValueError(f'Unsupported graph_type: {self.graph_type}')


def graph_laplacian_consistency(shared, adjacency):
    if adjacency.abs().sum() == 0:
        return shared.sum() * 0
    adjacency_sym = 0.5 * (adjacency + adjacency.transpose(1, 2))
    diff = shared.unsqueeze(2) - shared.unsqueeze(1)
    energy = 0.5 * adjacency_sym.unsqueeze(-1) * diff.pow(2)
    return energy.sum(dim=(1, 2, 3)).mean()


def decouple_loss(shared, private):
    shared = F.normalize(shared, dim=-1)
    private = F.normalize(private, dim=-1)
    return (shared * private).sum(dim=-1).pow(2).mean()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class LatentPrivateDiffusion(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, T: int = 20, time_dim: int = 32):
        super().__init__()
        self.T = T
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
        )
        betas = torch.linspace(1e-4, 0.02, T)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bar[t].unsqueeze(-1)
        zt = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise
        return zt, noise

    def predict_noise(self, zt, t, cond):
        t_emb = self.time_embed(t.float())
        return self.denoiser(torch.cat([zt, cond, t_emb], dim=-1))

    def diffusion_loss(self, z0, cond):
        batch_size = z0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=z0.device)
        zt, noise = self.q_sample(z0, t)
        pred = self.predict_noise(zt, t, cond)
        return F.mse_loss(pred, noise)

    def refine(self, z0, cond, t_value=None):
        if t_value is None:
            t_value = max(1, self.T // 2)
        t = torch.full((z0.shape[0],), min(t_value, self.T - 1), device=z0.device, dtype=torch.long)
        zt, _ = self.q_sample(z0, t)
        pred = self.predict_noise(zt, t, cond)
        alpha_bar = self.alpha_bar[t].unsqueeze(-1)
        return (zt - (1 - alpha_bar).sqrt() * pred) / (alpha_bar.sqrt() + 1e-8)


class GliomaGraphDiffusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_slices: int = 7,
        num_modalities: int = 4,
        node_mode: str = 'regions',
        num_regions: int = 3,
        feat_dim: int = 256,
        shared_dim: int = 128,
        private_dim: int = 128,
        graph_type: str = 'learnable',
        diffusion_T: int = 20,
        graph_ema_momentum: float = 0.95,
        graph_ema_blend: float = 0.5,
        diffusion_init_alpha: float = 0.05,
        shared_private_mix_init: float = 0.05,
        classifier_private_scale_init: float = 0.05,
        diffusion_max_ratio: float = 0.5,
        use_anchor: bool = True,
        use_private: bool = True,
        use_diffusion: bool = True,
    ):
        super().__init__()
        self.node_mode = node_mode
        self.num_input_modalities = num_modalities
        self.num_regions = num_regions
        self.num_nodes = num_regions if node_mode == 'regions' else num_modalities
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.use_anchor = use_anchor
        self.use_private = use_private
        self.use_diffusion = use_diffusion and use_private
        self.graph_type = graph_type
        self.graph_ema_momentum = graph_ema_momentum
        self.graph_ema_blend = graph_ema_blend
        self.diffusion_max_ratio = diffusion_max_ratio

        encoder_channels = z_slices * num_modalities if node_mode == 'regions' else z_slices
        self.encoders = nn.ModuleList([
            ROI2DEncoder(in_ch=encoder_channels, feat_dim=feat_dim)
            for _ in range(self.num_nodes)
        ])
        self.shared_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, shared_dim), nn.LayerNorm(shared_dim), nn.SiLU())
            for _ in range(self.num_nodes)
        ])
        self.private_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, private_dim), nn.LayerNorm(private_dim), nn.SiLU())
            for _ in range(self.num_nodes)
        ])

        self.graph_builder = SemanticGraphBuilder(shared_dim, graph_type=graph_type)
        self.graph_norm = nn.LayerNorm(shared_dim)
        self.private_to_shared = nn.Linear(private_dim, shared_dim)
        self.diffusion_alpha = nn.Parameter(torch.tensor(float(diffusion_init_alpha)))
        self.shared_private_mix = nn.Parameter(torch.tensor(float(shared_private_mix_init)))
        self.classifier_private_scale = nn.Parameter(torch.tensor(float(classifier_private_scale_init)))
        self.register_buffer('adj_ema', torch.zeros(self.num_nodes, self.num_nodes))
        self.register_buffer('adj_ema_ready', torch.tensor(0, dtype=torch.uint8))
        self.anchor_prototypes = nn.Parameter(torch.randn(num_classes, shared_dim) * 0.02)

        private_latent_dim = private_dim * self.num_nodes
        if self.use_diffusion:
            self.diffusion = LatentPrivateDiffusion(private_latent_dim, shared_dim, T=diffusion_T)
        else:
            self.diffusion = None

        classifier_in = shared_dim + (private_latent_dim if use_private else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def _smooth_adjacency(self, adjacency):
        if self.graph_type == 'no_graph':
            return adjacency
        batch_mean = adjacency.mean(dim=0).detach()
        if self.training:
            if int(self.adj_ema_ready.item()) == 0:
                self.adj_ema.copy_(batch_mean)
                self.adj_ema_ready.fill_(1)
            else:
                self.adj_ema.mul_(self.graph_ema_momentum).add_(batch_mean * (1.0 - self.graph_ema_momentum))
        if int(self.adj_ema_ready.item()) == 0:
            return adjacency
        ema = self.adj_ema.unsqueeze(0).expand_as(adjacency)
        return (1.0 - self.graph_ema_blend) * adjacency + self.graph_ema_blend * ema

    def encode_modalities(self, images):
        features = []
        for modality_idx, encoder in enumerate(self.encoders):
            features.append(encoder(images[:, modality_idx]))
        features = torch.stack(features, dim=1)
        shared = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.shared_heads)],
            dim=1,
        )
        private = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.private_heads)],
            dim=1,
        )
        return features, shared, private

    def encode_regions(self, images, region_masks):
        if region_masks is None:
            region_masks = torch.ones(
                images.shape[0],
                self.num_regions,
                images.shape[2],
                images.shape[3],
                images.shape[4],
                device=images.device,
                dtype=images.dtype,
            )

        features = []
        for region_idx, encoder in enumerate(self.encoders):
            mask = region_masks[:, region_idx:region_idx + 1]
            masked = images * mask
            region_input = masked.reshape(masked.shape[0], -1, masked.shape[-2], masked.shape[-1])
            features.append(encoder(region_input))
        features = torch.stack(features, dim=1)
        shared = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.shared_heads)],
            dim=1,
        )
        private = torch.stack(
            [head(features[:, idx]) for idx, head in enumerate(self.private_heads)],
            dim=1,
        )
        return features, shared, private

    def forward(self, images, labels=None, modality_mask=None, region_masks=None, return_extras=False, freeze_graph=False):
        if modality_mask is not None:
            images = images * modality_mask[:, :, None, None, None]

        if self.node_mode == 'regions':
            raw_features, shared_raw, private = self.encode_regions(images, region_masks)
        else:
            raw_features, shared_raw, private = self.encode_modalities(images)
        adjacency_raw = self.graph_builder(shared_raw)
        adjacency = self._smooth_adjacency(adjacency_raw)
        if freeze_graph and self.training:
            adjacency = adjacency.detach()

        if self.graph_type == 'no_graph':
            shared_graph = shared_raw
        else:
            message = torch.matmul(adjacency, shared_raw)
            shared_graph = self.graph_norm(shared_raw + message)

        cons = graph_laplacian_consistency(shared_raw, adjacency)

        private_flat = private.reshape(private.shape[0], -1) if self.use_private else None
        private_nodes_base = private
        diffusion_residual_nodes = None
        if self.use_private and self.use_diffusion:
            shared_cond = shared_graph.mean(dim=1)
            diff = self.diffusion.diffusion_loss(private_flat, shared_cond)
            diffusion_target = self.diffusion.refine(private_flat, shared_cond)
            diffusion_residual = diffusion_target - private_flat
            alpha = torch.clamp(self.diffusion_alpha, min=0.0, max=0.3)
            private_repr = private_flat + alpha * diffusion_residual
            diffusion_residual_nodes = diffusion_residual.reshape(private.shape[0], self.num_nodes, self.private_dim)
        elif self.use_private:
            diff = shared_raw.sum() * 0
            private_repr = private_flat
            diffusion_residual_nodes = torch.zeros_like(private)
        else:
            diff = shared_raw.sum() * 0
            private_repr = None
            diffusion_residual_nodes = None

        if private_repr is None:
            shared = shared_graph
            private_nodes_final = None
            decouple = shared_raw.sum() * 0
            leak = shared_raw.sum() * 0
            diff_norm = shared_raw.sum() * 0
        else:
            private_nodes_final = private_repr.reshape(private.shape[0], self.num_nodes, self.private_dim)
            if diffusion_residual_nodes is None:
                diffusion_residual_nodes = torch.zeros_like(private_nodes_final)
            shared_delta = self.private_to_shared(diffusion_residual_nodes)
            mix = torch.clamp(self.shared_private_mix, min=0.0, max=0.3)
            shared = shared_graph + mix * shared_delta
            private_shared = self.private_to_shared(private_nodes_final)
            decouple = decouple_loss(shared_raw, private_shared)
            leak = decouple_loss(shared_graph, self.private_to_shared(diffusion_residual_nodes))
            shared_norm = shared_graph.norm(dim=-1).mean()
            residual_norm = diffusion_residual_nodes.norm(dim=-1).mean()
            diff_norm = F.relu(residual_norm - self.diffusion_max_ratio * shared_norm)
        shared_mean = shared.mean(dim=1)

        if labels is not None and self.use_anchor:
            anchors = self.anchor_prototypes[labels]
            anchor = F.mse_loss(shared_mean, anchors)
        else:
            anchor = shared_raw.sum() * 0

        if private_repr is None:
            fused = shared_mean
        else:
            cls_scale = torch.clamp(self.classifier_private_scale, min=0.0, max=0.3)
            fused = torch.cat([shared_mean, cls_scale * private_repr], dim=-1)
        logits = self.classifier(fused)

        output = {
            'logits': logits,
            'losses': {
                'cons': cons,
                'anchor': anchor,
                'decouple': decouple,
                'leak': leak,
                'diff': diff,
                'diff_norm': diff_norm,
                'graph_energy': cons.detach(),
            },
        }

        if return_extras:
            output['extras'] = {
                'raw_features': raw_features,
                'shared_raw': shared_raw,
                'shared_graph': shared_graph,
                'shared': shared,
                'private': private_nodes_base,
                'private_final_nodes': private_nodes_final,
                'diffusion_residual_nodes': diffusion_residual_nodes,
                'private_repr': private_repr,
                'shared_mean': shared_mean,
                'shared_norm': shared.norm(dim=-1).mean().detach(),
                'private_norm': (
                    private_nodes_final.norm(dim=-1).mean().detach()
                    if private_nodes_final is not None
                    else torch.zeros((), device=shared.device)
                ),
                'diffusion_residual_norm': (
                    diffusion_residual_nodes.norm(dim=-1).mean().detach()
                    if diffusion_residual_nodes is not None
                    else torch.zeros((), device=shared.device)
                ),
                'adjacency': adjacency,
                'fused': fused,
            }

        return output
