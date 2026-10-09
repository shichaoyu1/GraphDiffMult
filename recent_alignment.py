"""MRI/metadata adaptations retaining published CARZero and RadZero alignment cores.

Official modules/versions/licenses: third_party/ALIGNMENT_SOURCES.json.
These are controlled adaptations, not original chest-X-ray checkpoint reproductions.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiment_model import GliomaGraphDiffusionNet
from third_party.carzero.dqn import TQN_Model
from third_party.radzero.similarity import SimilarityLogit
from third_party.radzero.dinov2_blocks import Dinov2Layer

METHODS = ('pasa_full', 'pasa_minimal', 'pasa_text_control', 'text_cosine',
           'carzero_metadata', 'radzero_metadata')
BENCHMARK_VERSION = 'pasa_recent_alignment_v1'


def masked_nce(scores, positives, known, per_positive=False, symmetric=True):
    """Known-label extension of InfoNCE / RadZero per-positive MP-NCE."""
    if scores.shape != positives.shape or scores.shape != known.shape:
        raise ValueError('Scores and label masks must match')
    if torch.any(positives & ~known):
        raise ValueError('Positive labels must be known')
    def direction(logits, pos, valid):
        active = pos.any(dim=1)
        if not active.any():
            return logits.sum() * 0
        logits, pos, valid = logits[active], pos[active], valid[active]
        if per_positive:
            negative_lse = logits.masked_fill(~(valid & ~pos), float('-inf')).logsumexp(dim=1)
            losses = []
            for row in range(len(logits)):
                values = logits[row, pos[row]]
                losses.append(torch.logaddexp(values, negative_lse[row]) - values)
            return torch.cat(losses).mean()
        denominator = logits.masked_fill(~valid, float('-inf')).logsumexp(dim=1)
        numerator = logits.masked_fill(~pos, float('-inf')).logsumexp(dim=1)
        return (denominator - numerator).mean()
    row_loss = direction(scores, positives, known)
    return (row_loss + direction(scores.T, positives.T, known.T)) / 2 if symmetric else row_loss


class RecentAlignmentModel(nn.Module):
    def __init__(self, method, num_anchors, text_features=None, z_slices=7,
                 feat_dim=256, shared_dim=128, private_dim=128, temperature=.07):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.temperature = temperature
        full = method in ('pasa_full', 'pasa_text_control')
        self.encoder = GliomaGraphDiffusionNet(num_classes=1, z_slices=z_slices,
            feat_dim=feat_dim, shared_dim=shared_dim, private_dim=private_dim,
            graph_type='learnable' if full else 'no_graph', use_anchor=False,
            use_private=full, use_diffusion=full)
        self.text_based = method not in ('pasa_full', 'pasa_minimal')
        if self.text_based:
            if text_features is None or len(text_features['global']) != num_anchors:
                raise ValueError('The frozen train-vocabulary text cache is required')
            self.register_buffer('text_global', text_features['global'].float())
            self.register_buffer('text_local', text_features['tokens'].float())
            self.register_buffer('text_mask', text_features['mask'].bool())
            self.text_projector = nn.Linear(self.text_global.shape[-1], shared_dim, bias=False)
        else:
            self.prototypes = nn.Parameter(torch.randn(num_anchors, shared_dim) * .02)
        if method == 'carzero_metadata':
            config = SimpleNamespace(model=SimpleNamespace(fusion=SimpleNamespace(
                d_model=shared_dim, class_num=1, decoder_number_layer=4)))
            self.carzero = TQN_Model(config)
        if method == 'radzero_metadata':
            config = SimpleNamespace(hidden_size=shared_dim, num_attention_heads=4, qkv_bias=True,
                attention_probs_dropout_prob=0., hidden_dropout_prob=0., layerscale_value=1.,
                drop_path_rate=0., mlp_ratio=4., hidden_act='gelu', layer_norm_eps=1e-6,
                use_swiglu_ffn=False, _attn_implementation='eager')
            self.radzero_vision_adapter = nn.ModuleList([Dinov2Layer(config) for _ in range(2)])
            # Official config: cos similarity, shared learned attention/loss temperature.
            self.radzero = SimilarityLogit('cos')
            self.radzero_norm = nn.LayerNorm(shared_dim)
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(self, images, region_masks=None, freeze_graph=False):
        output = self.encoder(images, region_masks=region_masks, return_extras=True, freeze_graph=freeze_graph)
        local = output['extras']['shared']
        if not torch.isfinite(local).all():
            raise ValueError('Nonfinite image features')
        global_image = local.mean(dim=1)
        if self.text_based:
            candidates = self.text_projector(self.text_global)
        else:
            candidates = self.prototypes
        result = {'encoder_output': output, 'local': local, 'candidates': candidates}
        if self.method == 'carzero_metadata':
            forward = self.carzero(local, candidates).squeeze(-1)
            text_tokens = self.text_projector(self.text_local)
            # Crop each candidate's padded word tokens before calling the unmodified official decoder.
            reverse = torch.cat([self.carzero(text_tokens[i, self.text_mask[i]].unsqueeze(0), global_image)
                                 for i in range(len(candidates))], dim=0).squeeze(-1).T
            result.update(scores=(forward + reverse) / 2, directions=(forward, reverse))
        elif self.method == 'radzero_metadata':
            vision = torch.cat([global_image.unsqueeze(1), local], dim=1)
            for layer in self.radzero_vision_adapter:
                vision = layer(vision)[0]
            temperature = self.log_temperature.exp()
            scores, _ = self.radzero(self.radzero_norm(candidates), self.radzero_norm(vision),
                                    temperature=temperature, need_attn_weights=True)
            result.update(scores=scores.T, temperature=temperature)
        elif self.method.startswith('pasa_'):
            node_scores = F.normalize(local, dim=-1) @ F.normalize(candidates, dim=-1).T
            result.update(scores=node_scores.mean(dim=1), node_scores=node_scores)
        else:
            result['scores'] = F.normalize(global_image, dim=-1) @ F.normalize(candidates, dim=-1).T
        return result

    def alignment_loss(self, result, positives, known):
        if self.method == 'carzero_metadata':
            # Original CARZero consumes raw MLP logits in CE; no extra 1/.07 scaling.
            return sum(masked_nce(scores, positives, known) for scores in result['directions']) / 2
        if self.method == 'radzero_metadata':
            return masked_nce(result['scores'] / result['temperature'], positives, known,
                              per_positive=True, symmetric=True)
        if self.method.startswith('pasa_'):
            nodes = result['node_scores'].shape[1]
            scores = result['node_scores'].reshape(-1, result['scores'].shape[-1]) / self.temperature
            return masked_nce(scores, positives.repeat_interleave(nodes, 0),
                              known.repeat_interleave(nodes, 0), symmetric=False)
        return masked_nce(result['scores'] / self.temperature, positives, known)
