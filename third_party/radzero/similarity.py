import math
import torch
from torch import nn
from torch.nn import functional as F

class SimilarityLogit(nn.Module):
    def __init__(self, sim_op="dot", **kwargs):
        super().__init__()
        self.sim_op = sim_op

    def forward(
        self,
        queries: torch.Tensor,
        local_tokens: torch.Tensor,
        need_attn_weights: bool = False,
        repeat: bool = True,
        **kwargs,
    ):
        if repeat:
            query_attn_features = queries.unsqueeze(0).expand(
                local_tokens.shape[0], queries.shape[0], queries.shape[1]
            )
        else:
            assert queries.dim() == 3
            query_attn_features = queries

        if self.sim_op == "cos":
            temperature = kwargs.get("temperature")
            assert temperature is not None
            denominator = temperature
            query_attn_features = F.normalize(query_attn_features, p=2, dim=-1)
            local_tokens = F.normalize(local_tokens, p=2, dim=-1)
        elif self.sim_op == "dot":
            denominator = math.sqrt(local_tokens.size(-1))
        else:
            raise NotImplementedError

        scores = (
            torch.bmm(query_attn_features, local_tokens.permute(0, 2, 1)) / denominator
        )
        attn_weights = F.softmax(scores, dim=-1)

        aggregated = torch.matmul(attn_weights, local_tokens)

        query_attn_features = F.normalize(query_attn_features, p=2, dim=-1)
        aggregated = F.normalize(aggregated, p=2, dim=-1)

        logits = torch.matmul(
            query_attn_features.unsqueeze(2), aggregated.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        logits = logits.T

        if need_attn_weights:
            attn_scores = [scores]
        else:
            attn_scores = None

        return logits, attn_scores
