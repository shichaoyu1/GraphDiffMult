import torch

def multi_positive_nce_loss(
    logits: torch.Tensor,
    group_map: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
    row_sum: bool = False,
    col_sum: bool = False,
):
    """
    Args:
        logits: tensor of shape (N_total, B_global), each row is a logit between a key phrase and each candidate image.
        group_map: tensor of shape (N_total,), source image index of each key phrase.
        temperature: scaling factor.

    For each key phrase row i, the positive is the candidate image index == group_map[i],
    and the rest are treated as negatives.

    For each column j, each positive for image j is considered independently.

    Returns:
        loss: scalar tensor.
    """
    scaled_logits = torch.exp(logits / temperature)  # (N_total, B_global)

    pos_logits = scaled_logits[
        torch.arange(scaled_logits.size(0)), group_map
    ]  # (N_total,)

    row_loss = get_row_loss(
        scaled_logits,
        pos_logits,
        group_map,
        eps,
        row_sum,
    )

    neg_mask = torch.ones_like(scaled_logits)
    neg_mask[torch.arange(scaled_logits.size(0)), group_map] = 0  # (N_total, B_global)

    column_loss = get_col_loss(
        scaled_logits,
        pos_logits,
        neg_mask,
        group_map,
        eps,
        col_sum,
    )

    loss = (row_loss.mean() + column_loss.mean()) / 2

    return loss

def get_row_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    group_map: torch.Tensor,
    eps: float = 1e-8,
    row_sum: bool = False,
):
    if row_sum:
        # Create a tensor to hold the summed values
        row_sum_logits = torch.zeros(
            logits.shape[-1], device=logits.device
        )  # (B_global)
        row_pos_sum_logits = torch.zeros(
            logits.shape[-1], device=logits.device
        )  # (B_global)

        # Use scatter_add to sum values based on group_map
        row_sum_logits.scatter_add_(0, group_map, logits.sum(dim=1))  # (B_global)
        row_pos_sum_logits.scatter_add_(0, group_map, pos_logits)  # (B_global)
        p_row = row_pos_sum_logits / (row_sum_logits + eps)  # (B_global)
    else:
        row_sum_logits = logits.sum(dim=1)  # (N_total)
        p_row = pos_logits / (row_sum_logits + eps)  # (N_total)

    return -torch.log(p_row + eps)

def get_col_loss(
    logits: torch.Tensor,
    pos_logits: torch.Tensor,
    neg_mask: torch.Tensor,
    group_map: torch.Tensor,
    eps: float = 1e-8,
    col_sum: bool = False,
):
    if col_sum:
        # MIL-NCE loss
        column_sum_logits = logits.sum(dim=0)  # (B_global,)
        pos_mask = torch.ones_like(logits) - neg_mask  # (N_total, B_global)
        column_pos_logits = (logits * pos_mask).sum(dim=0)  # (B_global,)
        p_column = column_pos_logits / (column_sum_logits + eps)  # (B_global,)
    else:
        # MP-NCE loss (UniCLIP)
        neg_logits = logits * neg_mask  # (N_total, B_global)
        sum_neg_logits = neg_logits.sum(dim=0)  # (B_global,)
        sum_neg_logits = sum_neg_logits[group_map]  # (N_total)
        p_column = pos_logits / (pos_logits + sum_neg_logits + eps)  # (N_total)

    return -torch.log(p_column + eps)
