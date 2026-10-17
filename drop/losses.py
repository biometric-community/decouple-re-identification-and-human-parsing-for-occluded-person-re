"""ID CE (label-smooth) + PCT + parsing CE."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTriplet(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = float(margin)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=-1)
        dist = torch.cdist(feats, feats, p=2)
        labels = labels.view(-1)
        mask_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_neg = ~mask_pos
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        mask_pos = mask_pos & ~eye
        dist_ap = dist.clone()
        dist_ap[~mask_pos] = -1.0
        hardest_pos, _ = dist_ap.max(dim=1)
        dist_an = dist.clone()
        dist_an[~mask_neg] = 1e6
        hardest_neg, _ = dist_an.min(dim=1)
        valid = (hardest_pos > -0.5) & (hardest_neg < 1e5)
        if valid.sum() == 0:
            return dist.new_zeros(())
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin).mean()


def part_compactness_triplet(
    parts: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """PCT: average batch-hard triplet over part embeddings (B,K,D)."""
    tri = BatchHardTriplet(margin=margin)
    b, k, d = parts.shape
    losses = []
    for i in range(k):
        losses.append(tri(parts[:, i, :], labels))
    return torch.stack(losses).mean()


def drop_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    parse_tgt: torch.Tensor,
    ce: nn.Module,
    margin: float = 0.3,
    lambda_tri: float = 1.0,
    lambda_pct: float = 1.0,
    lambda_parse: float = 2.0,
) -> Dict[str, torch.Tensor]:
    l_ce = ce(outputs["logits"], labels)
    l_tri = BatchHardTriplet(margin)(outputs["feat"], labels)
    l_pct = part_compactness_triplet(outputs["parts"], labels, margin=margin)
    # parsing CE — paper weights parsing higher after decoupling
    plog = outputs["parse_logits"]  # (B,K,H,W)
    l_parse = F.cross_entropy(plog, parse_tgt)
    loss = l_ce + lambda_tri * l_tri + lambda_pct * l_pct + lambda_parse * l_parse
    return {
        "loss": loss,
        "ce": l_ce.detach(),
        "tri": l_tri.detach(),
        "pct": l_pct.detach(),
        "parse": l_parse.detach(),
    }
