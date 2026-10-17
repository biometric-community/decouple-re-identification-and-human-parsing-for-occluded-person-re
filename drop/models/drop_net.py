"""DROP network: ResNet-50 + HBDA parsing branch + part ReID head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HierarchicalBiasDecoupledAdapter(nn.Module):
    """HBDA (simplified): bottleneck adapter for parsing-specific features."""

    def __init__(self, channels: int = 2048, bottleneck: int = 512):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.down(x)))
        y = self.bn2(self.up(y))
        return self.act(x + y)


class DROPNet(nn.Module):
    """Decoupled ReID + parsing heads (paper Sec. III)."""

    def __init__(
        self,
        num_classes: int,
        num_parts: int = 5,
        pretrained: bool = True,
        feat_dim: int = 256,
    ):
        super().__init__()
        self.num_parts = int(num_parts)
        weights = "DEFAULT" if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        c = 2048
        self.hbda = HierarchicalBiasDecoupledAdapter(c, bottleneck=512)
        # parsing head: K part logits per spatial location (stripe CE target)
        self.parse_head = nn.Conv2d(c, self.num_parts, 1)
        # ReID: global + K part embeddings
        self.global_fc = nn.Linear(c, feat_dim, bias=False)
        self.part_fcs = nn.ModuleList([nn.Linear(c, feat_dim, bias=False) for _ in range(self.num_parts)])
        self.bn = nn.BatchNorm1d(feat_dim * (self.num_parts + 1))
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim * (self.num_parts + 1), num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for m in [self.global_fc, *self.part_fcs]:
            nn.init.normal_(m.weight, std=0.001)

    def _part_pools(self, fmap: torch.Tensor):
        # PCB-style equal stripes — linspace so we always get exactly K parts
        # (torch.chunk can return fewer when H % K != 0).
        _, _, h, _ = fmap.shape
        edges = torch.linspace(0, h, self.num_parts + 1, device=fmap.device).long()
        parts = []
        for k in range(self.num_parts):
            lo, hi = int(edges[k].item()), int(edges[k + 1].item())
            if hi <= lo:
                hi = min(lo + 1, h)
            ch = fmap[:, :, lo:hi, :]
            parts.append(F.adaptive_avg_pool2d(ch, 1).flatten(1))
        g = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return g, parts

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(x)
        # parsing branch (decoupled)
        pmap = self.hbda(fmap)
        parse_logits = self.parse_head(pmap)  # (B,K,H,W)
        # ReID branch uses original fmap (decoupled from parsing adapter)
        g, parts = self._part_pools(fmap)
        g = self.global_fc(g)
        p_emb = [fc(p) for fc, p in zip(self.part_fcs, parts)]
        feat = torch.cat([g, *p_emb], dim=1)
        feat = self.bn(feat)
        feat_n = F.normalize(feat, dim=1)
        logits = self.classifier(feat)
        return {
            "feat": feat_n,
            "logits": logits,
            "parts": torch.stack(p_emb, dim=1),  # (B,K,D)
            "global": F.normalize(g, dim=1),
            "parse_logits": parse_logits,
        }


def build_drop(cfg: dict, num_classes: int) -> DROPNet:
    m = cfg.get("model", {})
    return DROPNet(
        num_classes=num_classes,
        num_parts=int(m.get("num_parts", 5)),
        pretrained=bool(m.get("pretrained", True)),
        feat_dim=int(m.get("feat_dim", 256)),
    )


def stripe_parse_targets(fmap_hw: tuple[int, int], num_parts: int, device, batch: int) -> torch.Tensor:
    """Pseudo parsing labels: spatial location → stripe id (stand-in for PifPaf)."""
    h, w = fmap_hw
    t = torch.zeros(batch, h, w, dtype=torch.long, device=device)
    edges = torch.linspace(0, h, num_parts + 1).long()
    for k in range(num_parts):
        t[:, edges[k] : edges[k + 1], :] = k
    return t
