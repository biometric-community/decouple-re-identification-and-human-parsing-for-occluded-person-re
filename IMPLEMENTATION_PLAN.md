# Implementation plan: DROP

Paper: Dou et al., TBIOM 2026, DOI [10.1109/TBIOM.2026.3680760](https://doi.org/10.1109/TBIOM.2026.3680760).

## Scope

Rebuild **DROP** (HBDA + parsing-guided ReID + PCT) as a PyTorch package `drop/` and train/eval on **Market-1501** (holistic, K=5). Occluded-Duke / PifPaf parsing labels are blocked → stripe pseudo-parse (see `DEVIATIONS.md`).

## Figure inventory (Results)

| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Table I Market row | bar mAP / Rank-1 | `outputs/figures/market_map_r1.svg` |
| Training curves (ours) | loss + mAP | `outputs/figures/train_curves.svg` |

## Hyperparameters (paper §IV-A-2)

- Image 256×128; ResNet-50; Adam 3.5e-4; 120 epochs; LR ×0.1 @ 40, 70
- Batch 64; label smoothing 0.1; triplet margin 0.3; λ_hp=0.4; K=5 (holistic)

## Dataset size gate

Market-1501 under `projects/datasets/market1501` ≈ 0.14 GiB → **full_train**.

## Deliverables

`drop/` package, `configs/`, `scripts/*.sh`, docs, REPORT from real logs, publish to `biometric-community/decouple-re-identification-and-human-parsing-for-occluded-person-re`.
