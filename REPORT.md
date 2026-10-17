# Report: DROP

Market-1501 rebuild of Dou et al., TBIOM 2026 (DOI [10.1109/TBIOM.2026.3680760](https://doi.org/10.1109/TBIOM.2026.3680760)).

## Setup

- Image: 256×128, ResNet-50 + HBDA + PCT
- Epochs: 120, Adam lr=0.00035
- Dataset size gate: full_train (0.1388 GiB)

## Market-1501 results

| | mAP | Rank-1 | Rank-5 |
|--|--:|--:|--:|
| **Ours** | 76.90 | 87.38 | 96.32 |

Paper headline Occluded-Duke Rank-1 **77.2%** not reproduced here (dataset blocked). See `DEVIATIONS.md`.

## Figures

### `outputs/figures/market_map_r1.svg`

![](outputs/figures/market_map_r1.svg)

### `outputs/figures/train_curves.svg`

![](outputs/figures/train_curves.svg)

## Regenerate

```bash
bash scripts/report.sh
```
