# DROP: Decouple Re-Identification and Human Parsing for Occluded Person Re-Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of Dou et al. (TBIOM 2026): Hierarchical Bias-Decoupled Adapter (HBDA) + parsing branch + part-aware compactness triplet (PCT) for person ReID on **Market-1501**.

**Repository:** https://github.com/biometric-community/decouple-re-identification-and-human-parsing-for-occluded-person-re

**Paper:** [10.1109/TBIOM.2026.3680760](https://doi.org/10.1109/TBIOM.2026.3680760)

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/report.sh
```

## Data

Point `configs/*.yaml` `data.root` at `projects/datasets/market1501` (official Market-1501 extract).

## Results (Market-1501, this rebuild)

| | mAP | Rank-1 | Rank-5 |
|--|--:|--:|--:|
| **Ours** | 76.90 | 87.38 | 96.32 |

See `REPORT.md`. Occluded-Duke headline numbers are not reproduced here — Market-only; see `DEVIATIONS.md`.

## Citation

```
@article{dou2026drop,
  title={Decouple Re-Identification and Human Parsing for Occluded Person Re-Identification},
  author={Dou, Shuguang and Tu, Yuanpeng and Lin, Yanna and Gao, Junyao and Zhao, Cairong},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2026},
  doi={10.1109/TBIOM.2026.3680760}
}
```
