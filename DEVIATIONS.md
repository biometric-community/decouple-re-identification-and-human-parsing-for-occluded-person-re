# Deviations from the paper

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | Occluded-Duke / Occluded-ReID / P-Duke / MSMT17 / Duke | **Market-1501 only** | Only Market registered under `projects/datasets/` |
| D2 | PifPaf 17-part confidence + 19-part affinity → K groups | Horizontal **K=5 stripe** pseudo-parse targets | PifPaf weights / affinity pipeline not shipped |
| D3 | HBDA full hierarchical bias modules | Single bottleneck adapter on layer-4 map | Exact HBDA blocks under-specified in extract; functional decoupling |
| D4 | WAMP + PEMB (memory bank size 4) for PCT | Batch-hard PCT over current-batch part embeddings | PEMB omitted for simplicity; same part-wise triplet spirit |
| D5 | Spatial-smoothed parsing loss (γ=0.5) + LS on pixels | Pixel CE on stripe labels; ID CE uses LS=0.1 | Spatial γ term omitted (D2 labels already smooth stripes) |
| D6 | Inference: foreground + visible parts (thresh 40%) | Concat global+parts L2 feature CMC | Standard holistic Market protocol |
| D7 | Two RTX 3090 | Single GPU `CUDA_VISIBLE_DEVICES=0` | Available hardware |
| D8 | torchreid codebase | Native PyTorch `drop/` package | Project layout requirement |

## Hyperparameters taken from Implementation (§IV-A-2)

- 256×128, ResNet-50, Adam 3.5e-4, 120 epochs, drops @40/70, batch 64, LS=0.1, margin=0.3, λ=0.4, K=5
