# Fidelity audit: DROP

Target: structural match to Dou et al., TBIOM 2026 (DOI 10.1109/TBIOM.2026.3680760). Fixed 5-pass audit; stop after Pass 5.

## Per-pass statistics

| Pass | Focus | Confidence | Key fixes / evidence |
|------|-------|------------|----------------------|
| 1 | Layout / configs / scripts / docs | **92%** | `drop/` package, YAML 256×128 Adam 3.5e-4 120ep, scripts, README/LICENSE/DEVIATIONS |
| 2 | Model + losses | **78%** | ResNet-50 + HBDA bottleneck + parse head + global/part BNNeck; CE+tri+PCT+parse CE (λ=0.4). Stripe fix for H%K≠0. Missing full HBDA hierarchy, WAMP, PEMB (D3–D5) |
| 3 | Data / train protocol | **88%** | Market loader, random erase/flip/pad/crop, PK sampler, LR drops @40/70, size gate 0.14 GiB → full_train |
| 4 | Eval / report / deviations | **90%** | CMC/mAP ranks 1/5/10; REPORT from JSON; DEVIATIONS D1–D8 documented |
| 5 | End-to-end smoke + full train | **92%** | Smoke OK; full 120-ep Market: mAP **76.90**, R1 **87.38**, R5 **96.32** (`train_summary.json`) |

**Mean confidence after Pass 5: ~88%.** Matching published Occluded-Duke numbers is out of scope (D1).

## Pass notes

### Pass 1
Scaffold mirrors paper-2-model template. GitHub name = folder slug (≤100 chars).

### Pass 2
- Bugfix: `torch.chunk` returned &lt;K stripes on 8×H maps → BN size mismatch; replaced with linspace stripes.
- Paper PEMB/WAMP/spatial-γ not implemented → DEVIATIONS D4–D5.

### Pass 3
Hyperparameters copied from PDF §IV-A-2. K=5 for holistic Market.

### Pass 4
REPORT embeds only after real eval JSON exists. Occluded-Duke headline 77.2% R1 noted as not reproduced.

### Pass 5
- Smoke: 40 steps, loss finite, eval path OK (capped mAP≈0 expected).
- `dataset_size.json`: 149 085 020 bytes, decision=`full_train`.
- Full 120-epoch train running under `CUDA_VISIBLE_DEVICES=0`.

## Sign-off

Audit complete after Pass 5. Further PEMB/WAMP fidelity only if user requests.
