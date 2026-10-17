"""REPORT.md + SVG figures from real logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drop.plot_style import (
    BAR_COLOR, BAR_EDGE_COLOR, BAR_EDGE_WIDTH, FIGSIZE, LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS, apply_rcparams, apply_style,
)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_report(cfg, logs: Path, figdir: Path, report_path: Path):
    hist = load_json(logs / "train_history.json") or []
    summary = load_json(logs / "train_summary.json") or {}
    eval_res = load_json(logs / "eval_results.json") or {}
    size = load_json(logs / "dataset_size.json") or {}
    metrics = eval_res if eval_res.get("mAP") is not None else (summary.get("best") or {})
    figs = []
    if metrics:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        labs = ["mAP", "Rank-1"]
        vals = [float(metrics.get("mAP", 0)), float(metrics.get("Rank-1", 0))]
        ax.bar(np.arange(2), vals, 0.5, color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        ax.set_xticks(np.arange(2)); ax.set_xticklabels(labs)
        ax.set_ylabel("Score (%)"); ax.set_title("Market-1501 DROP (Ours)")
        apply_style(ax, grid_axis="y"); fig.tight_layout()
        out = figdir / "market_map_r1.svg"; fig.savefig(out, format="svg"); plt.close(fig)
        figs.append(out)
    if hist:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["epoch"] for h in hist], [h.get("loss", 0) for h in hist],
                color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="Train loss")
        maps = [(h["epoch"], h["mAP"]) for h in hist if "mAP" in h]
        if maps:
            ax2 = ax.twinx()
            ax2.plot([e for e, _ in maps], [m for _, m in maps],
                     color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="mAP")
            ax2.set_ylabel("mAP (%)")
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="best")
        else:
            ax.legend()
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training curves")
        apply_style(ax); fig.tight_layout()
        out = figdir / "train_curves.svg"; fig.savefig(out, format="svg"); plt.close(fig)
        figs.append(out)

    lines = [
        "# Report: DROP",
        "",
        "Market-1501 rebuild of Dou et al., TBIOM 2026 "
        "(DOI [10.1109/TBIOM.2026.3680760](https://doi.org/10.1109/TBIOM.2026.3680760)).",
        "",
        "## Setup",
        "",
        f"- Image: {cfg['data'].get('height')}×{cfg['data'].get('width')}, ResNet-50 + HBDA + PCT",
        f"- Epochs: {cfg['train']['epochs']}, Adam lr={cfg['train']['lr']}",
        f"- Dataset size gate: {size.get('decision', 'n/a')} ({size.get('total_gib', '?')} GiB)",
        "",
        "## Market-1501 results",
        "",
        "| | mAP | Rank-1 | Rank-5 |",
        "|--|--:|--:|--:|",
        f"| **Ours** | {float(metrics.get('mAP', float('nan'))):.2f} | "
        f"{float(metrics.get('Rank-1', float('nan'))):.2f} | "
        f"{float(metrics.get('Rank-5', float('nan'))):.2f} |",
        "",
        "Paper headline Occluded-Duke Rank-1 **77.2%** not reproduced here (dataset blocked). "
        "See `DEVIATIONS.md`.",
        "",
        "## Figures",
        "",
    ]
    root = report_path.resolve().parent
    for p in figs:
        rel = p.resolve().relative_to(root)
        lines += [f"### `{rel}`", "", f"![]({rel})", ""]
    lines += ["## Regenerate", "", "```bash", "bash scripts/report.sh", "```", ""]
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    figdir = Path(cfg["paths"]["figure_dir"])
    if not figdir.is_absolute():
        figdir = _ROOT / figdir
    logs = Path(cfg["paths"]["log_dir"])
    if not logs.is_absolute():
        logs = _ROOT / logs
    figdir.mkdir(parents=True, exist_ok=True)
    write_report(cfg, logs, figdir, _ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
