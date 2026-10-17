"""Evaluate DROP checkpoint on Market-1501."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drop.data import Market1501, build_transforms, resolve_market_root
from drop.models import build_drop
from drop.train import load_config, run_eval


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    root = resolve_market_root(cfg)
    n = Market1501(root, "train", transform=build_transforms(cfg, False)).num_classes
    model = build_drop(cfg, num_classes=n).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    metrics = run_eval(model, cfg, device)
    out = {k: metrics[k] for k in metrics if k != "cmc"}
    out["checkpoint"] = args.checkpoint
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["log_dir"]) / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
