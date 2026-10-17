"""Train DROP on Market-1501."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drop.data import Market1501, RandomIdentitySampler, build_transforms, resolve_market_root
from drop.losses import drop_loss
from drop.metrics import evaluate_rank
from drop.models.drop_net import build_drop, stripe_parse_targets


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for k in ("checkpoint_dir", "log_dir", "pred_dir", "figure_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_features(model, loader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        out = model(img)
        feats.append(out["feat"].cpu())
        pids.append(batch["pid"].numpy())
        cams.append(batch["camid"].numpy())
    import numpy as np

    return {
        "feats": torch.cat(feats, 0),
        "pids": np.concatenate(pids),
        "cams": np.concatenate(cams),
    }


def run_eval(model, cfg, device: torch.device) -> Dict[str, float]:
    root = resolve_market_root(cfg)
    tf = build_transforms(cfg, train=False)
    q = Market1501(root, "query", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    g = Market1501(root, "gallery", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    nw = int(cfg["data"].get("num_workers", 4))
    bs = int(cfg["eval"].get("batch_size", 64))
    qq = extract_features(model, DataLoader(q, batch_size=bs, shuffle=False, num_workers=nw), device)
    gg = extract_features(model, DataLoader(g, batch_size=bs, shuffle=False, num_workers=nw), device)
    return evaluate_rank(
        qq["feats"], qq["pids"], qq["cams"], gg["feats"], gg["pids"], gg["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )


def lr_at(epoch: int, cfg: dict) -> float:
    base = float(cfg["train"]["lr"])
    drops = cfg["train"].get("lr_drops", [40, 70])
    lr = base
    for d in drops:
        if epoch >= int(d):
            lr *= 0.1
    return lr


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DROP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _ensure_dirs(cfg)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    size_json = log_dir / "dataset_size.json"
    if not size_json.exists():
        import subprocess

        data_root = Path(cfg["data"]["root"])
        if not data_root.is_absolute():
            data_root = (_ROOT / data_root).resolve()
        try:
            total = int(subprocess.check_output(["du", "-sb", str(data_root)], text=True).split()[0])
            size_json.write_text(
                json.dumps(
                    {
                        "total_bytes": total,
                        "total_gib": round(total / (1024**3), 4),
                        "decision": "full_train" if total < 5 * 1024**3 else "skip",
                        "threshold_bytes": 5 * 1024**3,
                    },
                    indent=2,
                )
            )
        except Exception as e:
            print(f"[size] warn: {e}", flush=True)

    root = resolve_market_root(cfg)
    train_ds = Market1501(
        root, "train", transform=build_transforms(cfg, True),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    model = build_drop(cfg, num_classes=train_ds.num_classes).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.1)))
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    batch = int(cfg["train"]["batch_size"])
    ni = int(cfg["train"].get("num_instances", 4))
    sampler = RandomIdentitySampler(train_ds, num_instances=ni, batch_size=batch)
    loader = DataLoader(
        train_ds, batch_size=batch, sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True, drop_last=True,
    )

    epochs = int(cfg["train"]["epochs"])
    eval_every = args.eval_every or int(cfg["eval"].get("every", 20))
    max_steps = cfg["train"].get("max_steps")
    k_parts = int(cfg["model"].get("num_parts", 5))

    history, best, global_step = [], {"mAP": -1.0}, 0

    for epoch in range(epochs):
        t0 = time.time()
        cur_lr = lr_at(epoch, cfg)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        running, n = 0.0, 0
        for bd in loader:
            x = bd["image"].to(device, non_blocking=True)
            y = bd["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x)
                _, _, h, w = out["parse_logits"].shape
                parse_tgt = stripe_parse_targets((h, w), k_parts, device, x.size(0))
                losses = drop_loss(
                    out, y, parse_tgt, ce,
                    margin=float(cfg["train"].get("margin", 0.3)),
                    lambda_tri=float(cfg["train"].get("lambda_tri", 1.0)),
                    lambda_pct=float(cfg["train"].get("lambda_pct", 1.0)),
                    lambda_parse=float(cfg["train"].get("lambda_parse", 0.4)),
                )
                loss = losses["loss"]
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            n += 1
            global_step += 1
            if max_steps is not None and global_step >= int(max_steps):
                break

        row = {"epoch": epoch + 1, "lr": cur_lr, "loss": running / max(n, 1), "sec": time.time() - t0}
        if ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs):
            metrics = run_eval(model, cfg, device)
            row["mAP"] = metrics["mAP"]
            row["Rank-1"] = metrics["Rank-1"]
            row["Rank-5"] = metrics.get("Rank-5", 0.0)
            print(
                f"[eval] epoch={epoch+1} mAP={metrics['mAP']:.2f} "
                f"R1={metrics['Rank-1']:.2f} R5={metrics.get('Rank-5', 0):.2f}",
                flush=True,
            )
            if metrics["mAP"] > best["mAP"]:
                best = {k: metrics[k] for k in metrics if k != "cmc"}
                best["epoch"] = epoch + 1
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, "best": best, "cfg": cfg},
                    ckpt_dir / "best.pt",
                )
        history.append(row)
        print(
            f"[epoch] {epoch+1}/{epochs} loss={row['loss']:.4f} lr={cur_lr:.6f} sec={row['sec']:.1f}",
            flush=True,
        )
        (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "epoch": epoch + 1, "history": history, "best": best},
            ckpt_dir / "last.pt",
        )
        if max_steps is not None and global_step >= int(max_steps):
            break

    summary = {
        "best": best,
        "paper_target_note": "Paper highlights Occluded-Duke R1=77.2%; Market comparable to SOTA (Table text)",
        "epochs_ran": len(history),
        "config": args.config,
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best={best}", flush=True)


if __name__ == "__main__":
    main()
