"""PCT loss, evaluation protocol and settings, checked against the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from drop.data.market1501 import Market1501, parse_market_name
from drop.data.sampler import RandomIdentitySampler
from drop.losses import BatchHardTriplet, part_compactness_triplet
from drop.metrics import evaluate_rank
from drop.models.drop_net import stripe_parse_targets

ROOT = Path(__file__).resolve().parents[1]


def eq_7(parts: torch.Tensor, labels: torch.Tensor, margin: float) -> float:
    """[Avg(hardest positives) - Avg(hardest negatives) + margin]+ per anchor."""
    feats = F.normalize(parts, dim=-1)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    ap, an = [], []
    for k in range(parts.shape[1]):
        dist = torch.cdist(feats[:, k], feats[:, k])
        ap.append(dist.masked_fill(~same | eye, -1.0).max(dim=1).values)
        an.append(dist.masked_fill(same, 1e6).min(dim=1).values)
    return F.relu(torch.stack(ap).mean(0) - torch.stack(an).mean(0) + margin).mean().item()


def test_pct_hinges_after_averaging_the_parts():
    """Eq. (7): a badly placed part can be offset by the others before the hinge."""
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    bad = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.01], [0.0, 1.01]])
    parts = torch.stack([good, good, good, bad], dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    assert part_compactness_triplet(parts, labels, 0.3).item() == pytest.approx(eq_7(parts, labels, 0.3), abs=1e-4)


def test_batch_hard_triplet_is_zero_when_identities_are_separated():
    feats = torch.tensor([[1.0, 0.0], [1.0, 0.01], [-1.0, 0.0], [-1.0, 0.01]])
    assert BatchHardTriplet(0.3)(feats, torch.tensor([0, 0, 1, 1])).item() == pytest.approx(0.0)


@pytest.mark.parametrize("key", ["lr", "lr_drops", "epochs", "margin", "label_smoothing", "lambda_parse"])
def test_training_settings_match_the_paper(default_cfg, targets, key):
    assert default_cfg["train"][key] == targets["training"][key]


def test_part_count_matches_the_holistic_setting(default_cfg, targets):
    assert default_cfg["model"]["num_parts"] == targets["holistic_parts"]


def test_parsing_targets_cover_every_part():
    target = stripe_parse_targets((16, 8), 5, torch.device("cpu"), 2)
    assert target.unique().tolist() == [0, 1, 2, 3, 4]


def test_market_names_are_parsed():
    assert parse_market_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market_name("-1_c3s1_000451_03.jpg") == (-1, 3)


def rank(q_pid, q_cam, g_pids, g_cams, g_feats) -> dict:
    return evaluate_rank(
        torch.tensor([[1.0, 0.0]]),
        np.array([q_pid]),
        np.array([q_cam]),
        torch.tensor(g_feats),
        np.array(g_pids),
        np.array(g_cams),
        ranks=[1],
    )


def test_same_camera_matches_are_excluded():
    result = rank(1, 1, [1, 1, 2], [1, 2, 2], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    assert result["Rank-1"] == 100.0


def test_average_precision_matches_a_worked_example():
    """Relevant items at ranks 1 and 3 give AP = (1/1 + 2/3) / 2."""
    feats = [[1.0, 0.0], [0.9, 0.3], [0.5, 0.8]]
    assert rank(1, 1, [1, 2, 1], [2, 2, 2], feats)["mAP"] == pytest.approx(100 * (1 + 2 / 3) / 2)


@pytest.fixture
def market(tmp_path):
    names = {
        "bounding_box_test": ["0001_c2s1_000001_01.jpg", "-1_c2s1_000002_01.jpg", "0000_c2s1_000003_01.jpg"],
        "query": ["0001_c1s1_000001_00.jpg"],
    }
    for folder, files in names.items():
        (tmp_path / folder).mkdir()
        for name in files:
            Image.new("RGB", (8, 8)).save(tmp_path / folder / name)
    return tmp_path


def test_junk_images_do_not_count_against_a_query(market):
    """Market-1501 protocol: id -1 images are junk and are ignored, unlike id 0 distractors."""
    gallery = Market1501(market, "gallery")
    feats = {-1: [1.0, 0.0], 1: [0.8, 0.6], 0: [0.0, 1.0]}
    result = rank(1, 1, gallery.pids, gallery.camids, [feats[p] for p in gallery.pids])
    assert result["Rank-1"] == 100.0


def test_sampler_draws_k_instances_per_identity():
    class Labels:
        labels = [i // 6 for i in range(48)]

    batch = list(RandomIdentitySampler(Labels(), num_instances=4, batch_size=16))[:16]
    counts = np.bincount([Labels.labels[i] for i in batch])
    assert sorted(set(counts[counts > 0])) == [4]


def test_checkpoints_are_not_selected_on_query_and_gallery():
    """Keeping the best epoch by test mAP makes the reported test mAP optimistic."""
    source = (ROOT / "drop" / "train.py").read_text(encoding="utf-8")
    assert 'metrics["mAP"] > best["mAP"]' not in source
