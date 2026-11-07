"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def ours(report: str) -> tuple[float, float]:
    match = re.search(r"\| \*\*Ours\*\* \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, "results row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


def test_market_map_matches_table_3(report, targets):
    assert ours(report)[0] == pytest.approx(targets["market1501"]["map"], abs=targets["tolerance_pp"])


def test_market_rank1_matches_table_3(report, targets):
    assert ours(report)[1] == pytest.approx(targets["market1501"]["rank1"], abs=targets["tolerance_pp"])
