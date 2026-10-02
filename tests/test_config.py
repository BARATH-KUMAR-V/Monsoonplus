"""The network config is load-bearing: every module reads it. These tests pin it."""

from __future__ import annotations

import json

import pytest

from config.segments import (
    HORIZONS_MINUTES,
    HORIZON_STEPS,
    JUNCTIONS,
    JUNCTION_BY_ID,
    LANDMARKS,
    SEGMENTS,
    SEGMENT_IDS,
    STEP_MINUTES,
    THRESHOLDS,
    junction_adjacency,
    monsoon_regime,
    rain_label,
    sample_weight,
    segment_lengths_km,
    validate,
)


def test_config_is_internally_valid():
    """The config's own validator must pass: no orphan junctions, no disconnected graph."""
    problems = validate()
    assert problems == [], f"config/segments.json has problems: {problems}"


def test_expected_network_size():
    assert len(SEGMENTS) == 10
    assert len(JUNCTIONS) == 8


def test_segment_lengths_are_plausible():
    """Chennai's inner corridors are a few km apart; anything outside 0.5-15 km is a typo."""
    for segment, length in zip(SEGMENTS, segment_lengths_km()):
        assert 0.5 < length < 15.0, f"{segment.id} length {length:.2f} km looks wrong"


def test_graph_is_connected():
    adjacency = junction_adjacency()
    seen = {JUNCTIONS[0].id}
    stack = [JUNCTIONS[0].id]
    while stack:
        node = stack.pop()
        for neighbour, _ in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    assert len(seen) == len(JUNCTIONS), "routing would silently fail on an unreachable junction"


@pytest.mark.parametrize(
    "rain,expected",
    [(0.0, "clear"), (2.4, "clear"), (2.5, "rain"), (14.9, "rain"), (15.0, "heavy_rain"), (90.0, "heavy_rain")],
)
def test_rain_label_boundaries(rain, expected):
    assert rain_label(rain) == expected


def test_sample_weights_match_the_spec():
    """Rain 3x, heavy rain 8x -- the cost-sensitive loss depends on exactly these."""
    assert sample_weight(0.0) == 1.0
    assert sample_weight(5.0) == 3.0
    assert sample_weight(50.0) == 8.0


def test_impact_weights_sum_to_one():
    weights = THRESHOLDS["impact_score_weights"]
    assert weights == {"traffic": 0.40, "rain": 0.25, "flood": 0.20, "deterioration": 0.15}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_horizons_divide_into_steps():
    assert HORIZONS_MINUTES == [15, 30, 60]
    assert HORIZON_STEPS == [h // STEP_MINUTES for h in HORIZONS_MINUTES]


def test_landmarks_map_to_real_junctions():
    for landmark in LANDMARKS:
        assert landmark["junction"] in JUNCTION_BY_ID, landmark["name"]


def test_chennai_is_northeast_monsoon():
    """November must be the peak. If this flips, the Overview badge lies to the user."""
    assert "peak" in monsoon_regime(11)["label"].lower()
    assert "onset" in monsoon_regime(10)["label"].lower()


def test_frontend_reads_the_same_file():
    """The UI must import config/segments.json, not keep its own copy of the roads."""
    from config.segments import REPO_ROOT

    network_js = (REPO_ROOT / "frontend" / "src" / "config" / "network.js").read_text(
        encoding="utf-8"
    )
    assert "@config/segments.json" in network_js, "frontend stopped using the shared config"

    # And no page should hardcode a segment id list of its own.
    for page in (REPO_ROOT / "frontend" / "src" / "pages").glob("*.jsx"):
        text = page.read_text(encoding="utf-8")
        assert "'r1', 'r2'" not in text, f"{page.name} looks like it hardcodes segment ids"
