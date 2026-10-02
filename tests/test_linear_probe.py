"""The linear-probe ablation must score the SAME windows the neural model is scored on.

The probe re-implements the chronological split and the error arithmetic in plain NumPy
so it can run without PyTorch. These tests pin that re-implementation to the real
pipeline (``ml.data.windowing`` and ``ml.training.losses``), so the two cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.segments import REPO_ROOT
from ml.data.windowing import LABELS, build_chennai_datasets
from ml.models.monsoonplus_net import load_checkpoint
from ml.training import linear_probe as lp
from ml.training.evaluate import MASKED_VARIANTS, _improvement, masked_baselines
from ml.training.losses import error_breakdown
from ml.training.train import build_graph_for, evaluate as run_eval

DAYS = 12  # small window keeps the suite fast; the logic is size-independent
CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "monsoonplus_final.pt"


@pytest.fixture(scope="module")
def window_and_splits():
    window = lp.generate_chennai_window(days=DAYS, seed=2024)
    splits = build_chennai_datasets(days=DAYS, seed=2024, train_stride=1)
    return window, splits


def test_split_indices_match_the_real_windowing(window_and_splits):
    window, splits = window_and_splits
    train, val, test = lp.split_indices(window.num_steps)
    assert np.array_equal(train, splits.train.indices)
    assert np.array_equal(val, splits.val.indices)
    assert np.array_equal(test, splits.test.indices)


def test_default_test_split_is_the_1908_windows_the_model_is_scored_on():
    window = lp.generate_chennai_window(days=45, seed=2024)
    _, _, test = lp.split_indices(window.num_steps)
    assert test.size == 1908
    labels = np.bincount(lp.window_labels(window, test), minlength=3)
    assert labels.tolist() == [1285, 330, 293]  # the counts recorded in the exported JSON


def test_targets_and_labels_match_the_dataset(window_and_splits):
    window, splits = window_and_splits
    test = lp.split_indices(window.num_steps)[2]
    last_speed, delta = lp.build_targets(window, test)
    labels = lp.window_labels(window, test)
    for position in (0, 7, len(splits.test) // 2, len(splits.test) - 1):
        sample = splits.test[position]
        assert np.allclose(
            last_speed[position][:, None] + delta[position], sample["target"].numpy(), atol=1e-4
        )
        assert np.allclose(last_speed[position], sample["traffic"][:, -1, 0].numpy(), atol=1e-4)
        assert labels[position] == int(sample["label_index"])


def test_error_breakdown_matches_losses_error_breakdown():
    rng = np.random.default_rng(0)
    predictions = rng.normal(20, 5, size=(60, 10, 3))
    targets = predictions + rng.normal(0, 1.5, size=predictions.shape)
    labels = rng.integers(0, 3, size=60)
    expected = error_breakdown(predictions, targets, labels).to_dict()
    assert lp.error_breakdown_dict(predictions, targets, labels) == expected


def test_improvement_matches_evaluate():
    for model_mae, baseline_mae in [(0.8, 1.6), (1.0, 0.9), (0.5, None), (None, 1.0)]:
        assert lp._improvement(model_mae, baseline_mae) == _improvement(
            float("nan") if model_mae is None else model_mae,
            float("nan") if baseline_mae is None else baseline_mae,
        )


@pytest.fixture(scope="module")
def probe():
    return lp.run_probe(days=45, seed=2024)


def test_probe_reports_three_variants_on_1908_windows(probe):
    assert set(probe["variants"]) == set(lp.PROBE_NAMES)
    assert probe["split"]["test_windows"] == 1908
    for row in probe["variants"].values():
        assert sum(row["count_by_label"].values()) == 1908
        assert row["mae_overall"] > 0


def test_weather_and_satellite_do_not_hurt_heavy_rain(probe):
    heavy = {name: row["mae_by_label"]["heavy_rain"] for name, row in probe["variants"].items()}
    assert heavy["traffic_weather_linear"] < heavy["traffic_only_linear"]
    # The generator is built so satellite carries information weather history lacks.
    assert heavy["traffic_weather_satellite_linear"] <= heavy["traffic_weather_linear"]


def test_probe_is_deterministic(probe):
    again = lp.run_probe(days=45, seed=2024)
    assert again == probe


def _fake_evaluation():
    return {
        "model": {
            "mae_overall": 0.8,
            "mae_by_label": {"clear": 0.7, "rain": 0.8, "heavy_rain": 1.0},
        },
        "baselines": [{"name": "persistence"}],
        "honesty_notes": ["an existing note"],
    }


def test_attach_is_idempotent_and_leaves_other_rows_alone(probe):
    evaluation = _fake_evaluation()
    lp.attach_to_evaluation(evaluation, probe)
    once = [b["name"] for b in evaluation["baselines"]]
    assert once == ["persistence", *lp.PROBE_NAMES]
    lp.attach_to_evaluation(evaluation, probe)
    assert [b["name"] for b in evaluation["baselines"]] == once
    assert evaluation["honesty_notes"].count("an existing note") == 1
    assert sum(n.startswith(lp.NOTE_PREFIX) for n in evaluation["honesty_notes"]) == 1
    assert evaluation["ablation_methodology"]


def test_baseline_rows_carry_model_improvement(probe):
    rows = {r["name"]: r for r in lp.baseline_rows(probe, _fake_evaluation()["model"])}
    only = rows["traffic_only_linear"]
    expected = round(100 * (only["mae_by_label"]["heavy_rain"] - 1.0) / only["mae_by_label"]["heavy_rain"], 2)
    assert only["model_improvement_percent_by_label"]["heavy_rain"] == expected


def test_honesty_note_is_generated_from_the_numbers(probe):
    note = lp.honesty_note(probe)
    heavy = probe["variants"]["traffic_only_linear"]["mae_by_label"]["heavy_rain"]
    assert f"{heavy:.2f} km/h" in note
    # a sign flip must change the wording, not just the digits
    flipped = {
        **probe,
        "variants": {
            **probe["variants"],
            "traffic_weather_satellite_linear": {
                **probe["variants"]["traffic_weather_satellite_linear"],
                "mae_by_label": {
                    **probe["variants"]["traffic_weather_satellite_linear"]["mae_by_label"],
                    "heavy_rain": probe["variants"]["traffic_weather_linear"]["mae_by_label"]["heavy_rain"] * 1.1,
                },
            },
        },
    }
    assert "WORSE" in lp.honesty_note(flipped)
    assert "WORSE" not in note


# ---- masked (inference-time) ablation on the trained network ----------------------
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no trained checkpoint in ml/checkpoints")
def test_masked_baselines_switch_modalities_off(window_and_splits):
    _, splits = window_and_splits
    model, _ = load_checkpoint(CHECKPOINT)
    model.set_normalisation(splits.normalisation)
    device = torch.device("cpu")
    edge_index, edge_weight, _ = build_graph_for(splits.test.traffic.shape[1], splits)

    full, _, _, _, _ = run_eval(model, splits.test, edge_index, edge_weight, device)
    identity, _, _, _, _ = run_eval(
        model, splits.test, edge_index, edge_weight, device,
        availability_override=torch.ones(3),
    )
    assert np.array_equal(full, identity)  # an all-ones override changes nothing

    masked = masked_baselines(model, splits.test, edge_index, edge_weight, device)
    assert set(masked) == set(MASKED_VARIANTS)
    for breakdown in masked.values():
        assert np.isfinite(breakdown.mae_overall)
    dropped, _, _, _, _ = run_eval(
        model, splits.test, edge_index, edge_weight, device,
        availability_override=torch.tensor(MASKED_VARIANTS["traffic_only_masked"]),
    )
    assert not np.array_equal(full, dropped)  # switching inputs off must matter
