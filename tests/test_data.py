"""Data loading, windowing, baselines and the JSON-safety rules."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from config.segments import HISTORY_STEPS, HORIZON_STEPS, SEGMENTS
from ml.data.chennai_synthetic import (
    generate_chennai_window,
    satellite_features,
    traffic_features,
    weather_features,
)
from ml.data.metr_la import to_traffic_features
from ml.data.ndwi import synthetic_ndwi
from ml.data.windowing import LABELS, build_chennai_datasets, chronological_split


@pytest.fixture(scope="module")
def window():
    return generate_chennai_window(days=12, seed=5)


# --------------------------------------------------------- synthetic data --


def test_window_shapes(window):
    assert window.speeds.shape[1] == len(SEGMENTS)
    assert window.rain.shape == window.speeds.shape
    assert window.ndwi.shape == window.speeds.shape
    assert len(window.minute_of_day) == window.speeds.shape[0]


def test_window_is_labelled_synthetic(window):
    """Nothing may present this as measured Chennai traffic."""
    assert window.is_real is False
    assert "synthetic" in window.label.lower()


def test_speeds_are_physically_plausible(window):
    assert window.speeds.min() >= 3.0
    assert window.speeds.max() <= max(s.free_flow_kmh for s in SEGMENTS) * 1.1
    # A monsoon window that never drops below 25 km/h would have no signal to learn.
    assert window.speeds.mean() < 33.0


def test_rain_reduces_speed(window):
    """The core physical relationship the model is supposed to learn."""
    city_rain = window.rain.mean(axis=1)
    mean_speed = window.speeds.mean(axis=1)
    dry = mean_speed[city_rain < 2.5]
    wet = mean_speed[city_rain > 30]
    assert dry.size and wet.size
    assert wet.mean() < dry.mean() - 2.0, "rain must visibly slow the network"


def test_wetness_is_not_saturated(window):
    """A wetness channel pinned at its ceiling carries no information.

    This caught a real calibration bug: an over-aggressive accumulator held wetness at
    1.0 almost permanently, which both destroyed the satellite signal and pushed every
    rainy road to the speed floor.
    """
    wetness = window.wetness
    assert wetness.min() < 0.15, "wetness never drains back toward dry"
    assert wetness.max() > 0.5, "wetness never meaningfully accumulates"
    saturated = float((wetness > 0.98).mean())
    assert saturated < 0.25, f"{saturated:.0%} of samples are saturated at the ceiling"


def test_satellite_adds_information_beyond_rain(window):
    """If NDWI were just a smoothed copy of rainfall, 'trimodal' would be decoration."""
    rain = window.rain.reshape(-1)
    wetness = window.wetness.reshape(-1)
    correlation = float(np.corrcoef(rain, wetness)[0, 1])
    assert 0.2 < correlation < 0.95, (
        f"rain-wetness correlation {correlation:.2f}: the satellite channel is either "
        "unrelated noise or a redundant copy of the weather channel"
    )


def test_drainage_ordering_matches_flood_history():
    """A 'severe' flood-history road must stay wet LONGEST after the rain stops.

    The exponent was inverted at one point, which made marsh-side roads dry fastest --
    the exact opposite of what config/segments.json documents about them.
    """
    rain = np.full(len(SEGMENTS), 50.0, dtype=np.float32)
    dry = np.zeros(len(SEGMENTS), dtype=np.float32)

    snapshot = synthetic_ndwi(rain)
    for _ in range(30):
        snapshot = synthetic_ndwi(rain, previous_wetness=snapshot.wetness)
    for _ in range(60):
        snapshot = synthetic_ndwi(dry, previous_wetness=snapshot.wetness)

    by_history = {}
    for segment, value in zip(SEGMENTS, snapshot.wetness):
        by_history.setdefault(segment.flood_history, []).append(float(value))

    severe = np.mean(by_history["severe"])
    moderate = np.mean(by_history["moderate"])
    low = np.mean(by_history["low"])
    assert severe > moderate > low, (
        f"after drying, severe={severe:.3f} moderate={moderate:.3f} low={low:.3f} -- "
        "roads with a severe flood history must retain the most water"
    )


# ------------------------------------------------------------- windowing --


def test_metr_la_feature_padding_shape():
    """Single-channel speeds must become 3 traffic features BEFORE any checkpoint load."""
    speeds = np.array([[30.0, 0.0], [31.0, 28.0], [0.0, 27.0], [33.0, 26.0]], dtype=np.float32)
    features = to_traffic_features(speeds)
    assert features.shape == (4, 2, 3)
    # confidence is 0 exactly where the reading was missing
    assert features[0, 1, 2] == 0.0
    assert features[0, 0, 2] == 1.0
    # missing speeds are filled, never left at the 0 km/h cliff
    assert features[..., 0].min() > 0.0
    # free-flow proxy is positive and constant down the time axis
    assert np.allclose(features[:, 0, 1], features[0, 0, 1])
    assert (features[..., 1] > 0).all()


def test_chronological_split_does_not_leak(window):
    splits = build_chennai_datasets(days=12, seed=5)
    train_max = int(splits.train.indices.max())
    val_min = int(splits.val.indices.min())
    test_min = int(splits.test.indices.min())

    assert val_min > train_max - HISTORY_STEPS, "validation window reaches into training data"
    assert test_min > int(splits.val.indices.max()) - HISTORY_STEPS
    assert len(splits.train) > len(splits.val)


def test_normalisation_is_fitted_on_training_only(window):
    """Fitting on the whole series would leak validation statistics into training."""
    traffic = traffic_features(window)
    weather = weather_features(window)
    satellite = satellite_features(window)
    rain = window.rain.mean(axis=1)

    splits = chronological_split(
        traffic, weather, satellite, rain, (1, 1, 1), "test", False, "unit-test"
    )
    train_end = int(traffic.shape[0] * 0.7)
    expected = traffic[:train_end].reshape(-1, traffic.shape[-1]).mean(axis=0)
    assert np.allclose(splits.normalisation.traffic_mean, expected, atol=1e-3)

    whole = traffic.reshape(-1, traffic.shape[-1]).mean(axis=0)
    assert not np.allclose(splits.normalisation.traffic_mean, whole, atol=1e-4)


def test_sample_contents(window):
    splits = build_chennai_datasets(days=12, seed=5)
    sample = splits.train[0]
    assert sample["traffic"].shape == (len(SEGMENTS), HISTORY_STEPS, 3)
    assert sample["weather"].shape == (len(SEGMENTS), HISTORY_STEPS, 3)
    assert sample["satellite"].shape == (len(SEGMENTS), 2)
    assert sample["target"].shape == (len(SEGMENTS), len(HORIZON_STEPS))
    assert float(sample["weight"]) in (1.0, 3.0, 8.0)
    assert 0 <= int(sample["label_index"]) < len(LABELS)


def test_metr_la_masks_weather_and_satellite():
    """LA has no rain channel; claiming otherwise would make 'trimodal' false."""
    from ml.data.windowing import build_metr_la_datasets

    splits = build_metr_la_datasets(max_sensors=8, quiet=True)
    assert list(splits.train.availability) == [1.0, 0.0, 0.0]


# ----------------------------------------------------------- baselines ----


def test_persistence_predicts_the_last_speed():
    from ml.training.baselines import persistence_predictions

    splits = build_chennai_datasets(days=10, seed=3)
    predictions, targets, labels = persistence_predictions(splits.test)
    assert predictions.shape == targets.shape
    # every horizon gets the same value
    assert np.allclose(predictions[:, :, 0], predictions[:, :, -1])

    sample = splits.test[0]
    assert np.allclose(predictions[0, :, 0], sample["traffic"][:, -1, 0].numpy(), atol=1e-4)


def test_historical_average_uses_time_of_day(window):
    from ml.training.baselines import HistoricalAverage

    model = HistoricalAverage().fit(window.speeds, window.minute_of_day)
    assert model.table.shape == (288, len(SEGMENTS))
    assert np.isfinite(model.table).all()
    # Rush hour should be slower than the small hours.
    rush = model.table[(18 * 60) // 5]
    night = model.table[(3 * 60) // 5]
    assert rush.mean() < night.mean()


# --------------------------------------------------------- json safety ----


def test_exports_contain_no_nan():
    """NaN is not valid JSON. json.dumps writes it happily; JSON.parse then throws."""
    from ml.training.evaluate import _json_safe_dumps

    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": [1.0, float("-inf")],
        "d": {"e": np.float32("nan")},
        "ok": 1.5,
    }
    text = _json_safe_dumps(payload)
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)  # must not raise
    assert parsed["a"] is None
    assert parsed["b"] is None
    assert parsed["c"][1] is None
    assert parsed["d"]["e"] is None
    assert parsed["ok"] == 1.5


def test_collector_store_cleans_nan(tmp_path):
    from collector.store import write_json

    path = write_json(tmp_path / "x.json", {"v": float("nan"), "w": [float("inf"), 2.0]})
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["v"] is None
    assert parsed["w"] == [None, 2.0]
