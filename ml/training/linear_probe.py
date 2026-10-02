"""Linear-probe modality ablation, scored on the project's own held-out test windows.

Why this exists
---------------
The baselines already in the evaluation (persistence, historical average, LSTM
without the graph) answer "does the trained network beat a naive guess, and does the
graph help". They do not answer the question a Deep Learning course asks first:
**does fusing three modalities beat using fewer of them?**

``ml/training/ablations.py`` answers it by retraining the full network per variant
(strongest evidence, but minutes of CPU). This module answers it with something that
runs in about a second and needs **no PyTorch**: a pooled ridge-regression "linear
probe" that is identical across variants except for which inputs its feature vector
includes:

    traffic_only_linear               last 6 speeds, free-flow speed
    traffic_weather_linear            + rain history (6 steps and 24-step mean), temp, wind
    traffic_weather_satellite_linear  + NDWI and the accumulated-wetness anomaly

It is deliberately a weaker model family than MonsoonPlusNet (linear, no recurrence,
no graph), so it isolates one question -- *does this input help a model that could use
it* -- not *is the network optimal*.

Same data, same split, same targets as the network
--------------------------------------------------
Everything is taken from the project's own pipeline so the numbers sit in the same
table as the network's, on the same windows:

* the window is ``ml.data.chennai_synthetic.generate_chennai_window`` (same days/seed
  as training and export);
* the split is the chronological 70/15/15 split of ``ml.data.windowing`` -- the probe
  fits on the training portion only, picks its ridge strength on the validation
  portion, and is scored on the test portion only (1,908 windows at the defaults, the
  exact count the neural model is scored on);
* targets are the speed at t+15/30/60 predicted as a *delta* from the last observed
  speed, like the network's head;
* the weather label is the city-mean rain at the last observed step, like
  ``ForecastDataset``.

``tests/test_linear_probe.py`` pins both the split indices and the error-breakdown
arithmetic against ``windowing.py`` / ``losses.py`` so this file cannot drift from the
real pipeline without a test failing.

Run it
------
    python -m ml.training.linear_probe          # prints the table, writes ml/reports/linear_probe.json
    python ml/reference/ablation_numpy.py       # same thing, runnable from anywhere

The only imports are NumPy and the project's own torch-free modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from config.segments import (
    HISTORY_STEPS,
    HORIZON_STEPS,
    HORIZONS_MINUTES,
    REPO_ROOT,
    THRESHOLDS,
    rain_label,
)
from ml.data.chennai_synthetic import (
    ChennaiWindow,
    generate_chennai_window,
    satellite_features,
    traffic_features,
    weather_features,
)

LABELS = ("clear", "rain", "heavy_rain")
RAIN_WEIGHTS = THRESHOLDS["rain_weights"]
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)  # keep in step with windowing.chronological_split
RECENT_STEPS = 6                      # 30 minutes of recent history per feature
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)  # chosen on the validation split, never on test

# Names are what the frontend's Model Lab looks for in evaluation.baselines.
VARIANTS = {
    "traffic_only": "traffic_only_linear",
    "traffic_weather": "traffic_weather_linear",
    "traffic_weather_satellite": "traffic_weather_satellite_linear",
}
PROBE_NAMES = tuple(VARIANTS.values())
REPORT_PATH = REPO_ROOT / "ml" / "reports" / "linear_probe.json"

# The honesty note this module writes starts with this prefix so re-running replaces
# its own note instead of stacking duplicates.
NOTE_PREFIX = "Modality ablation (linear probe"


# --------------------------------------------------------------------------------
# splits: a numpy mirror of ml.data.windowing.chronological_split (test pins it)
# --------------------------------------------------------------------------------
def split_indices(
    num_steps: int, fractions: tuple[float, float, float] = SPLIT_FRACTIONS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window indices ``t`` for train / val / test. A window at ``t`` sees steps
    ``t-24 .. t-1`` and predicts steps up to ``t+11``."""
    train_end = int(num_steps * fractions[0])
    val_end = int(num_steps * (fractions[0] + fractions[1]))
    last_valid = num_steps - max(HORIZON_STEPS)

    def indices(start: int, stop: int) -> np.ndarray:
        return np.arange(max(HISTORY_STEPS, start + HISTORY_STEPS), min(stop, last_valid))

    return indices(0, train_end), indices(train_end, val_end), indices(val_end, num_steps)


# --------------------------------------------------------------------------------
# features and targets
# --------------------------------------------------------------------------------
def _recent(series: np.ndarray, index: np.ndarray, steps: int) -> np.ndarray:
    """(M, N, steps): the last ``steps`` values of a (T, N) series before each t,
    oldest first."""
    offsets = np.arange(steps, 0, -1)
    gathered = series[index[:, None] - offsets[None, :]]  # (M, steps, N)
    return gathered.transpose(0, 2, 1)


def build_features(window: ChennaiWindow, index: np.ndarray) -> dict[str, np.ndarray]:
    """Per-(window, node) feature blocks, each (M, N, F)."""
    traffic = traffic_features(window)    # (T, N, 3): speed, free-flow, confidence
    weather = weather_features(window)    # (T, N, 3): rain, temp, wind
    satellite = satellite_features(window)  # (T, N, 2): ndwi, wetness

    speed = traffic[:, :, 0]
    free_flow = traffic[:, :, 1]
    rain = weather[:, :, 0]
    last = index - 1

    free_now = free_flow[last]
    traffic_block = np.concatenate(
        [
            _recent(speed, index, RECENT_STEPS) / free_now[:, :, None],
            (free_now / 35.0)[:, :, None],
        ],
        axis=-1,
    )

    weather_block = np.concatenate(
        [
            _recent(rain, index, RECENT_STEPS) / 100.0,
            (_recent(rain, index, HISTORY_STEPS).mean(axis=-1) / 100.0)[:, :, None],
            (weather[last, :, 1] / 35.0)[:, :, None],
            (weather[last, :, 2] / 20.0)[:, :, None],
        ],
        axis=-1,
    )

    satellite_block = satellite[last]  # (M, N, 2)
    return {"traffic": traffic_block, "weather": weather_block, "satellite": satellite_block}


def variant_matrix(blocks: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """Concatenate the blocks a variant is allowed to see -> (M, N, F)."""
    parts = [blocks["traffic"]]
    if variant in ("traffic_weather", "traffic_weather_satellite"):
        parts.append(blocks["weather"])
    if variant == "traffic_weather_satellite":
        parts.append(blocks["satellite"])
    return np.concatenate(parts, axis=-1)


def build_targets(window: ChennaiWindow, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (last_speed (M, N), delta (M, N, H)); speed at t+h is last + delta."""
    speed = window.speeds.astype(np.float64)
    last_speed = speed[index - 1]
    delta = np.stack([speed[index + h - 1] - last_speed for h in HORIZON_STEPS], axis=-1)
    return last_speed, delta


def window_labels(window: ChennaiWindow, index: np.ndarray) -> np.ndarray:
    """Weather class per window, from city-mean rain at the last observed step."""
    city_rain = window.rain.mean(axis=1)
    return np.array([LABELS.index(rain_label(float(city_rain[t - 1]))) for t in index])


# --------------------------------------------------------------------------------
# ridge regression
# --------------------------------------------------------------------------------
def _flatten(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(-1, matrix.shape[-1])


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    gram = xb.T @ xb + lam * np.eye(xb.shape[1])
    return np.linalg.solve(gram, xb.T @ y)


def _ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) @ weights


def fit_and_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Standardise on train, choose lambda on val, return (test deltas, lambda).

    x are (M, N, F), y are (M, N, H); samples are pooled over windows and roads, so
    one coefficient set serves all ten segments (the network shares weights across
    nodes in the same way).
    """
    tx, vx, sx = _flatten(train_x), _flatten(val_x), _flatten(test_x)
    mean = tx.mean(axis=0)
    std = tx.std(axis=0)
    std[std < 1e-9] = 1.0
    tx, vx, sx = (tx - mean) / std, (vx - mean) / std, (sx - mean) / std

    ty = train_y.reshape(-1, train_y.shape[-1])
    vy = val_y.reshape(-1, val_y.shape[-1])

    best_lam, best_error = RIDGE_GRID[0], float("inf")
    for lam in RIDGE_GRID:
        weights = _ridge_fit(tx, ty, lam)
        error = float(np.abs(_ridge_predict(vx, weights) - vy).mean())
        if error < best_error:
            best_lam, best_error = lam, error

    weights = _ridge_fit(tx, ty, best_lam)
    predicted = _ridge_predict(sx, weights).reshape(test_x.shape[0], test_x.shape[1], -1)
    return predicted, best_lam


# --------------------------------------------------------------------------------
# error breakdown -- same arithmetic and JSON shape as ml.training.losses.error_breakdown
# --------------------------------------------------------------------------------
def error_breakdown_dict(
    predictions: np.ndarray, targets: np.ndarray, label_index: np.ndarray
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    absolute = np.abs(predictions - targets)
    squared = (predictions - targets) ** 2

    out = {
        "mae_overall": float(absolute.mean()),
        "rmse_overall": float(np.sqrt(squared.mean())),
        "mae_by_label": {},
        "mae_by_horizon": {},
        "mae_by_label_horizon": {},
        "count_by_label": {},
    }
    for i, horizon in enumerate(HORIZONS_MINUTES):
        out["mae_by_horizon"][f"t+{horizon}"] = float(absolute[:, :, i].mean())

    weighted_total = weight_total = 0.0
    for i, label in enumerate(LABELS):
        mask = label_index == i
        count = int(mask.sum())
        out["count_by_label"][label] = count
        if count == 0:
            out["mae_by_label"][label] = float("nan")
            out["mae_by_label_horizon"][label] = {f"t+{h}": float("nan") for h in HORIZONS_MINUTES}
            continue
        subset = absolute[mask]
        label_mae = float(subset.mean())
        out["mae_by_label"][label] = label_mae
        out["mae_by_label_horizon"][label] = {
            f"t+{h}": float(subset[:, :, j].mean()) for j, h in enumerate(HORIZONS_MINUTES)
        }
        weight = float(RAIN_WEIGHTS[label])
        weighted_total += label_mae * weight * count
        weight_total += weight * count

    out["rain_weighted_mae"] = weighted_total / weight_total if weight_total else out["mae_overall"]
    return _rounded(out)


def _rounded(breakdown: dict) -> dict:
    """Round exactly as ErrorBreakdown.to_dict does, in the same key order."""

    def r(value):
        return None if not np.isfinite(value) else round(float(value), 3)

    return {
        "mae_overall": r(breakdown["mae_overall"]),
        "rmse_overall": r(breakdown["rmse_overall"]),
        "rain_weighted_mae": r(breakdown["rain_weighted_mae"]),
        "mae_by_label": {k: r(v) for k, v in breakdown["mae_by_label"].items()},
        "mae_by_horizon": {k: r(v) for k, v in breakdown["mae_by_horizon"].items()},
        "mae_by_label_horizon": {
            label: {h: r(v) for h, v in per.items()}
            for label, per in breakdown["mae_by_label_horizon"].items()
        },
        "count_by_label": dict(breakdown["count_by_label"]),
    }


# --------------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------------
def run_probe(days: int = 45, seed: int = 2024) -> dict:
    """Fit the three variants and score them on the held-out test windows."""
    window = generate_chennai_window(days=days, seed=seed)
    train_i, val_i, test_i = split_indices(window.num_steps)

    blocks = {name: build_features(window, idx) for name, idx in
              (("train", train_i), ("val", val_i), ("test", test_i))}
    _, train_y = build_targets(window, train_i)
    _, val_y = build_targets(window, val_i)
    test_last, test_y = build_targets(window, test_i)
    labels = window_labels(window, test_i)
    actual = test_last[:, :, None] + test_y

    variants = {}
    for variant, public_name in VARIANTS.items():
        delta, lam = fit_and_predict(
            variant_matrix(blocks["train"], variant),
            train_y,
            variant_matrix(blocks["val"], variant),
            val_y,
            variant_matrix(blocks["test"], variant),
        )
        predicted = test_last[:, :, None] + delta
        variants[public_name] = {
            "ridge_lambda": lam,
            **error_breakdown_dict(predicted, actual, labels),
        }

    return {
        "days": days,
        "seed": seed,
        "split": {
            "train_windows": int(train_i.size),
            "val_windows": int(val_i.size),
            "test_windows": int(test_i.size),
            "fractions": list(SPLIT_FRACTIONS),
        },
        "variants": variants,
    }


def _improvement(model_mae: Optional[float], baseline_mae: Optional[float]) -> Optional[float]:
    """Percent reduction in MAE; identical to ml.training.evaluate._improvement."""
    if model_mae is None or baseline_mae is None:
        return None
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return None
    return round(100.0 * (baseline_mae - model_mae) / baseline_mae, 2)


def baseline_rows(probe: dict, model_block: dict) -> list[dict]:
    """The probe variants as ``evaluation.baselines`` rows, with the network's
    improvement over each, computed the way evaluate.compare_to_baselines does."""
    rows = []
    for name in PROBE_NAMES:
        variant = {k: v for k, v in probe["variants"][name].items() if k != "ridge_lambda"}
        rows.append(
            {
                "name": name,
                **variant,
                "model_improvement_percent_overall": _improvement(
                    model_block["mae_overall"], variant["mae_overall"]
                ),
                "model_improvement_percent_by_label": {
                    label: _improvement(
                        model_block["mae_by_label"].get(label),
                        variant["mae_by_label"].get(label),
                    )
                    for label in LABELS
                },
            }
        )
    return rows


def methodology_text(probe: dict) -> str:
    split = probe["split"]
    return (
        "Three pooled ridge-regression linear probes, identical except for which "
        "inputs they see (traffic only / + weather / + satellite). They are fitted on "
        f"the chronological training split, tuned on validation, and scored on the same "
        f"{split['test_windows']} held-out test windows as the network, with the same "
        "t+15/30/60 delta targets. They are intentionally a simpler model family than "
        "monsoonplus (linear, no recurrence, no graph), so the comparison answers "
        "'does this input help a model that could use it', not 'is monsoonplus's exact "
        "architecture optimal'."
    )


def _gain(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if not before or after is None:
        return None
    return 100.0 * (before - after) / before


def honesty_note(probe: dict) -> str:
    """Plain-English reading of the probe, generated from its numbers (never typed)."""
    v = probe["variants"]
    only = v["traffic_only_linear"]["mae_by_label"].get("heavy_rain")
    weather = v["traffic_weather_linear"]["mae_by_label"].get("heavy_rain")
    full = v["traffic_weather_satellite_linear"]["mae_by_label"].get("heavy_rain")
    if None in (only, weather, full):
        return (
            f"{NOTE_PREFIX}, same test windows): not enough heavy-rain windows in the "
            "test split to read a modality effect from it."
        )

    weather_gain = _gain(only, weather)
    satellite_gain = _gain(weather, full)

    text = (
        f"{NOTE_PREFIX}, same held-out test windows as the network): a traffic-only "
        f"model reaches {only:.2f} km/h MAE in heavy rain. "
    )
    if weather_gain is not None:
        verb = "cuts that by" if weather_gain >= 0 else "makes that worse by"
        text += f"Adding weather history {verb} {abs(weather_gain):.1f}% ({weather:.2f} km/h). "
    if satellite_gain is not None and satellite_gain >= 1.0:
        text += (
            f"Adding the satellite NDWI/wetness features on top cuts it a further "
            f"{satellite_gain:.1f}% ({full:.2f} km/h) even for a LINEAR model, which is what "
            "this generator is built to produce: NDWI tracks accumulated wetness over many "
            "hours, which a 2-hour weather history cannot reveal. The satellite branch is "
            "therefore not decorative here -- but the signal is synthetic, and real "
            "Sentinel-2 NDWI may behave differently."
        )
    elif satellite_gain is not None and satellite_gain <= -1.0:
        text += (
            f"Adding the satellite features on top makes the linear probe WORSE by "
            f"{abs(satellite_gain):.1f}% ({full:.2f} km/h): with so few extra inputs a pooled "
            "linear model can overfit them. Reported as-is; the retrained-network ablation "
            "is the stronger test of whether the satellite branch earns its place."
        )
    else:
        text += (
            f"Adding the satellite features on top changes it by "
            f"{(satellite_gain or 0.0):+.1f}% ({full:.2f} km/h) -- essentially no gain for a "
            "LINEAR model. That does not show the satellite branch is useless: a pooled "
            "linear probe cannot combine that signal non-linearly with neighbouring "
            "roads' state the way the graph network can. The retrained-network ablation "
            "is the stronger test."
        )
    return text


def attach_to_evaluation(evaluation: dict, probe: dict) -> dict:
    """Merge the probe into an evaluation payload. Idempotent: re-running replaces the
    probe's own rows and note rather than duplicating them."""
    evaluation["baselines"] = [
        b for b in evaluation.get("baselines", []) if b.get("name") not in PROBE_NAMES
    ] + baseline_rows(probe, evaluation["model"])
    evaluation["ablation_methodology"] = methodology_text(probe)
    notes = [n for n in evaluation.get("honesty_notes", []) if not n.startswith(NOTE_PREFIX)]
    notes.append(honesty_note(probe))
    evaluation["honesty_notes"] = notes
    return evaluation


def print_report(probe: dict) -> None:
    split = probe["split"]
    print(
        f"linear-probe modality ablation  (days={probe['days']} seed={probe['seed']}, "
        f"test windows={split['test_windows']})"
    )
    header = f"  {'variant':<36}{'overall':>9}{'clear':>8}{'rain':>8}{'heavy':>8}{'lambda':>8}"
    print(header)
    for name, row in probe["variants"].items():
        m = row["mae_by_label"]
        print(
            f"  {name:<36}{row['mae_overall']:>9.3f}{m['clear']:>8.3f}{m['rain']:>8.3f}"
            f"{m['heavy_rain']:>8.3f}{row['ridge_lambda']:>8g}"
        )
    print("\n" + honesty_note(probe))


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description="Dependency-free linear-probe modality ablation")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no-write", action="store_true", help="print only, do not write the report")
    args = parser.parse_args(argv)

    probe = run_probe(days=args.days, seed=args.seed)
    print_report(probe)
    if not args.no_write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(f"\nwrote {REPORT_PATH}")
    return probe


if __name__ == "__main__":
    main()
