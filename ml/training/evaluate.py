"""Evaluation: the rain-vs-clear split sits next to every headline number.

The rule this module enforces
----------------------------
An aggregate "X% better than persistence" can be entirely driven by clear-weather
samples while the rain error gets *worse*. That would be the opposite of what
monsoonplus claims to do, so ``compare_to_baselines`` computes the improvement
separately for clear, rain and heavy rain, and ``honesty_notes`` writes a plain-English
warning into the output whenever the aggregate flatters the model relative to its
rain performance.

Those notes are exported to the frontend and shown in the Model Lab. Nothing filters
them out on the way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config.segments import HORIZONS_MINUTES, REPO_ROOT, SEGMENT_IDS
from ml.data.windowing import LABELS, SplitDatasets
from ml.models.monsoonplus_net import MonsoonPlusNet
from ml.training.baselines import (
    HistoricalAverage,
    persistence_predictions,
    predict_lstm,
    train_lstm_baseline,
)
from ml.training.losses import ErrorBreakdown, error_breakdown

REPORT_DIR = REPO_ROOT / "ml" / "reports"

# Inference-time modality ablation on the TRAINED network: (traffic, weather, satellite)
# availability, applied on top of what the dataset itself provides. Not retrained --
# it shows what the finished model currently leans on. ml/training/ablations.py is the
# retrain-without-it version; ml/training/linear_probe.py is the model-family-free one.
MASKED_VARIANTS = {
    "traffic_only_masked": (1.0, 0.0, 0.0),
    "traffic_weather_masked": (1.0, 1.0, 0.0),
}


def _improvement(model_mae: float, baseline_mae: float) -> Optional[float]:
    """Percent reduction in MAE. None when the baseline number is unavailable."""
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return None
    return round(100.0 * (baseline_mae - model_mae) / baseline_mae, 2)


@dataclass
class Comparison:
    name: str
    breakdown: ErrorBreakdown
    improvement_overall: Optional[float]
    improvement_by_label: dict[str, Optional[float]]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            **self.breakdown.to_dict(),
            "model_improvement_percent_overall": self.improvement_overall,
            "model_improvement_percent_by_label": self.improvement_by_label,
        }


def compare_to_baselines(
    model_breakdown: ErrorBreakdown, baselines: dict[str, ErrorBreakdown]
) -> list[Comparison]:
    out = []
    for name, baseline in baselines.items():
        out.append(
            Comparison(
                name=name,
                breakdown=baseline,
                improvement_overall=_improvement(
                    model_breakdown.mae_overall, baseline.mae_overall
                ),
                improvement_by_label={
                    label: _improvement(
                        model_breakdown.mae_by_label.get(label, float("nan")),
                        baseline.mae_by_label.get(label, float("nan")),
                    )
                    for label in LABELS
                },
            )
        )
    return out


def honesty_notes(
    model_breakdown: ErrorBreakdown, comparisons: list[Comparison]
) -> list[str]:
    """Plain-English caveats, generated from the numbers rather than written by hand."""
    notes: list[str] = []

    for label in ("rain", "heavy_rain"):
        count = model_breakdown.count_by_label.get(label, 0)
        if count == 0:
            notes.append(
                f"No {label.replace('_', ' ')} samples in this test split, so the "
                f"{label.replace('_', ' ')} column is not evidence of anything."
            )

    clear_mae = model_breakdown.mae_by_label.get("clear", float("nan"))
    heavy_mae = model_breakdown.mae_by_label.get("heavy_rain", float("nan"))
    if np.isfinite(clear_mae) and np.isfinite(heavy_mae) and heavy_mae > clear_mae:
        ratio = heavy_mae / max(clear_mae, 1e-6)
        notes.append(
            f"Heavy-rain error is {ratio:.1f}x the clear-weather error "
            f"({heavy_mae:.2f} vs {clear_mae:.2f} km/h). Heavy rain remains the hard case; "
            "the cost-sensitive loss narrows this gap but does not close it."
        )

    for comparison in comparisons:
        overall = comparison.improvement_overall
        heavy = comparison.improvement_by_label.get("heavy_rain")
        clear = comparison.improvement_by_label.get("clear")
        if overall is None:
            continue
        if overall <= 0:
            notes.append(
                f"The model does NOT beat {comparison.name} overall "
                f"({overall:+.1f}%). Reported as-is."
            )
        if (
            heavy is not None
            and clear is not None
            and overall > 0
            and heavy < 0.4 * overall
        ):
            notes.append(
                f"Against {comparison.name}, the {overall:+.1f}% aggregate gain is driven "
                f"mainly by clear weather ({clear:+.1f}%); the heavy-rain gain is only "
                f"{heavy:+.1f}%. The aggregate number overstates the monsoon benefit."
            )
    return notes


@torch.no_grad()
def model_predictions(
    model: MonsoonPlusNet,
    dataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ml.training.train import evaluate as run_eval

    predictions, targets, labels, gates, rain = run_eval(
        model, dataset, edge_index, edge_weight, device, batch_size, collect_gate=True
    )
    return predictions, targets, labels, gates, rain


def masked_baselines(
    model: MonsoonPlusNet,
    dataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, ErrorBreakdown]:
    """Score the trained network with weather and/or satellite switched off.

    Uses the same availability mask the METR-LA phase uses, so the gate renormalises
    over the modalities that remain (``MonsoonPlusNet.modality_masked`` does the same
    for a single batch). Only meaningful when the dataset carries all three modalities.
    """
    from ml.training.train import evaluate as run_eval

    out: dict[str, ErrorBreakdown] = {}
    for name, mask in MASKED_VARIANTS.items():
        predictions, targets, labels, _, _ = run_eval(
            model,
            dataset,
            edge_index,
            edge_weight,
            device,
            availability_override=torch.tensor(mask, dtype=torch.float32),
        )
        out[name] = error_breakdown(predictions, targets, labels)
    return out


def gate_by_regime(gates: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean gate weights per weather regime -- the Model Lab's modality breakdown.

    These are the trained model's own gate outputs. If the numbers do not support the
    "leans on weather in rain" story, they are reported unchanged.
    """
    from ml.models.fusion import MODALITIES

    out: dict[str, dict[str, float]] = {}
    for index, label in enumerate(LABELS):
        mask = labels == index
        if not mask.any():
            out[label] = {name: float("nan") for name in MODALITIES}
            continue
        mean = gates[mask].reshape(-1, len(MODALITIES)).mean(axis=0)
        out[label] = {
            name: round(float(mean[i]) * 100, 1) for i, name in enumerate(MODALITIES)
        }
    return out


def gate_shift_claim(gate_summary: dict[str, dict[str, float]]) -> dict:
    """Does the gate actually shift toward weather+satellite in heavy rain?

    Returns the verdict and the numbers behind it. ``supported`` is False when the
    evidence is absent -- the UI prints whichever answer comes back.
    """
    clear = gate_summary.get("clear", {})
    heavy = gate_summary.get("heavy_rain", {})
    if not clear or not heavy or not np.isfinite(heavy.get("weather", float("nan"))):
        return {
            "supported": False,
            "reason": "Not enough samples in one of the regimes to compare gates.",
        }

    clear_env = clear.get("weather", 0.0) + clear.get("satellite", 0.0)
    heavy_env = heavy.get("weather", 0.0) + heavy.get("satellite", 0.0)
    shift = heavy_env - clear_env
    return {
        "supported": bool(shift > 1.0),
        "clear_weather_plus_satellite_percent": round(clear_env, 1),
        "heavy_rain_weather_plus_satellite_percent": round(heavy_env, 1),
        "shift_percentage_points": round(shift, 1),
        "reason": (
            f"In heavy rain the gate puts {heavy_env:.1f}% of its weight on weather+satellite "
            f"versus {clear_env:.1f}% in clear weather ({shift:+.1f} points)."
            if shift > 1.0
            else f"The gate does NOT shift toward weather+satellite in heavy rain "
            f"({shift:+.1f} points). The central claim is not supported by this "
            f"training run and is reported as such."
        ),
    }


def per_segment_mae(
    predictions: np.ndarray, targets: np.ndarray, labels: np.ndarray
) -> list[dict]:
    """MAE per road segment, split by regime. Feeds the Model Lab's per-road table."""
    absolute = np.abs(predictions - targets)  # (samples, nodes, horizons)
    rows = []
    for node, segment_id in enumerate(SEGMENT_IDS):
        row: dict = {"segment_id": segment_id, "mae_overall": round(float(absolute[:, node].mean()), 3)}
        for index, label in enumerate(LABELS):
            mask = labels == index
            row[f"mae_{label}"] = (
                round(float(absolute[mask, node].mean()), 3) if mask.any() else None
            )
        rows.append(row)
    return rows


def evaluate_everything(
    model: MonsoonPlusNet,
    splits: SplitDatasets,
    device: torch.device,
    dataset_name: str,
    include_lstm: bool = True,
    speeds_for_historical: Optional[np.ndarray] = None,
    minute_of_day: Optional[np.ndarray] = None,
    verbose: bool = True,
    include_masked: bool = True,
) -> dict:
    """Full evaluation of the model plus every baseline, on the TEST split.

    ``include_masked`` adds the inference-time modality ablation rows
    (``traffic_only_masked`` / ``traffic_weather_masked``) when the dataset carries all
    three modalities; it is a no-op for METR-LA, which has no weather or satellite.
    """
    from ml.training.train import build_graph_for

    edge_index, edge_weight, graph_info = build_graph_for(
        splits.test.traffic.shape[1], splits
    )
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    if verbose:
        print(f"\n--- evaluating on {dataset_name} test split ({len(splits.test)} windows) ---")

    predictions, targets, labels, gates, rain = model_predictions(
        model, splits.test, edge_index, edge_weight, device
    )
    model_breakdown = error_breakdown(predictions, targets, labels)
    if verbose:
        print(f"  monsoonplus      {model_breakdown.headline()}")

    baselines: dict[str, ErrorBreakdown] = {}

    persistence_pred, persistence_target, persistence_labels = persistence_predictions(splits.test)
    baselines["persistence"] = error_breakdown(
        persistence_pred, persistence_target, persistence_labels
    )
    if verbose:
        print(f"  persistence      {baselines['persistence'].headline()}")

    if speeds_for_historical is not None and minute_of_day is not None:
        from ml.training.baselines import historical_average_predictions

        train_end = int(speeds_for_historical.shape[0] * 0.7)
        hist_pred, hist_target, hist_labels = historical_average_predictions(
            splits.test, speeds_for_historical, minute_of_day, train_end
        )
        baselines["historical_average"] = error_breakdown(hist_pred, hist_target, hist_labels)
        if verbose:
            print(f"  historical avg   {baselines['historical_average'].headline()}")

    if include_lstm:
        if verbose:
            print("  training LSTM baseline...")
        lstm = train_lstm_baseline(
            splits.train, splits.val, splits.normalisation, device=device, verbose=verbose
        )
        lstm_pred, lstm_target, lstm_labels = predict_lstm(lstm, splits.test, device=device)
        baselines["lstm_no_graph"] = error_breakdown(lstm_pred, lstm_target, lstm_labels)
        if verbose:
            print(f"  lstm (no graph)  {baselines['lstm_no_graph'].headline()}")

    comparisons = compare_to_baselines(model_breakdown, baselines)

    # Ablated copies of the model itself are reported next to the baselines but kept
    # out of the honesty notes: "does the model beat a worse version of itself" is the
    # ablation question, not the does-it-beat-a-naive-guess question the notes police.
    ablation_comparisons: list[Comparison] = []
    if include_masked and bool(np.all(splits.test.availability > 0)):
        masked = masked_baselines(model, splits.test, edge_index, edge_weight, device)
        ablation_comparisons = compare_to_baselines(model_breakdown, masked)
        if verbose:
            for name, breakdown in masked.items():
                print(f"  {name:<16} {breakdown.headline()}")

    gate_summary = gate_by_regime(gates, labels)
    notes = honesty_notes(model_breakdown, comparisons)

    if verbose:
        print("\n  gate weights by regime (the model's own numbers):")
        for label, weights in gate_summary.items():
            print(f"    {label:<11} {weights}")
        claim = gate_shift_claim(gate_summary)
        print(f"\n  central claim supported: {claim['supported']}")
        print(f"    {claim['reason']}")
        if notes:
            print("\n  honesty notes:")
            for note in notes:
                print(f"    - {note}")

    return {
        "dataset": dataset_name,
        "dataset_label": splits.label,
        "dataset_is_real": splits.is_real,
        "dataset_source": splits.source,
        "test_windows": len(splits.test),
        "horizons_minutes": HORIZONS_MINUTES,
        "graph": graph_info,
        "model": model_breakdown.to_dict(),
        "baselines": [c.to_dict() for c in comparisons + ablation_comparisons],
        "gate_by_regime": gate_summary,
        "gate_shift_claim": gate_shift_claim(gate_summary),
        "per_segment": per_segment_mae(predictions, targets, labels),
        "honesty_notes": notes,
    }


def save_report(report: dict, name: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}.json"
    path.write_text(_json_safe_dumps(report), encoding="utf-8")
    return path


def _json_safe_dumps(obj) -> str:
    """json.dump writes bare NaN, which is not valid JSON and breaks JSON.parse().

    Every number is funnelled through here before it reaches a file, so no exported
    artefact can contain NaN or Infinity.
    """

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else round(float(value), 6)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        return value

    return json.dumps(clean(obj), indent=2)


def main() -> None:
    import argparse

    from ml.data.windowing import build_chennai_datasets, build_metr_la_datasets
    from ml.models.monsoonplus_net import load_checkpoint
    from ml.training.train import CHECKPOINT_DIR, resolve_device

    parser = argparse.ArgumentParser(description="Evaluate MonsoonPlusNet honestly")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "monsoonplus_final.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--metr-la-sensors", type=int, default=64)
    parser.add_argument("--skip-metr-la", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, map_location=str(device))
    model.to(device)
    print(f"loaded {args.checkpoint}")
    print(f"  meta: {meta}")

    reports = {}

    chennai = build_chennai_datasets(days=args.days)
    from ml.data.chennai_synthetic import generate_chennai_window

    window = generate_chennai_window(days=args.days)
    model.set_normalisation(chennai.normalisation)
    reports["synthetic_chennai"] = evaluate_everything(
        model,
        chennai,
        device,
        "Synthetic Chennai window",
        include_lstm=not args.no_lstm,
        speeds_for_historical=window.speeds,
        minute_of_day=window.minute_of_day,
    )
    print(f"\nsaved {save_report(reports['synthetic_chennai'], 'eval_synthetic_chennai')}")

    if not args.skip_metr_la:
        metr = build_metr_la_datasets(max_sensors=args.metr_la_sensors)
        model.set_normalisation(metr.normalisation)
        reports["metr_la"] = evaluate_everything(
            model, metr, device, "METR-LA", include_lstm=not args.no_lstm
        )
        print(f"\nsaved {save_report(reports['metr_la'], 'eval_metr_la')}")


if __name__ == "__main__":
    main()
