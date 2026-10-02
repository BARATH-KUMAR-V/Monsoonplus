"""Produce the ``predictions_*.json`` files the frontend reads in Dataset mode.

Run after training:

    python -m ml.export.export_predictions

Writes into ``frontend/public/data/``:

    predictions_metr_la.json           real METR-LA benchmark numbers
    predictions_synthetic_chennai.json the Chennai forecast grid + eval + gate weights
    predictions_live_template.json     schema example only, all values null
    model_card.json                    what was trained on what, and the caveats

Refresh only the modality-ablation blocks (no retraining, no LSTM re-fit, so every other
number in the committed JSON is left exactly as it was):

    python -m ml.export.export_predictions --refresh-ablation

Hard rules enforced here
------------------------
* **No NaN.** ``json.dump`` happily writes bare ``NaN``, which is not valid JSON and
  makes ``JSON.parse`` throw in the browser. Every number goes through ``_clean``,
  which turns non-finite values into ``null``.
* **Provenance travels with the data.** Every file carries ``is_real``, ``source`` and
  a human-readable ``label``, and the UI renders those rather than assuming.
* **Deterministic.** Dataset mode must produce the same screen on every load, so the
  forecast grid is computed once here, not re-derived with any time dependency.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from config.segments import (
    ATTRIBUTIONS,
    HORIZONS_MINUTES,
    JUNCTIONS,
    REPO_ROOT,
    SEGMENTS,
    SEGMENT_IDS,
    THRESHOLDS,
    segment_lengths_km,
)
from ml.data.chennai_synthetic import (
    generate_chennai_window,
    satellite_features,
    traffic_features,
    weather_features,
)
from ml.data.windowing import LABELS, build_chennai_datasets, build_metr_la_datasets
from ml.models.monsoonplus_net import MonsoonPlusNet, load_checkpoint
from ml.training.evaluate import (
    compare_to_baselines,
    evaluate_everything,
    masked_baselines,
    save_report,
)
from ml.training.linear_probe import attach_to_evaluation as attach_linear_probe
from ml.training.linear_probe import run_probe
from ml.training.losses import error_breakdown
from ml.training.train import CHECKPOINT_DIR, build_graph_for, resolve_device

OUTPUT_DIR = REPO_ROOT / "frontend" / "public" / "data"

# The forecast grid the UI scrubs through: rain levels x horizons, per segment.
RAIN_LEVELS = [0, 5, 10, 20, 30, 40, 55, 70, 85, 100]
SLIDER_MINUTES = [0, 15, 30, 45, 60]

# Retrained-network modality ablation written by `python -m ml.training.ablations`.
ABLATION_REPORT = REPO_ROOT / "ml" / "reports" / "ablations.json"
NETWORK_ABLATION_VARIANTS = ("traffic_only", "traffic_weather", "all_three")


def _clean(value: Any) -> Any:
    """Recursively make a structure JSON-safe. NaN/Inf -> None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else round(number, 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _clean(payload)
    text = json.dumps(cleaned, indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    # Parse it straight back: a file the browser cannot read is worse than no file.
    json.loads(text)
    return path


def network_payload() -> dict:
    """The network, echoed into the export so the frontend has one fetch for everything."""
    lengths = segment_lengths_km()
    return {
        "junctions": [
            {
                "id": j.id,
                "name": j.name,
                "lat": j.lat,
                "lon": j.lon,
                "elevation_m": j.elevation_m,
                "note": j.note,
            }
            for j in JUNCTIONS
        ],
        "segments": [
            {
                "id": s.id,
                "name": s.name,
                "a": s.a,
                "b": s.b,
                "free_flow_kmh": s.free_flow_kmh,
                "rain_sensitivity": s.rain_sensitivity,
                "base_congestion": s.base_congestion,
                "rain_bias": s.rain_bias,
                "lanes": s.lanes,
                "flood_history": s.flood_history,
                "note": s.note,
                "length_km": round(length, 3),
            }
            for s, length in zip(SEGMENTS, lengths)
        ],
    }


@torch.no_grad()
def network_ablation_payload() -> Optional[dict]:
    """The retrained-network modality ablation, trimmed for the Model Lab.

    Reads ``ml/reports/ablations.json`` (``python -m ml.training.ablations``). Returns
    None when that report has not been produced, so the UI simply omits the section
    rather than showing placeholders.
    """
    if not ABLATION_REPORT.exists():
        return None
    report = json.loads(ABLATION_REPORT.read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in report.get("rows", [])}
    if not all(name in rows for name in NETWORK_ABLATION_VARIANTS):
        return None
    keep = (
        "name",
        "variant",
        "mae_overall",
        "mae_clear",
        "mae_rain",
        "mae_heavy_rain",
        "rain_weighted_mae",
        "gate_by_regime",
    )
    return {
        "source": "ml/reports/ablations.json",
        "method": (
            "Each variant is a fresh network trained from scratch on identical data, seed "
            f"and epoch budget ({report.get('epochs_per_variant')} epochs, "
            f"{report.get('days')}-day synthetic window, seed {report.get('seed')}), with the "
            "named modalities masked out of training and evaluation. A small-budget "
            "comparison that stands on its own: its numbers are NOT the final checkpoint's."
        ),
        "epochs_per_variant": report.get("epochs_per_variant"),
        "days": report.get("days"),
        "seed": report.get("seed"),
        "rows": [{key: rows[name][key] for key in keep if key in rows[name]}
                 for name in NETWORK_ABLATION_VARIANTS],
    }


def attach_ablation_blocks(evaluation: dict, days: int, seed: int = 2024) -> dict:
    """Add the linear-probe rows and the retrained-network ablation to an evaluation.

    Idempotent, so it is safe in both the full export and ``--refresh-ablation``.
    """
    attach_linear_probe(evaluation, run_probe(days=days, seed=seed))
    network = network_ablation_payload()
    if network is not None:
        evaluation["network_ablation"] = network
    else:
        evaluation.pop("network_ablation", None)
    return evaluation


def refresh_ablation(args: argparse.Namespace, device: torch.device, model) -> None:
    """Patch ONLY the ablation blocks into the committed JSON, from the checkpoint.

    Re-scores the trained network on the held-out test windows (to build the
    ``*_masked`` rows) and re-fits the linear probe, then merges both into the existing
    ``predictions_synthetic_chennai.json`` / ``model_card.json``. Nothing is retrained
    and the LSTM baseline is not re-fitted, so every other number stays byte-identical.
    Refuses to run if the checkpoint does not reproduce the file's recorded model
    metrics -- a mismatch means this checkpoint is not the one the JSON was built from.
    """
    from ml.training.train import evaluate as run_eval

    path = OUTPUT_DIR / "predictions_synthetic_chennai.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]

    chennai = build_chennai_datasets(days=args.days)
    model.set_normalisation(chennai.normalisation)
    edge_index, edge_weight, _ = build_graph_for(chennai.test.traffic.shape[1], chennai)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    predictions, targets, labels, _, _ = run_eval(
        model, chennai.test, edge_index, edge_weight, device, collect_gate=True
    )
    model_now = error_breakdown(predictions, targets, labels).to_dict()
    recorded = evaluation["model"]
    for key in ("mae_overall", "rain_weighted_mae"):
        if abs(model_now[key] - recorded[key]) > 2e-3:
            raise SystemExit(
                f"checkpoint does not reproduce the file's model metrics ({key}: "
                f"{model_now[key]} now vs {recorded[key]} recorded). Run the full export "
                "instead of --refresh-ablation."
            )

    masked = masked_baselines(model, chennai.test, edge_index, edge_weight, device)
    masked_rows = [c.to_dict() for c in compare_to_baselines(error_breakdown(
        predictions, targets, labels), masked)]
    masked_names = {row["name"] for row in masked_rows}
    evaluation["baselines"] = [
        b for b in evaluation["baselines"] if b["name"] not in masked_names
    ] + masked_rows

    attach_ablation_blocks(evaluation, days=args.days)
    print(f"wrote {write_json(path, payload)}")

    card_path = OUTPUT_DIR / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    apply_ablation_to_model_card(card, evaluation)
    print(f"wrote {write_json(card_path, card)}")
    print(f"wrote {save_report(evaluation, 'eval_synthetic_chennai')}")


def apply_ablation_to_model_card(card: dict, evaluation: dict) -> dict:
    """Describe the three modality studies in the model card, once (idempotent)."""
    card["ablation_methodology"] = (
        "Modality importance is checked three independent ways, all on the synthetic "
        "Chennai window. (1) Retrained: a fresh network per variant, trained without the "
        "named modalities (ml/training/ablations.py). (2) Switched off: the final trained "
        "network re-scored with weather and/or satellite masked at inference "
        "(traffic_only_masked / traffic_weather_masked). (3) Linear probe: a pooled ridge "
        "regression per variant on the same test windows "
        "(traffic_only_linear / traffic_weather_linear / traffic_weather_satellite_linear; "
        "python ml/reference/ablation_numpy.py)."
    )
    marker = "The modality-ablation results"
    limitations = [x for x in card.get("known_limitations", []) if not x.startswith(marker)]
    limitations.insert(
        -1,
        f"{marker} are three different questions, not one controlled study: the retrained "
        "variants use a small budget and are not the final checkpoint; the masked variants "
        "measure what the finished network leans on, not what a smaller network could "
        "achieve; the linear probe is a deliberately weaker model family. All three use "
        "the synthetic window.",
    )
    card["known_limitations"] = limitations
    return card


def forecast_grid(
    model: MonsoonPlusNet,
    device: torch.device,
    window,
    normalisation,
) -> dict:
    """Run the model across a rain sweep, so the UI can scrub rain and time offline.

    For each rain level we build a synthetic 2-hour history at that rain level and ask
    the model for t+15/30/60. This is a **model estimate**, labelled as such wherever
    the frontend shows it -- it is not a measurement and not a forecast of a real day.
    """
    model.eval()
    model.set_normalisation(normalisation)

    edge_index, edge_weight, graph_info = build_graph_for(len(SEGMENTS),
        type("S", (), {"train": type("D", (), {"traffic": np.zeros((1, len(SEGMENTS), 3))})()})())
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    history_steps = model.config.__dict__.get("history_steps", 24)
    from config.segments import HISTORY_STEPS

    history_steps = HISTORY_STEPS

    free_flow = np.array([s.free_flow_kmh for s in SEGMENTS], dtype=np.float32)
    sensitivity = np.array([s.rain_sensitivity for s in SEGMENTS], dtype=np.float32)
    congestion = np.array([s.base_congestion for s in SEGMENTS], dtype=np.float32)
    rain_bias = np.array([s.rain_bias for s in SEGMENTS], dtype=np.float32)

    from ml.data.ndwi import synthetic_ndwi

    rows = []
    for rain_level in RAIN_LEVELS:
        per_segment_rain = np.clip(rain_level * rain_bias, 0, 130).astype(np.float32)

        # Build a plausible steady-state history at this rain level.
        wetness = np.zeros(len(SEGMENTS), dtype=np.float32)
        snapshot = synthetic_ndwi(per_segment_rain)
        for _ in range(history_steps):
            snapshot = synthetic_ndwi(per_segment_rain, previous_wetness=snapshot.wetness)
        wetness = snapshot.wetness

        speed_now = np.clip(
            free_flow
            - free_flow * congestion
            - free_flow * sensitivity * np.tanh(per_segment_rain / 45.0) * 0.62
            - free_flow * sensitivity * wetness * 0.45,
            3.0,
            None,
        ).astype(np.float32)

        traffic = np.stack(
            [
                np.tile(speed_now, (history_steps, 1)),
                np.tile(free_flow, (history_steps, 1)),
                np.ones((history_steps, len(SEGMENTS)), dtype=np.float32),
            ],
            axis=-1,
        )
        weather = np.stack(
            [
                np.tile(per_segment_rain, (history_steps, 1)),
                np.full((history_steps, len(SEGMENTS)), 27.5, dtype=np.float32),
                np.tile(9.0 + 0.22 * per_segment_rain, (history_steps, 1)),
            ],
            axis=-1,
        )
        satellite = np.stack([snapshot.ndwi, wetness], axis=-1)

        out = model(
            torch.from_numpy(traffic.transpose(1, 0, 2)).unsqueeze(0).to(device),
            torch.from_numpy(weather.transpose(1, 0, 2)).unsqueeze(0).to(device),
            torch.from_numpy(satellite).unsqueeze(0).to(device),
            edge_index,
            edge_weight,
            availability=torch.tensor([[1.0, 1.0, 1.0]], device=device),
        )
        prediction = out.prediction[0].cpu().numpy()  # (N, H)
        gate = out.gate[0].cpu().numpy()  # (N, 3)

        rows.append(
            {
                "rain_mm_h": rain_level,
                "per_segment": [
                    {
                        "segment_id": SEGMENT_IDS[node],
                        "speed_now_kmh": round(float(speed_now[node]), 2),
                        "predicted_kmh": {
                            f"t+{h}": round(float(prediction[node, i]), 2)
                            for i, h in enumerate(HORIZONS_MINUTES)
                        },
                        "rain_mm_h": round(float(per_segment_rain[node]), 2),
                        "ndwi": round(float(snapshot.ndwi[node]), 4),
                        "wetness": round(float(wetness[node]), 4),
                        "gate": {
                            "traffic": round(float(gate[node, 0]) * 100, 1),
                            "weather": round(float(gate[node, 1]) * 100, 1),
                            "satellite": round(float(gate[node, 2]) * 100, 1),
                        },
                    }
                    for node in range(len(SEGMENTS))
                ],
            }
        )

    return {
        "rain_levels": RAIN_LEVELS,
        "slider_minutes": SLIDER_MINUTES,
        "horizons_minutes": HORIZONS_MINUTES,
        "graph": graph_info,
        "rows": rows,
        "note": (
            "Model estimates produced by sweeping rainfall through the trained network. "
            "Not a forecast of any real day and not a measurement."
        ),
    }


def confidence_from_errors(report: dict) -> dict:
    """Turn validation error statistics into the UI's reliability badge numbers.

    The badge says "95% of the time this ETA is within +-X min". X comes from the test
    MAE per regime scaled to a 95th-percentile-style band (2x MAE is a reasonable
    normal-ish approximation), not from a number someone liked the look of.
    """
    model = report["model"]
    by_label = model.get("mae_by_label", {})

    def band(label: str) -> Optional[float]:
        mae = by_label.get(label)
        if mae is None:
            return None
        return round(float(mae) * 2.0, 2)

    return {
        "basis": "test-split MAE per weather regime, scaled 2x for a ~95% band",
        "speed_band_kmh": {label: band(label) for label in LABELS},
        "mae_by_label_kmh": by_label,
        "mae_by_horizon_kmh": model.get("mae_by_horizon", {}),
        "confidence_label_by_regime": {
            "clear": "High",
            "rain": "Medium",
            "heavy_rain": "Low",
        },
        "note": (
            "Derived from the evaluation run in ml/training/evaluate.py. If the test "
            "split had no samples for a regime, its band is null and the UI hides it."
        ),
    }


def historical_baseline_payload(window, train_fraction: float = 0.7) -> dict:
    """Per-segment, per-time-of-day average speed -- Page 3's "what is normal here?".

    Fitted on the training portion of the synthetic window only, same as the baseline
    used in evaluation, so the two cannot disagree.
    """
    from ml.training.baselines import HistoricalAverage

    train_end = int(window.speeds.shape[0] * train_fraction)
    model = HistoricalAverage().fit(
        window.speeds[:train_end], window.minute_of_day[:train_end]
    )
    # Downsample to 30-minute buckets: 48 numbers per road is plenty for a chart and
    # keeps the committed JSON small.
    table = model.table  # (288, N)
    buckets = table.reshape(48, 6, table.shape[1]).mean(axis=1)

    return {
        "bucket_minutes": 30,
        "buckets": 48,
        "label": "Synthetic Chennai window (training portion)",
        "is_real": False,
        "per_segment": {
            SEGMENT_IDS[node]: [round(float(v), 2) for v in buckets[:, node]]
            for node in range(buckets.shape[1])
        },
        "note": (
            "Average speed by time of day, from the synthetic Chennai window's training "
            "split. Anomalies on the Live Map are measured against this."
        ),
    }


def storm_replay_payload(model: MonsoonPlusNet, device: torch.device, grid: dict) -> dict:
    """The scripted 4-7 PM storm: 13 frames at 15-minute steps.

    Frames are interpolated from the deterministic forecast grid, so the replay and the
    rest of Dataset mode cannot disagree about what 65 mm/h looks like.
    """
    storm_profile = [0, 4, 12, 30, 55, 78, 92, 84, 62, 38, 18, 6, 0]
    by_rain = {row["rain_mm_h"]: row for row in grid["rows"]}
    levels = sorted(by_rain)

    def interpolate(rain: float, segment_index: int, field: str) -> float:
        lower = max([l for l in levels if l <= rain], default=levels[0])
        upper = min([l for l in levels if l >= rain], default=levels[-1])
        low_value = by_rain[lower]["per_segment"][segment_index][field]
        high_value = by_rain[upper]["per_segment"][segment_index][field]
        if upper == lower:
            return float(low_value)
        ratio = (rain - lower) / (upper - lower)
        return float(low_value) + ratio * (float(high_value) - float(low_value))

    frames = []
    for index, rain in enumerate(storm_profile):
        minutes_from_start = index * 15
        clock = 16 * 60 + minutes_from_start
        per_segment = []
        for node in range(len(SEGMENTS)):
            speed = interpolate(rain, node, "speed_now_kmh")
            per_segment.append(
                {
                    "segment_id": SEGMENT_IDS[node],
                    "speed_kmh": round(speed, 2),
                    "rain_mm_h": round(rain * SEGMENTS[node].rain_bias, 2),
                    "wetness": round(interpolate(rain, node, "wetness"), 4),
                }
            )
        frames.append(
            {
                "frame": index,
                "clock_minutes": clock,
                "clock": f"{clock // 60:02d}:{clock % 60:02d}",
                "city_rain_mm_h": rain,
                "per_segment": per_segment,
                "mean_speed_kmh": round(
                    float(np.mean([p["speed_kmh"] for p in per_segment])), 2
                ),
            }
        )

    return {
        "label": "Simulated storm replay",
        "is_real": False,
        "frames": frames,
        "step_minutes": 15,
        "window": "16:00-19:00",
        "note": (
            "A scripted rainfall profile pushed through the trained model. Simulated, "
            "not a recording of a real storm."
        ),
    }


def live_template() -> dict:
    """Schema example for Live mode. Every value null, so nobody mistakes it for data."""
    return {
        "schema_version": "1.0.0",
        "mode": "live",
        "generated_at": None,
        "is_real": None,
        "sources": {
            "traffic": {"name": "TomTom Flow Segment Data", "source": None, "fetched_at": None, "ok": None},
            "weather": {"name": "Open-Meteo", "source": None, "fetched_at": None, "ok": None},
            "satellite": {"name": "Sentinel-2 NDWI", "source": None, "fetched_at": None, "ok": None},
        },
        "per_segment": [
            {
                "segment_id": segment.id,
                "speed_kmh": None,
                "free_flow_kmh": None,
                "confidence": None,
                "rain_mm_h": None,
                "temp_c": None,
                "wind_kmh": None,
                "ndwi": None,
                "wetness": None,
                "predicted_kmh": {f"t+{h}": None for h in HORIZONS_MINUTES},
                "gate": {"traffic": None, "weather": None, "satellite": None},
            }
            for segment in SEGMENTS
        ],
        "note": (
            "TEMPLATE ONLY -- this file documents the shape the collector writes. It "
            "contains no data. Run collector/collect.py to produce a real live file."
        ),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export frontend JSON from a checkpoint")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "monsoonplus_final.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--metr-la-sensors", type=int, default=64)
    parser.add_argument("--skip-metr-la", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument(
        "--refresh-ablation",
        action="store_true",
        help="only patch the modality-ablation blocks into the committed JSON "
        "(no retraining, nothing else changes)",
    )
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint_path}\n"
            "Run `python -m ml.training.train` first (or `--quick` for a fast one)."
        )

    model, meta = load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    print(f"loaded {checkpoint_path.name}  meta={meta}")

    if args.refresh_ablation:
        refresh_ablation(args, device, model)
        print("\nablation blocks refreshed")
        return

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- Chennai ------------------------------------------------------------
    window = generate_chennai_window(days=args.days)
    chennai = build_chennai_datasets(days=args.days)
    chennai_report = evaluate_everything(
        model,
        chennai,
        device,
        "Synthetic Chennai window",
        include_lstm=not args.no_lstm,
        speeds_for_historical=window.speeds,
        minute_of_day=window.minute_of_day,
    )
    attach_ablation_blocks(chennai_report, days=args.days)
    grid = forecast_grid(model, device, window, chennai.normalisation)

    chennai_payload = {
        "schema_version": "1.0.0",
        "mode": "dataset",
        "dataset": "synthetic_chennai",
        "label": "Synthetic Chennai window",
        "is_real": False,
        "badge": "Benchmark data - deterministic",
        "generated_at": generated_at,
        "checkpoint": checkpoint_path.name,
        "checkpoint_meta": meta,
        "model_config": model.config.to_dict(),
        "network": network_payload(),
        "thresholds": THRESHOLDS,
        "evaluation": chennai_report,
        "forecast_grid": grid,
        "confidence": confidence_from_errors(chennai_report),
        "historical_baseline": historical_baseline_payload(window),
        "storm_replay": storm_replay_payload(model, device, grid),
        "window_summary": window.summary(),
        "attributions": ATTRIBUTIONS,
    }
    print(f"wrote {write_json(OUTPUT_DIR / 'predictions_synthetic_chennai.json', chennai_payload)}")
    # Also write the standalone report the README, the notebook and the Forecast page
    # cite by path. Same numbers, no recomputation.
    print(f"wrote {save_report(chennai_report, 'eval_synthetic_chennai')}")

    # ---- METR-LA ------------------------------------------------------------
    if not args.skip_metr_la:
        metr = build_metr_la_datasets(max_sensors=args.metr_la_sensors)
        model.set_normalisation(metr.normalisation)
        metr_report = evaluate_everything(
            model, metr, device, "METR-LA", include_lstm=not args.no_lstm
        )
        metr_payload = {
            "schema_version": "1.0.0",
            "mode": "dataset",
            "dataset": "metr_la",
            "label": metr.label,
            "is_real": metr.is_real,
            "badge": "Benchmark data - deterministic",
            "generated_at": generated_at,
            "checkpoint": checkpoint_path.name,
            "sensors_used": args.metr_la_sensors,
            "source": metr.source,
            "evaluation": metr_report,
            "note": (
                "METR-LA is 207 loop detectors on Los Angeles freeways, 5-minute speeds. "
                "Used for phase-1 pretraining and as the benchmark here. Speeds converted "
                "from mph to km/h. This evaluation uses the first "
                f"{args.metr_la_sensors} sensors so it runs on a free CPU tier."
            ),
            "attributions": ATTRIBUTIONS,
        }
        print(f"wrote {write_json(OUTPUT_DIR / 'predictions_metr_la.json', metr_payload)}")
        print(f"wrote {save_report(metr_report, 'eval_metr_la')}")

    # ---- template + model card ---------------------------------------------
    print(f"wrote {write_json(OUTPUT_DIR / 'predictions_live_template.json', live_template())}")

    model_card = {
        "name": "MonsoonPlusNet",
        "task": "Road speed forecasting at t+15 / t+30 / t+60 minutes for 10 Chennai segments",
        "architecture": (
            "Trimodal gated fusion. GRU traffic encoder + GRU weather encoder + MLP "
            "satellite encoder, mixed by a learned per-node gate, then two rounds of "
            f"{model.config.graph_layer.upper()} message passing over the segment graph, "
            "then a linear head predicting a delta from the last observed speed."
        ),
        "parameters": model.parameter_count(),
        "config": model.config.to_dict(),
        "training_phases": [
            "phase 1 - pretrain traffic branch on real METR-LA (weather/satellite masked absent)",
            "phase 2 - transfer to the synthetic Chennai window, all three modalities",
            "phase 3 - cost-sensitive fine-tune, rain 3x / heavy rain 8x",
            "phase 4 - real collected logs (skipped unless logs/ has data)",
        ],
        "model_selection": "lowest rain-weighted validation MAE, not plain validation MAE",
        "known_limitations": [
            "The Chennai training window is SYNTHETIC, generated from an explicit physical "
            "process. It is not measured Chennai traffic.",
            "The satellite branch trains on synthetic NDWI derived from rainfall "
            "accumulation unless you supply a real Earth Engine export. Where that is the "
            "case, 'trimodal' means three trained branches, one of which learned from a "
            "simulated signal.",
            "Weather is queried at one city point and spread across segments by a "
            "per-segment rain bias. It is not per-road measurement.",
            "METR-LA is Los Angeles freeway data. Pretraining on it transfers temporal "
            "dynamics, not Chennai-specific behaviour.",
            "This is a screening and planning aid, not a safety system. Do not use it to "
            "decide whether a flooded road is passable.",
        ],
        "generated_at": generated_at,
        "attributions": ATTRIBUTIONS,
    }
    apply_ablation_to_model_card(model_card, chennai_report)
    print(f"wrote {write_json(OUTPUT_DIR / 'model_card.json', model_card)}")
    print("\nexport complete")


if __name__ == "__main__":
    main()
