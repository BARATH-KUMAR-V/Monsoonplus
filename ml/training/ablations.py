"""Ablations: GCN vs GAT, and the modality drop.

Two questions this answers, and both answers are reported whichever way they come out:

1. **Does the graph layer choice matter?** Train the identical model with GCNConv and
   with GATConv, same seed, same data, same epochs.

2. **Is the model actually trimodal, or is it a traffic model with decoration?** Train
   traffic-only, traffic+weather, and all three. If traffic-only matches all-three,
   then the extra modalities are not pulling their weight and the project should say so
   rather than claim a trimodal benefit.

The modality drop is implemented with the availability mask, not by zeroing inputs.
Zeroing would still let the encoder and gate consume a (constant) signal; masking makes
the gate renormalise over the remaining modalities, which is a genuine ablation.

    python -m ml.training.ablations --epochs 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config.segments import REPO_ROOT
from ml.data.windowing import build_chennai_datasets
from ml.models.monsoonplus_net import build_model
from ml.training.evaluate import _json_safe_dumps, gate_by_regime
from ml.training.losses import error_breakdown
from ml.training.train import build_graph_for, evaluate, resolve_device, run_phase

REPORT_DIR = REPO_ROOT / "ml" / "reports"

MODALITY_SETTINGS = {
    "traffic_only": (1, 0, 0),
    "traffic_weather": (1, 1, 0),
    "all_three": (1, 1, 1),
}


@dataclass
class AblationRow:
    name: str
    variant: str
    mae_overall: float
    mae_clear: float
    mae_rain: float
    mae_heavy_rain: float
    rain_weighted_mae: float
    gate_by_regime: dict
    epochs_run: int
    seconds: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "mae_overall": round(self.mae_overall, 3),
            "mae_clear": round(self.mae_clear, 3),
            "mae_rain": round(self.mae_rain, 3),
            "mae_heavy_rain": round(self.mae_heavy_rain, 3),
            "rain_weighted_mae": round(self.rain_weighted_mae, 3),
            "gate_by_regime": self.gate_by_regime,
            "epochs_run": self.epochs_run,
            "seconds": self.seconds,
        }


def _override_availability(splits, availability: tuple[int, int, int]) -> None:
    """Point every split in this SplitDatasets at a new availability mask."""
    for dataset in (splits.train, splits.val, splits.test):
        dataset.availability = np.asarray(availability, dtype=np.float32)


def run_one(
    name: str,
    variant: str,
    graph_layer: str,
    availability: tuple[int, int, int],
    days: int,
    epochs: int,
    device: torch.device,
    seed: int,
    verbose: bool,
) -> AblationRow:
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = build_chennai_datasets(days=days, seed=seed)
    _override_availability(splits, availability)

    model = build_model(graph_layer=graph_layer)
    result, _ = run_phase(
        f"ablation_{name}",
        model,
        splits,
        device,
        epochs=epochs,
        weighted=True,
        verbose=verbose,
    )

    edge_index, edge_weight, _ = build_graph_for(splits.test.traffic.shape[1], splits)
    predictions, targets, labels, gates, _ = evaluate(
        model,
        splits.test,
        edge_index.to(device),
        edge_weight.to(device),
        device,
        collect_gate=True,
    )
    breakdown = error_breakdown(predictions, targets, labels)

    return AblationRow(
        name=name,
        variant=variant,
        mae_overall=breakdown.mae_overall,
        mae_clear=breakdown.mae_by_label.get("clear", float("nan")),
        mae_rain=breakdown.mae_by_label.get("rain", float("nan")),
        mae_heavy_rain=breakdown.mae_by_label.get("heavy_rain", float("nan")),
        rain_weighted_mae=breakdown.rain_weighted_mae,
        gate_by_regime=gate_by_regime(gates, labels),
        epochs_run=result.epochs_run,
        seconds=result.seconds,
    )


def interpret(rows: list[AblationRow]) -> list[str]:
    """Turn the table into honest sentences, including unflattering ones."""
    notes: list[str] = []
    by_name = {row.name: row for row in rows}

    gcn, gat = by_name.get("gcn"), by_name.get("gat")
    if gcn and gat:
        delta = gat.rain_weighted_mae - gcn.rain_weighted_mae
        if abs(delta) < 0.02:
            notes.append(
                f"GCN and GAT are effectively tied on rain-weighted MAE "
                f"({gcn.rain_weighted_mae:.3f} vs {gat.rain_weighted_mae:.3f}). On a "
                f"10-node graph attention has little room to help; GCN stays the default "
                f"because it is cheaper."
            )
        elif delta > 0:
            notes.append(
                f"GCN beats GAT on rain-weighted MAE ({gcn.rain_weighted_mae:.3f} vs "
                f"{gat.rain_weighted_mae:.3f}), so GCN remains the default."
            )
        else:
            notes.append(
                f"GAT beats GCN on rain-weighted MAE ({gat.rain_weighted_mae:.3f} vs "
                f"{gcn.rain_weighted_mae:.3f}). GCN is still the documented default, but "
                f"this run favours GAT -- switch with --graph-layer gat."
            )

    traffic_only = by_name.get("traffic_only")
    all_three = by_name.get("all_three")
    if traffic_only and all_three:
        overall_gain = traffic_only.mae_overall - all_three.mae_overall
        heavy_gain = traffic_only.mae_heavy_rain - all_three.mae_heavy_rain
        if heavy_gain > 0.05:
            notes.append(
                f"Adding weather and satellite cuts heavy-rain MAE by "
                f"{heavy_gain:.3f} km/h ({traffic_only.mae_heavy_rain:.3f} -> "
                f"{all_three.mae_heavy_rain:.3f}), which is the trimodal claim holding up "
                f"where it matters."
            )
        else:
            notes.append(
                f"Adding weather and satellite does NOT meaningfully improve heavy-rain "
                f"MAE ({traffic_only.mae_heavy_rain:.3f} -> {all_three.mae_heavy_rain:.3f}). "
                f"On this run the trimodal claim is not supported by the ablation, and the "
                f"Model Lab reports that."
            )
        if overall_gain <= 0:
            notes.append(
                f"Overall MAE is no better with all three modalities "
                f"({traffic_only.mae_overall:.3f} -> {all_three.mae_overall:.3f}); the "
                f"benefit, where it exists, is concentrated in rain."
            )

    weather_only = by_name.get("traffic_weather")
    if weather_only and all_three:
        satellite_gain = weather_only.mae_heavy_rain - all_three.mae_heavy_rain
        if satellite_gain > 0.03:
            notes.append(
                f"The satellite channel alone is worth {satellite_gain:.3f} km/h of "
                f"heavy-rain MAE on top of traffic+weather."
            )
        else:
            notes.append(
                f"The satellite channel adds little beyond traffic+weather in heavy rain "
                f"({satellite_gain:+.3f} km/h). Note the satellite signal here is synthetic "
                f"NDWI unless a GEE export was supplied, which limits what this measures."
            )
    return notes


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description="Run monsoonplus ablations")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    verbose = not args.quiet
    rows: list[AblationRow] = []

    print("=== ablation 1: graph layer (GCN vs GAT) ===")
    for layer in ("gcn", "gat"):
        rows.append(
            run_one(
                layer,
                f"graph_layer={layer}",
                layer,
                (1, 1, 1),
                args.days,
                args.epochs,
                device,
                args.seed,
                verbose,
            )
        )

    print("\n=== ablation 2: modality drop ===")
    for name, availability in MODALITY_SETTINGS.items():
        if name == "all_three":
            continue  # the GCN row above already is all_three
        rows.append(
            run_one(
                name,
                f"modalities={name}",
                "gcn",
                availability,
                args.days,
                args.epochs,
                device,
                args.seed,
                verbose,
            )
        )
    # Reuse the GCN all-three run so the comparison shares a seed and epoch budget.
    all_three = AblationRow(**{**rows[0].__dict__, "name": "all_three", "variant": "modalities=all_three"})
    rows.append(all_three)

    notes = interpret(rows)

    print("\n--- ablation results (test split, rain-weighted MAE drives the verdict) ---")
    header = f"{'variant':<28}{'MAE':>7}{'clear':>8}{'rain':>8}{'heavy':>8}{'rain-wtd':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.variant:<28}{row.mae_overall:>7.3f}{row.mae_clear:>8.3f}"
            f"{row.mae_rain:>8.3f}{row.mae_heavy_rain:>8.3f}{row.rain_weighted_mae:>10.3f}"
        )

    print("\n--- interpretation ---")
    for note in notes:
        print(f"  - {note}")

    report = {
        "epochs_per_variant": args.epochs,
        "days": args.days,
        "seed": args.seed,
        "rows": [row.to_dict() for row in rows],
        "interpretation": notes,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "ablations.json"
    path.write_text(_json_safe_dumps(report), encoding="utf-8")
    print(f"\nsaved {path}")
    return report


if __name__ == "__main__":
    main()
