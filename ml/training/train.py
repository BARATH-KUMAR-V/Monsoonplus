"""Four-phase training for MonsoonPlusNet.

    phase 1  pretrain on METR-LA        traffic branch only (weather/satellite masked)
    phase 2  transfer to Chennai        all three modalities, unweighted loss
    phase 3  cost-sensitive fine-tune   rain 3x, heavy rain 8x
    phase 4  real collected logs        only if logs/ has data; skipped loudly otherwise

Every phase keeps the checkpoint with the lowest **rain-weighted** validation MAE, not
the lowest plain validation MAE. See ml/training/losses.py for why that distinction
decides whether the tool is useful in a storm.

Usage
-----
    python -m ml.training.train                      # all phases, CPU-friendly defaults
    python -m ml.training.train --graph-layer gat    # the documented one-line swap
    python -m ml.training.train --quick              # tiny run, for smoke-testing
    python -m ml.training.train --skip-pretrain      # straight to Chennai
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config.segments import HORIZONS_MINUTES, REPO_ROOT
from ml.data.graph import (
    assert_graph_contributes,
    chennai_adjacency,
    correlation_adjacency,
    dense_to_edge_index,
    gaussian_kernel_adjacency,
    graph_stats,
    metr_la_distance_matrix,
)
from ml.data.windowing import (
    SplitDatasets,
    build_chennai_datasets,
    build_metr_la_datasets,
    make_loader,
)
from ml.models.monsoonplus_net import MonsoonPlusNet, build_model
from ml.training.losses import CostSensitiveLoss, error_breakdown

CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints"
REPORT_DIR = REPO_ROOT / "ml" / "reports"


@dataclass
class PhaseResult:
    name: str
    epochs_run: int
    best_rain_weighted_val_mae: float
    best_epoch: int
    seconds: float
    dataset_label: str
    dataset_is_real: bool
    skipped: bool = False
    skip_reason: str = ""
    history: list[dict] = field(default_factory=list)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_graph_for(
    num_nodes: int, splits: Optional[SplitDatasets] = None
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Chennai uses its road graph; METR-LA uses the published sensor distances.

    Falls back to a correlation graph for METR-LA when the distance table is missing,
    and says so in the returned stats rather than pretending it is the road graph.
    """
    from ml.data.metr_la import DISTANCES_PATH, load_metr_la

    if num_nodes == len(chennai_adjacency()):
        weights = chennai_adjacency()
        source = "Chennai road graph (shared junctions, Gaussian kernel)"
    else:
        data = load_metr_la(quiet=True)
        distances = (
            metr_la_distance_matrix(data.sensor_ids, DISTANCES_PATH)
            if data.sensor_ids
            else None
        )
        if distances is not None:
            weights = gaussian_kernel_adjacency(distances, sigma_mode="std")
            source = "METR-LA published sensor distances (Gaussian kernel, sigma=std)"
        else:
            speeds = data.speeds[:, :num_nodes]
            weights = correlation_adjacency(speeds)
            source = "METR-LA correlation fallback (distance table unavailable)"
        if weights.shape[0] != num_nodes:
            weights = weights[:num_nodes, :num_nodes]

    assert_graph_contributes(weights)
    stats = graph_stats(weights)
    stats["source"] = source
    edge_index, edge_weight = dense_to_edge_index(weights)
    return edge_index, edge_weight, stats


@torch.no_grad()
def evaluate(
    model: MonsoonPlusNet,
    dataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
    collect_gate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run the model over a dataset. Returns predictions, targets, labels, gates, rain."""
    model.eval()
    predictions, targets, labels, gates, rains = [], [], [], [], []

    for batch in make_loader(dataset, batch_size=batch_size):
        out = model(
            batch["traffic"].to(device),
            batch["weather"].to(device),
            batch["satellite"].to(device),
            edge_index,
            edge_weight,
            availability=batch["availability"].to(device),
        )
        predictions.append(out.prediction.cpu().numpy())
        targets.append(batch["target"].numpy())
        labels.append(batch["label_index"].numpy())
        rains.append(batch["rain_mm_h"].numpy())
        if collect_gate:
            gates.append(out.gate.cpu().numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(labels),
        np.concatenate(gates) if collect_gate else None,
        np.concatenate(rains),
    )


def run_phase(
    name: str,
    model: MonsoonPlusNet,
    splits: SplitDatasets,
    device: torch.device,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weighted: bool = True,
    patience: int = 6,
    freeze: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[PhaseResult, dict]:
    """Train one phase; return its result and the best state_dict found."""
    started = time.time()
    num_nodes = splits.train.traffic.shape[1]
    edge_index, edge_weight, graph_info = build_graph_for(num_nodes, splits)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model.set_normalisation(splits.normalisation)
    model.to(device)

    # Freezing lets phase 3 refine the gate and head without disturbing a traffic
    # encoder that already learned temporal dynamics from 34k steps of METR-LA.
    frozen_names: list[str] = []
    for module_name, module in model.named_children():
        requires_grad = not (freeze and module_name in freeze)
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad
        if not requires_grad:
            frozen_names.append(module_name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )
    criterion = CostSensitiveLoss()
    train_loader = make_loader(splits.train, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  data   {splits.label}  ({'real' if splits.is_real else 'synthetic'})")
        print(f"  graph  {graph_info['source']}")
        print(
            f"         {graph_info['nodes']} nodes, {graph_info['edges']} edges, "
            f"mean edge weight {graph_info['mean_edge_weight']:.3f}"
        )
        print(f"  loss   {'rain-weighted (3x/8x)' if weighted else 'unweighted'}")
        if frozen_names:
            print(f"  frozen {', '.join(frozen_names)}")

    best_score = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimiser.zero_grad()
            out = model(
                batch["traffic"].to(device),
                batch["weather"].to(device),
                batch["satellite"].to(device),
                edge_index,
                edge_weight,
                availability=batch["availability"].to(device),
            )
            loss = criterion(
                out.prediction,
                batch["target"].to(device),
                batch["weight"].to(device) if weighted else None,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimiser.step()
            running_loss += float(loss.detach())
            batches += 1

        predictions, targets, labels, _, _ = evaluate(
            model, splits.val, edge_index, edge_weight, device
        )
        breakdown = error_breakdown(predictions, targets, labels)
        score = breakdown.rain_weighted_mae
        scheduler.step(score)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(running_loss / max(1, batches), 4),
                "val_mae": round(breakdown.mae_overall, 4),
                "val_rain_weighted_mae": round(score, 4),
                "val_mae_clear": round(breakdown.mae_by_label.get("clear", float("nan")), 4),
                "val_mae_heavy": round(
                    breakdown.mae_by_label.get("heavy_rain", float("nan")), 4
                ),
            }
        )

        improved = score < best_score - 1e-4
        if improved:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = " *" if improved else ""
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  loss {running_loss / max(1, batches):7.4f}"
                f"  val MAE {breakdown.mae_overall:6.3f}"
                f"  rain-weighted {score:6.3f}{marker}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stop: no rain-weighted improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)
    result = PhaseResult(
        name=name,
        epochs_run=len(history),
        best_rain_weighted_val_mae=round(best_score, 4),
        best_epoch=best_epoch,
        seconds=round(time.time() - started, 1),
        dataset_label=splits.label,
        dataset_is_real=splits.is_real,
        history=history,
    )
    return result, best_state


def save_checkpoint(
    model: MonsoonPlusNet, path: Path, meta: dict, normalisation=None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "meta": meta,
    }
    if normalisation is not None:
        blob["normalisation"] = normalisation.to_dict()
    torch.save(blob, path)


def find_log_files() -> list[Path]:
    log_dir = REPO_ROOT / "logs"
    if not log_dir.exists():
        return []
    return sorted(p for p in log_dir.glob("*.json") if p.name != "README.md")


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description="Train MonsoonPlusNet")
    parser.add_argument("--graph-layer", choices=("gcn", "gat"), default="gcn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--transfer-epochs", type=int, default=20)
    parser.add_argument("--cost-epochs", type=int, default=16)
    parser.add_argument("--days", type=int, default=45, help="synthetic Chennai window length")
    parser.add_argument("--metr-la-sensors", type=int, default=64,
                        help="subset of METR-LA sensors for pretraining (CPU-friendly)")
    parser.add_argument("--metr-la-stride", type=int, default=6,
                        help="thin overlapping pretraining windows (6 = one every 30 min)")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--allow-download", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(argv)

    if args.quick:
        args.pretrain_epochs, args.transfer_epochs, args.cost_epochs = 1, 2, 1
        args.days, args.metr_la_sensors = 8, 24
        args.metr_la_stride = 12

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"monsoonplus training  device={device}  graph_layer={args.graph_layer}")

    model = build_model(graph_layer=args.graph_layer, hidden=args.hidden)
    print(model.describe())

    results: list[PhaseResult] = []

    # ---- phase 1: METR-LA pretraining -------------------------------------
    if args.skip_pretrain:
        results.append(
            PhaseResult(
                name="phase1_metr_la_pretrain",
                epochs_run=0,
                best_rain_weighted_val_mae=float("nan"),
                best_epoch=-1,
                seconds=0.0,
                dataset_label="skipped",
                dataset_is_real=False,
                skipped=True,
                skip_reason="--skip-pretrain passed",
            )
        )
    else:
        metr_splits = build_metr_la_datasets(
            allow_download=args.allow_download,
            max_sensors=args.metr_la_sensors,
            train_stride=args.metr_la_stride,
        )
        print("\n" + metr_splits.describe())
        result, _ = run_phase(
            "phase1_metr_la_pretrain",
            model,
            metr_splits,
            device,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            weighted=False,  # LA has no rain channel; weighting would be meaningless
        )
        results.append(result)
        save_checkpoint(
            model,
            CHECKPOINT_DIR / "phase1_metr_la.pt",
            {"phase": "phase1", "dataset": metr_splits.label, "is_real": metr_splits.is_real},
            metr_splits.normalisation,
        )

    # ---- phase 2: transfer to Chennai -------------------------------------
    chennai = build_chennai_datasets(days=args.days, seed=args.seed)
    print("\n" + chennai.describe())
    result, _ = run_phase(
        "phase2_chennai_transfer",
        model,
        chennai,
        device,
        epochs=args.transfer_epochs,
        batch_size=args.batch_size,
        learning_rate=8e-4,
        weighted=False,
    )
    results.append(result)
    save_checkpoint(
        model,
        CHECKPOINT_DIR / "phase2_chennai.pt",
        {"phase": "phase2", "dataset": chennai.label, "is_real": chennai.is_real},
        chennai.normalisation,
    )

    # ---- phase 3: cost-sensitive fine-tune --------------------------------
    result, _ = run_phase(
        "phase3_cost_sensitive",
        model,
        chennai,
        device,
        epochs=args.cost_epochs,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        weighted=True,
    )
    results.append(result)
    save_checkpoint(
        model,
        CHECKPOINT_DIR / "monsoonplus_final.pt",
        {
            "phase": "phase3",
            "dataset": chennai.label,
            "is_real": chennai.is_real,
            "graph_layer": args.graph_layer,
            "horizons_minutes": HORIZONS_MINUTES,
            "note": (
                "Trained on the synthetic Chennai window after METR-LA pretraining. "
                "Satellite branch trained on synthetic NDWI unless a GEE export was supplied."
            ),
        },
        chennai.normalisation,
    )

    # ---- phase 4: real collected logs -------------------------------------
    log_files = find_log_files()
    if not log_files:
        print(
            "\n=== phase4_real_logs ===\n"
            "  SKIPPED: logs/ contains no collected log files.\n"
            "  Run the collector during real Chennai rain to populate it "
            "(see README section 9). The model ships trained through phase 3."
        )
        results.append(
            PhaseResult(
                name="phase4_real_logs",
                epochs_run=0,
                best_rain_weighted_val_mae=float("nan"),
                best_epoch=-1,
                seconds=0.0,
                dataset_label="no logs collected yet",
                dataset_is_real=False,
                skipped=True,
                skip_reason="logs/ is empty -- collect live data to enable this phase",
            )
        )
    else:
        print(f"\n=== phase4_real_logs ===\n  found {len(log_files)} log files")
        print("  Fine-tuning on real logs requires at least a few hundred windows;")
        print("  see ml/training/finetune_logs.py for the dedicated entry point.")
        results.append(
            PhaseResult(
                name="phase4_real_logs",
                epochs_run=0,
                best_rain_weighted_val_mae=float("nan"),
                best_epoch=-1,
                seconds=0.0,
                dataset_label=f"{len(log_files)} log files present",
                dataset_is_real=True,
                skipped=True,
                skip_reason="run ml/training/finetune_logs.py to use them",
            )
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "graph_layer": args.graph_layer,
        "device": str(device),
        "seed": args.seed,
        "phases": [asdict(r) for r in results],
    }
    report_path = REPORT_DIR / "training_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- summary ---")
    for item in results:
        if item.skipped:
            print(f"  {item.name:<28} SKIPPED  ({item.skip_reason})")
        else:
            print(
                f"  {item.name:<28} best rain-weighted val MAE "
                f"{item.best_rain_weighted_val_mae:.3f} @ epoch {item.best_epoch} "
                f"({item.seconds:.0f}s)"
            )
    print(f"\ncheckpoint: {CHECKPOINT_DIR / 'monsoonplus_final.pt'}")
    print(f"report    : {report_path}")
    return summary


if __name__ == "__main__":
    main()
