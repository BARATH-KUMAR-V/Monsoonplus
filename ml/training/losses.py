"""Cost-sensitive loss and the rain-weighted metrics early stopping uses.

Why weight the loss at all
--------------------------
About 60% of the window is clear weather. An unweighted loss therefore spends 60% of
its gradient on exactly the situation a monsoon tool does not need help with, and a
model can reach a flattering aggregate MAE while being useless in a cloudburst.

The weights come from ``config/segments.json`` so they are stated in one place:

    clear 1x    rain 3x    heavy_rain 8x

Early stopping on rain-weighted validation MAE
----------------------------------------------
This matters as much as the loss. Plain validation MAE is dominated by clear samples,
so selecting a checkpoint on it picks the epoch that is best at dry weather. The
checkpoint this project keeps is the one with the lowest *rain-weighted* validation
MAE -- the epoch that is best at the job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from config.segments import HORIZONS_MINUTES, THRESHOLDS
from ml.data.windowing import LABELS

RAIN_WEIGHTS = THRESHOLDS["rain_weights"]


class CostSensitiveLoss(nn.Module):
    """Per-sample-weighted Huber loss over all nodes and horizons.

    Huber rather than plain L2: speed series contain genuine spikes (an accident, a
    sensor glitch) and squared error lets a handful of them dominate the gradient.
    ``delta=2.0`` km/h is about the noise floor of the speed signal.
    """

    def __init__(self, delta: float = 2.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = prediction - target
        absolute = error.abs()
        per_element = torch.where(
            absolute <= self.delta,
            0.5 * error.pow(2),
            self.delta * (absolute - 0.5 * self.delta),
        )
        per_sample = per_element.flatten(start_dim=1).mean(dim=1)  # (B,)

        if weight is None:
            return per_sample.mean()
        # Normalise by the weight sum so the learning rate does not implicitly scale
        # with how rainy the batch happened to be.
        weight = weight.to(per_sample.dtype)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)


@dataclass
class ErrorBreakdown:
    """MAE sliced the way the project promises to report it: never one number alone."""

    mae_overall: float
    rmse_overall: float
    mae_by_label: dict[str, float] = field(default_factory=dict)
    mae_by_horizon: dict[str, float] = field(default_factory=dict)
    mae_by_label_horizon: dict[str, dict[str, float]] = field(default_factory=dict)
    count_by_label: dict[str, int] = field(default_factory=dict)
    rain_weighted_mae: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mae_overall": round(self.mae_overall, 3),
            "rmse_overall": round(self.rmse_overall, 3),
            "rain_weighted_mae": round(self.rain_weighted_mae, 3),
            "mae_by_label": {k: round(v, 3) for k, v in self.mae_by_label.items()},
            "mae_by_horizon": {k: round(v, 3) for k, v in self.mae_by_horizon.items()},
            "mae_by_label_horizon": {
                label: {h: round(v, 3) for h, v in per_horizon.items()}
                for label, per_horizon in self.mae_by_label_horizon.items()
            },
            "count_by_label": dict(self.count_by_label),
        }

    def headline(self) -> str:
        clear = self.mae_by_label.get("clear", float("nan"))
        rain = self.mae_by_label.get("rain", float("nan"))
        heavy = self.mae_by_label.get("heavy_rain", float("nan"))
        return (
            f"MAE {self.mae_overall:.2f} km/h overall "
            f"(clear {clear:.2f} | rain {rain:.2f} | heavy {heavy:.2f}) "
            f"rain-weighted {self.rain_weighted_mae:.2f}"
        )


def error_breakdown(
    predictions: np.ndarray, targets: np.ndarray, label_indices: np.ndarray
) -> ErrorBreakdown:
    """Compute the full MAE breakdown.

    predictions/targets: (samples, nodes, horizons)
    label_indices:       (samples,) index into LABELS
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    label_indices = np.asarray(label_indices)

    absolute = np.abs(predictions - targets)
    squared = (predictions - targets) ** 2

    breakdown = ErrorBreakdown(
        mae_overall=float(absolute.mean()),
        rmse_overall=float(np.sqrt(squared.mean())),
    )

    for index, horizon in enumerate(HORIZONS_MINUTES):
        breakdown.mae_by_horizon[f"t+{horizon}"] = float(absolute[:, :, index].mean())

    weighted_total = 0.0
    weight_total = 0.0
    for index, label in enumerate(LABELS):
        mask = label_indices == index
        count = int(mask.sum())
        breakdown.count_by_label[label] = count
        if count == 0:
            breakdown.mae_by_label[label] = float("nan")
            breakdown.mae_by_label_horizon[label] = {
                f"t+{h}": float("nan") for h in HORIZONS_MINUTES
            }
            continue

        subset = absolute[mask]
        label_mae = float(subset.mean())
        breakdown.mae_by_label[label] = label_mae
        breakdown.mae_by_label_horizon[label] = {
            f"t+{h}": float(subset[:, :, i].mean()) for i, h in enumerate(HORIZONS_MINUTES)
        }

        weight = float(RAIN_WEIGHTS[label])
        weighted_total += label_mae * weight * count
        weight_total += weight * count

    breakdown.rain_weighted_mae = (
        weighted_total / weight_total if weight_total > 0 else breakdown.mae_overall
    )
    return breakdown


def rain_weighted_mae(
    predictions: np.ndarray, targets: np.ndarray, label_indices: np.ndarray
) -> float:
    """The single number early stopping watches."""
    return error_breakdown(predictions, targets, label_indices).rain_weighted_mae
