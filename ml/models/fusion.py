"""Gated trimodal fusion -- the core of the monsoonplus novelty claim.

The claim being defended
------------------------
"In heavy rain the model should demonstrably lean on weather and satellite more than
on traffic history alone."

That is only a real claim if the mixing weights are (a) learned, (b) input-dependent,
and (c) readable from outside. So the gate here is:

  * learned        -- a small network trained jointly with everything else,
  * conditional    -- it reads all three embeddings plus the raw rain level, per node
                      and per sample, so the mix can differ between a dry road and a
                      flooding one in the same forward pass,
  * observable     -- ``forward`` returns the weights, and nothing in the Model Lab
                      display is invented. If the gate says traffic 0.9 in a storm,
                      the UI shows 0.9 and the write-up has to explain it.

Availability masking
--------------------
METR-LA has no rainfall or satellite channel. Rather than feed the weather and
satellite encoders zeros and pretend they trained (which would make "trimodal" a lie),
phase-1 pretraining passes a mask marking those modalities absent. The gate then
renormalises over the available modalities only, and the absent encoders receive no
gradient at all. Phases 2+ supply all three on Chennai data, where they do train.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("traffic", "weather", "satellite")


class GatedFusion(nn.Module):
    """Mix three per-node modality embeddings with learned, input-dependent weights.

    Inputs  (B, N, H) each, plus an optional (B, N, 3) availability mask and an
            optional (B, N, 1) rain context scalar.
    Returns ``(fused, gate)`` with fused (B, N, H) and gate (B, N, 3) summing to 1
            over the modality axis.
    """

    def __init__(self, hidden: int, rain_context: bool = True, temperature: float = 1.0):
        super().__init__()
        self.hidden = hidden
        self.rain_context = rain_context
        self.temperature = temperature

        # The gate sees all three embeddings at once -- it has to compare them, not
        # score each in isolation -- plus the raw rain level, which is the single most
        # informative scalar for "should I trust weather right now".
        gate_in = hidden * len(MODALITIES) + (1 if rain_context else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(MODALITIES)),
        )

        # Projections keep each modality in its own subspace before mixing, so the
        # gate weights stay interpretable as "how much of this signal".
        self.project = nn.ModuleDict(
            {name: nn.Linear(hidden, hidden) for name in MODALITIES}
        )
        self.out_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        traffic: torch.Tensor,
        weather: torch.Tensor,
        satellite: torch.Tensor,
        availability: Optional[torch.Tensor] = None,
        rain_context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([traffic, weather, satellite], dim=2)  # (B, N, 3, H)

        gate_input = torch.cat([traffic, weather, satellite], dim=-1)
        if self.rain_context:
            if rain_context is None:
                rain_context = torch.zeros(
                    *traffic.shape[:2], 1, device=traffic.device, dtype=traffic.dtype
                )
            gate_input = torch.cat([gate_input, rain_context], dim=-1)

        logits = self.gate(gate_input) / self.temperature  # (B, N, 3)

        if availability is not None:
            # Absent modalities get -inf so softmax renormalises over what exists.
            logits = logits.masked_fill(availability <= 0, float("-inf"))

        weights = torch.softmax(logits, dim=-1)
        # A row with every modality masked out would be all-NaN; fall back to traffic.
        weights = torch.nan_to_num(weights, nan=0.0)
        degenerate = weights.sum(dim=-1, keepdim=True) <= 0
        if degenerate.any():
            fallback = torch.zeros_like(weights)
            fallback[..., 0] = 1.0
            weights = torch.where(degenerate, fallback, weights)

        projected = torch.stack(
            [self.project[name](stacked[:, :, i]) for i, name in enumerate(MODALITIES)],
            dim=2,
        )  # (B, N, 3, H)

        fused = (projected * weights.unsqueeze(-1)).sum(dim=2)
        return self.out_norm(fused), weights


def gate_summary(gate: torch.Tensor) -> dict[str, float]:
    """Mean gate weight per modality, as the percentages the Model Lab shows.

    Averaged over batch and nodes. These are the trained model's own numbers; the
    frontend reads them from the export and never synthesises them.
    """
    mean = gate.detach().float().mean(dim=(0, 1))
    return {name: round(float(mean[i]) * 100, 1) for i, name in enumerate(MODALITIES)}
