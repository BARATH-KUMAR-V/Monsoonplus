"""Per-modality encoders. Each maps its raw signal to a common hidden width so the
gate can compare them on equal terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrafficEncoder(nn.Module):
    """GRU over each segment's recent speed history.

    Input  (B, N, T, F) -- B samples, N segments, T history steps, F traffic features.
    Output (B, N, H).

    The GRU is shared across segments: segments are folded into the batch dimension so
    one set of recurrent weights sees every road. That is what makes the encoder
    transferable from METR-LA's 207 sensors to Chennai's 10 -- node count never
    appears in a weight shape.
    """

    def __init__(self, in_features: int, hidden: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, traffic: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, feats = traffic.shape
        flat = traffic.reshape(batch * nodes, steps, feats)
        _, hidden_state = self.gru(flat)
        last = hidden_state[-1]  # (B*N, H) -- top layer's final state
        return self.norm(last.reshape(batch, nodes, self.hidden))


class WeatherEncoder(nn.Module):
    """GRU over the weather history at each segment.

    Input  (B, N, T, F) -- rainfall, temperature, wind.
    Output (B, N, H).

    Rainfall is the signal that actually carries the monsoon effect, so it gets a
    recurrent encoder too: the model needs "it has been raining for 40 minutes"
    to look different from "it just started".
    """

    def __init__(self, in_features: int, hidden: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, feats = weather.shape
        flat = weather.reshape(batch * nodes, steps, feats)
        _, hidden_state = self.gru(flat)
        last = hidden_state[-1]
        return self.norm(last.reshape(batch, nodes, self.hidden))


class SatelliteEncoder(nn.Module):
    """MLP over the per-segment satellite signal (NDWI + wetness anomaly).

    Input  (B, N, F) -- no time axis: Sentinel-2 revisits every ~5 days, so within a
    one-hour forecast window the satellite view is effectively static.
    Output (B, N, H).
    """

    def __init__(self, in_features: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, satellite: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(satellite))
