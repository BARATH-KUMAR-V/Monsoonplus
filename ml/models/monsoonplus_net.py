"""MonsoonPlusNet -- the assembled trimodal gated-fusion GNN.

    traffic  (B,N,T,3) --GRU--+
    weather  (B,N,T,3) --GRU--+--> gated fusion (B,N,3 weights) --> GCN/GAT --> head
    satellite(B,N,2)   --MLP--+                                                  |
                                                                                 v
                                                      speeds at t+15 / t+30 / t+60

Design points that matter for the project's claims
--------------------------------------------------
* Node count never appears in a weight shape, so a checkpoint pretrained on METR-LA's
  207 sensors loads unchanged onto Chennai's 10 segments. That is the transfer-learning
  claim, and ``test_transfer_shapes`` in the test suite pins it.
* ``forward`` returns the gate weights alongside the prediction. Modality importance in
  the Model Lab is read from here, not invented.
* The head predicts a *delta* from the last observed speed rather than an absolute
  speed. Persistence is a strong traffic baseline, so predicting the residual means the
  model starts at roughly persistence and has to earn any further improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn

from config.segments import (
    HORIZONS_MINUTES,
    NUM_SATELLITE_FEATURES,
    NUM_TRAFFIC_FEATURES,
    NUM_WEATHER_FEATURES,
)
from ml.models.encoders import SatelliteEncoder, TrafficEncoder, WeatherEncoder
from ml.models.fusion import MODALITIES, GatedFusion, gate_summary
from ml.models.graph_layers import SpatialBlock


@dataclass
class ModelOutput:
    """What a forward pass hands back.

    prediction: (B, N, H) speeds in km/h, one column per horizon.
    gate:       (B, N, 3) modality weights, summing to 1. Genuinely the model's.
    """

    prediction: torch.Tensor
    gate: torch.Tensor

    def gate_percentages(self) -> dict[str, float]:
        return gate_summary(self.gate)


@dataclass
class ModelConfig:
    hidden: int = 64
    traffic_features: int = NUM_TRAFFIC_FEATURES
    weather_features: int = NUM_WEATHER_FEATURES
    satellite_features: int = NUM_SATELLITE_FEATURES
    horizons: list[int] = field(default_factory=lambda: list(HORIZONS_MINUTES))
    graph_layer: str = "gcn"
    gru_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "hidden": self.hidden,
            "traffic_features": self.traffic_features,
            "weather_features": self.weather_features,
            "satellite_features": self.satellite_features,
            "horizons": list(self.horizons),
            "graph_layer": self.graph_layer,
            "gru_layers": self.gru_layers,
            "gat_heads": self.gat_heads,
            "dropout": self.dropout,
        }


def _broadcast_availability(
    availability: Optional[torch.Tensor], batch: int, nodes: int
) -> Optional[torch.Tensor]:
    """Accept any sensible availability shape and return (B, N, 3).

    Three shapes turn up in practice and all three are legitimate:
      (3,)      -- a dataset-level mask written by hand
      (B, 3)    -- what the DataLoader produces by collating per-sample (3,) masks
      (B, N, 3) -- a per-node mask, e.g. one segment's satellite tile is cloud-covered
    """
    if availability is None:
        return None
    if availability.dim() == 1:
        return availability.view(1, 1, -1).expand(batch, nodes, len(MODALITIES))
    if availability.dim() == 2:
        return availability.unsqueeze(1).expand(batch, nodes, len(MODALITIES))
    return availability


class MonsoonPlusNet(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, **overrides):
        super().__init__()
        self.config = config or ModelConfig(**overrides)
        cfg = self.config
        hidden = cfg.hidden

        self.traffic_encoder = TrafficEncoder(
            cfg.traffic_features, hidden, layers=cfg.gru_layers, dropout=cfg.dropout
        )
        self.weather_encoder = WeatherEncoder(cfg.weather_features, hidden)
        self.satellite_encoder = SatelliteEncoder(
            cfg.satellite_features, hidden, dropout=cfg.dropout
        )

        self.fusion = GatedFusion(hidden)
        self.spatial = SpatialBlock(
            hidden,
            graph_layer=cfg.graph_layer,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, len(cfg.horizons)),
        )

        # Normalisation lives inside the model, as buffers, so it travels with the
        # checkpoint. Inference therefore cannot silently use different statistics
        # from training -- a classic source of "it worked in the notebook" bugs.
        # Identity until set_normalisation() is called.
        self.register_buffer("traffic_mean", torch.zeros(cfg.traffic_features))
        self.register_buffer("traffic_std", torch.ones(cfg.traffic_features))
        self.register_buffer("weather_mean", torch.zeros(cfg.weather_features))
        self.register_buffer("weather_std", torch.ones(cfg.weather_features))
        self.register_buffer("satellite_mean", torch.zeros(cfg.satellite_features))
        self.register_buffer("satellite_std", torch.ones(cfg.satellite_features))

    def set_normalisation(self, normalisation) -> None:
        """Copy per-feature statistics (fitted on the training split) into buffers."""

        def put(name: str, values) -> None:
            buffer = getattr(self, name)
            buffer.copy_(torch.as_tensor(values, dtype=buffer.dtype).reshape(buffer.shape))

        put("traffic_mean", normalisation.traffic_mean)
        put("traffic_std", normalisation.traffic_std)
        put("weather_mean", normalisation.weather_mean)
        put("weather_std", normalisation.weather_std)
        put("satellite_mean", normalisation.satellite_mean)
        put("satellite_std", normalisation.satellite_std)

    @property
    def num_horizons(self) -> int:
        return len(self.config.horizons)

    def modality_masked(
        self,
        traffic: torch.Tensor,
        weather: torch.Tensor,
        satellite: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        drop: Iterable[str] = ("weather", "satellite"),
    ) -> ModelOutput:
        """Run the *already-trained* network with some modalities switched off.

        This is an inference-time sensitivity probe, not a retrained ablation: it
        measures what the trained network currently leans on, using the same
        availability mask the METR-LA phase uses, so the gate renormalises over the
        remaining modalities and the dropped branches contribute nothing. (For the
        stronger "retrain without it" test see ``ml/training/ablations.py``.)

        ``drop`` is any subset of ``{"weather", "satellite"}``. Traffic is never
        droppable -- it is the one signal always available live, and the head predicts
        a residual around its last speed.
        """
        drop = set(drop)
        unknown = drop - {"weather", "satellite"}
        if unknown:
            raise ValueError(
                f"cannot drop {sorted(unknown)}; only 'weather' and 'satellite' are optional"
            )
        availability = torch.tensor(
            [1.0, 0.0 if "weather" in drop else 1.0, 0.0 if "satellite" in drop else 1.0],
            device=traffic.device,
            dtype=traffic.dtype,
        )
        return self(
            traffic, weather, satellite, edge_index, edge_weight, availability=availability
        )

    def forward(
        self,
        traffic: torch.Tensor,
        weather: torch.Tensor,
        satellite: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        traffic      (B, N, T, Ft)
        weather      (B, N, T, Fw)
        satellite    (B, N, Fs)
        availability (B, N, 3) or (3,) -- 1 where a modality is present, 0 where absent.
        """
        # Encoders see standardised inputs; the residual below uses the raw km/h.
        traffic_embedding = self.traffic_encoder(
            (traffic - self.traffic_mean) / self.traffic_std
        )
        weather_embedding = self.weather_encoder(
            (weather - self.weather_mean) / self.weather_std
        )
        satellite_embedding = self.satellite_encoder(
            (satellite - self.satellite_mean) / self.satellite_std
        )

        availability = _broadcast_availability(
            availability, traffic.shape[0], traffic.shape[1]
        )

        # Rain now, as the gate's context scalar: index 0 of the weather features at
        # the most recent history step, standardised so the gate's input scale is
        # stable whatever the units upstream.
        rain_now = (weather[:, :, -1, 0:1] - self.weather_mean[0]) / self.weather_std[0]

        fused, gate = self.fusion(
            traffic_embedding,
            weather_embedding,
            satellite_embedding,
            availability=availability,
            rain_context=rain_now,
        )

        spatial = self.spatial(fused, edge_index, edge_weight)
        delta = self.head(spatial)

        # Residual around the last observed speed: the model predicts the change.
        last_speed = traffic[:, :, -1, 0:1]
        prediction = last_speed + delta

        return ModelOutput(prediction=prediction, gate=gate)

    def parameter_count(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "traffic_encoder": count(self.traffic_encoder),
            "weather_encoder": count(self.weather_encoder),
            "satellite_encoder": count(self.satellite_encoder),
            "fusion_gate": count(self.fusion),
            "spatial": count(self.spatial),
            "head": count(self.head),
            "total": count(self),
        }

    def describe(self) -> str:
        counts = self.parameter_count()
        lines = [
            f"MonsoonPlusNet  graph_layer={self.config.graph_layer}  hidden={self.config.hidden}",
            f"  horizons {self.config.horizons} min",
        ]
        for name, value in counts.items():
            lines.append(f"  {name:<18} {value:>9,}")
        return "\n".join(lines)


def build_model(graph_layer: str = "gcn", **overrides) -> MonsoonPlusNet:
    """Single construction point, so training / eval / export can never disagree."""
    return MonsoonPlusNet(ModelConfig(graph_layer=graph_layer, **overrides))


def load_checkpoint(
    path,
    map_location: str = "cpu",
    strict: bool = True,
    graph_layer: Optional[str] = None,
) -> tuple[MonsoonPlusNet, dict]:
    """Rebuild a model from a checkpoint written by ``ml/training/train.py``.

    The config travels *inside* the checkpoint, so the architecture is reconstructed
    exactly rather than guessed. Note ``strict=True`` by default: a shape mismatch
    should be a loud failure here, because ``strict=False`` only forgives missing or
    unexpected keys and will still raise on a shape conflict -- better to see it at
    load time than to silently load half a model.

    ``graph_layer`` swaps the spatial backend at load time. The one supported swap is
    between a PyG layer and its pure-PyTorch twin (``gcn`` <-> ``dense_gcn``,
    ``gat`` <-> ``dense_gat``): the twins share parameter names and shapes, so a
    checkpoint trained with torch_geometric runs without it. Swapping GCN for GAT is
    a different architecture and fails loudly under ``strict=True``.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**blob["config"])
    if graph_layer is not None:
        config.graph_layer = graph_layer
    model = MonsoonPlusNet(config)
    model.load_state_dict(blob["state_dict"], strict=strict)
    model.eval()
    return model, blob.get("meta", {})


if __name__ == "__main__":
    from ml.data.graph import chennai_adjacency, dense_to_edge_index

    torch.manual_seed(0)
    edge_index, edge_weight = dense_to_edge_index(chennai_adjacency())

    for layer in ("gcn", "gat"):
        model = build_model(graph_layer=layer)
        batch, nodes, steps = 4, 10, 24
        out = model(
            torch.randn(batch, nodes, steps, NUM_TRAFFIC_FEATURES).abs() * 30,
            torch.randn(batch, nodes, steps, NUM_WEATHER_FEATURES).abs() * 10,
            torch.randn(batch, nodes, NUM_SATELLITE_FEATURES),
            edge_index,
            edge_weight,
        )
        print(model.describe())
        print(f"  prediction {tuple(out.prediction.shape)}  gate {tuple(out.gate.shape)}")
        print(f"  gate sums to 1: {torch.allclose(out.gate.sum(-1), torch.ones(batch, nodes))}")
        print(f"  gate means: {out.gate_percentages()}\n")
