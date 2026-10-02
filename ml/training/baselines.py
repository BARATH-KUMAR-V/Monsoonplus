"""Baselines the model has to beat. All three are evaluated with the same code path
and the same test split as MonsoonPlusNet, so the comparison is fair.

1. **Persistence** -- predict the last observed speed for every horizon. In traffic
   forecasting this is a genuinely strong baseline at t+15, and papers that omit it are
   usually hiding something. If MonsoonPlusNet cannot beat persistence at t+15, the
   honest conclusion is that it has not learned much at that horizon, and the Model Lab
   says so.

2. **Historical average** -- per segment, per time-of-day, averaged over the training
   split. Captures the daily rhythm with no model at all. This is also what Page 3's
   "what is normal for this road at this time?" reads from.

3. **LSTM** -- a per-segment recurrent model with no graph and no weather. Isolates
   how much the graph plus the extra modalities are worth, rather than just "a neural
   network helps".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config.segments import HORIZON_STEPS, STEP_MINUTES
from ml.data.windowing import ForecastDataset, make_loader


def persistence_predictions(dataset: ForecastDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last observed speed, repeated across horizons."""
    predictions, targets, labels = [], [], []
    for position in range(len(dataset)):
        sample = dataset[position]
        last_speed = sample["traffic"][:, -1, 0].numpy()  # (N,)
        horizons = sample["target"].shape[-1]
        predictions.append(np.repeat(last_speed[:, None], horizons, axis=1))
        targets.append(sample["target"].numpy())
        labels.append(int(sample["label_index"]))
    return np.stack(predictions), np.stack(targets), np.array(labels)


class HistoricalAverage:
    """Mean speed per (segment, time-of-day bucket), fitted on the training split.

    Buckets are one model step wide (5 minutes), so 288 buckets a day. Buckets never
    seen in training fall back to the segment's overall mean.
    """

    def __init__(self, step_minutes: int = STEP_MINUTES):
        self.step_minutes = step_minutes
        self.buckets_per_day = (24 * 60) // step_minutes
        self.table: np.ndarray | None = None  # (buckets, nodes)
        self.segment_mean: np.ndarray | None = None

    def fit(self, speeds: np.ndarray, minute_of_day: np.ndarray) -> "HistoricalAverage":
        """speeds (T, N); minute_of_day (T,)."""
        num_nodes = speeds.shape[1]
        bucket = (np.asarray(minute_of_day) // self.step_minutes).astype(int)
        bucket = np.clip(bucket, 0, self.buckets_per_day - 1)

        totals = np.zeros((self.buckets_per_day, num_nodes), dtype=np.float64)
        counts = np.zeros((self.buckets_per_day, 1), dtype=np.float64)
        np.add.at(totals, bucket, speeds)
        np.add.at(counts, bucket, 1.0)

        self.segment_mean = speeds.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            table = totals / counts
        empty = counts.squeeze(-1) == 0
        table[empty] = self.segment_mean[None, :]
        self.table = table
        return self

    def predict(self, minute_of_day: np.ndarray, horizons: list[int] | None = None) -> np.ndarray:
        """Speed at each horizon for the given forecast-issue minutes.

        Returns (samples, nodes, horizons).
        """
        if self.table is None:
            raise RuntimeError("HistoricalAverage.fit() must run before predict()")
        horizon_steps = horizons or HORIZON_STEPS
        minute_of_day = np.asarray(minute_of_day)

        out = []
        for steps in horizon_steps:
            offset = minute_of_day + steps * self.step_minutes
            bucket = ((offset % (24 * 60)) // self.step_minutes).astype(int)
            out.append(self.table[bucket])
        return np.stack(out, axis=-1)


class LSTMBaseline(nn.Module):
    """Per-segment LSTM over speed history. No graph, no weather, no satellite.

    Deliberately given the same hidden width and the same residual-from-last-speed
    output as MonsoonPlusNet, so the ablation measures the architecture rather than
    capacity or output parameterisation.
    """

    def __init__(self, in_features: int = 3, hidden: int = 64, horizons: int = 3, layers: int = 2):
        super().__init__()
        self.register_buffer("traffic_mean", torch.zeros(in_features))
        self.register_buffer("traffic_std", torch.ones(in_features))
        self.lstm = nn.LSTM(in_features, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizons))

    def set_normalisation(self, normalisation) -> None:
        self.traffic_mean.copy_(torch.as_tensor(normalisation.traffic_mean, dtype=torch.float32))
        self.traffic_std.copy_(torch.as_tensor(normalisation.traffic_std, dtype=torch.float32))

    def forward(self, traffic: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, feats = traffic.shape
        normalised = (traffic - self.traffic_mean) / self.traffic_std
        flat = normalised.reshape(batch * nodes, steps, feats)
        _, (hidden_state, _) = self.lstm(flat)
        delta = self.head(hidden_state[-1]).reshape(batch, nodes, -1)
        return traffic[:, :, -1, 0:1] + delta


def train_lstm_baseline(
    train_dataset: ForecastDataset,
    val_dataset: ForecastDataset,
    normalisation,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    device: str = "cpu",
    weighted: bool = True,
    verbose: bool = True,
) -> LSTMBaseline:
    """Train the LSTM baseline under the same cost-sensitive loss as the main model."""
    from ml.training.losses import CostSensitiveLoss, rain_weighted_mae

    torch.manual_seed(13)
    model = LSTMBaseline().to(device)
    model.set_normalisation(normalisation)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = CostSensitiveLoss()

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    best_score = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            prediction = model(batch["traffic"].to(device))
            loss = criterion(
                prediction,
                batch["target"].to(device),
                batch["weight"].to(device) if weighted else None,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

        predictions, targets, labels = predict_lstm(model, val_dataset, device=device)
        score = rain_weighted_mae(predictions, targets, labels)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    [lstm] epoch {epoch + 1}/{epochs} rain-weighted val MAE {score:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_lstm(
    model: LSTMBaseline, dataset: ForecastDataset, batch_size: int = 128, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets, labels = [], [], []
    for batch in make_loader(dataset, batch_size=batch_size):
        predictions.append(model(batch["traffic"].to(device)).cpu().numpy())
        targets.append(batch["target"].numpy())
        labels.append(batch["label_index"].numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(labels)


def historical_average_predictions(
    dataset: ForecastDataset,
    speeds: np.ndarray,
    minute_of_day: np.ndarray,
    train_end_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on the training portion only, then predict for this dataset's windows."""
    model = HistoricalAverage().fit(speeds[:train_end_index], minute_of_day[:train_end_index])

    issue_minutes = np.array([minute_of_day[int(i) - 1] for i in dataset.indices])
    predictions = model.predict(issue_minutes)

    targets, labels = [], []
    for position in range(len(dataset)):
        sample = dataset[position]
        targets.append(sample["target"].numpy())
        labels.append(int(sample["label_index"]))
    return predictions, np.stack(targets), np.array(labels)
