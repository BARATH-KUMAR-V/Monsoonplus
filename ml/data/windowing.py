"""Turn a long (time, node, feature) series into supervised forecasting samples.

One sample is:

    inputs   traffic   (N, T, 3)   the last T=24 steps -> 2 hours of history
             weather   (N, T, 3)
             satellite (N, 2)      most recent satellite view
    target   speeds    (N, H)      speed at t+15, t+30, t+60
    meta     rain_mm_h scalar, weather label, minute_of_day

Splitting is **chronological**, never shuffled-then-split. Traffic is strongly
autocorrelated, so a random split would put a sample's near-identical neighbour in the
training set and report a validation MAE that is really a memorisation score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.segments import (
    HISTORY_STEPS,
    HORIZON_STEPS,
    HORIZONS_MINUTES,
    THRESHOLDS,
    rain_label,
    sample_weight,
)

MPH_TO_KMH = 1.60934


@dataclass
class Normalisation:
    """Per-feature mean/std, computed on the training split only.

    Computing these over the whole series would leak validation statistics into
    training. Stored in the checkpoint so inference normalises identically.
    """

    traffic_mean: np.ndarray
    traffic_std: np.ndarray
    weather_mean: np.ndarray
    weather_std: np.ndarray
    satellite_mean: np.ndarray
    satellite_std: np.ndarray

    def to_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, blob: dict) -> "Normalisation":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in blob.items()})

    @classmethod
    def fit(
        cls, traffic: np.ndarray, weather: np.ndarray, satellite: np.ndarray
    ) -> "Normalisation":
        def stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            flat = array.reshape(-1, array.shape[-1])
            mean = flat.mean(axis=0).astype(np.float32)
            std = flat.std(axis=0).astype(np.float32)
            std[std < 1e-3] = 1.0  # a constant channel must not blow up
            return mean, std

        traffic_mean, traffic_std = stats(traffic)
        weather_mean, weather_std = stats(weather)
        satellite_mean, satellite_std = stats(satellite)
        return cls(
            traffic_mean, traffic_std, weather_mean, weather_std, satellite_mean, satellite_std
        )


class ForecastDataset(Dataset):
    """Windowed samples over one continuous series.

    ``availability`` marks which modalities this dataset actually carries, so the
    gate can renormalise instead of being fed zeros and asked to pretend.
    """

    def __init__(
        self,
        traffic: np.ndarray,
        weather: np.ndarray,
        satellite: np.ndarray,
        rain_for_label: np.ndarray,
        availability: tuple[int, int, int] = (1, 1, 1),
        history: int = HISTORY_STEPS,
        horizon_steps: Optional[list[int]] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.traffic = np.asarray(traffic, dtype=np.float32)
        self.weather = np.asarray(weather, dtype=np.float32)
        self.satellite = np.asarray(satellite, dtype=np.float32)
        self.rain_for_label = np.asarray(rain_for_label, dtype=np.float32)
        self.availability = np.asarray(availability, dtype=np.float32)
        self.history = history
        self.horizon_steps = list(horizon_steps or HORIZON_STEPS)

        max_horizon = max(self.horizon_steps)
        last_valid = self.traffic.shape[0] - max_horizon
        self.indices = (
            np.arange(self.history, last_valid)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, position: int) -> dict:
        t = int(self.indices[position])
        history_slice = slice(t - self.history, t)

        traffic = self.traffic[history_slice]       # (T, N, F) -> transpose to (N, T, F)
        weather = self.weather[history_slice]
        satellite = self.satellite[t - 1]           # most recent view

        target = np.stack(
            [self.traffic[t + h - 1, :, 0] for h in self.horizon_steps], axis=-1
        )  # (N, H) -- channel 0 is speed

        rain_now = float(self.rain_for_label[t - 1])
        return {
            "traffic": torch.from_numpy(np.ascontiguousarray(traffic.transpose(1, 0, 2))),
            "weather": torch.from_numpy(np.ascontiguousarray(weather.transpose(1, 0, 2))),
            "satellite": torch.from_numpy(np.ascontiguousarray(satellite)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "availability": torch.from_numpy(self.availability),
            "rain_mm_h": torch.tensor(rain_now, dtype=torch.float32),
            "weight": torch.tensor(sample_weight(rain_now), dtype=torch.float32),
            "label_index": torch.tensor(_label_index(rain_now), dtype=torch.long),
            "index": torch.tensor(t, dtype=torch.long),
        }


LABELS = ("clear", "rain", "heavy_rain")


def _label_index(rain_mm_h: float) -> int:
    return LABELS.index(rain_label(rain_mm_h))


@dataclass
class SplitDatasets:
    train: ForecastDataset
    val: ForecastDataset
    test: ForecastDataset
    normalisation: Normalisation
    label: str
    is_real: bool
    source: str

    def describe(self) -> str:
        def wet(dataset: ForecastDataset) -> str:
            labels = [
                _label_index(float(dataset.rain_for_label[int(i) - 1]))
                for i in dataset.indices
            ]
            counts = np.bincount(labels, minlength=3)
            total = max(1, counts.sum())
            return " ".join(f"{LABELS[i]} {100 * counts[i] / total:.0f}%" for i in range(3))

        return (
            f"{self.label}: train {len(self.train)} / val {len(self.val)} / test {len(self.test)}\n"
            f"  train mix  {wet(self.train)}\n"
            f"  val mix    {wet(self.val)}\n"
            f"  test mix   {wet(self.test)}"
        )


def chronological_split(
    traffic: np.ndarray,
    weather: np.ndarray,
    satellite: np.ndarray,
    rain_for_label: np.ndarray,
    availability: tuple[int, int, int],
    label: str,
    is_real: bool,
    source: str,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    train_stride: int = 1,
) -> SplitDatasets:
    """Split by time, fit normalisation on the training portion only.

    ``train_stride`` thins the *training* windows only. Consecutive windows sit one
    step (5 min) apart and overlap in 23 of 24 history steps, so a stride of 6 keeps
    windows 30 min apart and discards very little real information while cutting an
    epoch's cost by 6x. Validation and test always use every window, so the reported
    metrics are never thinned.
    """
    num_steps = traffic.shape[0]
    train_end = int(num_steps * fractions[0])
    val_end = int(num_steps * (fractions[0] + fractions[1]))

    normalisation = Normalisation.fit(
        traffic[:train_end], weather[:train_end], satellite[:train_end]
    )

    max_horizon = max(HORIZON_STEPS)
    last_valid = num_steps - max_horizon

    def indices(start: int, stop: int, stride: int = 1) -> np.ndarray:
        # Each window needs HISTORY_STEPS of history behind it, so a split's first
        # usable index sits HISTORY_STEPS after its boundary. That also stops a
        # validation window from reaching back into training data.
        return np.arange(
            max(HISTORY_STEPS, start + HISTORY_STEPS), min(stop, last_valid), max(1, stride)
        )

    def make(start: int, stop: int, stride: int = 1) -> ForecastDataset:
        return ForecastDataset(
            traffic,
            weather,
            satellite,
            rain_for_label,
            availability=availability,
            indices=indices(start, stop, stride),
        )

    return SplitDatasets(
        train=make(0, train_end, train_stride),
        val=make(train_end, val_end),
        test=make(val_end, num_steps),
        normalisation=normalisation,
        label=label,
        is_real=is_real,
        source=source,
    )


def build_metr_la_datasets(
    allow_download: bool = False,
    max_sensors: Optional[int] = None,
    quiet: bool = False,
    train_stride: int = 6,
) -> SplitDatasets:
    """Phase-1 pretraining data: METR-LA traffic only.

    Weather and satellite tensors are present but zero-filled AND marked unavailable,
    so the gate masks them out and their encoders get no gradient. This is the
    difference between "we pretrained the traffic branch on METR-LA" (true) and "we
    pretrained a trimodal model on METR-LA" (would be false).

    Speeds are converted mph -> km/h so every number in the project shares one unit.
    """
    from ml.data.metr_la import load_metr_la, to_traffic_features

    data = load_metr_la(allow_download=allow_download, quiet=quiet)
    speeds = data.speeds
    if max_sensors is not None:
        speeds = speeds[:, :max_sensors]

    traffic = to_traffic_features(speeds)
    traffic[..., 0] *= MPH_TO_KMH
    traffic[..., 1] *= MPH_TO_KMH

    num_steps, num_nodes = traffic.shape[0], traffic.shape[1]
    weather = np.zeros((num_steps, num_nodes, 3), dtype=np.float32)
    satellite = np.zeros((num_steps, num_nodes, 2), dtype=np.float32)
    rain_for_label = np.zeros(num_steps, dtype=np.float32)  # LA, no rain channel

    return chronological_split(
        traffic,
        weather,
        satellite,
        rain_for_label,
        availability=(1, 0, 0),
        label=data.label,
        is_real=data.is_real,
        source=data.source,
        train_stride=train_stride,
    )


def build_chennai_datasets(
    days: int = 45,
    seed: int = 2024,
    satellite_enabled: bool = True,
    train_stride: int = 1,
) -> SplitDatasets:
    """Phase-2+ data: the synthetic Chennai window, all three modalities live."""
    from ml.data.chennai_synthetic import (
        generate_chennai_window,
        satellite_features,
        traffic_features,
        weather_features,
    )

    window = generate_chennai_window(days=days, seed=seed)
    traffic = traffic_features(window)
    weather = weather_features(window)
    satellite = satellite_features(window)
    if not satellite_enabled:
        satellite = np.zeros_like(satellite)

    return chronological_split(
        traffic,
        weather,
        satellite,
        window.rain.mean(axis=1),
        availability=(1, 1, 1 if satellite_enabled else 0),
        label=window.label,
        is_real=window.is_real,
        source=window.source,
        train_stride=train_stride,
    )


def make_loader(
    dataset: ForecastDataset, batch_size: int = 32, shuffle: bool = False
) -> DataLoader:
    """Loader with num_workers=0 on purpose: Windows spawn plus Colab both behave
    better single-process here, and these datasets are small enough that workers
    cost more than they save."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    chennai = build_chennai_datasets()
    print(chennai.describe())
    sample = chennai.train[0]
    print("\nsample shapes")
    for key in ("traffic", "weather", "satellite", "target", "availability"):
        print(f"  {key:<12} {tuple(sample[key].shape)}")
    print(f"  horizons     {HORIZONS_MINUTES} min")
    print(f"  weight       {float(sample['weight'])} (rain {float(sample['rain_mm_h']):.1f} mm/h)")

    metr = build_metr_la_datasets()
    print()
    print(metr.describe())
    print(f"  availability {metr.train.availability.tolist()}  <- weather/satellite masked off")
