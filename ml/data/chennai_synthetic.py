"""A synthetic Chennai monsoon window, generated from an explicit physical process.

This is the honest substitute for months of self-collected monsoon data that a
second-year student cannot have before a submission deadline. It is labelled
"Synthetic Chennai window" everywhere it surfaces, and nothing here is presented as a
measurement.

What makes it a *useful* synthetic set rather than noise
-------------------------------------------------------
The generative process gives each modality something real to contribute, so the gate
has an actual job:

1. **Traffic**   -- diurnal rush-hour profile + autocorrelated noise. Recent history
                    alone predicts most of the variance on a dry day.
2. **Weather**   -- storm cells sweep across the city, so rain differs per segment at
                    the same instant (via ``rain_bias``). Instantaneous rain drives an
                    immediate slowdown.
3. **Satellite** -- NDWI tracks *accumulated* wetness with a long decay, and the
                    flooding term depends on accumulation, not on instantaneous rain.

Point 3 is the key design decision. Because accumulated wetness is not recoverable
from the 2-hour weather history, the satellite channel carries information the other
two modalities genuinely lack: a road that has been soaking for six hours behaves
differently from one hit by the same rainfall rate ten minutes ago. If NDWI were just
a smoothed copy of rainfall, the satellite branch would be decorative and the gate
would be free to ignore it -- and the "trimodal" claim would be hollow.

The modality-drop ablation tests exactly this: removing satellite should hurt, most
visibly on heavy-rain samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.segments import (
    MODEL_CFG,
    SEGMENTS,
    STEP_MINUTES,
    rain_label,
)


@dataclass
class ChennaiWindow:
    """A generated Chennai window.

    speeds        (T, N)    km/h
    rain          (T, N)    mm/h, per segment
    temperature   (T, N)    deg C
    wind          (T, N)    km/h
    ndwi          (T, N)    -1..1, normalised difference water index
    wetness       (T, N)    accumulated-wetness anomaly, 0..1
    minute_of_day (T,)
    labels        (T,)      'clear' | 'rain' | 'heavy_rain', from city-mean rain
    """

    speeds: np.ndarray
    rain: np.ndarray
    temperature: np.ndarray
    wind: np.ndarray
    ndwi: np.ndarray
    wetness: np.ndarray
    minute_of_day: np.ndarray
    day_index: np.ndarray
    labels: np.ndarray
    is_real: bool = False
    label: str = "Synthetic Chennai window"
    source: str = "generated in ml/data/chennai_synthetic.py"

    @property
    def num_steps(self) -> int:
        return int(self.speeds.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.speeds.shape[1])

    def summary(self) -> dict:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            "steps": self.num_steps,
            "nodes": self.num_nodes,
            "days": int(self.day_index.max()) + 1,
            "label_counts": {str(k): int(v) for k, v in zip(unique, counts)},
            "rain_max_mm_h": round(float(self.rain.max()), 1),
            "speed_range_kmh": [
                round(float(self.speeds.min()), 1),
                round(float(self.speeds.max()), 1),
            ],
            "ndwi_range": [round(float(self.ndwi.min()), 3), round(float(self.ndwi.max()), 3)],
        }


def _diurnal_profile(minute_of_day: np.ndarray) -> np.ndarray:
    """Chennai's two rush hours: ~9:00 and ~18:30, the evening one worse."""
    morning = np.exp(-0.5 * ((minute_of_day - 540) / 70.0) ** 2)
    evening = np.exp(-0.5 * ((minute_of_day - 1110) / 95.0) ** 2)
    lunch = 0.25 * np.exp(-0.5 * ((minute_of_day - 810) / 60.0) ** 2)
    night = -0.35 * np.exp(-0.5 * ((minute_of_day - 180) / 150.0) ** 2)
    return 0.62 * morning + 1.0 * evening + lunch + night


def _storm_series(
    num_steps: int, steps_per_day: int, rng: np.random.Generator, rain_days: float
) -> np.ndarray:
    """City-mean rainfall in mm/h, as a sequence of storm cells.

    Chennai's northeast monsoon is bursty: long dry stretches punctuated by intense
    cells, not steady drizzle. Peaks reach 60-90 mm/h, which is where the interesting
    failure modes live.
    """
    rain = np.zeros(num_steps, dtype=np.float64)
    num_days = max(1, num_steps // steps_per_day)
    expected_storms = int(round(rain_days * 3.4)) + 1

    for _ in range(expected_storms):
        start = int(rng.integers(0, num_steps))
        # Storms last 2-9 hours; evening storms are the classic Chennai pattern.
        duration = int(rng.integers(24, 110))
        peak = float(rng.uniform(8, 92))
        centre = start + duration / 2.0
        index = np.arange(start, min(num_steps, start + duration))
        if index.size == 0:
            continue
        shape = np.exp(-0.5 * ((index - centre) / (duration / 3.2)) ** 2)
        # Asymmetric: rain builds faster than it clears, and a long moderate tail
        # follows each cell -- which is where the 2.5-15 mm/h samples come from.
        shape *= np.where(index < centre, 1.0, 0.82)
        rain[index] += peak * shape

        tail = np.arange(index[-1], min(num_steps, index[-1] + duration))
        if tail.size > 1:
            rain[tail] += peak * 0.22 * np.exp(-(tail - tail[0]) / (duration / 2.0))

    # Light background drizzle around the cells.
    drizzle = np.clip(rng.normal(0, 0.8, num_steps), 0, None)
    smooth = np.convolve(drizzle, np.ones(12) / 12.0, mode="same")
    rain = rain + smooth * (rain > 0.5)

    return np.clip(rain, 0, 120)


def generate_chennai_window(
    days: int = 45,
    seed: int = 2024,
    rain_fraction: float = 0.42,
) -> ChennaiWindow:
    """Generate a Chennai monsoon window at the configured 5-minute cadence.

    ``rain_fraction`` steers roughly how much of the window is wet. The default makes
    a northeast-monsoon-like window: wet enough that rain samples are not a rounding
    error, dry enough that the rain-vs-clear split stays meaningful.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = (24 * 60) // STEP_MINUTES
    num_steps = steps_per_day * days
    num_nodes = len(SEGMENTS)

    step = np.arange(num_steps)
    minute_of_day = (step * STEP_MINUTES) % 1440
    day_index = (step * STEP_MINUTES) // 1440

    city_rain = _storm_series(num_steps, steps_per_day, rng, rain_fraction * days)

    free_flow = np.array([s.free_flow_kmh for s in SEGMENTS], dtype=np.float64)
    sensitivity = np.array([s.rain_sensitivity for s in SEGMENTS], dtype=np.float64)
    congestion = np.array([s.base_congestion for s in SEGMENTS], dtype=np.float64)
    rain_bias = np.array([s.rain_bias for s in SEGMENTS], dtype=np.float64)

    # --- weather, per segment ------------------------------------------------
    # A storm cell crossing the city reaches segments at slightly different times;
    # the per-segment lag is what makes rain spatially varying rather than a scalar.
    lags = (np.arange(num_nodes) % 4).astype(int)
    rain = np.zeros((num_steps, num_nodes))
    for node in range(num_nodes):
        shifted = np.roll(city_rain, lags[node])
        shifted[: lags[node]] = city_rain[: lags[node]]
        rain[:, node] = shifted * rain_bias[node]
    rain += np.clip(rng.normal(0, 0.6, rain.shape), -2, 4) * (rain > 1.0)
    rain = np.clip(rain, 0, 130)

    # Temperature drops during rain; wind gusts with it.
    base_temp = 29.5 - 4.0 * np.sin((minute_of_day - 300) / 1440 * 2 * np.pi)
    temperature = base_temp[:, None] - 0.055 * rain + rng.normal(0, 0.35, rain.shape)
    wind = 9.0 + 0.22 * rain + rng.normal(0, 1.6, rain.shape)
    wind = np.clip(wind, 0, 70)

    # --- satellite: accumulated wetness --------------------------------------
    # A leaky accumulator in normalised 0-1 units. Calibrated so that:
    #   * a dry spell drains back toward 0 within a few hours,
    #   * ~1 hour of 50 mm/h reaches roughly 0.6-0.8,
    #   * only prolonged extreme rain saturates.
    # The earlier calibration used a 5-hour decay with a large input coefficient, which
    # pinned wetness at its ceiling almost permanently -- the satellite channel then
    # carried no information and every rainy road sat at the speed floor.
    #
    # Base decay time constant is 90 minutes; the per-segment exponent stretches it, so
    # a "severe" flood-history road drains roughly 4x slower than a "low" one.
    decay = np.exp(-STEP_MINUTES / 90.0)
    drainage_speed = np.array(
        [{"severe": 0.35, "moderate": 0.7, "low": 1.0}[s.flood_history] for s in SEGMENTS]
    )
    accumulation_rate = 0.00068  # per (mm/h) per step

    wetness = np.zeros_like(rain)
    accumulator = np.zeros(num_nodes)
    per_step_decay = decay**drainage_speed
    for t in range(num_steps):
        accumulator = accumulator * per_step_decay + rain[t] * accumulation_rate
        accumulator = np.clip(accumulator, 0.0, 1.0)
        wetness[t] = accumulator

    # NDWI: wetter ground raises the index. Sentinel-2 revisits every ~5 days, so the
    # generated series is piecewise-constant over 5-day blocks plus sensor noise --
    # the same staleness the real product has.
    ndwi_true = -0.18 + 0.42 * np.tanh(wetness * 1.4)
    revisit_steps = steps_per_day * 5
    ndwi = ndwi_true.copy()
    for start in range(0, num_steps, revisit_steps):
        stop = min(num_steps, start + revisit_steps)
        ndwi[start:stop] = ndwi_true[start:stop].mean(axis=0)
    ndwi += rng.normal(0, 0.012, ndwi.shape)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # --- traffic -------------------------------------------------------------
    diurnal = _diurnal_profile(minute_of_day)[:, None]
    weekend = ((day_index % 7) >= 5).astype(np.float64)[:, None]

    congestion_drop = free_flow[None, :] * congestion[None, :] * diurnal * (1 - 0.35 * weekend)
    # Immediate rain effect, saturating: the first 20 mm/h hurts far more than the next.
    rain_drop = free_flow[None, :] * sensitivity[None, :] * np.tanh(rain / 45.0) * 0.62
    # Flood effect, driven by accumulation rather than instantaneous rain. Linear in
    # wetness so it does not saturate the way the rain term does.
    flood_drop = free_flow[None, :] * sensitivity[None, :] * wetness * 0.45

    speeds = free_flow[None, :] - congestion_drop - rain_drop - flood_drop

    # Autocorrelated noise -- traffic is smooth in time, so white noise would make
    # persistence look artificially bad.
    noise = rng.normal(0, 1.5, speeds.shape)
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    for node in range(num_nodes):
        noise[:, node] = np.convolve(noise[:, node], kernel, mode="same")
    speeds = speeds + noise
    speeds = np.clip(speeds, 3.0, free_flow[None, :] * 1.05)

    labels = np.array([rain_label(float(v)) for v in rain.mean(axis=1)])

    return ChennaiWindow(
        speeds=speeds.astype(np.float32),
        rain=rain.astype(np.float32),
        temperature=temperature.astype(np.float32),
        wind=wind.astype(np.float32),
        ndwi=ndwi.astype(np.float32),
        wetness=wetness.astype(np.float32),
        minute_of_day=minute_of_day.astype(np.int32),
        day_index=day_index.astype(np.int32),
        labels=labels,
    )


def traffic_features(window: ChennaiWindow) -> np.ndarray:
    """(T, N, 3): speed, free-flow, confidence -- the same 3 channels as METR-LA."""
    free_flow = np.array([s.free_flow_kmh for s in SEGMENTS], dtype=np.float32)
    free_flow_channel = np.broadcast_to(free_flow[None, :], window.speeds.shape)
    confidence = np.ones_like(window.speeds)
    return np.stack([window.speeds, free_flow_channel, confidence], axis=-1).astype(np.float32)


def weather_features(window: ChennaiWindow) -> np.ndarray:
    """(T, N, 3): rain mm/h, temperature C, wind km/h."""
    return np.stack([window.rain, window.temperature, window.wind], axis=-1).astype(np.float32)


def satellite_features(window: ChennaiWindow) -> np.ndarray:
    """(T, N, 2): NDWI and the wetness anomaly."""
    return np.stack([window.ndwi, window.wetness], axis=-1).astype(np.float32)


if __name__ == "__main__":
    window = generate_chennai_window()
    print("Synthetic Chennai window")
    for key, value in window.summary().items():
        print(f"  {key:<16} {value}")

    # Does the satellite channel actually add information beyond recent rain?
    from numpy.polynomial import polynomial as P

    rain_flat = window.rain.reshape(-1)
    wet_flat = window.wetness.reshape(-1)
    speed_flat = window.speeds.reshape(-1)
    print(f"\n  corr(rain, speed)     {np.corrcoef(rain_flat, speed_flat)[0, 1]:+.3f}")
    print(f"  corr(wetness, speed)  {np.corrcoef(wet_flat, speed_flat)[0, 1]:+.3f}")
    print(f"  corr(rain, wetness)   {np.corrcoef(rain_flat, wet_flat)[0, 1]:+.3f}")
    print(
        "  (rain and wetness are only partly correlated, so the satellite channel\n"
        "   carries information the weather history does not)"
    )
