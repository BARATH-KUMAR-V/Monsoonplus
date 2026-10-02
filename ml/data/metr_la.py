"""METR-LA loader: real data when available, a clearly-labelled stand-in when not.

METR-LA is 207 loop-detector sensors on LA freeways, 5-minute speeds, Mar-Jun 2012.
monsoonplus uses it for phase-1 pretraining (transfer learning US -> Chennai) and as
the "Dataset mode" benchmark.

Three sources are tried, in order:

1. ``ml/data/raw/metr-la.h5``        -- the canonical HDF5 release.
2. ``ml/data/raw/METR-LA/*.npy``     -- the graphmining.ai mirror (adj + node values).
3. a synthetic stand-in             -- generated, and flagged ``is_real=False`` so the
                                       UI can label it honestly.

Why a raw h5py reader instead of ``pandas.read_hdf``
----------------------------------------------------
``metr-la.h5`` is a pandas "fixed" format block with byte-string axis labels. Under
pandas 3.x ``read_hdf`` raises ``TypeError`` decoding those bytes, so we read the HDF5
datasets directly with h5py and never hand the file to pandas at all. This is not a
style preference -- ``read_hdf`` genuinely cannot open this file on a modern pandas.
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config.segments import REPO_ROOT

RAW_DIR = REPO_ROOT / "ml" / "data" / "raw"
H5_PATH = RAW_DIR / "metr-la.h5"
MIRROR_DIR = RAW_DIR / "METR-LA"

# Mirrors are tried in order. The first is the canonical ``metr-la.h5`` (~57 MB) as
# republished by the DL-Traff benchmark repo; the historical graphmining.ai host that
# torch_geometric_temporal points at no longer resolves, so it sits last as a courtesy.
MIRROR_URLS = (
    (
        "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/main/METRLA/metr-la.h5",
        "h5",
    ),
    (
        "https://graphmining.ai/temporal_datasets/METR-LA.zip",
        "zip",
    ),
)

# Sensor-to-sensor road distances, used to build the Gaussian-kernel adjacency.
DISTANCES_URL = (
    "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/main/METRLA/distances_la_2012.csv"
)
DISTANCES_PATH = RAW_DIR / "distances_la_2012.csv"

SPEED_FLOOR = 1.0
SPEED_CEIL = 80.0


@dataclass
class MetrLaData:
    """Raw METR-LA speeds plus provenance.

    speeds: (time_steps, num_sensors) in mph as published; 0.0 marks a missing reading.
    adjacency: (num_sensors, num_sensors) if a mirror supplied one, else None.
    sensor_ids: the 207 detector ids, in column order, for the distance table.
    minute_of_day: (time_steps,) 0-1435, for the time-of-day historical baseline.
    """

    speeds: np.ndarray
    adjacency: Optional[np.ndarray]
    is_real: bool
    source: str
    label: str
    sensor_ids: Optional[list[str]] = None
    minute_of_day: Optional[np.ndarray] = None
    day_of_week: Optional[np.ndarray] = None

    @property
    def num_sensors(self) -> int:
        return int(self.speeds.shape[1])

    @property
    def num_steps(self) -> int:
        return int(self.speeds.shape[0])


def _read_h5_raw(
    path: Path,
) -> Tuple[np.ndarray, Optional[list[str]], Optional[np.ndarray]]:
    """Pull the speed block, sensor ids and timestamps out of metr-la.h5 with h5py.

    Layout of the published file (a pandas "fixed" block, written years ago):

        df/axis0         (207,)         |S6    sensor ids as BYTE strings
        df/axis1         (34272,)       int64  nanosecond epoch timestamps
        df/block0_values (34272, 207)   f8     speeds in mph, 0.0 == missing

    ``pandas.read_hdf`` is avoided deliberately. It needs the pytables package (which
    has no wheel on many modern Python builds) and, once installed, trips over
    decoding those ``|S6`` byte labels on pandas 3.x. h5py reads the arrays directly
    and we decode the labels ourselves, so neither problem can occur.
    """
    import h5py

    with h5py.File(path, "r") as fh:
        candidates: list[tuple[str, tuple]] = []

        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                candidates.append((name, obj.shape))

        fh.visititems(visit)
        if not candidates:
            raise ValueError(f"{path} contains no 2-D dataset; not a METR-LA release")

        # Prefer an explicit block0_values; otherwise take the largest 2-D block.
        values_key = next(
            (n for n, _ in candidates if n.endswith("block0_values")),
            max(candidates, key=lambda c: c[1][0] * c[1][1])[0],
        )
        speeds = np.asarray(fh[values_key][:], dtype=np.float32)

        group = values_key.rsplit("/", 1)[0] if "/" in values_key else ""
        prefix = f"{group}/" if group else ""

        sensor_ids: Optional[list[str]] = None
        for key in (f"{prefix}axis0", f"{prefix}block0_items"):
            if key in fh:
                raw = fh[key][:]
                sensor_ids = [
                    v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                    for v in raw
                ]
                break

        timestamps: Optional[np.ndarray] = None
        if f"{prefix}axis1" in fh:
            timestamps = np.asarray(fh[f"{prefix}axis1"][:], dtype=np.int64)

    if speeds.shape[0] < speeds.shape[1]:
        speeds = speeds.T  # stored sensor-major; we want time-major
    return speeds, sensor_ids, timestamps


def _clock_from_timestamps(
    timestamps: Optional[np.ndarray], num_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Minute-of-day and day-of-week from nanosecond epoch stamps.

    Falls back to a synthetic 5-minute clock starting at midnight Monday when the
    file carries no usable axis, so downstream time-of-day features always exist.
    """
    if timestamps is not None and timestamps.shape[0] == num_steps:
        seconds = timestamps.astype(np.int64) // 1_000_000_000
        minute_of_day = ((seconds % 86_400) // 60).astype(np.int32)
        # 1970-01-01 was a Thursday, hence the +3 to land Monday on 0.
        day_of_week = (((seconds // 86_400) + 3) % 7).astype(np.int32)
        return minute_of_day, day_of_week

    step = np.arange(num_steps, dtype=np.int64)
    return ((step * 5) % 1440).astype(np.int32), ((step * 5) // 1440 % 7).astype(np.int32)


def _read_mirror(directory: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read the graphmining.ai mirror: node_values.npy (+ adj_mat.npy)."""
    values_path = next(
        (p for p in (directory / "node_values.npy", directory / "X.npy") if p.exists()),
        None,
    )
    if values_path is None:
        raise FileNotFoundError(f"no node_values.npy under {directory}")

    values = np.load(values_path)
    # Mirror ships (time, nodes, features) with feature 0 = speed.
    if values.ndim == 3:
        values = values[:, :, 0]
    speeds = np.asarray(values, dtype=np.float32)
    if speeds.shape[0] < speeds.shape[1]:
        speeds = speeds.T

    adjacency = None
    for name in ("adj_mat.npy", "A.npy", "adj.npy"):
        if (directory / name).exists():
            adjacency = np.asarray(np.load(directory / name), dtype=np.float32)
            break
    return speeds, adjacency


def _fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "monsoonplus/1.0 (student project)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_metr_la(timeout: int = 300) -> bool:
    """Fetch METR-LA into ``ml/data/raw/``.

    Entirely optional -- every caller falls back to the synthetic stand-in, so a
    firewalled machine still gets a working demo. Returns True when real data is on
    disk afterwards.
    """
    if H5_PATH.exists() or (MIRROR_DIR / "node_values.npy").exists():
        print("[metr-la] real data already present")
        _download_distances(timeout)
        return True

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, kind in MIRROR_URLS:
        print(f"[metr-la] trying {url}")
        try:
            payload = _fetch(url, timeout)
        except Exception as exc:  # noqa: BLE001 - any network failure is non-fatal
            print(f"[metr-la]   unavailable ({exc})")
            continue

        print(f"[metr-la]   got {len(payload) / 1e6:.1f} MB")
        if kind == "h5":
            H5_PATH.write_bytes(payload)
        else:
            MIRROR_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    for member in archive.namelist():
                        if member.endswith("/"):
                            continue
                        target = MIRROR_DIR / Path(member).name
                        with archive.open(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            except zipfile.BadZipFile:
                H5_PATH.write_bytes(payload)

        if H5_PATH.exists() or (MIRROR_DIR / "node_values.npy").exists():
            _download_distances(timeout)
            return True

    print("[metr-la] every mirror failed; the synthetic stand-in will be used")
    return False


def _download_distances(timeout: int) -> None:
    """Grab the sensor distance table so the adjacency is the real road graph.

    Optional: without it, ``ml/data/graph.py`` falls back to a correlation-derived
    adjacency and says so.
    """
    # Sensor ids come from the HDF5 axis labels, so only the distance table is fetched.
    for url, path in ((DISTANCES_URL, DISTANCES_PATH),):
        if path.exists():
            continue
        try:
            path.write_bytes(_fetch(url, timeout))
            print(f"[metr-la] fetched {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[metr-la] optional {path.name} unavailable ({exc})")


def _synthetic_metr_la(
    num_sensors: int = 207, num_days: int = 30, seed: int = 7
) -> np.ndarray:
    """A METR-LA-shaped stand-in: same cadence, same units, same missing-data habit.

    Deliberately NOT presented as METR-LA anywhere. Every consumer carries
    ``is_real=False`` through to the UI label.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 288  # 5-minute cadence
    num_steps = steps_per_day * num_days
    minute_of_day = (np.arange(num_steps) % steps_per_day) * 5

    # Two rush-hour troughs, per-sensor depth and phase.
    free_flow = rng.uniform(55, 70, size=num_sensors)
    am_depth = rng.uniform(5, 28, size=num_sensors)
    pm_depth = rng.uniform(8, 32, size=num_sensors)
    am_peak = rng.uniform(420, 480, size=num_sensors)
    pm_peak = rng.uniform(1020, 1110, size=num_sensors)

    t = minute_of_day[:, None]
    am = np.exp(-0.5 * ((t - am_peak[None, :]) / 55.0) ** 2)
    pm = np.exp(-0.5 * ((t - pm_peak[None, :]) / 75.0) ** 2)
    speeds = free_flow[None, :] - am_depth[None, :] * am - pm_depth[None, :] * pm

    # Weekends are lighter.
    day_index = np.arange(num_steps) // steps_per_day
    weekend = ((day_index % 7) >= 5).astype(np.float32)[:, None]
    speeds += weekend * 6.0

    # Spatially correlated noise, so neighbouring sensors move together.
    noise = rng.normal(0, 3.0, size=(num_steps, num_sensors))
    kernel = np.array([0.15, 0.7, 0.15])
    noise = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, noise)
    speeds = speeds + noise

    speeds = np.clip(speeds, SPEED_FLOOR, SPEED_CEIL).astype(np.float32)

    # METR-LA marks missing readings as exactly 0.0, roughly 8% of cells.
    missing = rng.random(speeds.shape) < 0.08
    speeds[missing] = 0.0
    return speeds


def load_metr_la(allow_download: bool = False, quiet: bool = False) -> MetrLaData:
    """Load METR-LA, preferring real data, falling back to a labelled stand-in."""
    if allow_download:
        download_metr_la()

    if H5_PATH.exists():
        try:
            speeds, sensor_ids, timestamps = _read_h5_raw(H5_PATH)
            minute_of_day, day_of_week = _clock_from_timestamps(timestamps, speeds.shape[0])
            if not quiet:
                print(f"[metr-la] real HDF5: {speeds.shape} from {H5_PATH.name}")
            return MetrLaData(
                speeds=speeds,
                adjacency=None,
                is_real=True,
                source=str(H5_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                label="METR-LA (real)",
                sensor_ids=sensor_ids,
                minute_of_day=minute_of_day,
                day_of_week=day_of_week,
            )
        except Exception as exc:  # noqa: BLE001 - corrupt file must not kill the run
            print(f"[metr-la] could not read {H5_PATH}: {exc}")

    if (MIRROR_DIR / "node_values.npy").exists() or (MIRROR_DIR / "X.npy").exists():
        try:
            speeds, adjacency = _read_mirror(MIRROR_DIR)
            minute_of_day, day_of_week = _clock_from_timestamps(None, speeds.shape[0])
            if not quiet:
                print(f"[metr-la] real mirror: {speeds.shape} from {MIRROR_DIR.name}/")
            return MetrLaData(
                speeds=speeds,
                adjacency=adjacency,
                is_real=True,
                source=str(MIRROR_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
                label="METR-LA (real)",
                minute_of_day=minute_of_day,
                day_of_week=day_of_week,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[metr-la] could not read mirror: {exc}")

    if not quiet:
        print("[metr-la] no real data found -> synthetic METR-LA-shaped stand-in")
    speeds = _synthetic_metr_la()
    minute_of_day, day_of_week = _clock_from_timestamps(None, speeds.shape[0])
    return MetrLaData(
        speeds=speeds,
        adjacency=None,
        is_real=False,
        source="generated in ml/data/metr_la.py",
        label="Synthetic METR-LA-shaped stand-in",
        minute_of_day=minute_of_day,
        day_of_week=day_of_week,
    )


def to_traffic_features(speeds: np.ndarray) -> np.ndarray:
    """Expand single-channel speeds to the 3 traffic features the model expects.

    The checkpoint's traffic input width is fixed at 3 (speed, free-flow, confidence).
    METR-LA ships speed only, so we pad it *here*, before anything touches a
    checkpoint:

      * free-flow proxy -- the per-sensor 95th percentile of observed speed, a
        standard stand-in for the uncongested speed of a link.
      * confidence      -- 1.0 where a reading exists, 0.0 where METR-LA wrote 0.0.

    Doing this at load time matters: ``load_state_dict(strict=False)`` forgives
    missing or unexpected keys but will NOT rescue a shape mismatch on
    ``traffic_encoder.gru.weight_ih_l0``. Padding after the fact is too late.

    Returns (time_steps, num_sensors, 3). Missing speeds are filled by
    forward-fill then the sensor mean, so the GRU never sees a 0 km/h cliff.
    """
    speeds = np.asarray(speeds, dtype=np.float32)
    observed = speeds > 0.0
    confidence = observed.astype(np.float32)

    filled = speeds.copy()
    filled[~observed] = np.nan

    # Forward-fill down the time axis.
    for sensor in range(filled.shape[1]):
        column = filled[:, sensor]
        valid = ~np.isnan(column)
        if not valid.any():
            column[:] = 30.0
            continue
        index = np.where(valid, np.arange(len(column)), 0)
        np.maximum.accumulate(index, out=index)
        column[:] = column[index]
        # Leading NaNs have nothing to carry forward; use the sensor mean.
        column[np.isnan(column)] = float(np.nanmean(column))
    filled = np.nan_to_num(filled, nan=30.0)

    free_flow = np.percentile(filled, 95, axis=0).astype(np.float32)
    free_flow = np.maximum(free_flow, 5.0)
    free_flow_channel = np.broadcast_to(free_flow[None, :], filled.shape)

    return np.stack([filled, free_flow_channel, confidence], axis=-1).astype(np.float32)


if __name__ == "__main__":
    data = load_metr_la(allow_download=True)
    features = to_traffic_features(data.speeds)
    print(f"source      : {data.source}")
    print(f"label       : {data.label}  (is_real={data.is_real})")
    print(f"speeds      : {data.speeds.shape}")
    print(f"features    : {features.shape}  (speed, free-flow proxy, confidence)")
    print(f"observed    : {100 * float((data.speeds > 0).mean()):.1f}% of cells")
    if data.sensor_ids:
        print(f"sensor ids  : {data.sensor_ids[:4]} ... ({len(data.sensor_ids)} total)")
    if data.minute_of_day is not None:
        print(f"clock       : first stamp at minute {int(data.minute_of_day[0])} of day")
    print(f"speed range : {features[..., 0].min():.1f} - {features[..., 0].max():.1f}")
