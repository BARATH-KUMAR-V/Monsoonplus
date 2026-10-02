"""Sentinel-2 NDWI (the satellite modality), with a labelled synthetic fallback.

NDWI = (Green - NIR) / (Green + NIR), Sentinel-2 bands B3 and B8. Water and saturated
ground reflect green while absorbing near-infrared, so NDWI rises as ground gets
wetter. Over the Pallikaranai marsh and the low-lying Velachery/Thoraipakkam stretches
that is a usable flood-risk proxy.

Three honest states
-------------------
``source="gee_sentinel2"``   a real Earth Engine export was available.
``source="local_export"``    a CSV/JSON exported from GEE earlier sits in ml/data/raw/.
``source="heuristic_fallback"``
                            generated from rainfall accumulation. Clearly labelled in
                            the UI as "Synthetic NDWI", and the README's limitations
                            section states plainly that the shipped model's satellite
                            branch is trained on this unless you wire up GEE yourself.

Earth Engine is entirely optional. It needs a free Google account and a one-time
project registration that can take a day or two to approve, so nothing in the base
deliverable waits on it.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config.segments import REPO_ROOT, SEGMENTS, segment_midpoints

RAW_DIR = REPO_ROOT / "ml" / "data" / "raw"
NDWI_EXPORT_PATH = RAW_DIR / "ndwi_chennai.csv"

# Sentinel-2 revisits the same point every ~5 days at the equator with two satellites.
REVISIT_DAYS = 5

# NDWI above this over a road corridor means standing water is likely.
FLOOD_NDWI_THRESHOLD = 0.05


@dataclass
class NdwiSnapshot:
    """NDWI per segment at one instant.

    ndwi      (N,)  -1..1
    wetness   (N,)  0..1 accumulated-wetness anomaly derived from NDWI
    """

    ndwi: np.ndarray
    wetness: np.ndarray
    source: str
    captured: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_measured(self) -> bool:
        return self.source in ("gee_sentinel2", "local_export")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "is_measured": self.is_measured,
            "captured": self.captured,
            "per_segment": [
                {
                    "segment_id": segment.id,
                    "ndwi": None if not np.isfinite(n) else round(float(n), 4),
                    "wetness": None if not np.isfinite(w) else round(float(w), 4),
                    "flood_likely": bool(np.isfinite(n) and n >= FLOOD_NDWI_THRESHOLD),
                }
                for segment, n, w in zip(SEGMENTS, self.ndwi, self.wetness)
            ],
            "notes": list(self.notes),
            "threshold": FLOOD_NDWI_THRESHOLD,
        }


def _get_ee_project() -> Optional[str]:
    """Read EARTHENGINE_PROJECT from environment or .env file."""
    if "EARTHENGINE_PROJECT" in os.environ:
        return os.environ.get("EARTHENGINE_PROJECT") or None
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EARTHENGINE_PROJECT="):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                return val or None
    return None


def earth_engine_available() -> bool:
    """True only if the ``ee`` package is installed *and* already authenticated.

    Deliberately does not trigger an interactive login: an import side effect that
    opens a browser during a demo would be worse than a labelled fallback.
    """
    try:
        import ee  # noqa: PLC0415
    except ImportError:
        return False
    try:
        project = _get_ee_project()
        ee.Initialize(project=project)
        return True
    except Exception:  # noqa: BLE001 - not authenticated, or no project set
        return False


def fetch_ndwi_from_gee(date: Optional[str] = None, buffer_m: int = 300) -> Optional[NdwiSnapshot]:
    """Mean NDWI in a buffer around each segment midpoint, from Sentinel-2 SR.

    Returns None when Earth Engine is unavailable, so the caller falls back. Uses the
    least-cloudy scene in a 20-day window ending at ``date``, because a single-date
    composite over Chennai in monsoon season is usually solid cloud.
    """
    if not earth_engine_available():
        return None

    try:
        import ee  # noqa: PLC0415
        from datetime import date as date_cls, timedelta  # noqa: PLC0415

        end = date_cls.fromisoformat(date) if date else date_cls.today()
        start = end - timedelta(days=20)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(str(start), str(end))
            .filterBounds(ee.Geometry.Point([80.232, 12.992]).buffer(20000))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        image = ee.Image(collection.first())
        ndwi_image = image.normalizedDifference(["B3", "B8"]).rename("ndwi")

        points = [
            ee.Feature(ee.Geometry.Point([lon, lat]).buffer(buffer_m), {"segment": segment.id})
            for segment, (lat, lon) in zip(SEGMENTS, segment_midpoints())
        ]
        sampled = ndwi_image.reduceRegions(
            collection=ee.FeatureCollection(points), reducer=ee.Reducer.mean(), scale=10
        ).getInfo()

        by_segment = {
            feature["properties"]["segment"]: feature["properties"].get("ndwi")
            for feature in sampled["features"]
        }
        values = np.array(
            [
                np.nan if by_segment.get(segment.id) is None else float(by_segment[segment.id])
                for segment in SEGMENTS
            ],
            dtype=np.float32,
        )
        captured = image.date().format("YYYY-MM-dd").getInfo()
        return NdwiSnapshot(
            ndwi=values,
            wetness=ndwi_to_wetness(values),
            source="gee_sentinel2",
            captured=captured,
            notes=["Contains modified Copernicus Sentinel data."],
        )
    except Exception as exc:  # noqa: BLE001 - never let GEE break a run
        print(f"[ndwi] Earth Engine query failed ({exc}); falling back")
        return None


def load_local_export(path: Path = NDWI_EXPORT_PATH) -> Optional[NdwiSnapshot]:
    """Read an NDWI table exported from GEE earlier.

    Expected CSV columns: ``segment_id,ndwi`` plus an optional ``captured`` date.
    This is the realistic path for a student: run the Earth Engine Code Editor
    snippet in the README once, export a CSV, drop it here.
    """
    if not path.exists():
        return None
    try:
        by_segment: dict[str, float] = {}
        captured: Optional[str] = None
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                segment_id = (row.get("segment_id") or row.get("segment") or "").strip()
                raw_value = row.get("ndwi")
                if not segment_id or raw_value in (None, ""):
                    continue
                by_segment[segment_id] = float(raw_value)
                captured = captured or (row.get("captured") or None)

        if not by_segment:
            return None
        values = np.array(
            [by_segment.get(segment.id, np.nan) for segment in SEGMENTS], dtype=np.float32
        )
        missing = [s.id for s in SEGMENTS if s.id not in by_segment]
        notes = ["Contains modified Copernicus Sentinel data."]
        if missing:
            notes.append(f"No NDWI value for {', '.join(missing)} (cloud or missing row).")
        return NdwiSnapshot(
            ndwi=values,
            wetness=ndwi_to_wetness(values),
            source="local_export",
            captured=captured,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ndwi] could not parse {path} ({exc})")
        return None


def ndwi_to_wetness(ndwi: np.ndarray) -> np.ndarray:
    """Map NDWI to the 0..1 wetness anomaly the model consumes.

    Dry urban ground sits near -0.2; saturated ground and standing water push toward
    +0.3. The linear rescale of that band keeps the model's second satellite feature
    on the same scale as the synthetic generator's, so a GEE export and the fallback
    are interchangeable at the model's input.
    """
    ndwi = np.asarray(ndwi, dtype=np.float32)
    return np.clip((ndwi + 0.2) / 0.5, 0.0, 1.0).astype(np.float32)


def synthetic_ndwi(
    rain_mm_h: np.ndarray, previous_wetness: Optional[np.ndarray] = None, step_minutes: int = 5
) -> NdwiSnapshot:
    """Derive NDWI from rainfall accumulation. Always labelled as a fallback.

    Mirrors the accumulator in ``ml/data/chennai_synthetic.py`` so the offline demo and
    the training data agree on what "wet" means. Drainage differs per segment, from the
    ``flood_history`` field in the config.
    """
    rain_mm_h = np.asarray(rain_mm_h, dtype=np.float32)
    if rain_mm_h.ndim == 0:
        rain_mm_h = np.full(len(SEGMENTS), float(rain_mm_h), dtype=np.float32)

    # These three constants MUST stay identical to ml/data/chennai_synthetic.py, or the
    # offline demo and the training data would disagree about what "wet" means.
    # A lower drainage exponent means slower decay, so "severe" flood history == water
    # lingers longest.
    drainage_speed = np.array(
        [{"severe": 0.35, "moderate": 0.7, "low": 1.0}[s.flood_history] for s in SEGMENTS],
        dtype=np.float32,
    )
    decay = float(np.exp(-step_minutes / 90.0))
    accumulation_rate = 0.00068

    previous = (
        np.zeros(len(SEGMENTS), dtype=np.float32)
        if previous_wetness is None
        else np.asarray(previous_wetness, dtype=np.float32)
    )

    accumulator = previous * (decay**drainage_speed) + rain_mm_h * accumulation_rate
    accumulator = np.clip(accumulator, 0.0, 1.0)
    ndwi = (-0.18 + 0.42 * np.tanh(accumulator * 1.4)).astype(np.float32)

    return NdwiSnapshot(
        ndwi=ndwi,
        wetness=accumulator.astype(np.float32),
        source="heuristic_fallback",
        captured=None,
        notes=[
            "Synthetic NDWI derived from rainfall accumulation, not a Sentinel-2 observation.",
        ],
    )


def get_ndwi(
    rain_mm_h: Optional[np.ndarray] = None,
    date: Optional[str] = None,
    allow_gee: bool = True,
) -> NdwiSnapshot:
    """Best available NDWI: live GEE, then a local export, then the labelled fallback."""
    if allow_gee:
        snapshot = fetch_ndwi_from_gee(date=date)
        if snapshot is not None:
            return snapshot

    snapshot = load_local_export()
    if snapshot is not None:
        return snapshot

    return synthetic_ndwi(
        np.zeros(len(SEGMENTS), dtype=np.float32) if rain_mm_h is None else rain_mm_h
    )


# The Earth Engine Code Editor snippet referenced by README section 7. Kept here so the
# code and the documentation cannot drift apart.
GEE_CODE_EDITOR_SNIPPET = """\
// monsoonplus -- export per-segment NDWI from Sentinel-2.
// Paste into https://code.earthengine.google.com, press Run, then open the Tasks tab.
var segments = ee.FeatureCollection([
%s
]);
var scene = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2024-10-01', '2024-12-31')
  .filterBounds(segments)
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();
var ndwi = ee.Image(scene).normalizedDifference(['B3', 'B8']).rename('ndwi');
var sampled = ndwi.reduceRegions({
  collection: segments.map(function (f) { return f.buffer(300); }),
  reducer: ee.Reducer.mean(),
  scale: 10
});
Export.table.toDrive({
  collection: sampled,
  description: 'ndwi_chennai',
  fileFormat: 'CSV',
  selectors: ['segment_id', 'ndwi']
});
"""


def gee_snippet() -> str:
    """Render the Code Editor snippet with this project's actual segment midpoints."""
    lines = []
    for segment, (lat, lon) in zip(SEGMENTS, segment_midpoints()):
        lines.append(
            f"  ee.Feature(ee.Geometry.Point([{lon:.5f}, {lat:.5f}]), "
            f"{{segment_id: '{segment.id}'}}),"
        )
    return GEE_CODE_EDITOR_SNIPPET % "\n".join(lines)


if __name__ == "__main__":
    print(f"Earth Engine authenticated: {earth_engine_available()}")
    # Accumulate over 4 hours of 45 mm/h rain, as the collector does step by step.
    # One isolated step would give every segment the same value; the per-segment
    # drainage rates only diverge once there is wetness to carry forward.
    # 2 hours of 45 mm/h rain, then 5 hours dry. During the rain every segment
    # saturates alike; the per-segment drainage rates (from flood_history in the
    # config) only separate them once the rain stops -- which is exactly the lag the
    # satellite modality is there to capture.
    rain = np.full(len(SEGMENTS), 45.0, dtype=np.float32)
    dry = np.zeros(len(SEGMENTS), dtype=np.float32)
    snapshot = synthetic_ndwi(rain)
    for _ in range(23):
        snapshot = synthetic_ndwi(rain, previous_wetness=snapshot.wetness)
    for _ in range(60):
        snapshot = synthetic_ndwi(dry, previous_wetness=snapshot.wetness)
    print("2 h of 45 mm/h rain, then 5 h dry -- marsh-side roads stay wet longest:")
    print(f"source  : {snapshot.source}  (measured={snapshot.is_measured})")
    for note in snapshot.notes:
        print(f"note    : {note}")
    print(f"\n{'segment':<6}{'ndwi':>9}{'wetness':>10}  flood likely")
    for row, segment in zip(snapshot.to_dict()["per_segment"], SEGMENTS):
        print(
            f"{row['segment_id']:<6}{row['ndwi']:>9.4f}{row['wetness']:>10.4f}"
            f"  {'yes' if row['flood_likely'] else 'no':<3}"
            f"  ({segment.flood_history} drainage)"
        )
    print("\n--- Earth Engine Code Editor snippet (README section 7) ---")
    print(gee_snippet()[:420] + "...")
