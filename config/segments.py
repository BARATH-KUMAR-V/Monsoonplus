"""Typed access to ``config/segments.json`` -- the one source of truth.

Every module (model, training, export, backend, collector) imports the network from
here. Nothing re-declares a segment list of its own.

    from config.segments import NETWORK, SEGMENTS, JUNCTIONS, segment_lengths_km

Geometry (segment length, adjacency) is *derived* from the junction coordinates
rather than stored, so moving a junction updates everything downstream.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

CONFIG_PATH = Path(__file__).resolve().parent / "segments.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_network() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


NETWORK: Dict[str, Any] = load_network()

CITY: Dict[str, Any] = NETWORK["city"]
MODEL_CFG: Dict[str, Any] = NETWORK["model"]
THRESHOLDS: Dict[str, Any] = NETWORK["thresholds"]
ATTRIBUTIONS: List[str] = NETWORK["attributions"]
MONSOON_CALENDAR: Dict[str, Any] = NETWORK["monsoon_calendar"]
LANDMARKS: List[Dict[str, Any]] = NETWORK["landmarks"]


@dataclass(frozen=True)
class Junction:
    id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    note: str = ""

    @property
    def latlon(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass(frozen=True)
class Segment:
    id: str
    name: str
    a: str
    b: str
    free_flow_kmh: float
    rain_sensitivity: float
    base_congestion: float
    rain_bias: float
    lanes: int
    flood_history: str
    note: str = ""


JUNCTIONS: List[Junction] = [Junction(**j) for j in NETWORK["junctions"]]
SEGMENTS: List[Segment] = [Segment(**s) for s in NETWORK["segments"]]

JUNCTION_BY_ID: Dict[str, Junction] = {j.id: j for j in JUNCTIONS}
SEGMENT_BY_ID: Dict[str, Segment] = {s.id: s for s in SEGMENTS}

SEGMENT_IDS: List[str] = [s.id for s in SEGMENTS]
JUNCTION_IDS: List[str] = [j.id for j in JUNCTIONS]

NUM_NODES: int = len(SEGMENTS)  # graph nodes are road segments, not junctions
HISTORY_STEPS: int = int(MODEL_CFG["history_steps"])
STEP_MINUTES: int = int(MODEL_CFG["step_minutes"])
HORIZONS_MINUTES: List[int] = list(MODEL_CFG["horizons_minutes"])
HORIZON_STEPS: List[int] = [h // STEP_MINUTES for h in HORIZONS_MINUTES]
NUM_TRAFFIC_FEATURES: int = len(MODEL_CFG["traffic_features"])
NUM_WEATHER_FEATURES: int = len(MODEL_CFG["weather_features"])
NUM_SATELLITE_FEATURES: int = len(MODEL_CFG["satellite_features"])


def haversine_km(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) pairs."""
    r = math.pi / 180.0
    d_lat = (b[0] - a[0]) * r
    d_lon = (b[1] - a[1]) * r
    h = (
        math.sin(d_lat / 2) ** 2
        + math.cos(a[0] * r) * math.cos(b[0] * r) * math.sin(d_lon / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))


@lru_cache(maxsize=1)
def segment_lengths_km() -> Tuple[float, ...]:
    """Road length per segment, straight line inflated by the detour factor.

    Real roads are not straight lines between junctions, so the straight-line
    distance is scaled by ``road_length_detour_factor`` to approximate driven
    distance. Same number the frontend computes, from the same constants.
    """
    factor = float(MODEL_CFG["road_length_detour_factor"])
    return tuple(
        haversine_km(JUNCTION_BY_ID[s.a].latlon, JUNCTION_BY_ID[s.b].latlon) * factor
        for s in SEGMENTS
    )


@lru_cache(maxsize=1)
def segment_midpoints() -> Tuple[Tuple[float, float], ...]:
    out = []
    for s in SEGMENTS:
        ja, jb = JUNCTION_BY_ID[s.a], JUNCTION_BY_ID[s.b]
        out.append(((ja.lat + jb.lat) / 2.0, (ja.lon + jb.lon) / 2.0))
    return tuple(out)


@lru_cache(maxsize=1)
def junction_adjacency() -> Dict[str, List[Tuple[str, str]]]:
    """junction id -> list of (neighbour junction id, segment id). For routing."""
    adj: Dict[str, List[Tuple[str, str]]] = {j.id: [] for j in JUNCTIONS}
    for s in SEGMENTS:
        adj[s.a].append((s.b, s.id))
        adj[s.b].append((s.a, s.id))
    return adj


def monsoon_regime(month: int) -> Dict[str, str]:
    """Which Chennai monsoon regime a calendar month falls in."""
    for regime in MONSOON_CALENDAR["regimes"]:
        if month in regime["months"]:
            return {"label": regime["label"], "note": regime["note"]}
    return {"label": "Unknown", "note": ""}


def rain_label(rain_mm_h: float) -> str:
    """Ground-truth weather class used for cost-sensitive weighting and the
    rain-vs-clear evaluation split."""
    cfg = THRESHOLDS["rain_label_mm_h"]
    if rain_mm_h >= cfg["heavy_rain_min"]:
        return "heavy_rain"
    if rain_mm_h >= cfg["rain_min"]:
        return "rain"
    return "clear"


def sample_weight(rain_mm_h: float) -> float:
    """3x for rain, 8x for heavy rain. The cost-sensitive loss uses this."""
    return float(THRESHOLDS["rain_weights"][rain_label(rain_mm_h)])


def rain_bucket(rain_mm_h: float) -> str:
    b = THRESHOLDS["rain_buckets"]
    if rain_mm_h < b["dry_max"]:
        return "Dry"
    if rain_mm_h < b["light_max"]:
        return "Light"
    if rain_mm_h < b["moderate_max"]:
        return "Moderate"
    if rain_mm_h < b["heavy_max"]:
        return "Heavy"
    return "Extreme"


def validate() -> List[str]:
    """Sanity checks on the config. Returns a list of problems (empty == healthy).

    Called by the test suite and by ``python -m config.segments``.
    """
    problems: List[str] = []
    if len(SEGMENT_IDS) != len(set(SEGMENT_IDS)):
        problems.append("duplicate segment ids")
    if len(JUNCTION_IDS) != len(set(JUNCTION_IDS)):
        problems.append("duplicate junction ids")
    for s in SEGMENTS:
        for end in (s.a, s.b):
            if end not in JUNCTION_BY_ID:
                problems.append(f"segment {s.id} references unknown junction {end}")
        if s.a == s.b:
            problems.append(f"segment {s.id} is a self-loop")
        if not 0.0 <= s.rain_sensitivity <= 1.0:
            problems.append(f"segment {s.id} rain_sensitivity out of range")
        if s.free_flow_kmh <= 0:
            problems.append(f"segment {s.id} non-positive free_flow_kmh")
    for lm in LANDMARKS:
        if lm["junction"] not in JUNCTION_BY_ID:
            problems.append(f"landmark {lm['name']} maps to unknown junction")

    # Every junction must be reachable, or routing silently returns no route.
    adj = junction_adjacency()
    seen = {JUNCTION_IDS[0]}
    stack = [JUNCTION_IDS[0]]
    while stack:
        node = stack.pop()
        for nxt, _ in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != len(JUNCTION_IDS):
        problems.append(f"graph is not connected: unreachable {set(JUNCTION_IDS) - seen}")

    w = THRESHOLDS["impact_score_weights"]
    if abs(sum(w.values()) - 1.0) > 1e-9:
        problems.append("impact_score_weights do not sum to 1.0")
    for h in HORIZONS_MINUTES:
        if h % STEP_MINUTES:
            problems.append(f"horizon {h} is not a multiple of step_minutes")
    return problems


if __name__ == "__main__":
    issues = validate()
    lengths = segment_lengths_km()
    print(f"monsoonplus network: {NUM_NODES} segments, {len(JUNCTIONS)} junctions")
    for s, km in zip(SEGMENTS, lengths):
        print(f"  {s.id:>4}  {s.name:<34} {s.a}->{s.b}  {km:5.2f} km")
    print(f"horizons: {HORIZONS_MINUTES} min  (steps {HORIZON_STEPS}, {STEP_MINUTES} min each)")
    print("config valid" if not issues else "PROBLEMS: " + "; ".join(issues))
