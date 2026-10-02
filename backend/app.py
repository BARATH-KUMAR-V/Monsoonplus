"""Optional local FastAPI service for Live mode.

This backend is **never deployed**. PyTorch plus PyTorch Geometric is far too large for
a Vercel or Netlify serverless function, and the deployed site does not need it: the
built frontend reads committed JSON and works entirely on its own. This process exists
so you can run Live mode on your own machine.

    uvicorn backend.app:app --reload --port 8000

A note on the POST endpoints
----------------------------
FastAPI reads a plain scalar parameter from the **query string**, not from the JSON
body. A React caller doing `fetch(url, {method:'POST', body: JSON.stringify({...}),
headers:{'Content-Type':'application/json'}})` therefore gets a 422, or worse, silently
has its values ignored. Every endpoint here that the frontend POSTs to declares an
explicit Pydantic body model, which is the fix.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.segments import (
    ATTRIBUTIONS,
    CITY,
    HISTORY_STEPS,
    HORIZONS_MINUTES,
    JUNCTIONS,
    SEGMENTS,
    SEGMENT_BY_ID,
    rain_label,
)
from ml.data.graph import chennai_adjacency, dense_to_edge_index
from ml.data.ndwi import get_ndwi, synthetic_ndwi
from ml.data.weather import fetch_air_quality, fetch_forecast, per_segment_rain
from ml.models.monsoonplus_net import load_checkpoint
from ml.training.train import CHECKPOINT_DIR

# ---------------------------------------------------------------- schemas --


class PredictRequest(BaseModel):
    """Body model for POST /api/predict.

    Declared as a Pydantic model on purpose: without it FastAPI would look for
    ``rain_mm_h`` in the query string and ignore the JSON the frontend sends.
    """

    rain_mm_h: float = Field(0.0, ge=0, le=200, description="Rainfall in mm/h")
    minutes_ahead: int = Field(30, ge=0, le=60, description="Forecast offset")
    satellite_enabled: bool = Field(True, description="Include the satellite modality")
    segment_ids: Optional[list[str]] = Field(
        None, description="Subset of segments; all of them when omitted"
    )


class SegmentPrediction(BaseModel):
    segment_id: str
    name: str
    speed_now_kmh: float
    predicted_kmh: dict[str, float]
    rain_mm_h: float
    ndwi: Optional[float]
    wetness: Optional[float]
    gate: dict[str, float]


class PredictResponse(BaseModel):
    generated_at: str
    rain_mm_h: float
    minutes_ahead: int
    satellite_enabled: bool
    horizons_minutes: list[int]
    per_segment: list[SegmentPrediction]
    note: str


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    checkpoint: Optional[str]
    segments: int
    junctions: int
    torch_version: str
    uptime_seconds: float


# ------------------------------------------------------------------ state --

STATE: dict[str, Any] = {"model": None, "meta": {}, "started": time.time(), "checkpoint": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the checkpoint once at startup.

    A missing checkpoint is NOT fatal: the service still serves weather and health, and
    /api/predict returns a clear 503 telling you to train first.
    """
    path = CHECKPOINT_DIR / "monsoonplus_final.pt"
    if path.exists():
        try:
            model, meta = load_checkpoint(path)
            STATE["model"] = model
            STATE["meta"] = meta
            STATE["checkpoint"] = path.name
            print(f"[backend] loaded {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[backend] could not load checkpoint: {exc}")
    else:
        print(f"[backend] no checkpoint at {path} -- /api/predict will return 503")

    edge_index, edge_weight = dense_to_edge_index(chennai_adjacency())
    STATE["edge_index"] = edge_index
    STATE["edge_weight"] = edge_weight
    yield


app = FastAPI(
    title="monsoonplus local API",
    version="1.0.0",
    description="Optional local service for Live mode. Never deployed to a serverless host.",
    lifespan=lifespan,
)

# The frontend runs on a different port in dev. Loopback only -- this service is not
# meant to be exposed beyond the machine it runs on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- routes --


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        model_loaded=STATE["model"] is not None,
        checkpoint=STATE["checkpoint"],
        segments=len(SEGMENTS),
        junctions=len(JUNCTIONS),
        torch_version=torch.__version__,
        uptime_seconds=round(time.time() - STATE["started"], 1),
    )


@app.get("/api/network")
def network() -> dict:
    """The road network, straight from the single source of truth."""
    return {
        "city": CITY,
        "segments": [
            {
                "id": s.id,
                "name": s.name,
                "a": s.a,
                "b": s.b,
                "free_flow_kmh": s.free_flow_kmh,
                "flood_history": s.flood_history,
            }
            for s in SEGMENTS
        ],
        "junctions": [
            {"id": j.id, "name": j.name, "lat": j.lat, "lon": j.lon} for j in JUNCTIONS
        ],
        "attributions": ATTRIBUTIONS,
    }


@app.get("/api/weather")
def weather(hours: int = Query(24, ge=1, le=168)) -> dict:
    """Hourly forecast from Open-Meteo. No API key is required or accepted."""
    series = fetch_forecast(hours=hours)
    return {
        "source": series.source,
        "is_measured": series.is_measured,
        "notes": series.notes,
        "hourly": [
            {
                "time": series.time[i],
                "precipitation": float(series.precipitation_mm_h[i]),
                "temperature": float(series.temperature_c[i]),
                "wind": float(series.wind_kmh[i]),
                "humidity": float(series.humidity_pct[i]),
                "visibility": float(series.visibility_m[i]),
            }
            for i in range(len(series.time))
        ],
    }


@app.get("/api/air-quality")
def air_quality() -> dict:
    return fetch_air_quality()


def _build_inputs(rain_mm_h: float, satellite_enabled: bool):
    """Assemble a single model input batch at a steady rainfall level."""
    num_nodes = len(SEGMENTS)
    free_flow = np.array([s.free_flow_kmh for s in SEGMENTS], dtype=np.float32)
    sensitivity = np.array([s.rain_sensitivity for s in SEGMENTS], dtype=np.float32)
    congestion = np.array([s.base_congestion for s in SEGMENTS], dtype=np.float32)
    rain_bias = np.array([s.rain_bias for s in SEGMENTS], dtype=np.float32)

    segment_rain = np.clip(rain_mm_h * rain_bias, 0, 130).astype(np.float32)

    snapshot = synthetic_ndwi(segment_rain)
    for _ in range(HISTORY_STEPS):
        snapshot = synthetic_ndwi(segment_rain, previous_wetness=snapshot.wetness)

    speed_now = np.clip(
        free_flow
        - free_flow * congestion
        - free_flow * sensitivity * np.tanh(segment_rain / 45.0) * 0.62
        - free_flow * sensitivity * snapshot.wetness * 0.45,
        3.0,
        None,
    ).astype(np.float32)

    traffic = np.stack(
        [
            np.tile(speed_now, (HISTORY_STEPS, 1)),
            np.tile(free_flow, (HISTORY_STEPS, 1)),
            np.ones((HISTORY_STEPS, num_nodes), dtype=np.float32),
        ],
        axis=-1,
    )
    weather_input = np.stack(
        [
            np.tile(segment_rain, (HISTORY_STEPS, 1)),
            np.full((HISTORY_STEPS, num_nodes), 27.5, dtype=np.float32),
            np.tile(9.0 + 0.22 * segment_rain, (HISTORY_STEPS, 1)),
        ],
        axis=-1,
    )
    satellite = np.stack([snapshot.ndwi, snapshot.wetness], axis=-1).astype(np.float32)
    if not satellite_enabled:
        satellite = np.zeros_like(satellite)

    return speed_now, segment_rain, snapshot, traffic, weather_input, satellite


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Forecast speeds at the configured horizons.

    Takes a JSON **body** (see the module docstring for why that matters).
    """
    model = STATE.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained checkpoint loaded. Run `python -m ml.training.train` "
                "(or `--quick`) from the repository root, then restart this service."
            ),
        )

    speed_now, segment_rain, snapshot, traffic, weather_input, satellite = _build_inputs(
        request.rain_mm_h, request.satellite_enabled
    )

    availability = torch.tensor(
        [[1.0, 1.0, 1.0 if request.satellite_enabled else 0.0]], dtype=torch.float32
    )
    with torch.no_grad():
        out = model(
            torch.from_numpy(traffic.transpose(1, 0, 2)).unsqueeze(0),
            torch.from_numpy(weather_input.transpose(1, 0, 2)).unsqueeze(0),
            torch.from_numpy(satellite).unsqueeze(0),
            STATE["edge_index"],
            STATE["edge_weight"],
            availability=availability,
        )
    prediction = out.prediction[0].numpy()
    gate = out.gate[0].numpy()

    wanted = set(request.segment_ids) if request.segment_ids else None
    rows: list[SegmentPrediction] = []
    for index, segment in enumerate(SEGMENTS):
        if wanted and segment.id not in wanted:
            continue
        rows.append(
            SegmentPrediction(
                segment_id=segment.id,
                name=segment.name,
                speed_now_kmh=round(float(speed_now[index]), 2),
                predicted_kmh={
                    f"t+{h}": round(float(prediction[index, i]), 2)
                    for i, h in enumerate(HORIZONS_MINUTES)
                },
                rain_mm_h=round(float(segment_rain[index]), 2),
                ndwi=round(float(snapshot.ndwi[index]), 4) if request.satellite_enabled else None,
                wetness=round(float(snapshot.wetness[index]), 4)
                if request.satellite_enabled
                else None,
                gate={
                    "traffic": round(float(gate[index, 0]) * 100, 1),
                    "weather": round(float(gate[index, 1]) * 100, 1),
                    "satellite": round(float(gate[index, 2]) * 100, 1),
                },
            )
        )

    return PredictResponse(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rain_mm_h=request.rain_mm_h,
        minutes_ahead=request.minutes_ahead,
        satellite_enabled=request.satellite_enabled,
        horizons_minutes=HORIZONS_MINUTES,
        per_segment=rows,
        note=(
            "Model estimate from MonsoonPlusNet. The traffic history is reconstructed from a "
            "steady-state assumption at the requested rainfall, not measured. Use the collector "
            "for genuinely live traffic."
        ),
    )


@app.get("/api/live")
def live() -> dict:
    """What Live mode on the frontend polls.

    Combines real Open-Meteo weather with a model forecast. Traffic is live only when
    the collector has written a recent snapshot; otherwise this endpoint says so in
    ``sources.traffic`` and the UI labels it.
    """
    from collector.store import latest_snapshot

    forecast = fetch_forecast(hours=12)
    rain_now = float(forecast.precipitation_mm_h[0]) if len(forecast.precipitation_mm_h) else 0.0

    snapshot = latest_snapshot()
    traffic_source = snapshot.get("source") if snapshot else None
    # A snapshot existing is not the same as traffic having been collected. Without a
    # TomTom key the collector still writes a snapshot, with an empty traffic list and
    # source "not_configured" -- reporting ok=true for that would be a quiet lie.
    traffic_rows = (snapshot or {}).get("per_segment") or []
    traffic_ok = bool(
        traffic_rows
        and traffic_source not in (None, "not_configured", "budget_exhausted")
        and any(row.get("ok") for row in traffic_rows)
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "live",
        "rain_mm_h": rain_now,
        "weather_label": rain_label(rain_now),
        "sources": {
            "traffic": {
                "name": "TomTom Flow Segment Data",
                "source": traffic_source or "not collected",
                "fetched_at": snapshot.get("fetched_at") if snapshot else None,
                "ok": traffic_ok,
            },
            "weather": {
                "name": "Open-Meteo",
                "source": forecast.source,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ok": forecast.is_measured,
            },
            "satellite": {
                "name": "Sentinel-2 NDWI",
                "source": get_ndwi(rain_mm_h=np.full(len(SEGMENTS), rain_now)).source,
                "fetched_at": None,
                "ok": True,
            },
        },
        "weather": {
            "source": forecast.source,
            "hourly": [
                {
                    "time": forecast.time[i],
                    "precipitation": float(forecast.precipitation_mm_h[i]),
                    "temperature": float(forecast.temperature_c[i]),
                    "wind": float(forecast.wind_kmh[i]),
                    "humidity": float(forecast.humidity_pct[i]),
                    "visibility": float(forecast.visibility_m[i]),
                }
                for i in range(len(forecast.time))
            ],
        },
    }

    if STATE.get("model") is not None:
        forecast_rows = predict(
            PredictRequest(rain_mm_h=rain_now, minutes_ahead=30, satellite_enabled=True)
        )
        payload["per_segment"] = [row.model_dump() for row in forecast_rows.per_segment]
    else:
        payload["per_segment"] = []
        payload["note"] = "No checkpoint loaded; predictions unavailable."

    if snapshot:
        payload["observed_traffic"] = snapshot.get("per_segment", [])

    return payload


@app.get("/")
def index() -> dict:
    return {
        "service": "monsoonplus local API",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/network", "/api/weather", "/api/air-quality", "/api/predict", "/api/live"],
        "note": "Optional. The deployed frontend does not need this service.",
    }
