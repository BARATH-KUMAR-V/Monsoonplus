"""Live data collector: TomTom traffic + Open-Meteo weather -> snapshots and logs.

Everything here is optional. The site works fully without it; this is what turns on
Live mode and, over time, fills Log mode with real predicted-vs-actual records.

Budget
------
TomTom's free tier is, at the time of writing, around **20,000 requests per month**
(not 2,500 per day -- that was an older tier, and the README says to re-check because
the terms move). One poll costs one request per segment, so 10 segments = 10 requests.

The schedule below is deliberately uneven, because a monsoon tool only needs fine time
resolution when traffic is actually changing:

    peak hours  (07:00-11:00, 16:00-21:00)   every 15 min
    daytime     (11:00-16:00)                every 30 min
    night       (21:00-07:00)                every 60 min

That is 16 + 10 + 10 = 36 polls a day, 360 requests a day, about 10,800 a month -- a
little over half the allowance, leaving room for manual runs. ``--budget`` enforces a
hard monthly ceiling regardless, tracked in SQLite so restarting does not reset it.

Usage
-----
    python -m collector.collect --once          # single poll, then exit
    python -m collector.collect                 # run on the schedule, forever
    python -m collector.collect --dry-run       # show the plan, make no requests
    python -m collector.collect --publish       # copy logs into the frontend

The TomTom key goes in a ``.env`` file at the repository root:

    TOMTOM_API_KEY=your_key_here

With no key the collector still runs: it records Open-Meteo weather and the model's
predictions, and marks traffic as unavailable. Nothing is faked.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.segments import (
    HORIZONS_MINUTES,
    REPO_ROOT,
    SEGMENTS,
    segment_midpoints,
    rain_label,
)
from collector.store import (
    budget_remaining,
    build_logs_index,
    publish_logs_to_frontend,
    record_requests,
    requests_this_month,
    save_log,
    save_snapshot,
)

TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
DEFAULT_MONTHLY_BUDGET = 18000  # headroom under the ~20k free tier

PEAK_HOURS = set(range(7, 11)) | set(range(16, 21))
DAY_HOURS = set(range(11, 16))


def load_env() -> None:
    """Read .env at the repo root. No external dependency, no key ever logged."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def poll_interval_minutes(when: Optional[datetime] = None) -> int:
    hour = (when or datetime.now()).hour
    if hour in PEAK_HOURS:
        return 15
    if hour in DAY_HOURS:
        return 30
    return 60


def fetch_tomtom_segment(latitude: float, longitude: float, api_key: str, timeout: int = 15) -> Optional[dict]:
    """One TomTom Flow Segment Data reading.

    The key travels in the query string because that is the only form TomTom accepts.
    It is never written to a log line, a snapshot or a committed file.
    """
    query = urllib.parse.urlencode(
        {"point": f"{latitude},{longitude}", "unit": "KMPH", "key": api_key}
    )
    request = urllib.request.Request(
        f"{TOMTOM_URL}?{query}", headers={"User-Agent": "monsoonplus/1.0 (student project)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - never let one dead segment kill the poll
        # Deliberately prints the exception type, not the URL, so the key cannot leak.
        print(f"  [tomtom] request failed ({type(exc).__name__})")
        return None

    data = payload.get("flowSegmentData") or {}
    if not data:
        return None
    return {
        "speed_kmh": float(data.get("currentSpeed", 0.0)),
        "free_flow_kmh": float(data.get("freeFlowSpeed", 0.0)),
        "confidence": float(data.get("confidence", 0.0)),
        "road_closure": bool(data.get("roadClosure", False)),
    }


def collect_traffic(api_key: Optional[str], budget: int) -> dict:
    """Poll every segment, respecting the monthly budget."""
    if not api_key:
        return {
            "source": "not_configured",
            "ok": False,
            "note": (
                "No TOMTOM_API_KEY in .env. Traffic is not collected. Weather and model "
                "predictions still are."
            ),
            "per_segment": [],
        }

    remaining = budget_remaining("tomtom", budget)
    if remaining < len(SEGMENTS):
        return {
            "source": "budget_exhausted",
            "ok": False,
            "note": (
                f"Monthly TomTom budget reached ({requests_this_month('tomtom')} of {budget}). "
                "Traffic collection paused until next month."
            ),
            "per_segment": [],
        }

    rows = []
    used = 0
    for segment, (latitude, longitude) in zip(SEGMENTS, segment_midpoints()):
        reading = fetch_tomtom_segment(latitude, longitude, api_key)
        used += 1
        if reading is None:
            rows.append({"segment_id": segment.id, "ok": False})
            continue
        rows.append(
            {
                "segment_id": segment.id,
                "name": segment.name,
                "ok": True,
                "speed_kmh": round(reading["speed_kmh"], 2),
                "free_flow_kmh": round(reading["free_flow_kmh"], 2),
                "confidence": round(reading["confidence"], 3),
                "road_closure": reading["road_closure"],
            }
        )

    record_requests("tomtom", used)
    good = [r for r in rows if r.get("ok")]
    return {
        "source": "tomtom_flow_segment_data",
        "ok": bool(good),
        "requests_used": used,
        "requests_this_month": requests_this_month("tomtom"),
        "monthly_budget": budget,
        "per_segment": rows,
    }


def collect_weather() -> dict:
    from ml.data.weather import fetch_forecast

    series = fetch_forecast(hours=12)
    return {
        "source": series.source,
        "ok": series.is_measured,
        "rain_mm_h": float(series.precipitation_mm_h[0]) if len(series.precipitation_mm_h) else 0.0,
        "temp_c": float(series.temperature_c[0]) if len(series.temperature_c) else None,
        "wind_kmh": float(series.wind_kmh[0]) if len(series.wind_kmh) else None,
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
        "notes": series.notes,
    }


def collect_satellite(rain_mm_h: float) -> dict:
    from ml.data.ndwi import get_ndwi

    snapshot = get_ndwi(rain_mm_h=np.full(len(SEGMENTS), rain_mm_h))
    return snapshot.to_dict()


def model_predictions(rain_mm_h: float) -> dict:
    """Ask the trained checkpoint what it expects, so the log can be scored later."""
    try:
        import torch

        from ml.data.graph import chennai_adjacency, dense_to_edge_index
        from ml.models.monsoonplus_net import load_checkpoint
        from ml.training.train import CHECKPOINT_DIR
        from backend.app import _build_inputs

        path = CHECKPOINT_DIR / "monsoonplus_final.pt"
        if not path.exists():
            return {"ok": False, "note": "no checkpoint; run ml.training.train first"}

        model, _ = load_checkpoint(path)
        edge_index, edge_weight = dense_to_edge_index(chennai_adjacency())
        _, segment_rain, _, traffic, weather, satellite = _build_inputs(rain_mm_h, True)

        with torch.no_grad():
            out = model(
                torch.from_numpy(traffic.transpose(1, 0, 2)).unsqueeze(0),
                torch.from_numpy(weather.transpose(1, 0, 2)).unsqueeze(0),
                torch.from_numpy(satellite).unsqueeze(0),
                edge_index,
                edge_weight,
                availability=torch.tensor([[1.0, 1.0, 1.0]]),
            )
        prediction = out.prediction[0].numpy()
        gate = out.gate[0].numpy()
        return {
            "ok": True,
            "per_segment": [
                {
                    "segment_id": segment.id,
                    "predicted_kmh": {
                        f"t+{h}": round(float(prediction[i, j]), 2)
                        for j, h in enumerate(HORIZONS_MINUTES)
                    },
                    "gate": {
                        "traffic": round(float(gate[i, 0]) * 100, 1),
                        "weather": round(float(gate[i, 1]) * 100, 1),
                        "satellite": round(float(gate[i, 2]) * 100, 1),
                    },
                }
                for i, segment in enumerate(SEGMENTS)
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "note": f"prediction unavailable ({type(exc).__name__}: {exc})"}


def run_once(api_key: Optional[str], budget: int, write_log: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    print(f"[collector] poll at {now.isoformat(timespec='seconds')}")

    weather = collect_weather()
    rain_now = weather["rain_mm_h"]
    label = rain_label(rain_now)
    print(f"  weather  {weather['source']}  rain {rain_now:.1f} mm/h -> {label}")

    traffic = collect_traffic(api_key, budget)
    print(f"  traffic  {traffic['source']}  ok={traffic['ok']}")

    satellite = collect_satellite(rain_now)
    print(f"  satellite {satellite['source']}")

    predictions = model_predictions(rain_now)
    print(f"  model    ok={predictions.get('ok')}")

    snapshot = {
        "fetched_at": now.isoformat(timespec="seconds"),
        "source": traffic["source"],
        "rain_mm_h": rain_now,
        "weather_label": label,
        "per_segment": traffic["per_segment"],
        "weather": weather,
        "satellite": satellite,
        "predictions": predictions,
    }
    save_snapshot(snapshot)

    if write_log:
        # A log pairs the prediction made now with the observation made now. Scoring
        # predicted-vs-actual needs a later observation, which ml/training/score_logs.py
        # does by joining a log's t+15 prediction to the observation 15 minutes later.
        observed = {
            row["segment_id"]: row.get("speed_kmh")
            for row in traffic["per_segment"]
            if row.get("ok")
        }
        log = {
            "schema_version": "1.0.0",
            "collected_at": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "label": label,
            "rain_mm_h": rain_now,
            "traffic_source": traffic["source"],
            "traffic_ok": traffic["ok"],
            "per_segment": [
                {
                    "segment_id": segment.id,
                    "name": segment.name,
                    "observed_speed_kmh": observed.get(segment.id),
                    "predicted_kmh": next(
                        (
                            p["predicted_kmh"]
                            for p in predictions.get("per_segment", [])
                            if p["segment_id"] == segment.id
                        ),
                        None,
                    ),
                    "rain_mm_h": round(rain_now * segment.rain_bias, 2),
                    "ndwi": next(
                        (
                            row["ndwi"]
                            for row in satellite.get("per_segment", [])
                            if row["segment_id"] == segment.id
                        ),
                        None,
                    ),
                }
                for segment in SEGMENTS
            ],
            "note": (
                "One collection point. Predicted values are for t+15/30/60 from this "
                "instant; score them against the observations in the logs that follow."
            ),
        }
        path = save_log(log, when=now, label=label)
        print(f"  log      {path.name}")

    return snapshot


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="monsoonplus live collector")
    parser.add_argument("--once", action="store_true", help="single poll then exit")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, make no requests")
    parser.add_argument("--budget", type=int, default=DEFAULT_MONTHLY_BUDGET)
    parser.add_argument("--publish", action="store_true", help="copy logs into frontend/public/data")
    parser.add_argument("--no-log", action="store_true", help="snapshot only, write no log file")
    args = parser.parse_args(argv)

    load_env()
    api_key = os.environ.get("TOMTOM_API_KEY")

    if args.publish:
        path = publish_logs_to_frontend()
        index = build_logs_index()
        print(f"published {index['count']} log file(s) -> {path}")
        return

    if args.dry_run:
        interval = poll_interval_minutes()
        used = requests_this_month("tomtom")
        print("monsoonplus collector -- dry run")
        print(f"  TomTom key present : {bool(api_key)}")
        print(f"  segments to poll   : {len(SEGMENTS)} (= {len(SEGMENTS)} requests per poll)")
        print(f"  interval right now : every {interval} min")
        print(f"  requests this month: {used} of {args.budget}")
        print(f"  remaining          : {budget_remaining('tomtom', args.budget)}")
        print("  schedule           : 15 min peak / 30 min daytime / 60 min overnight")
        print("  estimated monthly  : ~10,800 requests at the default schedule")
        print("\nNo requests were made.")
        return

    if args.once:
        run_once(api_key, args.budget, write_log=not args.no_log)
        return

    print("[collector] running on schedule. Ctrl+C to stop.")
    try:
        while True:
            run_once(api_key, args.budget, write_log=not args.no_log)
            minutes = poll_interval_minutes()
            nxt = datetime.now() + timedelta(minutes=minutes)
            print(f"[collector] next poll in {minutes} min (at {nxt.strftime('%H:%M')})\n")
            time.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("\n[collector] stopped")


if __name__ == "__main__":
    main()
