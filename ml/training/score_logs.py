"""Score collected logs: join each prediction to the observation it was predicting.

A single log file cannot be scored. It records the model's t+15 / t+30 / t+60
predictions made at one instant, and the traffic observed at that same instant. To
score the t+15 prediction you need the observation from the log collected 15 minutes
later.

This script does that join across every file in ``logs/`` and writes
``frontend/public/data/logs_index.json``, which is what Page 5's real replay and the
pattern cards read.

    python -m ml.training.score_logs

With no logs it exits cleanly and says so. It never invents a number, and it refuses to
emit pattern statistics until enough rainy days exist to compute a spread.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from config.segments import HORIZONS_MINUTES, REPO_ROOT, SEGMENT_BY_ID
from collector.store import LOG_DIR, write_json

# A prediction and its observation rarely land exactly 15 minutes apart, because the
# collector polls on a 15/30/60-minute schedule and requests take time. Anything within
# this tolerance of the target offset counts as a match.
MATCH_TOLERANCE_MINUTES = 4

# Thresholds below which a statistic is not worth stating. Mirrors logs/README.md.
MIN_DAYS_FOR_RAIN_COMPARISON = 3   # at least 2 rainy + 1 clear
MIN_RAINY_DAYS_FOR_PATTERNS = 3


def load_logs(directory: Path = LOG_DIR) -> list[dict]:
    """Every log file, parsed, sorted by collection time."""
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop scoring
            print(f"[score] skipping {path.name}: {exc}")
            continue
        collected = payload.get("collected_at")
        if not collected:
            continue
        try:
            payload["_when"] = datetime.fromisoformat(collected)
        except ValueError:
            continue
        payload["_file"] = path.name
        entries.append(payload)
    entries.sort(key=lambda item: item["_when"])
    return entries


def observations_by_segment(entry: dict) -> dict[str, Optional[float]]:
    return {
        row["segment_id"]: row.get("observed_speed_kmh")
        for row in entry.get("per_segment", [])
    }


def score(entries: list[dict]) -> dict:
    """Match every prediction to its later observation and compute MAE.

    Returns a payload with per-horizon and per-label error, plus the matched pairs the
    frontend charts.
    """
    matched: list[dict] = []
    unmatched = 0

    for index, entry in enumerate(entries):
        issued = entry["_when"]
        for horizon in HORIZONS_MINUTES:
            target = issued + timedelta(minutes=horizon)

            # Find the log closest to the target time, within tolerance.
            best = None
            best_gap = None
            for later in entries[index + 1 :]:
                gap = abs((later["_when"] - target).total_seconds()) / 60.0
                if later["_when"] > target + timedelta(minutes=MATCH_TOLERANCE_MINUTES):
                    break  # entries are sorted; nothing later will be closer
                if best_gap is None or gap < best_gap:
                    best, best_gap = later, gap
            if best is None or best_gap is None or best_gap > MATCH_TOLERANCE_MINUTES:
                unmatched += 1
                continue

            actuals = observations_by_segment(best)
            for row in entry.get("per_segment", []):
                predicted_map = row.get("predicted_kmh") or {}
                predicted = predicted_map.get(f"t+{horizon}")
                actual = actuals.get(row["segment_id"])
                if predicted is None or actual is None:
                    continue
                matched.append(
                    {
                        "segment_id": row["segment_id"],
                        "date": entry.get("date"),
                        "time": entry.get("time"),
                        "horizon": horizon,
                        "label": entry.get("label", "clear"),
                        "rain_mm_h": entry.get("rain_mm_h"),
                        "predicted": float(predicted),
                        "actual": float(actual),
                        "abs_error": abs(float(predicted) - float(actual)),
                    }
                )

    if not matched:
        return {
            "scored": False,
            "matched_pairs": 0,
            "unmatched_predictions": unmatched,
            "reason": (
                "No prediction could be matched to a later observation. You need at "
                "least two consecutive polls, and traffic collection must be working "
                "(a TomTom key in .env)."
            ),
        }

    errors = np.array([m["abs_error"] for m in matched], dtype=np.float64)
    labels = np.array([m["label"] for m in matched])
    horizons = np.array([m["horizon"] for m in matched])

    def mae_where(mask: np.ndarray) -> Optional[float]:
        return round(float(errors[mask].mean()), 3) if mask.any() else None

    by_label = {label: mae_where(labels == label) for label in ("clear", "rain", "heavy_rain")}
    by_horizon = {f"t+{h}": mae_where(horizons == h) for h in HORIZONS_MINUTES}

    dates = sorted({m["date"] for m in matched if m.get("date")})
    rainy_dates = sorted(
        {m["date"] for m in matched if m["label"] in ("rain", "heavy_rain") and m.get("date")}
    )
    clear_dates = sorted({m["date"] for m in matched if m["label"] == "clear" and m.get("date")})

    notes: list[str] = []
    rain_comparison_ready = len(rainy_dates) >= 2 and len(clear_dates) >= 1
    if not rain_comparison_ready:
        notes.append(
            f"Rain-vs-clear comparison needs at least 2 rainy days and 1 clear day; "
            f"you have {len(rainy_dates)} rainy and {len(clear_dates)} clear. "
            f"The comparison is withheld rather than computed from too little data."
        )
    if len(rainy_dates) < MIN_RAINY_DAYS_FOR_PATTERNS:
        notes.append(
            f"Pattern cards need at least {MIN_RAINY_DAYS_FOR_PATTERNS} rainy days so a "
            f"spread can be computed; you have {len(rainy_dates)}."
        )

    per_segment = []
    for segment_id, segment in SEGMENT_BY_ID.items():
        mask = np.array([m["segment_id"] == segment_id for m in matched])
        if not mask.any():
            continue
        per_segment.append(
            {
                "segment_id": segment_id,
                "name": segment.name,
                "mae": round(float(errors[mask].mean()), 3),
                "samples": int(mask.sum()),
                "mae_rain": mae_where(mask & np.isin(labels, ["rain", "heavy_rain"])),
                "mae_clear": mae_where(mask & (labels == "clear")),
            }
        )

    return {
        "scored": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": len(matched),
        "unmatched_predictions": unmatched,
        "days": dates,
        "rainy_days": rainy_dates,
        "clear_days": clear_dates,
        "mae_overall": round(float(errors.mean()), 3),
        "mae_by_label": by_label,
        "mae_by_horizon": by_horizon,
        "rain_comparison_ready": rain_comparison_ready,
        "patterns_ready": len(rainy_dates) >= MIN_RAINY_DAYS_FOR_PATTERNS,
        "per_segment": per_segment,
        "samples": matched[:4000],  # cap so the committed file stays small
        "notes": notes,
    }


def build_index(entries: list[dict], scoring: dict) -> dict:
    """What the frontend's Log-mode date picker reads."""
    by_date: dict[str, dict] = {}
    for entry in entries:
        date = entry.get("date")
        if not date:
            continue
        bucket = by_date.setdefault(
            date, {"date": date, "file": entry["_file"], "polls": 0, "labels": {}}
        )
        bucket["polls"] += 1
        label = entry.get("label", "clear")
        bucket["labels"][label] = bucket["labels"].get(label, 0) + 1

    for bucket in by_date.values():
        # A day's headline label is whichever wet class dominates, else clear.
        labels = bucket["labels"]
        bucket["label"] = max(labels, key=labels.get) if labels else "clear"

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(by_date),
        "entries": sorted(by_date.values(), key=lambda b: b["date"], reverse=True),
        "scoring": scoring,
        "note": (
            "Written by ml/training/score_logs.py. Page 5's real replay and the pattern "
            "cards read this. Statistics are withheld, not estimated, below the data "
            "thresholds documented in logs/README.md."
        ),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score collected monsoonplus logs")
    parser.add_argument("--logs", default=str(LOG_DIR))
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "frontend" / "public" / "data" / "logs_index.json"),
    )
    args = parser.parse_args(argv)

    entries = load_logs(Path(args.logs))
    if not entries:
        print(
            f"No log files in {args.logs}.\n"
            "This is the expected state in a fresh clone. Collect some first:\n"
            "    python -m collector.collect --once\n"
            "Page 5 shows an explicit empty state until then -- nothing is fabricated."
        )
        return

    print(f"[score] {len(entries)} log file(s)")
    scoring = score(entries)

    if not scoring["scored"]:
        print(f"[score] {scoring['reason']}")
    else:
        print(f"[score] matched {scoring['matched_pairs']} prediction/observation pairs")
        print(f"[score] overall MAE {scoring['mae_overall']} km/h")
        for label, value in scoring["mae_by_label"].items():
            print(f"         {label:<11} {value if value is not None else 'no samples'}")
        for note in scoring["notes"]:
            print(f"[score] note: {note}")

    index = build_index(entries, scoring)
    path = write_json(Path(args.out), index)
    print(f"[score] wrote {path}")


if __name__ == "__main__":
    main()
