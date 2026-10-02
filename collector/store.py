"""Where the collector keeps what it gathers.

Two things live here:

* ``snapshots/latest.json`` -- the most recent traffic+weather reading, which the
  backend's ``/api/live`` serves.
* ``logs/YYYY_MM_DD_HH_MM_<label>.json`` -- the append-only record of what the model
  predicted versus what actually happened. These are what Log mode and the real
  predicted-vs-actual replay read.

A tiny SQLite database tracks the monthly TomTom request count so the budget guard
cannot be fooled by restarting the process.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.segments import REPO_ROOT

LOG_DIR = REPO_ROOT / "logs"
SNAPSHOT_DIR = REPO_ROOT / "collector" / "snapshots"
DB_PATH = REPO_ROOT / "collector" / "collector.db"


def _clean(value: Any) -> Any:
    """NaN and Infinity are not valid JSON. Everything written here goes through this."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else round(number, 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(payload), indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    json.loads(text)  # fail here rather than in the browser
    return path


# ------------------------------------------------------------- snapshots --


def save_snapshot(payload: dict) -> Path:
    return write_json(SNAPSHOT_DIR / "latest.json", payload)


def latest_snapshot() -> Optional[dict]:
    path = SNAPSHOT_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ logs --


def log_filename(when: datetime, label: str) -> str:
    return f"{when.strftime('%Y_%m_%d_%H_%M')}_{label}.json"


def save_log(payload: dict, when: Optional[datetime] = None, label: str = "clear") -> Path:
    when = when or datetime.now(timezone.utc)
    return write_json(LOG_DIR / log_filename(when, label), payload)


def list_logs() -> list[dict]:
    """Every collected log file, newest first."""
    if not LOG_DIR.exists():
        return []
    entries = []
    for path in sorted(LOG_DIR.glob("*.json"), reverse=True):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 6:
            continue
        date = "-".join(parts[0:3])
        label = "_".join(parts[5:])
        entries.append(
            {
                "file": path.name,
                "date": date,
                "time": f"{parts[3]}:{parts[4]}",
                "label": label,
                "bytes": path.stat().st_size,
            }
        )
    return entries


def build_logs_index() -> dict:
    """Index the frontend fetches to populate the Log-mode date picker."""
    entries = list_logs()
    labels = [e["label"] for e in entries]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(entries),
        "entries": entries,
        "label_counts": {label: labels.count(label) for label in set(labels)},
        "note": (
            "Written by collector/store.py. Page 5's real replay and the pattern cards "
            "need several rainy days plus a clear day before they say anything useful."
        ),
    }


def publish_logs_to_frontend() -> Optional[Path]:
    """Copy logs and their index into frontend/public/data so the static site can read them."""
    import shutil

    entries = list_logs()
    target_dir = REPO_ROOT / "frontend" / "public" / "data" / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copy2(LOG_DIR / entry["file"], target_dir / entry["file"])
    return write_json(
        REPO_ROOT / "frontend" / "public" / "data" / "logs_index.json", build_logs_index()
    )


# ---------------------------------------------------------- budget guard --


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS request_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " provider TEXT NOT NULL,"
        " month TEXT NOT NULL,"
        " requested_at TEXT NOT NULL,"
        " count INTEGER NOT NULL DEFAULT 1)"
    )
    connection.commit()
    return connection


def record_requests(provider: str, count: int = 1) -> None:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with _connect() as connection:
        connection.execute(
            "INSERT INTO request_log (provider, month, requested_at, count) VALUES (?, ?, ?, ?)",
            (provider, month, datetime.now(timezone.utc).isoformat(timespec="seconds"), count),
        )


def requests_this_month(provider: str) -> int:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with _connect() as connection:
        row = connection.execute(
            "SELECT COALESCE(SUM(count), 0) FROM request_log WHERE provider = ? AND month = ?",
            (provider, month),
        ).fetchone()
    return int(row[0] if row else 0)


def budget_remaining(provider: str, monthly_limit: int) -> int:
    return max(0, monthly_limit - requests_this_month(provider))
