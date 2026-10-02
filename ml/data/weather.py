"""Weather from Open-Meteo, with an offline fallback.

Open-Meteo needs **no API key**. There is nothing to register for, no credit card and
no token to paste anywhere -- the archive and forecast endpoints are open for
non-commercial use. If any code here ever starts asking for a key, that is a bug.

Two endpoints are used:

  * archive-api.open-meteo.com  -- ERA5 reanalysis, for historical training windows
  * api.open-meteo.com          -- forecast, for Live mode and the hourly timeline

Every function degrades to generated data labelled ``heuristic_fallback`` when the
network is unavailable, so training and the demo never depend on connectivity.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.segments import CITY, JUNCTIONS, SEGMENTS, JUNCTION_BY_ID

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOURLY_VARIABLES = (
    "precipitation",
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "visibility",
)


@dataclass
class WeatherSeries:
    """Hourly weather at one point.

    source is one of ``"open_meteo_archive"``, ``"open_meteo_forecast"`` or
    ``"heuristic_fallback"``. The UI shows this verbatim -- a fallback is never
    presented as a measurement.
    """

    time: list[str]
    precipitation_mm_h: np.ndarray
    temperature_c: np.ndarray
    wind_kmh: np.ndarray
    humidity_pct: np.ndarray
    visibility_m: np.ndarray
    source: str
    latitude: float
    longitude: float
    notes: list[str] = field(default_factory=list)

    @property
    def is_measured(self) -> bool:
        return self.source != "heuristic_fallback"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "is_measured": self.is_measured,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "time": list(self.time),
            "precipitation_mm_h": _clean(self.precipitation_mm_h),
            "temperature_c": _clean(self.temperature_c),
            "wind_kmh": _clean(self.wind_kmh),
            "humidity_pct": _clean(self.humidity_pct),
            "visibility_m": _clean(self.visibility_m),
            "notes": list(self.notes),
        }


def _clean(array: np.ndarray) -> list:
    """NaN is not valid JSON. Convert to None so json.dump produces null."""
    out = []
    for value in np.asarray(array, dtype=np.float64).ravel():
        out.append(None if not np.isfinite(value) else round(float(value), 3))
    return out


def _request_json(url: str, params: dict, timeout: int = 25) -> Optional[dict]:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "monsoonplus/1.0 (student project; no API key needed)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - offline is an expected state here
        print(f"[weather] {url} unavailable ({type(exc).__name__}: {exc})")
        return None


def _from_payload(payload: dict, latitude: float, longitude: float, source: str) -> WeatherSeries:
    hourly = payload.get("hourly", {})

    def column(name: str, default: float) -> np.ndarray:
        values = hourly.get(name)
        if not values:
            length = len(hourly.get("time", []))
            return np.full(length, default, dtype=np.float32)
        return np.array(
            [default if v is None else float(v) for v in values], dtype=np.float32
        )

    return WeatherSeries(
        time=list(hourly.get("time", [])),
        precipitation_mm_h=column("precipitation", 0.0),
        temperature_c=column("temperature_2m", 29.0),
        wind_kmh=column("wind_speed_10m", 10.0),
        humidity_pct=column("relative_humidity_2m", 70.0),
        visibility_m=column("visibility", 10000.0),
        source=source,
        latitude=latitude,
        longitude=longitude,
    )


def fetch_archive(
    start_date: str,
    end_date: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> WeatherSeries:
    """ERA5 reanalysis for a past date range. No key required.

    Dates are ``YYYY-MM-DD``. ERA5 lags real time by about five days.
    """
    latitude = CITY["center"][0] if latitude is None else latitude
    longitude = CITY["center"][1] if longitude is None else longitude

    payload = _request_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": CITY["timezone"],
        },
    )
    if payload and payload.get("hourly", {}).get("time"):
        return _from_payload(payload, latitude, longitude, "open_meteo_archive")

    hours = _hours_between(start_date, end_date)
    series = synthetic_weather(hours, latitude, longitude, start_date=start_date)
    series.notes.append(
        f"Open-Meteo archive unavailable for {start_date}..{end_date}; generated instead."
    )
    return series


def fetch_forecast(
    hours: int = 24, latitude: Optional[float] = None, longitude: Optional[float] = None
) -> WeatherSeries:
    """Hourly forecast for the next ``hours``. Powers the Forecast page timeline."""
    latitude = CITY["center"][0] if latitude is None else latitude
    longitude = CITY["center"][1] if longitude is None else longitude

    payload = _request_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "forecast_days": max(1, min(7, (hours + 23) // 24)),
            "timezone": CITY["timezone"],
        },
    )
    if payload and payload.get("hourly", {}).get("time"):
        series = _from_payload(payload, latitude, longitude, "open_meteo_forecast")
        return _truncate(series, hours)

    series = synthetic_weather(hours, latitude, longitude)
    series.notes.append("Open-Meteo forecast unavailable; generated instead.")
    return series


def fetch_air_quality(
    latitude: Optional[float] = None, longitude: Optional[float] = None
) -> dict:
    """Current AQI and pollen. Open-Meteo's air-quality endpoint, also keyless.

    Returns a dict carrying its own ``source``; the badge on the Forecast page prints
    that label, so demo values are never shown as measured.
    """
    latitude = CITY["center"][0] if latitude is None else latitude
    longitude = CITY["center"][1] if longitude is None else longitude

    payload = _request_json(
        AIR_QUALITY_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm10,pm2_5,us_aqi",
            "timezone": CITY["timezone"],
        },
    )
    if payload and payload.get("current"):
        current = payload["current"]
        return {
            "source": "open_meteo_air_quality",
            "is_measured": True,
            "us_aqi": current.get("us_aqi"),
            "pm2_5": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "pollen": None,
            "pollen_note": "Open-Meteo publishes pollen for Europe only; absent for Chennai.",
            "time": current.get("time"),
        }

    return {
        "source": "heuristic_fallback",
        "is_measured": False,
        "us_aqi": 78,
        "pm2_5": 23.4,
        "pm10": 52.1,
        "pollen": "low",
        "pollen_note": "Demo values. Air-quality endpoint unavailable.",
        "time": None,
    }


def _hours_between(start_date: str, end_date: str) -> int:
    from datetime import date

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    return max(1, (end - start).days + 1) * 24


def _truncate(series: WeatherSeries, hours: int) -> WeatherSeries:
    limit = min(hours, len(series.time))
    series.time = series.time[:limit]
    for name in (
        "precipitation_mm_h",
        "temperature_c",
        "wind_kmh",
        "humidity_pct",
        "visibility_m",
    ):
        setattr(series, name, getattr(series, name)[:limit])
    return series


def synthetic_weather(
    hours: int,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    seed: int = 11,
    start_date: Optional[str] = None,
) -> WeatherSeries:
    """Generated northeast-monsoon weather. Always labelled ``heuristic_fallback``."""
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)
    start = (
        datetime.fromisoformat(start_date)
        if start_date
        else datetime.now().replace(minute=0, second=0, microsecond=0)
    )
    times = [(start + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(hours)]

    hour_of_day = np.array([(start + timedelta(hours=i)).hour for i in range(hours)])
    # Chennai's monsoon rain peaks late afternoon into night.
    storm_bias = np.exp(-0.5 * ((hour_of_day - 18) / 4.5) ** 2)
    precipitation = np.clip(rng.gamma(1.3, 6.0, hours) * storm_bias - 1.2, 0, None)

    temperature = 29.0 - 4.0 * np.sin((hour_of_day - 5) / 24 * 2 * np.pi) - 0.05 * precipitation
    wind = np.clip(9.0 + 0.2 * precipitation + rng.normal(0, 1.8, hours), 0, 60)
    humidity = np.clip(68 + 0.7 * precipitation + rng.normal(0, 4, hours), 35, 100)
    visibility = np.clip(10000 - 230 * precipitation + rng.normal(0, 400, hours), 400, 20000)

    return WeatherSeries(
        time=times,
        precipitation_mm_h=precipitation.astype(np.float32),
        temperature_c=temperature.astype(np.float32),
        wind_kmh=wind.astype(np.float32),
        humidity_pct=humidity.astype(np.float32),
        visibility_m=visibility.astype(np.float32),
        source="heuristic_fallback",
        latitude=CITY["center"][0] if latitude is None else latitude,
        longitude=CITY["center"][1] if longitude is None else longitude,
        notes=["Synthetic northeast-monsoon weather. Not a measurement."],
    )


def per_segment_rain(series: WeatherSeries) -> np.ndarray:
    """Spread a single-point rainfall series across segments using ``rain_bias``.

    Open-Meteo is queried at one city point to stay well inside its fair-use limits.
    Scaling by each segment's documented rain bias is an approximation, and the README
    lists it under limitations -- it is not a claim of per-road measurement.
    """
    bias = np.array([s.rain_bias for s in SEGMENTS], dtype=np.float32)
    return np.outer(series.precipitation_mm_h, bias).astype(np.float32)


if __name__ == "__main__":
    print("Open-Meteo needs no API key. Trying the live forecast endpoint...\n")
    forecast = fetch_forecast(hours=12)
    print(f"source : {forecast.source}  (measured={forecast.is_measured})")
    for note in forecast.notes:
        print(f"note   : {note}")
    print(f"{'time':<18}{'rain mm/h':>10}{'temp C':>8}{'wind':>7}{'hum %':>7}")
    for i in range(min(8, len(forecast.time))):
        print(
            f"{forecast.time[i]:<18}{forecast.precipitation_mm_h[i]:>10.2f}"
            f"{forecast.temperature_c[i]:>8.1f}{forecast.wind_kmh[i]:>7.1f}"
            f"{forecast.humidity_pct[i]:>7.0f}"
        )

    air = fetch_air_quality()
    print(f"\nair quality source: {air['source']}  US AQI {air['us_aqi']}")
