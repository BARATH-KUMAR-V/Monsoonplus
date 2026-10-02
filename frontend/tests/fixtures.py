"""Response fixtures shaped like the real services, for offline UI testing."""
import json, math
from datetime import datetime, timedelta

BASE = datetime(2026, 10, 2, 0, 0)

def _hours(n=72):
    return [(BASE + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(n)]

def _series(n, fn):
    return [round(fn(i), 2) for i in range(n)]

def forecast(lat=13.0, lon=80.25, days=14, wet=True):
    n = 24 * days
    times = [(BASE + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(n)]
    rain = _series(n, lambda i: max(0.0, 9 * math.sin((i - 14) / 3.0)) if wet and 10 < i % 24 < 22 else 0.0)
    return {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
        "utc_offset_seconds": 19800, "elevation": 11.0,
        "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
        "current": {
            "time": (BASE + timedelta(hours=11)).strftime("%Y-%m-%dT%H:30"), "interval": 900,
            "temperature_2m": 31.4, "relative_humidity_2m": 74, "apparent_temperature": 36.1,
            "is_day": 1, "precipitation": 2.1, "rain": 2.1, "weather_code": 63,
            "cloud_cover": 82, "pressure_msl": 1006.4, "wind_speed_10m": 17.3,
            "wind_direction_10m": 118, "wind_gusts_10m": 34.2,
        },
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": times,
            "temperature_2m": _series(n, lambda i: 28 + 5 * math.sin((i % 24 - 6) / 24 * 2 * math.pi)),
            "relative_humidity_2m": _series(n, lambda i: 70 + 12 * math.sin(i / 7)),
            "apparent_temperature": _series(n, lambda i: 31 + 5 * math.sin((i % 24 - 6) / 24 * 2 * math.pi)),
            "precipitation_probability": _series(n, lambda i: min(95, max(0, 40 + 45 * math.sin(i / 5)))),
            "precipitation": rain,
            "rain": rain,
            "weather_code": [63 if r > 1 else 3 for r in rain],
            "cloud_cover": _series(n, lambda i: 60 + 30 * math.sin(i / 6)),
            "visibility": _series(n, lambda i: 12000 + 6000 * math.sin(i / 9)),
            "wind_speed_10m": _series(n, lambda i: 14 + 9 * math.sin(i / 8)),
            "wind_direction_10m": _series(n, lambda i: (i * 13) % 360),
            "wind_gusts_10m": _series(n, lambda i: 28 + 22 * math.sin(i / 8)),
            "uv_index": _series(n, lambda i: max(0, 8 * math.sin((i % 24 - 6) / 12 * math.pi))),
        },
        "daily_units": {"temperature_2m_max": "°C"},
        "daily": {
            "time": [(BASE + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days)],
            "weather_code": [63, 80, 3, 2, 95, 61, 1, 2, 3, 63, 80, 2, 1, 3][:days],
            "temperature_2m_max": _series(days, lambda i: 33 + 2 * math.sin(i)),
            "temperature_2m_min": _series(days, lambda i: 25 + 1.5 * math.sin(i + 1)),
            "apparent_temperature_max": _series(days, lambda i: 38 + 2 * math.sin(i)),
            "sunrise": [(BASE + timedelta(days=d, hours=6, minutes=5)).strftime("%Y-%m-%dT%H:%M") for d in range(days)],
            "sunset": [(BASE + timedelta(days=d, hours=18, minutes=2)).strftime("%Y-%m-%dT%H:%M") for d in range(days)],
            "uv_index_max": _series(days, lambda i: 8 + math.sin(i)),
            "precipitation_sum": _series(days, lambda i: max(0, 18 * math.sin(i / 2))),
            "precipitation_hours": _series(days, lambda i: max(0, 8 * math.sin(i / 2))),
            "precipitation_probability_max": _series(days, lambda i: min(100, 45 + 50 * math.sin(i / 2))),
            "wind_speed_10m_max": _series(days, lambda i: 20 + 8 * math.sin(i)),
            "wind_gusts_10m_max": _series(days, lambda i: 38 + 14 * math.sin(i)),
        },
    }

def forecast_multi(count):
    return [forecast(13.0 + i * 0.01, 80.2 + i * 0.01, days=2) for i in range(count)]

def elevation(count):
    return {"elevation": [round(3 + 9 * math.sin(i / 2.0), 1) for i in range(count)]}

def air():
    return {
        "latitude": 13.0, "longitude": 80.25,
        "current_units": {"pm2_5": "μg/m³", "us_aqi": "USAQI"},
        "current": {"time": "2026-10-02T11:00", "pm10": 58.2, "pm2_5": 31.4,
                    "us_aqi": 92, "carbon_monoxide": 310, "nitrogen_dioxide": 18.2, "ozone": 44.0},
    }

PLACES = [
    ("Velachery", "Chennai, Tamil Nadu", 12.9791, 80.2210),
    ("Velachery MRTS Station", "Chennai, Tamil Nadu", 12.9760, 80.2180),
    ("Thoraipakkam", "Chennai, Tamil Nadu", 12.9360, 80.2320),
    ("Guindy", "Chennai, Tamil Nadu", 13.0067, 80.2206),
]

def photon(q=""):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
         "properties": {"osm_id": 1000 + i, "osm_type": "N", "name": name, "city": "Chennai",
                        "state": "Tamil Nadu", "country": "India", "osm_value": "suburb"}}
        for i, (name, _ctx, lat, lon) in enumerate(PLACES)]}

def osrm(alts=3):
    def geom(offset):
        return [[80.221 + i * 0.004, 12.979 + i * 0.003 + offset] for i in range(26)]
    routes = []
    for a in range(alts):
        routes.append({
            "distance": 12400 + a * 900,
            "duration": 1680 + a * 260,
            "geometry": {"type": "LineString", "coordinates": geom(a * 0.004)},
            "legs": [{"distance": 12400 + a * 900, "duration": 1680 + a * 260, "steps": [
                {"name": "Velachery Main Road", "distance": 2400, "duration": 300, "maneuver": {"type": "depart"}},
                {"name": "100 Feet Road", "distance": 3800, "duration": 420, "maneuver": {"type": "turn"}},
                {"name": "Rajiv Gandhi Salai", "distance": 5200, "duration": 640, "maneuver": {"type": "turn"}},
                {"name": "", "distance": 120, "duration": 20, "maneuver": {"type": "arrive"}},
            ]}],
        })
    return {"code": "Ok", "routes": routes, "waypoints": [{"name": "A"}, {"name": "B"}]}
