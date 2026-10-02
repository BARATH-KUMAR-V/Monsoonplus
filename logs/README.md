# Collected logs

This directory holds the real predicted-vs-actual records that power **Log mode** on
the site: Page 5's real replay and Page 5's pattern cards.

It is **empty in a fresh clone**, and that is correct. Nothing in the project fabricates
a log file. Until you run the collector during real Chennai weather, the UI says
"No collected logs yet — collect live data to populate this view" and shows nothing else.

## How files get here

```bash
python -m collector.collect --once     # one poll
python -m collector.collect            # keep polling on the schedule
python -m collector.collect --publish  # copy logs into frontend/public/data/logs/
```

The collector writes one file per poll, named:

```
YYYY_MM_DD_HH_MM_<label>.json
```

where `<label>` is the ground-truth weather class for that moment, derived from the
Open-Meteo rainfall reading using the thresholds in `config/segments.json`:

| label        | rainfall           |
| ------------ | ------------------ |
| `clear`      | below 2.5 mm/h     |
| `rain`       | 2.5 – 15 mm/h      |
| `heavy_rain` | 15 mm/h and above  |

Example: `2026_11_14_18_30_heavy_rain.json`

## File schema

```jsonc
{
  "schema_version": "1.0.0",
  "collected_at": "2026-11-14T13:00:11+00:00",  // UTC, ISO 8601
  "date": "2026-11-14",
  "time": "13:00",
  "label": "heavy_rain",                        // clear | rain | heavy_rain
  "rain_mm_h": 47.2,                            // city-level, from Open-Meteo
  "traffic_source": "tomtom_flow_segment_data", // or not_configured / budget_exhausted
  "traffic_ok": true,
  "per_segment": [
    {
      "segment_id": "r2",                       // matches config/segments.json
      "name": "Velachery Main Road",
      "observed_speed_kmh": 14.3,               // null when traffic was not collected
      "predicted_kmh": {                        // what the model said AT THIS INSTANT
        "t+15": 13.1,
        "t+30": 12.4,
        "t+60": 11.9
      },
      "rain_mm_h": 51.9,                        // city rain x this segment's rain_bias
      "ndwi": -0.041                            // null if no satellite signal
    }
    // ... one entry per segment, always all 10, in config order
  ],
  "note": "..."
}
```

Two rules the writer enforces:

* **No NaN.** `NaN` is not valid JSON and `JSON.parse` throws on it. Every number is
  passed through `_clean()` in `collector/store.py`, which converts non-finite values
  to `null`.
* **Every segment appears**, even when its reading failed. A missing segment would be
  ambiguous; `"observed_speed_kmh": null` is not.

## Scoring predicted against actual

A single file cannot be scored: it holds a prediction for *t+15* and an observation for
*t+0*. Scoring joins each file's `t+15` prediction to the observation in the file
collected 15 minutes later:

```bash
python -m ml.training.score_logs
```

That writes `frontend/public/data/logs_index.json` with per-day MAE, split by weather
label, which is what Page 5 charts.

## How much data before this means anything

| What you want                        | Roughly what it needs                         |
| ------------------------------------ | --------------------------------------------- |
| The replay chart draws at all        | 1 day, any weather, ~20 polls                 |
| A believable MAE for one day         | 1 full day, ~36 polls                         |
| A rain-vs-clear comparison           | **2+ rainy days and 1+ clear day**            |
| Pattern cards ("floods in 47±8 min") | **3+ rainy days**, so a spread can be computed |
| Phase-4 fine-tuning on real logs     | ~2 weeks of continuous collection             |

Below those thresholds the UI says "not enough data yet" rather than printing a number
derived from two samples. That is deliberate: a pattern card claiming "47 ± 8 min" from
one observation would be worse than showing nothing.

## What is NOT stored here

* **No API keys.** The TomTom key lives in `.env` at the repository root and is never
  written to a log, a snapshot, or any committed file.
* **No personal data.** The collector reads public road-speed and weather endpoints. It
  has no users and records nothing about anyone.

## Version control

`.gitignore` ignores `logs/*.json` by default so a long collection run does not bloat
the repository. If you *want* to commit a few representative days (useful for a demo on
a machine that has never collected anything), force-add them:

```bash
git add -f logs/2026_11_14_18_30_heavy_rain.json
```
