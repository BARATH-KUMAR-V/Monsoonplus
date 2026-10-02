# Deploying MonsoonPlus to Vercel

The consumer app is a static site. It calls free, keyless public services straight from
the browser, so there is no server to run and no secret to configure.

## 1. Push to GitHub

```bash
git init                     # if this is not a repo yet
git add .
git commit -m "MonsoonPlus: weather-aware navigation"
git branch -M main
git remote add origin https://github.com/<you>/monsoonplus.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `node_modules/`, `.venv/` and the model
checkpoints. Check `git status` before the first push and confirm no `.env` is listed.

## 2. Connect Vercel

1. vercel.com → **Add New Project** → import the repository.
2. Leave every build setting alone. `vercel.json` at the repository root already sets the
   build command and output directory.
3. **Deploy.**

Every later `git push` to `main` redeploys automatically.

## Environment variables

None are required. Two are optional:

| Variable | Default | What it does |
|---|---|---|
| `VITE_OSRM_URL` | `https://router.project-osrm.org` | Point at your own OSRM instance. The public one is a demo server with no uptime guarantee, which is the single biggest reliability risk in this deployment. |
| `VITE_MONSOONPLUS_API` | `http://127.0.0.1:8000` | Where the developer view's optional local collector lives. Irrelevant to the consumer app. |

`TOMTOM_API_KEY` and `EARTHENGINE_PROJECT` belong to the research pipeline, not the web
app. Do not add them to Vercel — the browser app never reads them, and a key in a
`VITE_`-prefixed variable would be published in the JavaScript bundle for anyone to take.

## Services the deployed site calls

| Service | Used for | Key | Fair use |
|---|---|---|---|
| Open-Meteo | Weather, air quality, elevation | none | ~10k requests/day, non-commercial |
| Photon (Komoot) | Place search | none | be reasonable; the app debounces and caches |
| Nominatim (OSM) | Place search fallback | none | 1 request/second |
| OSRM demo | Routing | none | demo server, no guarantee |
| OpenStreetMap tiles | Map background | none | no bulk downloading |

Responses are cached in memory for the session and searches are debounced, which keeps a
normal user well inside every limit. If the site ever gets real traffic, host your own
OSRM and switch to a paid geocoder — the app reads both from environment variables, so
that is a configuration change, not a code change.

## Attribution

The footer and the About page already credit OpenStreetMap, Open-Meteo and OSRM, which
their licences require. Keep them.
