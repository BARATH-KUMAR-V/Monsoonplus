# monsoonplus

**Chennai monsoon traffic intelligence — a trimodal gated-fusion graph neural network
that forecasts road speeds at t+15 / t+30 / t+60 minutes.**

Built by Barath Kumar V, B.Tech AI & Data Science (year 2), as the Deep Learning
experiential-learning submission for course 23AD53C.

Runs end to end with **zero API keys, zero accounts, and zero cost**. Everything below
works on a laptop with nothing but Python and Node installed.

```bash
cd frontend && npm install && npm run dev     # the whole site, offline, no keys
```

---

## Table of contents

1. [What this is, in plain English](#1-what-this-is-in-plain-english)
2. [Architecture overview](#2-architecture-overview)
3. [Folder-by-folder map](#3-folder-by-folder-map)
4. [Setup from a blank computer](#4-setup-from-a-blank-computer)
5. [Running it locally](#5-running-it-locally)
6. [Turning on Live mode (TomTom)](#6-turning-on-live-mode-tomtom)
7. [Turning on satellite data (Earth Engine)](#7-turning-on-satellite-data-earth-engine)
8. [Training and retraining the model](#8-training-and-retraining-the-model)
9. [Collecting real logs for Log mode](#9-collecting-real-logs-for-log-mode)
10. [Deploying for free](#10-deploying-for-free)
11. [The 7 pages and every feature — real vs labelled](#11-the-7-pages-and-every-feature--real-vs-labelled)
12. [Known limitations](#12-known-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [How to extend it](#14-how-to-extend-it)
15. [Credits and required attributions](#15-credits-and-required-attributions)

---

## 1. What this is, in plain English

### For a non-technical reader

When it rains hard in Chennai, some roads slow to a crawl and some flood outright — and
which ones, and how fast, depends on more than just how busy the road was ten minutes
ago. monsoonplus is a website that predicts how fast traffic will be moving on ten
major Chennai roads, fifteen minutes, half an hour and an hour from now. It looks at
three things at once: how traffic has been flowing, what the weather is doing, and how
waterlogged the ground already is as seen from satellite. It then tells you which roads
to avoid, whether to leave now or wait, and which route keeps you away from the flooded
stretches. It is a planning aid for a rainy commute — **not** a safety system, and never
something to trust about whether a flooded road is passable.

### For a technical reviewer

monsoonplus is a spatio-temporal forecasting model over a 10-segment, 8-junction road
graph covering Velachery, Guindy, Saidapet, Adyar, Taramani, Perungudi, Thoraipakkam
and Pallikaranai. Three modality encoders — a GRU over a 2-hour traffic history, a GRU
over per-segment weather, and an MLP over Sentinel-2 NDWI — are combined by a **learned
gate** that produces per-node, per-sample mixing weights, then passed through two rounds
of `GCNConv` message passing over a Gaussian-kernel adjacency, then a linear head that
predicts a *residual* from the last observed speed at three horizons.

Training is four-phase: pretrain the traffic branch on real METR-LA (207 LA freeway
detectors) with weather and satellite masked *absent* rather than zero-filled; transfer
to Chennai with all three modalities live; fine-tune with a cost-sensitive loss (rain
×3, heavy rain ×8); and optionally fine-tune on real collected logs. Model selection at
every phase uses **rain-weighted validation MAE**, not plain validation MAE.

### The novelty claim, in one sentence

> Three modalities (traffic, weather, satellite wetness) fused by a *learned, readable*
> gate, transferred from a US benchmark to an Indian city, and evaluated with the
> rain-versus-clear split always shown next to any aggregate number.

### Does the claim hold up?

It is checkable rather than rhetorical, because the gate is readable from outside. From
the shipped training run (`ml/reports/eval_synthetic_chennai.json`):

| Weather regime | Traffic | Weather | Satellite |
| -------------- | ------- | ------- | --------- |
| Clear          | 45.7%   | 48.4%   | 5.9%      |
| Rain           | 44.8%   | 47.7%   | 7.5%      |
| **Heavy rain** | **33.0%** | **49.0%** | **18.0%** |

In heavy rain the gate puts **67.0%** of its weight on weather + satellite, versus
**54.3%** in clear weather — a **+12.7 point** shift, with the satellite share tripling.
That is the claimed behaviour, measured rather than asserted.

And the error numbers, on the held-out test split (1908 windows):

| Model | MAE overall | Clear | Rain | Heavy rain | Rain-weighted |
| --- | --- | --- | --- | --- | --- |
| **monsoonplus** | **0.81** | **0.73** | **0.85** | **1.08** | **0.93** |
| persistence | 1.13 | 0.98 | 1.26 | 1.64 | 1.37 |
| LSTM (no graph) | 1.02 | 0.93 | 1.05 | 1.36 | 1.18 |
| historical average | 2.94 | 2.66 | 1.43 | 5.87 | 4.02 |

Improvement over persistence: **+28.5% overall, +32.4% in rain, +34.2% in heavy rain.**
The rain-side gains *exceed* the aggregate, which is the honest direction — the headline
number is not being propped up by clear weather.

### Do the design choices earn their place?

`python -m ml.training.ablations` trains each variant from scratch on identical data,
seed and epoch budget (`ml/reports/ablations.json`):

| Variant | MAE | Clear | Rain | Heavy | Rain-weighted |
| --- | --- | --- | --- | --- | --- |
| `graph_layer=gcn` (default) | 0.858 | 0.801 | 0.975 | 1.023 | **0.946** |
| `graph_layer=gat` | 0.871 | 0.820 | 0.972 | 1.019 | 0.949 |
| `modalities=traffic_only` | 0.925 | 0.866 | 1.031 | 1.109 | 1.019 |
| `modalities=traffic_weather` | 0.899 | 0.839 | 1.044 | 1.055 | 0.986 |
| `modalities=all_three` | 0.858 | 0.801 | 0.975 | 1.023 | **0.946** |

Read honestly:

* **GCN and GAT are tied** (0.946 vs 0.949). On a 10-node graph attention has almost no
  room to help. GCN stays the default because it is cheaper, not because it won.
* **Each modality earns its place in heavy rain**, which is where it matters:
  1.109 → 1.055 → 1.023 km/h as weather and then satellite are added. The satellite
  channel is worth 0.032 km/h of heavy-rain MAE on top of traffic+weather — a real but
  modest contribution, and one measured against a *synthetic* NDWI signal.

> ⚠️ **Read this before quoting those numbers.** The Chennai figures come from a
> **synthetic** window generated by an explicit physical process, not from measured
> Chennai traffic. See [section 12](#12-known-limitations). The METR-LA figures are from
> real data, and there the model's margin is thin: **+1.3% over persistence, and it
> *loses* to the LSTM baseline by 10.2%.** Both facts are displayed on the site itself.

---

## 2. Architecture overview

```
                         DATA SOURCES
  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
  │   TomTom     │   │  Open-Meteo  │   │   Sentinel-2    │
  │  Flow Segment│   │ (no API key) │   │  NDWI via GEE   │
  │  (optional)  │   │              │   │   (optional)    │
  └──────┬───────┘   └──────┬───────┘   └────────┬────────┘
         │                  │                    │
         │   each has a labelled offline fallback │
         ▼                  ▼                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  config/segments.json  — THE single source of truth    │
  │  10 segments · 8 junctions · thresholds · attributions │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                       MODEL  (ml/)
  traffic   (N,24,3) ──GRU──┐
  weather   (N,24,3) ──GRU──┼──▶ GATED FUSION ──▶ GCN ×2 ──▶ head
  satellite (N,2)    ──MLP──┘    (N,3 weights,      over the      │
                                  per sample)     road graph      ▼
                                                        speed @ t+15/30/60
                           │
                           ▼
              ml/export/export_predictions.py
                           │
                           ▼
            frontend/public/data/*.json   (committed)
                           │
                           ▼
             FRONTEND  (React 19 + Vite 7)
         7 pages · Leaflet/OSM · Recharts · no backend
                           │
                           ▼
              Vercel / Netlify  (static, free)
```

### The gated-fusion idea

A naive trimodal model concatenates the three embeddings and lets a dense layer sort it
out. The mixing is then fixed after training: the model weights weather the same way on
a dry Tuesday as during a cloudburst.

A **gate** instead computes the mixing weights *from the inputs themselves*:

```python
logits  = gate_mlp([h_traffic, h_weather, h_satellite, rain_now])   # (B, N, 3)
weights = softmax(logits)                                            # sums to 1
fused   = Σ weights[m] · project[m](h[m])
```

Three properties make this a real claim rather than an architectural flourish:

* **Learned** — trained jointly with everything else, not hand-tuned.
* **Conditional** — it reads the rainfall scalar and all three embeddings, per node and
  per sample, so one forward pass can weight a flooding marsh road differently from a
  dry arterial.
* **Observable** — `forward()` returns the weights. The Model Lab page reads them
  directly. If the gate had *not* shifted toward weather in rain, the page would say so;
  the verdict text is generated from the numbers.

### Why trimodal, specifically

The satellite channel only earns its place if it carries information the other two
lack. In the generative process, NDWI tracks **accumulated** wetness with a multi-hour
decay, and the flooding term depends on accumulation rather than instantaneous rainfall.
A road soaking for six hours therefore behaves differently from one hit by the same
rainfall rate ten minutes ago — and that difference is *not* recoverable from a 2-hour
weather history. Measured correlation between rainfall and wetness in the shipped
window is ≈0.75: related, but far from redundant. `tests/test_data.py` asserts that
correlation stays inside 0.2–0.95, so the channel can neither become noise nor collapse
into a copy of the weather channel.

### Availability masking (why "trimodal" stays honest)

METR-LA has no rainfall or satellite channel. Feeding the weather and satellite encoders
zeros and calling the result "a trimodal model pretrained on METR-LA" would be false. So
phase-1 passes an **availability mask**: absent modalities get `-inf` gate logits, the
softmax renormalises over what exists, and those encoders receive *exactly zero*
gradient. `tests/test_model.py::test_absent_modalities_receive_no_gradient` pins it.

---

## 3. Folder-by-folder map

| Path | What's in it | Why it exists | Will you touch it? |
| --- | --- | --- | --- |
| `config/segments.json` | The 10 roads, 8 junctions, coordinates, thresholds, attributions, monsoon calendar | **The single source of truth.** Python, the frontend, the collector and the backend all read this one file | **Yes** — to add a road or change a threshold |
| `config/segments.py` | Typed accessors, derived geometry, a `validate()` self-check | Gives Python a dataclass view and derives lengths/adjacency from coordinates so nothing is duplicated | Rarely |
| `ml/models/encoders.py` | `TrafficEncoder`, `WeatherEncoder`, `SatelliteEncoder` | The three modality branches | Only to change capacity |
| `ml/models/fusion.py` | `GatedFusion` + `gate_summary` | The novelty. Produces and exposes the mixing weights | Only to change the gate |
| `ml/models/graph_layers.py` | `SpatialBlock` (GCN or GAT), batch-graph tiling | The spatial half; where the one-line GCN↔GAT swap lives | No |
| `ml/models/monsoonplus_net.py` | The assembled model, config, checkpoint I/O | Single construction point so training/eval/export can't disagree | No |
| `ml/data/metr_la.py` | METR-LA download + raw `h5py` reader + labelled stand-in | Real benchmark data; avoids the `read_hdf` failure entirely | No |
| `ml/data/chennai_synthetic.py` | The generated Chennai monsoon window | Honest substitute for months of real data | Maybe — to retune the physics |
| `ml/data/weather.py` | Open-Meteo archive + forecast + air quality | Keyless weather, with a labelled fallback | No |
| `ml/data/ndwi.py` | Earth Engine query, local CSV import, synthetic fallback, GEE snippet | The satellite modality and its three honest states | Only to wire up real GEE |
| `ml/data/graph.py` | Gaussian-kernel adjacency + `assert_graph_contributes` | Builds both graphs; guards the self-loop-domination bug | No |
| `ml/data/windowing.py` | Windowing, chronological split, normalisation | Turns series into supervised samples without leakage | No |
| `ml/training/losses.py` | `CostSensitiveLoss`, `error_breakdown` | Rain ×3 / heavy ×8, and the never-just-one-number reporting | No |
| `ml/training/train.py` | The 4-phase training script | The main entry point | **Yes** — to run it |
| `ml/training/baselines.py` | Persistence, historical average, LSTM | What the model must beat | No |
| `ml/training/evaluate.py` | Full evaluation + auto-generated honesty notes | Produces the Model Lab numbers | **Yes** — to run it |
| `ml/training/ablations.py` | GCN vs GAT, modality drop | Tests whether the design choices earn their place | Optional |
| `ml/training/score_logs.py` | Joins predictions to later observations | Scores Log mode once you have collected data | Only with logs |
| `ml/export/export_predictions.py` | Writes `frontend/public/data/*.json` | The handoff from model to website | **Yes** — after training |
| `ml/export/check_palette.py` | WCAG contrast + CIEDE2000 under 3 CVD simulations | Measures the accessibility claim instead of asserting it | Only if you change colours |
| `ml/checkpoints/` | Trained `.pt` files | Gitignored — regenerate with training | No |
| `ml/reports/` | Evaluation JSON and training logs | Evidence for the viva | Read only |
| `backend/app.py` | Optional FastAPI service | Powers Live mode locally. **Never deployed** | Only for Live mode |
| `collector/collect.py` | TomTom + Open-Meteo poller, budget guard | Fills Live and Log modes | Only for Live mode |
| `collector/store.py` | Snapshot/log writing, SQLite budget ledger | NaN-safe JSON writing; monthly quota tracking | No |
| `frontend/src/config/network.js` | Imports `config/segments.json` | The UI's only view of the road network | No |
| `frontend/src/lib/model.js` | Reads the forecast grid; impact score, status, flood | Turns model outputs into what the UI shows | Rarely |
| `frontend/src/lib/routing.js` | Time-dependent Dijkstra, flood-safe penalties | Real routing, not a lookup table | Rarely |
| `frontend/src/lib/alerts.js` | Threshold rules, documented as rules | Keeps "alert" from implying a second ML model | Maybe |
| `frontend/src/pages/` | The 7 screens | The product | **Yes** |
| `frontend/public/data/` | The committed prediction JSON | The deployed site's entire data layer | **Yes** — overwrite from Colab |
| `notebooks/` | The Colab training notebook | Runs the whole pipeline on free GPU | **Yes** |
| `tests/` | 48 pytest tests | Pins the invariants | Run them |
| `.github/workflows/` | Optional scheduled refresh | Purely optional; the app never needs it | No |
| `logs/` | Collected log files + schema README | Empty until you collect. Never fabricated | Only via the collector |

---

## 4. Setup from a blank computer

### 4.1 Install the prerequisites

| Tool | Version | Where |
| --- | --- | --- |
| Python | **3.10 or newer** (3.11/3.12 tested) | <https://www.python.org/downloads/> — tick **"Add Python to PATH"** |
| Node.js | **20 or newer** (24 tested) | <https://nodejs.org/> — the LTS installer |
| Git | any recent | <https://git-scm.com/downloads> |

Verify:

```bash
python --version
node --version
npm --version
```

### 4.2 Get the code

```bash
git clone https://github.com/YOUR_USERNAME/monsoonplus.git
cd monsoonplus
```

### 4.3 Python environment

```bash
python -m venv .venv
```

Activate it — **this differs per platform**:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

Install PyTorch first (it needs its own index for the CPU build), then the rest:

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Check the network config loads:

```bash
python -m config.segments
```

You should see the 10 segments listed and `config valid` on the last line.

### 4.4 Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4.5 Credentials — all optional

**Nothing here is required.** The project runs fully without any of it.

```bash
cp .env.example .env      # Windows PowerShell: copy .env.example .env
```

Open **`.env`** (at the repository root) and fill in only what you have:

| Variable | File | Needed for | Required? |
| --- | --- | --- | --- |
| `TOMTOM_API_KEY` | `.env`, line starting `TOMTOM_API_KEY=` | Live traffic | No — [section 6](#6-turning-on-live-mode-tomtom) |
| `EARTHENGINE_PROJECT` | `.env`, line starting `EARTHENGINE_PROJECT=` | Real Sentinel-2 NDWI | No — [section 7](#7-turning-on-satellite-data-earth-engine) |
| `VITE_MONSOONPLUS_API` | `.env` | Only if the backend runs on a non-default port | No |

**Open-Meteo needs no key.** There is deliberately no variable for it. If you find code
asking for one, that is a bug.

### 4.6 What must never be committed

| Never commit | Protected by |
| --- | --- |
| `.env` (your keys) | `.gitignore` line `.env` |
| `ml/checkpoints/*.pt` | `.gitignore` — regenerate by training |
| `ml/data/raw/` (METR-LA, 57 MB) | `.gitignore` — re-download |
| `node_modules/`, `.venv/` | `.gitignore` |
| `collector/collector.db`, snapshots | `.gitignore` |

`.env.example` **is** committed — it documents the variable names with empty values.

Before your first push, confirm nothing sensitive is staged:

```bash
git status --porcelain | grep -E "\.env$|\.pt$|\.h5$"   # should print nothing
```

---

## 5. Running it locally

**The full site, no accounts, no keys, no backend:**

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. You get all 7 pages in **Dataset mode**, reading the
committed JSON in `frontend/public/data/`. There are no network calls except
OpenStreetMap map tiles — and if those fail, the map still draws the roads over a plain
background and says tiles are unavailable.

Production build:

```bash
npm run build      # outputs frontend/dist/
npm run preview    # serves the built site at http://localhost:4173
```

Run the tests:

```bash
python -m pytest -q          # 48 tests
```

Check the colour palette:

```bash
python -m ml.export.check_palette
```

### Accessibility checking

`axe-core` runs automatically in development and reports WCAG violations to the browser
console. It is imported inside an `import.meta.env.DEV` branch, so Vite strips it from
production builds entirely — the smoke test asserts it never reaches the bundle.

```bash
cd frontend && npm run dev
# open the browser console, then:
#   window.__axeScan()   sweep all 7 routes, one line per page
#   window.__axe()       re-check the current page after interacting with it
```

**Measured result for this build** — WCAG 2.0/2.1 level A and AA, all 7 routes, in both
normal and high-contrast mode:

```
#/          CLEAN
#/trip      CLEAN
#/map       CLEAN
#/forecast  CLEAN
#/replay    CLEAN
#/model     CLEAN
#/system    CLEAN
```

Two real violations were found and fixed getting there, both worth knowing about if you
extend the UI:

* **`scrollable-region-focusable`** — a wide table that scrolls sideways but contains
  nothing focusable is unreachable by keyboard. Wrap wide tables in `<TableScroll>`
  (`frontend/src/components/ui.jsx`), which adds `tabIndex={0}` and a labelled region.
* **`svg-img-alt`** — Recharts gives every scatter point `role="img"` with no accessible
  name, which is 100 unlabelled images to a screen reader. The chart is now
  `aria-hidden` with an equivalent text summary beside it, the same pattern every map on
  the site uses.

---

## 6. Turning on Live mode (TomTom)

Live mode is **optional**. Everything works without it.

### 6.1 Register (free, no credit card)

1. Go to <https://developer.tomtom.com/> and create an account.
2. Dashboard → **My Apps** → **Create App**.
3. Tick the **Traffic API** product.
4. Copy the generated key.

### 6.2 Where the key goes

Open **`.env`** at the repository root. Find this line:

```
TOMTOM_API_KEY=
```

Paste your key after the `=`, with no quotes and no spaces:

```
TOMTOM_API_KEY=AbCdEf123456YourActualKeyHere
```

Save. That is the only place the key is ever written. It is never logged, never put in
a snapshot, never committed.

### 6.3 Quota and the budget guard

> **Check the current terms yourself.** TomTom has changed its free tier more than once.
> At the time of writing it is roughly **20,000 requests per month** — note this is a
> *monthly* allowance, not the "2,500 per day" figure that older tutorials quote. If the
> terms have changed since, the figures below need adjusting.

One poll costs **one request per segment** = 10 requests. The collector polls unevenly,
because a monsoon tool only needs fine resolution when things are changing:

| Window | Interval | Polls/day |
| --- | --- | --- |
| Peak (07:00–11:00, 16:00–21:00) | 15 min | 16 |
| Daytime (11:00–16:00) | 30 min | 10 |
| Night (21:00–07:00) | 60 min | 10 |

= 36 polls/day = 360 requests/day ≈ **10,800 per month**, a little over half the
allowance. A hard ceiling (`--budget`, default 18,000) is tracked in SQLite, so
restarting the process cannot reset the count.

### 6.4 Start it

```bash
python -m collector.collect --dry-run   # shows the plan, makes NO requests
python -m collector.collect --once      # one poll
python -m collector.collect             # run on the schedule
```

In a second terminal, start the backend that serves Live mode:

```bash
uvicorn backend.app:app --reload --port 8000
```

### 6.5 Verify it works

```bash
curl http://127.0.0.1:8000/api/health
```

Expect `{"ok":true,"model_loaded":true,...}`. Then in the browser, click **Live** in the
top bar. The badge should read `LIVE`. Go to **Data & System** — it states exactly which
source is live and which is falling back.

If the collector is not running, the badge reads `DATASET (live unavailable)` and the
app keeps working. That fallback is deliberate and labelled; it never silently serves
precomputed numbers as live ones.

**Cost: $0.** No step above requires a payment method.

---

## 7. Turning on satellite data (Earth Engine)

Also **optional**. Without it the satellite modality uses synthetic NDWI derived from
rainfall accumulation, labelled `heuristic_fallback` everywhere it surfaces.

### 7.1 Register

1. <https://earthengine.google.com/signup/> — sign in with any Google account.
2. Choose **non-commercial / research** use.
3. **Approval typically takes a few hours to a couple of days.** Plan around this; it is
   the only step in the whole project with a waiting period.

### 7.2 Two ways in

**Option A — the Code Editor (recommended; no Python setup).**

Print the ready-made snippet, already filled in with this project's segment midpoints:

```bash
python -c "from ml.data.ndwi import gee_snippet; print(gee_snippet())"
```

1. Paste it into <https://code.earthengine.google.com>.
2. Press **Run**, then open the **Tasks** tab and click **Run** on `ndwi_chennai`.
3. It exports a CSV to your Google Drive.
4. Download it and save it as exactly:

   ```
   ml/data/raw/ndwi_chennai.csv
   ```

   Columns: `segment_id,ndwi`. The loader picks it up automatically and reports
   `source: "local_export"`.

**Option B — the Python API.**

```bash
pip install earthengine-api
earthengine authenticate
```

If your account requires a project id, put it in `.env`:

```
EARTHENGINE_PROJECT=your-project-id
```

Then `ml/data/ndwi.py` queries Sentinel-2 directly and reports `source: "gee_sentinel2"`.

### 7.3 What the fallback looks like if you skip this

Everything still works. The site shows the satellite layer, the wetness overlay is
clickable, and flood risk is computed — but every surface says **"Synthetic NDWI"** and
the model card lists it under limitations. Specifically:

* Data & System → Input sources → Satellite reads
  *"synthetic NDWI derived from rainfall accumulation"*.
* `model_card.json` states the satellite branch trained on a simulated signal.
* Skipping this does **not** break the trimodal architecture — all three branches still
  train. It means one of them learned from a simulated signal, which is stated plainly
  rather than hidden.

---

## 8. Training and retraining the model

### 8.1 On Google Colab (free GPU — recommended)

1. Open <https://colab.research.google.com> → **GitHub** tab → paste your repo URL →
   open `notebooks/monsoonplus_colab_training.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.**
   (CPU also works, just slower.)
3. Edit the first code cell: set `REPO_URL` to your own repository.
4. **Runtime → Run all.**

| Tier | Time |
| --- | --- |
| Free T4 GPU | ~8–12 min |
| Free CPU | ~25–35 min |

The notebook downloads METR-LA, trains all four phases, evaluates against all three
baselines, runs the ablations, plots the gate shift, exports the JSON, and bundles
everything into `monsoonplus_outputs.zip` which it downloads for you.

> **Colab session timeouts.** Free Colab disconnects after ~90 minutes idle. Keep the
> tab visible while it runs. If it does drop, re-run from the top — nothing is lost
> because every phase writes its own checkpoint.

### 8.2 Locally

```bash
python -m ml.training.train                    # all four phases, GCN
python -m ml.training.train --graph-layer gat  # the documented one-line swap
python -m ml.training.train --quick            # ~2 min smoke test
python -m ml.training.evaluate                 # model + all baselines
python -m ml.training.ablations --epochs 8     # GCN vs GAT, modality drop
python -m ml.export.export_predictions         # write the frontend JSON
```

On a laptop CPU the full run takes roughly 13 minutes (phase 1 dominates).

### 8.3 Which files to copy back

From `monsoonplus_outputs.zip`, copy these **four** files into
`frontend/public/data/`, **overwriting** what is there:

```
predictions_synthetic_chennai.json
predictions_metr_la.json
predictions_live_template.json
model_card.json
```

Optionally copy `monsoonplus_final.pt` into `ml/checkpoints/` if you want to run the
backend or collector without retraining locally. It is gitignored — the website does
not need it.

Then commit the four JSON files and push. That is the entire deployment step.

### 8.4 The four phases, and why model selection matters

| Phase | Data | Loss | Notes |
| --- | --- | --- | --- |
| 1 | Real METR-LA | unweighted | Traffic branch only; weather + satellite masked **absent**. LA has no rain channel, so weighting would be meaningless |
| 2 | Synthetic Chennai | unweighted | Transfer; all three modalities live |
| 3 | Synthetic Chennai | **rain ×3, heavy ×8** | Cost-sensitive fine-tune |
| 4 | Real collected logs | rain-weighted | Skipped (loudly) unless `logs/` has data |

Every phase keeps the checkpoint with the lowest **rain-weighted validation MAE**. This
is not a detail. About 60% of the window is clear weather, so plain validation MAE is
dominated by exactly the conditions a monsoon tool does not need help with — selecting
on it picks whichever epoch is best at dry days.

---

## 9. Collecting real logs for Log mode

`logs/` is **empty in a fresh clone**, and that is correct. Nothing fabricates a log.
Until you collect, Page 5 says *"No collected logs yet — collect live data to populate
this view"* and shows nothing else.

### 9.1 How files are written

```bash
python -m collector.collect --once      # one poll
python -m collector.collect             # keep polling
python -m collector.collect --publish   # copy into frontend/public/data/logs/
```

Filename: `YYYY_MM_DD_HH_MM_<label>.json`, e.g. `2026_11_14_18_30_heavy_rain.json`.
The label is the ground-truth weather class from the Open-Meteo reading:

| label | rainfall |
| --- | --- |
| `clear` | < 2.5 mm/h |
| `rain` | 2.5 – 15 mm/h |
| `heavy_rain` | ≥ 15 mm/h |

### 9.2 Schema

```jsonc
{
  "schema_version": "1.0.0",
  "collected_at": "2026-11-14T13:00:11+00:00",
  "date": "2026-11-14",
  "time": "13:00",
  "label": "heavy_rain",
  "rain_mm_h": 47.2,
  "traffic_source": "tomtom_flow_segment_data",
  "traffic_ok": true,
  "per_segment": [
    {
      "segment_id": "r2",
      "name": "Velachery Main Road",
      "observed_speed_kmh": 14.3,          // null if traffic wasn't collected
      "predicted_kmh": { "t+15": 13.1, "t+30": 12.4, "t+60": 11.9 },
      "rain_mm_h": 51.9,                   // city rain × this segment's rain_bias
      "ndwi": -0.041                       // null if no satellite signal
    }
    // ... always all 10 segments, in config order
  ]
}
```

Two enforced rules: **no NaN** (invalid JSON — `JSON.parse` throws), and **every segment
always appears**, with `null` rather than omission when a reading failed.

### 9.3 Scoring

A single file can't be scored — it holds a prediction for t+15 and an observation for
t+0. Scoring joins each file's t+15 prediction to the observation 15 minutes later:

```bash
python -m ml.training.score_logs
```

### 9.4 How much data before it means anything

| What you want | Needs |
| --- | --- |
| The replay chart draws at all | 1 day, ~20 polls |
| A believable single-day MAE | 1 full day, ~36 polls |
| **A rain-vs-clear comparison** | **2+ rainy days and 1+ clear day** |
| **Pattern cards ("floods in 47±8 min")** | **3+ rainy days**, so a spread exists |
| Phase-4 fine-tuning | ~2 weeks of continuous collection |

Below those thresholds the UI says *"not enough data yet"* rather than printing a number
derived from two samples. A pattern card claiming "47 ± 8 min" from one observation
would be worse than showing nothing.

> **Timing note for a submission:** Chennai's northeast monsoon runs October–December,
> peaking in November. Outside that window you may wait a long time for a rainy day.
> This is exactly why the project does not depend on Log mode for anything.

---

## 10. Deploying for free

The deployed site is **static**. No backend, no server, no database, no cold starts.

> **Why the backend is never deployed:** PyTorch + PyTorch Geometric is well over the
> ~250 MB unzipped limit for a Vercel or Netlify serverless function. Rather than fight
> that, the model runs *ahead of time* and the site reads committed JSON.

### 10.1 Push to GitHub

```bash
git init
git add .
git commit -m "monsoonplus: trimodal gated-fusion GNN for Chennai monsoon traffic"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/monsoonplus.git
git push -u origin main
```

Confirm `.env` did **not** go up:

```bash
git ls-files | grep -E "^\.env$"    # must print nothing
```

### 10.2 Vercel

1. <https://vercel.com> → sign in with GitHub (free, no card).
2. **Add New → Project** → import your repository.
3. Set exactly:

   | Setting | Value |
   | --- | --- |
   | Framework Preset | **Vite** |
   | Root Directory | **`frontend`** |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |
   | Install Command | `npm install` |

4. **Environment variables: none required.** Dataset mode needs nothing. (Do *not* set
   `VITE_MONSOONPLUS_API` in production — the deployed site has no backend to talk to,
   and Live mode correctly falls back.)
5. **Deploy.**

### 10.3 Confirm it works

Open the Vercel URL and check:

- [ ] The Overview page shows KPIs, not skeletons
- [ ] The badge top-right reads **DATASET**
- [ ] The map draws roads and junctions
- [ ] Model Lab shows the baseline table with real numbers
- [ ] Changing the rain slider changes every number
- [ ] It works on your phone

### 10.4 Netlify / Cloudflare Pages

Same idea: base directory `frontend`, build `npm run build`, publish `frontend/dist`.
The app uses `HashRouter`, so **no SPA rewrite rule is needed** — refreshing on
`/#/forecast` works anywhere, including from a local file.

### 10.5 The optional GitHub Action

`.github/workflows/refresh-data.yml` can refresh live data on a schedule. It is
**entirely optional** — the app works perfectly with zero Actions runs, which is the
default. It is disabled unless you add a `TOMTOM_API_KEY` repository secret.

---

## 11. The 7 pages and every feature — real vs labelled

Legend: **Real** = computed from the trained model or measured data · **Rule** =
deterministic threshold logic, documented as such · **Labelled** = synthetic,
illustrative or demo, and says so on screen.

### Page 1 — Overview

| Feature | Status | Notes |
| --- | --- | --- |
| 4 KPI stats | **Real** | Recomputed live from the rain slider; nothing hardcoded |
| City-at-a-glance map | **Real** | OSM tiles + model-coloured segments |
| Priority roads by Impact Score | **Rule** | `0.40×traffic + 0.25×rain + 0.20×flood + 0.15×deterioration`, weights from config; inputs are model outputs |
| Saved / My routes | **Real** | Persisted in `localStorage` |
| Alert log | **Rule** | Threshold rules, listed in full on the Forecast page |
| Data health dashboard | **Real** | Freshness dots + required attributions |
| Monsoon regime badge | **Real** | From the Chennai calendar in config |
| Mode selector | **Real** | Global; visible on every page |

### Page 2 — Smart Trip

| Feature | Status | Notes |
| --- | --- | --- |
| From/To/Depart/Mode | **Real** | |
| Fastest vs Flood-safe | **Real** | High-flood edges ×4.0, moderate ×1.35 in the cost function |
| Up to 3 alternates | **Real** | Time-dependent Dijkstra with progressive penalties |
| ETA at now/+30/+60 | **Real** | Each edge evaluated at its arrival time |
| "Leave now or wait?" | **Real** | Same route at several departure offsets |
| Trip impact breakdown | **Real** | Model re-run at 0/25/75 mm/h |
| Rain forecast strip | **Real** | |
| **"What if it weren't raining?"** | **Real, labelled estimate** | A genuine second model pass at rain=0; the card says *"model estimate, not a measured fact"* |
| Landmark quick-picks | **Real** | 10 landmarks → nearest junction, from config |
| Reliability badge | **Real** | Derived from measured test-split MAE per regime, not invented |
| Route PDF export | **Real** | jsPDF + html2canvas, client-side |
| **Peer count** | **Labelled fake** | The card says plainly *"This number is fake. monsoonplus has no users and collects no telemetry."* |

### Page 3 — Live Map

| Feature | Status | Notes |
| --- | --- | --- |
| OSM tiles via Leaflet | **Real** | Degrades to a plain background offline |
| Time slider Now→+60 | **Real** | +15/+30/+60 are direct model outputs; +45 interpolated (stated on the page) |
| 5 togglable layers | **Real** | Traffic, Rain, Flood, Prediction, Combined |
| Click road → side panel | **Real** | Confidence band, Impact Score, deterioration callout |
| Compare roads (up to 4) | **Real** | |
| **Satellite wetness overlay** | **Real geometry, labelled signal** | Clickable polygons showing rain intensity + NDWI; the signal is synthetic NDWI unless GEE is configured |
| Historical speed baseline | **Real** | Per-road, per-time-of-day average from the training split |
| "Play next hour" | **Real** | Auto-steps the slider, rewriting the headline each frame |
| What-if slider | **Real** | "Leaving 15 min earlier saves X min" |

### Page 4 — Forecast

| Feature | Status | Notes |
| --- | --- | --- |
| Rain slider 0–100 + trend | **Real** | |
| 3-bucket regime tag | **Rule** | Dry / Steady rain / Cloudburst |
| Per-horizon KPI cards | **Real** | |
| Chart with uncertainty band | **Real** | Band from measured per-regime test error |
| What-if threshold table | **Real** | Which road gives way at which mm/h |
| Scenario battle | **Real** | Normal evening vs heavy monsoon |
| Hourly 24 h timeline | **Real or labelled** | Open-Meteo when reachable, otherwise a clearly-labelled generated series |
| Weather alerts | **Rule** | Full rule set printed in a `<details>` block |
| Rain-vs-slowdown scatter | **Real** | Every road at every sampled rainfall level |
| Historical baseline card | **Real** | Grouped historical data, or "not enough history" |
| AQI badge | **Real or labelled** | Open-Meteo air quality; demo values labelled if unreachable. Pollen is not published for Chennai and says so |
| CSV export | **Real** | |

### Page 5 — Event Replay

| Feature | Status | Notes |
| --- | --- | --- |
| Scripted storm, 13 frames | **Labelled simulated** | Banner reads `SIMULATED`; a rainfall curve pushed through the trained model |
| Auto-named milestones | **Real** | Computed from the frames, not hand-written |
| **Real predicted-vs-actual replay** | **Real when logs exist** | Otherwise an explicit empty state with the collector command |
| Pattern cards | **Real when logs exist** | Otherwise "not enough data yet" — never invented numbers |

### Page 6 — Model Lab

| Feature | Status | Notes |
| --- | --- | --- |
| Architecture diagram | **Real** | |
| METR-LA evaluation | **Real** | Actually computed on real METR-LA; shows the model *losing* to the LSTM there |
| Synthetic Chennai table | **Real, labelled synthetic** | Banner reads `SYNTHETIC` |
| Rain-vs-clear bars | **Real** | The project's central result |
| **Gate / modality breakdown** | **Real** | Read straight out of the trained network. The "claim supported / NOT supported" verdict is generated from the numbers |
| Model playground | **Labelled illustrative** | Interpolates the exported grid; says so |
| "Where it fails" | **Real** | Read from `model_card.json` |
| Baselines + ablations | **Real** | Persistence, historical average, LSTM; GCN-vs-GAT and modality-drop in `ml/training/ablations.py` |
| Honesty notes | **Real, auto-generated** | Generated from the results, so an unflattering outcome surfaces automatically |

### Page 7 — Data & System

| Feature | Status | Notes |
| --- | --- | --- |
| Model status | **Real** | Flips to "Degraded" when satellite is off |
| Input freshness per source | **Real** | |
| Satellite on/off toggle | **Real** | Genuinely removes the modality from the pipeline |
| "Powered by" attributions | **Real** | From config, so they can't drift |
| 3-mode inspector | **Real** | States which mode is active *and why*, including silent fallbacks |

### Cross-cutting

| Feature | Status |
| --- | --- |
| LIVE / REPLAY / DATASET badge + "last updated" | **Real**, always visible |
| High-contrast toggle | **Real**, persisted |
| Full keyboard navigation + skip link | **Real** |
| Every map accompanied by an equivalent list/table | **Real** |
| Loading / error / empty states | **Real**, everywhere data could be missing |
| Responsive to phone width | **Real** |

---

## 12. Known limitations

Read this section before the viva. Every point here is also stated on the site itself.

1. **The Chennai training data is synthetic.** It is generated by an explicit physical
   process (storm cells, accumulation-driven flooding, per-road rain sensitivity) — not
   measured Chennai traffic. The MAE figures describe how well the model learned *that
   process*. They are not evidence about real Chennai roads. Everything in the UI that
   uses it is labelled "Synthetic Chennai window".

2. **The satellite branch trains on synthetic NDWI** unless you supply a real Earth
   Engine export. "Trimodal" means three branches genuinely trained with gradient flow —
   one of which learned from a simulated signal. That is a real architecture with a
   simulated input, not a fake third modality.

3. **Weather is sampled at one city point** and spread across segments by each road's
   documented `rain_bias`. It is not per-road measurement. This keeps the project well
   inside Open-Meteo's fair-use limits, but it is an approximation.

4. **METR-LA is Los Angeles freeway data.** Pretraining on it transfers general temporal
   dynamics, not anything Chennai-specific. And on METR-LA itself the model manages only
   **+1.3% over persistence (6.09 vs 6.17 km/h MAE) and loses to the LSTM baseline by
   10.2%** (5.52 km/h) — displayed on the Model Lab page rather than buried. A model
   sized and tuned for a 10-segment graph, given 6 pretraining epochs, is not
   competitive on a 207-sensor freeway benchmark, and the project does not pretend
   otherwise. The honest reading is that METR-LA earns its place as *pretraining* and as
   a sanity benchmark, not as a result.

5. **Real logs are empty on arrival.** Page 5's real replay and the pattern cards stay
   empty until you collect data during actual Chennai weather. Nothing is fabricated to
   fill the space.

6. **The road network is a simplification.** 10 segments and 8 junctions is a model of
   South Chennai, not a map of it. Landmarks snap to the nearest junction, which can be
   over a kilometre away.

7. **Routing uses straight-line distance × 1.25.** No real road geometry, no turn
   restrictions, no signal timing.

8. **The font is self-hosted, not loaded from Google Fonts.** The brief asked for
   IBM Plex Sans or Public Sans "via Google Fonts", but it also requires that the page
   never fetch anything external. A Google Fonts `<link>` would contact Google on every
   page load and undo the privacy argument. **Public Sans** therefore ships as an npm
   package (`@fontsource-variable/public-sans`) bundled by Vite, so the specified
   typeface is used and served from our own origin. The smoke test asserts no
   `fonts.googleapis.com` reference survives the build.

9. **Two status colours are not distinguishable by colour alone.** Measured: "Slow" and
   "Disruption likely" fall to ΔE 1.0 under deuteranopia (see
   `ml/reports/palette_check.json`). This is why every status carries a **glyph, a text
   label and a distinct line dash** in addition to colour. All text passes 4.5:1 and all
   swatches pass 3:1 against the page.

10. **Not a safety system.** A screening and planning aid. Never use it to judge whether
   a flooded road is passable.

### What would need a paid service

| Want | Needs | Why it's out of scope |
| --- | --- | --- |
| Per-road rainfall measurement | Commercial weather API | Open-Meteo's free tier is a point forecast |
| Sub-5-minute live traffic | Paid TomTom tier | Free tier quota |
| Historical traffic for real training | Paid archive, or months of self-collection | The reason the Chennai window is synthetic |
| Hosted backend for public Live mode | Paid container host | PyTorch exceeds serverless limits |
| Persistent multi-user storage | Supabase/Neon beyond free tier | Out of scope; SQLite is local-only |

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `TypeError` / `ImportError: Import pytables failed` from `pandas.read_hdf` on `metr-la.h5` | The file is an old pandas fixed-format block with `\|S6` byte labels. `read_hdf` needs pytables (often unavailable) and trips over decoding those labels on pandas 3.x | **Already handled** — `ml/data/metr_la.py` reads it with `h5py` directly and never involves pandas. Do not "fix" it by adding pandas back |
| `ValueError: Out of range float values are not JSON compliant` / `JSON.parse` throws in the browser | NaN reached a JSON file. `json.dumps` writes bare `NaN`, which is not valid JSON | All writers funnel through `_clean()` (`ml/export/export_predictions.py`, `collector/store.py`). If you add a writer, use `write_json()` |
| Map is blank, roads invisible | OSM tiles unreachable | Expected offline. The map still draws roads and shows "Map tiles unavailable offline" |
| `ValueError: ... GCN self-loops (weight 1.0) would dominate aggregation` | Edge weights too small relative to the self-loop | **The guard working as designed.** Use `gaussian_kernel_adjacency()`, never raw `1/travel_time` (which gives ~0.003) |
| Adjacency has **zero edges** on a small graph | `sigma_mode="std"` collapses when all distances are similar | Use `sigma_mode="mean"` for the Chennai graph. METR-LA's wider spread suits `"std"` |
| `RuntimeError: size mismatch for traffic_encoder.gru.weight_ih_l0` | Traffic feature width differs from the checkpoint | Pad single-channel speeds to 3 features **before** loading, with `to_traffic_features()`. `strict=False` only forgives missing/extra keys — it will **not** rescue a shape mismatch |
| FastAPI returns 422, or ignores POSTed JSON | FastAPI reads scalar params from the **query string**, not the body | Declare a Pydantic body model. Every POST endpoint in `backend/app.py` already does |
| Colab session disconnects mid-training | Free tier idle timeout (~90 min) | Keep the tab visible. Re-run from the top — each phase writes its own checkpoint |
| `ModuleNotFoundError: No module named 'config'` | Running a script from inside a subdirectory | Run from the **repository root** with `python -m ml.training.train` |
| `npm run dev` fails resolving `@config/segments.json` | Vite root is `frontend/`, config is above it | `vite.config.js` already sets `server.fs.allow`. Run `npm run dev` from inside `frontend/` |
| Everything shows ~7 km/h and all roads jammed | Over-saturated wetness accumulator | **Fixed** — the accumulator is calibrated so wetness spans 0–0.9. `tests/test_data.py::test_wetness_is_not_saturated` guards it |
| Marsh roads dry *fastest* after rain | Inverted drainage exponent | **Fixed** — lower exponent = slower decay. `test_drainage_ordering_matches_flood_history` guards it |
| `osmnx` errors about bounding-box argument order | osmnx 2.x changed its API | This project does **not** depend on osmnx. If you add it, pin the version and use the current signature |
| Live mode badge says "live unavailable" | No collector/backend running | Expected. Start `uvicorn backend.app:app --port 8000`, or stay in Dataset mode |
| `/api/predict` returns 503 | No trained checkpoint | `python -m ml.training.train --quick`, then restart the backend |
| Model Lab shows "--" everywhere | `predictions_*.json` missing or stale | `python -m ml.export.export_predictions` |

---

## 14. How to extend it

### Add a road segment

**Edit one file: `config/segments.json`.**

```jsonc
{
  "id": "r11",
  "name": "ECR - Thiruvanmiyur to Injambakkam",
  "a": "ADY",                    // must be an existing junction id
  "b": "THO",
  "free_flow_kmh": 38,
  "rain_sensitivity": 0.42,      // 0-1, how much rain slows it
  "base_congestion": 0.13,       // 0-1, rush-hour baseline
  "rain_bias": 1.05,             // local rainfall multiplier
  "lanes": 4,
  "flood_history": "moderate",   // low | moderate | severe
  "note": "Coastal road"
}
```

Then:

```bash
python -m config.segments     # must print "config valid"
python -m pytest -q
python -m ml.training.train   # node count changed; retrain
python -m ml.export.export_predictions
```

The model, backend, collector and all 7 pages pick it up automatically — segment length
and adjacency are *derived* from the junction coordinates, never stored. **Do not** add
a segment list anywhere else; `tests/test_config.py` fails if the frontend stops reading
the shared file.

### Add a junction

Add to `"junctions"` with a unique id and coordinates, then reference it from at least
one segment. `validate()` fails if it would be unreachable, because routing would
silently never find it.

### Add a page

1. `frontend/src/pages/MyPage.jsx`, default-exporting a component.
2. Register a lazy import and a `<Route>` in `frontend/src/App.jsx`.
3. Add an entry to `NAV` in `frontend/src/components/Shell.jsx`.

Use `<Card>`, `<Stat>`, `<StatusChip>` and `<Provenance>` from `@/components/ui` so it
inherits the design system and the labelling conventions.

### Add a data source

1. Write a loader in `ml/data/` with the three honest states: live, local export, and a
   **labelled** fallback. Follow `ml/data/ndwi.py`.
2. Return a `source` string the UI can print verbatim.
3. Never let it raise — one dead source must not sink a run.
4. Add its attribution to `"attributions"` in `config/segments.json`.
5. Add a parse test with a realistic payload.

### Add a modality

Add an encoder in `ml/models/encoders.py`, extend `MODALITIES` in
`ml/models/fusion.py`, widen the gate input, and extend the availability mask. The mask
is what keeps the claim honest — a modality without real data must be marked absent, not
zero-filled.

### Swap GCN for GAT

```bash
python -m ml.training.train --graph-layer gat
```

That is the entire change. `ml/training/ablations.py` compares them on identical data,
seed and epoch budget.

---

## 15. Credits and required attributions

These must appear on the live site. They are stored in `config/segments.json` and
rendered on **Overview → Data health** and **Data & System → Powered by**, so they
cannot drift out of sync:

```
Traffic flow data (c) TomTom - Traffic Flow Segment Data
Weather data from Open-Meteo.com (CC BY 4.0)
Air quality data from Open-Meteo.com (CC BY 4.0)
(c) OpenStreetMap contributors
Contains modified Copernicus Sentinel data
```

> **Note on air quality.** The brief suggested OpenAQ. OpenAQ's current v3 API requires
> a registered API key, which conflicts with the hard "$0, no key, no card" constraint,
> so the AQI badge calls **Open-Meteo's air-quality endpoint** instead — same data class,
> genuinely keyless. The attribution credits what the code actually calls; crediting
> OpenAQ for data we do not fetch would be wrong.

### Data sources

| Source | Use | Licence |
| --- | --- | --- |
| [METR-LA](https://github.com/liyaguang/DCRNN) (Li et al., 2018) | Pretraining + benchmark | Research use |
| [Open-Meteo](https://open-meteo.com/) | Weather, forecast, air quality | CC BY 4.0, free, no key |
| [Public Sans](https://public-sans.digital.gov/) | Typeface, self-hosted via npm | Public domain (USWDS) |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Map tiles | ODbL |
| [Copernicus Sentinel-2](https://sentinel.esa.int/) | NDWI | Copernicus open licence |
| [TomTom](https://developer.tomtom.com/) | Live traffic (optional) | Free tier terms |

### Software

React 19 · Vite 7 · react-leaflet 5 · Leaflet · Recharts · jsPDF · html2canvas ·
Public Sans (@fontsource) · axe-core (dev only) · PyTorch · PyTorch Geometric ·
FastAPI · h5py · NumPy

### Method

The Gaussian-kernel adjacency follows the construction used for METR-LA in
*Diffusion Convolutional Recurrent Neural Network* (Li, Yu, Shahabi & Liu, ICLR 2018).
Gated multimodal fusion follows the general approach of Arevalo et al.,
*Gated Multimodal Units for Information Fusion* (ICLR Workshop, 2017).

### Licence

MIT — see [`LICENSE`](LICENSE).

---

*monsoonplus is a student project and a planning aid. It is not a freedom-to-operate
analysis, not a safety system, and not a substitute for official flood warnings.*
