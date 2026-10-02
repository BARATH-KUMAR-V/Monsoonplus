# ml/ — model, training, evaluation, export

Everything the frontend's numbers come from. The site only *reads* the JSON this folder
writes (`frontend/public/data/`); nothing in the UI is typed in by hand.

```
ml/
  data/       chennai_synthetic.py  explicit-physics Chennai generator
              metr_la.py            real METR-LA loader (synthetic fallback, labelled)
              weather.py, ndwi.py   ERA5 / Open-Meteo alignment, satellite NDWI
              windowing.py          windows + the chronological 70/15/15 split
              graph.py              Gaussian-kernel road adjacency
  models/     encoders.py           GRU (traffic, weather) + MLP (satellite)
              fusion.py             learned per-node gate over the three modalities
              graph_layers.py       SpatialBlock; PyG (gcn, gat) or dense (dense_gcn, dense_gat)
              dense_graph.py        pure-PyTorch twins of GCNConv / GATConv   (no torch_geometric)
              monsoonplus_net.py    the assembled model, load_checkpoint, modality_masked()
  training/   train.py              4 phases; --graph-layer gcn|gat|dense_gcn|dense_gat
              losses.py             CostSensitiveLoss, error_breakdown
              baselines.py          persistence, historical average, LSTM
              evaluate.py           rain-vs-clear MAE, baselines, inference-time masked ablation
              ablations.py         retrained GCN-vs-GAT and modality ablations
              linear_probe.py       NumPy ridge-regression modality probe   (no PyTorch)
              score_logs.py         joins collected predictions to what happened
  export/     export_predictions.py writes the JSON the site reads
  reference/  ablation_numpy.py     thin launcher for linear_probe (runs without PyTorch)
  reports/    eval_*.json, ablations.json, linear_probe.json, training_summary.json
  checkpoints/
```

## Everyday commands

```bash
python -m ml.training.train                         # all 4 phases, CPU, ~13 min
python -m ml.training.train --graph-layer gat       # one-flag layer swap
python -m ml.export.export_predictions              # regenerate frontend/public/data/*.json
python -m ml.training.score_logs                    # after the collector has run for a while
```

## Is each modality earning its place? Three methods, three different questions

The Model Lab shows all three side by side. They are *not* interchangeable, and they do
not have to agree.

| Method | What it does | Cost | Where |
| --- | --- | --- | --- |
| **Retrained** | Train a fresh network with a modality removed (availability mask). | minutes | `ml/training/ablations.py`, `ml/reports/ablations.json` |
| **Masked at inference** | Take the *trained* network and switch a modality off at test time. Measures how much the shipped model leans on it. | seconds | `ml/training/evaluate.py` (`masked_baselines`), `MonsoonPlusNet.modality_masked` |
| **Linear probe** | A ridge regression on the same generator and the exact same 1,908 test windows. Asks whether the signal is there at all, independent of any architecture. | under a second, NumPy only | `python -m ml.training.linear_probe` or `python ml/reference/ablation_numpy.py` |

Heavy-rain MAE (km/h) on the committed data, traffic only → +weather → +satellite:

| Method | traffic only | + weather | + satellite |
| --- | --- | --- | --- |
| Retrained network | 1.109 | 1.055 | 1.023 |
| Masked at inference | 2.064 | 1.149 | 1.077 |
| Linear probe | 1.576 | 1.272 | 1.246 |

Read the masked row carefully: it is large because a network trained with all three
modalities is *not* expected to cope when one vanishes. It shows reliance, not the value
of the data. The retrained row is the fairer measure of value.

All of this is on **synthetic** Chennai data, so what is demonstrated is that the model
uses the signals the generator puts there — not that real Sentinel-2 NDWI helps. The
probe's honesty note is generated from its own numbers and changes wording if the sign of
an effect flips.

```bash
python -m ml.training.linear_probe                  # prints + writes ml/reports/linear_probe.json
python -m ml.export.export_predictions --refresh-ablation   # re-attach all ablation blocks
                                                    # to the committed JSON from the existing
                                                    # checkpoint (no retrain); aborts if the
                                                    # checkpoint no longer matches recorded metrics
```

## Running without torch_geometric

`graph_layers.py` imports PyG lazily, so the dense backend never touches it:

```python
from ml.models.monsoonplus_net import load_checkpoint
model, _ = load_checkpoint("ml/checkpoints/monsoonplus_final.pt", graph_layer="dense_gcn")
```

The dense layers use the same parameter names and shapes as `GCNConv` / `GATConv`, so a
PyG-trained checkpoint loads with `strict=True` and reproduces its outputs (~2e-6 km/h on
the shipped model). `tests/test_dense_graph.py` pins this. Swapping *GCN* for *GAT* at load
time still fails loudly, as it should — those are different weights.

## Switching a modality off in a trained model

```python
out = model.modality_masked(traffic, weather, satellite, edge_index, edge_weight,
                            drop=("satellite",))
```

Traffic can't be dropped (it is the target signal); the gate renormalises over what is
left, exactly as in training with an availability mask.

## Known gaps

- The Chennai window is a documented physical process, not measured traffic. METR-LA
  numbers are real, and there the model is only marginally better than persistence.
- Dense layers scale as N² in the number of road segments — right for 10 (or METR-LA's
  207), wrong for a city-sized graph.
- The ONNX route (dense twin exported at opset 17, matches onnxruntime to ~4e-6) was
  checked once but is **not** wired into the frontend.
