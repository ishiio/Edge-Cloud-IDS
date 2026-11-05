# Edge-Cloud IDS

A working implementation of *A Lightweight Edge-Cloud AIoT Framework for Efficient Intrusion Detection in IoT Networks*:
a compact INT8 MLP on the gateway classifies every flow, a calibrated confidence gate keeps the decisions it is sure of,
and the rest go to an XGBoost cloud model. Includes the training pipeline, the two services, a live dashboard, a
threshold-analysis lab and the five-seed benchmark.

## Quick start
```bash
pip install -r requirements.txt
python -m ids.train --dataset all          # trains edge + cloud models on the CSVs in data/ (~15 min on 2 cores)
python -m service.run                      # dashboard at http://127.0.0.1:8000
python -m service.run --split              # cloud tier on :8001, gateway on :8000 (real HTTP hop)
python -m ids.train --benchmark --seeds 5  # five-seed study -> artifacts/benchmark.json (~35 min on 2 cores)
pytest                                     # 23 tests
python scripts/audit_real_data.py          # leakage / overlap audit -> artifacts/audit.json
```
Trained artifacts for both datasets are already in `artifacts/`, so `python -m service.run` works immediately.

## Data and evaluation
`data/` holds the real Edge-IIoTset (ML file) and TON_IoT (network file) CSVs, checksums in `data/manifest.json`. Exact
de-duplication gives **156,986** and **190,474** flows, identical to the paper. All numbers below are **real data**, five seeds
(0-4), test set, mean ± std, simulated Δ = 50 ms. They come from an independent implementation and are not a verified
reproduction of the paper (the paper's absolute scores differ, see notes).

| | Edge-Only INT8 per-tensor (paper) | Edge-Only INT8 per-channel (deployed) | Cloud-Only | Policy A | Policy B |
|---|---|---|---|---|---|
| **Edge-IIoTset** macro-F1 % | 90.57 ± 0.87 | 91.46 ± 0.45 | 92.12 ± 0.13 | 91.75 ± 0.15 | 92.12 ± 0.13 |
| forwarded % / latency ms | - | 0 / 0.01 | 100 / 50.7 | **8.0 ± 1.8 / 4.1** | 84.5 / 42.9 |
| **TON_IoT** macro-F1 % | 86.82 ± 5.36 | 94.67 ± 0.30 | 96.84 ± 0.11 | 96.43 ± 0.17 | 96.62 ± 0.24 |
| forwarded % / latency ms | - | 0 / 0.02 | 100 / 50.7 | **3.8 ± 1.9 / 1.9** | 78.0 / 39.5 |

What held and what did not, versus the paper:
* **Held:** a calibrated gate keeps Policy A within ~0.4 macro-F1 of Cloud-Only while sending a small share of flows (latency -92% / -96%);
  the paper's per-tensor INT8 edge model is unstable on TON_IoT (σ = 5.4 F1); the forwarding rule, not the model alone, sets the operating point.
* **Differed:** the real-data gap between Edge and Cloud is smaller than in the paper (Cloud-Only 92.1 / 96.8 vs the paper's 87.0 / 95.7), so Policy A forwards
  far fewer flows (4-8%, paper 42-46%). Policy B is *not* useless on Edge-IIoTset: +0.37 ± 0.10 F1 over A (consistent across seeds) for ~10× the cloud traffic;
  on TON_IoT the difference (+0.19 ± 0.25) is within noise.

**Deviations from the paper, and why**
* **INT8 is per-output-channel**, not per-tensor. Per-tensor INT8 rounds 22-28% of the weights to zero when a few outliers set the scale
  (TON_IoT layer-1 |w|max = 19.8) and drops F1 from 95.1 to 78.1 on seed 0. Per-channel keeps 8-bit weights (+0.4 KB of scales) and matches FP32.
  The paper's per-tensor scheme is still evaluated and shown as its own row (`edge_int8_pt`).
* **Identifier handling follows the paper literally**: only IP addresses and timestamps (plus the target-leaking binary `label` / `Attack_label`) are dropped;
  ports stay on TON_IoT; text-like categoricals are encoded with their top-12 training values. For Edge-IIoTset the dataset authors' standard ML-file column drops
  are used (payload/URI text fields and port columns) - see `ids/config.py`.
* The five-seed benchmark uses XGBoost only (`--no-dnn`) for run time; the deployed seed-0 artifacts compared XGBoost and the DNN on validation, and XGBoost won, as in the paper.

**Integrity audit** (`python scripts/audit_real_data.py` -> `artifacts/audit.json`, shown on the Benchmark page): row counts match the paper; no label column reaches the models;
train/test split is disjoint; only 3.8% / 6.6% of test flows have an identical model-input vector in training and Policy A scores are the same or higher on the never-seen-vector subset
(Edge-IIoTset 93.3 vs 91.8, TON_IoT 96.2 vs 96.3 F1); the best single feature alone reaches 16% / 41% macro-F1, so no column leaks the class.
Weak spots remain: Edge-IIoTset Fingerprinting recall is ~58%.

Bug fixed in the supplied v2 project: the identifier heuristic matched loose name tokens and silently dropped real features (`src_ip_bytes`, `dst_ip_bytes`,
`mbtcp.unit_id`, `mbtcp.trans_id`, `udp.time_delta`) and the config dropped ports and all text columns on TON_IoT; that cost ~9 F1 on TON_IoT (87.5 vs 96.9 Cloud-Only).

## What maps to the paper
| Paper | Where |
|---|---|
| Preprocessing: dedupe, drop identifiers, encode, log-transform, min-max (train stats only), |r|>0.95 filter + tree importance, split *before* fitting | `ids/preprocess.py`, `ids/pipeline.py` |
| Edge MLP 64-32, Adam, early stopping, INT8 (simulated; per-channel deployed, per-tensor from the paper also reported) | `ids/models.py` |
| Cloud: XGBoost (250 trees, depth 8) vs 256-128-64 DNN, best on validation | `ids/models.py` |
| Temperature scaling Tc, confidence c, Policies A and B (Eqs. 1-3, Alg. 1) | `ids/gate.py`, `service/engine.py` |
| Latency model, break-even, τ* selection (Eqs. 4-6, ε = 0.005, validation only) | `ids/gate.py` |
| Stratified 80/20 split, 10% validation, seeds 0-4, mean ± std, Tables II-III, Figs. 3-4 | `ids/train.py`, dashboard Benchmark / Threshold lab |

## Services
* **Gateway** `service/app.py` — dashboard, live stream, `/api/classify` (JSON), `/api/classify-csv`, `/api/config`, `/api/curve`, `/api/benchmark`.
  OpenAPI docs at `/docs`.
* **Cloud** `service/cloud.py` — `POST /v1/classify`; only forwarded flows reach it. Set `CLOUD_URL` on the gateway to use it.

In single-process mode the WAN round trip Δ is **simulated** (adjustable, default 50 ms), as in the paper; `Le`/`Lc` are measured median single-flow times.
In split mode the cloud inference + HTTP time is measured and Δ is added on top.

## Limitations (same as the paper, plus)
Recorded/synthetic flows replace live capture; Δ and INT8 are simulated; timings are from a cloud container, not a gateway device;
the edge model is not distilled; no drift or unseen-attack handling. The live stream replays held-out test flows. No authentication on the API — run it on a trusted network.

## Dashboard v2: datasets included

Start with `START_HERE.txt` after extracting the complete archive.

- Double-click `OPEN_DASHBOARD.html` for a portable, read-only dashboard. It embeds actual dataset previews, source statistics, seed-0 model results and threshold curves. No Python server is needed for browsing.
- Double-click `START_WINDOWS.cmd`, or run `python3 launch.py` on macOS/Linux, for live replay and new classifications. The launcher creates `.venv`, installs requirements on the first run and opens the browser once the API is ready. Python 3.11+ is required.
- The Datasets page shows both real CSVs, class distributions, row search, class filters, pagination, all-column previews, a feature dictionary, and full CSV downloads.
- The preview includes up to five actual rows per class. Counts and distributions describe the full raw CSV; model metrics describe the deduplicated evaluation data. The preview is not a random sample of traffic proportions.
- If the API becomes unavailable, the UI explicitly switches to the saved snapshot and disables live inference controls. Reconnect reloads the app.

The snapshot is bundled data, not a continuously refreshed catalogue. After replacing CSVs or retraining models, regenerate it and the portable HTML:

```bash
python scripts/build_snapshot.py
python scripts/build_offline.py
```

Pandas 3 or newer is required to load the bundled serialized replay data.
Full CSVs and seed-0 models are included. Live replay uses held-out test records.

## Dark and light themes
The dashboard now opens in dark mode. Use **Theme: Dark / Theme: Light** in the top bar to switch. Your browser remembers the choice when local storage is available. This works in both the portable HTML and the Python-served dashboard.

Windows cloud-model compatibility fix
-------------------------------------
Both bundled XGBoost cloud models now use native cloud.json files instead of
cloud.joblib memory snapshots. This avoids the XGBoost pickle-restore path
that raised "input stream corrupted" on Windows. No retraining is needed.
The launcher checks all models before starting the server.
Extract this updated ZIP into a NEW folder and run START_WINDOWS.cmd.
The dark/light theme switch and both real datasets are included.
