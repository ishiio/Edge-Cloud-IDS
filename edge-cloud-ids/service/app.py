"""Edge gateway: REST API + live dashboard.

Run:  uvicorn service.app:app --port 8000        (or: python -m service.run)
Set CLOUD_URL=http://host:8001 to use the separately deployed cloud tier; otherwise the cloud
model runs in-process and the wide-area round trip is simulated (as in the paper).
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ids.config import ARTIFACTS, DATASETS
from service.engine import Bundle, Gateway, RemoteCloud, available_datasets

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_ROWS = 5000

app = FastAPI(title="Edge-Cloud AIoT Intrusion Detection", version="1.0")
_gw: Gateway | None = None


def gateway() -> Gateway:
    global _gw
    if _gw is None:
        keys = available_datasets()
        if not keys:
            raise HTTPException(503, "no trained artifacts - run `python -m ids.train` first")
        bundles = {k: Bundle.load(k) for k in keys}
        url = os.environ.get("CLOUD_URL")
        _gw = Gateway(bundles, cloud=RemoteCloud(url) if url else None)
    return _gw


# --------------------------------------------------------------------- live stream
class Streamer:
    def __init__(self):
        self.running = False
        self.rate = 20
        self.scenario = "natural"
        self.thread: threading.Thread | None = None
        self.burst_until = 0.0

    def start(self, rate=None, scenario=None):
        if rate is not None:
            self.rate = int(max(1, min(rate, 500)))
        if scenario is not None:
            self.scenario = scenario
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        gw = gateway()
        tick = 0.25
        carry = 0.0
        while self.running:
            t0 = time.time()
            carry += self.rate * tick
            n = int(carry)
            carry -= n
            if n:
                scen = self.scenario
                try:
                    if time.time() < self.burst_until:
                        scen = "attack"
                    df, y = gw.sample(n, scen)
                    gw.classify_frame(df, truth=y)
                except Exception as e:  # keep the stream alive; surface the error in the state
                    gw.last_error = str(e)
            time.sleep(max(0.0, tick - (time.time() - t0)))


stream = Streamer()


# --------------------------------------------------------------------- schemas
class ConfigIn(BaseModel):
    dataset: str | None = None
    policy: str | None = None
    tau: float | None = Field(None, description="null -> keep, use auto=true for tau*")
    auto_tau: bool | None = None
    delta_ms: float | None = None


class StreamIn(BaseModel):
    running: bool
    rate: int | None = None
    scenario: str | None = None


class SimIn(BaseModel):
    n: int = Field(200, ge=1, le=2000)
    scenario: str = "natural"


class RecordsIn(BaseModel):
    records: list[dict] = Field(..., min_length=1, max_length=MAX_UPLOAD_ROWS)


# --------------------------------------------------------------------- routes
@app.get("/api/state")
def state():
    gw = gateway()
    s = gw.snapshot()
    s["stream"] = dict(running=stream.running, rate=stream.rate, scenario=stream.scenario,
                       burst=time.time() < stream.burst_until)
    s["datasets"] = {k: dict(title=b.meta["title"], source=b.meta["source"], classes=b.classes,
                             benign=b.meta["benign"], temperature=b.meta["temperature"],
                             size_kb=b.meta["edge_size_kb_int8"], size_kb_fp32=b.meta["edge_size_kb_fp32"],
                             cloud=b.meta["cloud_kind"], n_edge=b.meta["n_edge_features"],
                             n_full=b.meta["n_full_features"], le_ms=b.meta["le_ms"], lc_ms=b.meta["lc_ms"])
                     for k, b in gw.bundles.items()}
    return s


@app.post("/api/config")
def set_config(c: ConfigIn):
    gw = gateway()
    try:
        kw = dict(dataset=c.dataset, policy=c.policy, delta_ms=c.delta_ms)
        if c.auto_tau:
            kw["tau"] = None
        elif c.tau is not None:
            kw["tau"] = c.tau
        gw.set_config(**kw)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return state()


@app.post("/api/stream")
def set_stream(s: StreamIn):
    if s.scenario is not None:
        b = gateway().bundle
        if s.scenario not in ("natural", "benign", "attack", *b.classes):
            raise HTTPException(422, f"unknown scenario {s.scenario!r}")
    stream.start(s.rate, s.scenario) if s.running else stream.stop()
    if not s.running and s.scenario:
        stream.scenario = s.scenario
    return dict(running=stream.running, rate=stream.rate, scenario=stream.scenario)


@app.post("/api/burst")
def burst(seconds: float = 8.0):
    """Simulate an attack burst: the stream switches to attack-only traffic for a few seconds."""
    stream.burst_until = time.time() + max(1.0, min(seconds, 60.0))
    stream.start()
    return dict(burst_until=stream.burst_until)


@app.post("/api/reset")
def reset():
    gateway().reset()
    return dict(ok=True)


@app.post("/api/simulate")
def simulate(s: SimIn):
    gw = gateway()
    if s.scenario not in ("natural", "benign", "attack", *gw.bundle.classes):
        raise HTTPException(422, f"unknown scenario {s.scenario!r}")
    df, y = gw.sample(s.n, s.scenario)
    res = gw.classify_frame(df, truth=y)
    return dict(n=len(res), forwarded=sum(r["forwarded"] for r in res),
                accuracy=100 * float(np.mean([r["correct"] for r in res])))


@app.get("/api/sample")
def sample(n: int = 5, scenario: str = "natural"):
    """Raw example flows (with ground truth) that can be POSTed back to /api/classify."""
    gw = gateway()
    df, y = gw.sample(min(max(n, 1), 50), scenario)
    recs = json.loads(df.to_json(orient="records"))
    return dict(dataset=gw.cfg.dataset, flows=[dict(features=r, truth=gw.bundle.classes[int(t)])
                                               for r, t in zip(recs, y)])


@app.post("/api/classify")
def classify(body: RecordsIn):
    gw = gateway()
    try:
        out = gw.classify_records(body.records)
    except Exception as e:
        raise HTTPException(422, f"could not classify records: {e}")
    return dict(dataset=gw.cfg.dataset, policy=gw.cfg.policy, tau=gw.effective_tau(), results=out)


@app.post("/api/classify-csv")
async def classify_csv(file: UploadFile = File(...)):
    gw = gateway()
    raw = await file.read()
    if len(raw) > 20_000_000:
        raise HTTPException(413, "file too large (20 MB max)")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=MAX_UPLOAD_ROWS, low_memory=False)
    except Exception as e:
        raise HTTPException(422, f"not a readable CSV: {e}")
    if df.empty:
        raise HTTPException(422, "CSV has no rows")
    b = gw.bundle
    known = set(b.pre.input_columns)
    overlap = len(known & set(df.columns))
    if overlap < max(3, 0.3 * len(known)):
        raise HTTPException(422, f"CSV shares only {overlap}/{len(known)} expected columns with the "
                                 f"{b.meta['title']} schema; see /api/schema")
    truth = None
    lab = DATASETS[gw.cfg.dataset].label_column
    if lab in df.columns:
        cmap = {c: i for i, c in enumerate(b.classes)}
        t = df[lab].map(cmap)
        if t.notna().all():
            truth = t.astype(int).to_numpy()
    res = gw.classify_frame(df.drop(columns=[lab], errors="ignore"), truth=truth, record=False)
    counts: dict[str, int] = {}
    for r in res:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    f = sum(r["forwarded"] for r in res)
    return dict(dataset=gw.cfg.dataset, n=len(res), forwarded=f, forwarded_pct=100 * f / len(res),
                alerts=sum(r["alert"] for r in res), by_label=counts,
                accuracy=(100 * float(np.mean([r["correct"] for r in res])) if truth is not None else None),
                matched_columns=overlap, expected_columns=len(known), results=res[:500])


@app.get("/api/schema")
def schema():
    b = gateway().bundle
    return dict(dataset=b.key, input_columns=b.pre.input_columns, edge_features=b.pre.edge_feature_names,
                classes=b.classes, benign=b.meta["benign"])


@app.get("/api/curve")
def curve(dataset: str | None = None, delta_ms: float | None = None):
    gw = gateway()
    if dataset and dataset not in gw.bundles:
        raise HTTPException(404, "unknown dataset")
    return gw.curve(dataset, delta_ms)


@app.get("/api/results/{dataset}")
def results(dataset: str):
    gw = gateway()
    if dataset not in gw.bundles:
        raise HTTPException(404, "unknown dataset")
    r = dict(gw.bundles[dataset].results)
    return r


@app.get("/api/benchmark")
def benchmark():
    p = ARTIFACTS / "benchmark.json"
    if not p.exists():
        return JSONResponse(dict(available=False))
    data = json.loads(p.read_text())
    # Never mix historical synthetic benchmarks with newly trained real models.
    for key, bundle in gateway().bundles.items():
        meta = json.loads((ARTIFACTS / key / "meta.json").read_text())
        if key not in data or data[key].get("source") != meta.get("source"):
            return dict(available=False, reason="Benchmark source does not match deployed models")
    return dict(available=True, **data)


@app.get("/api/audit")
def audit():
    """Real-data integrity audit written by scripts/audit_real_data.py."""
    p = ARTIFACTS / "audit.json"
    return dict(available=True, **json.loads(p.read_text())) if p.exists() else dict(available=False)


@app.get("/api/health")
def health():
    gw = gateway()
    return dict(status="ok", datasets=list(gw.bundles), cloud=gw.cloud.kind)


@app.get("/api/datasets/{dataset}/download")
def download_dataset(dataset: str):
    from ids.data import find_real_file
    if dataset not in DATASETS:
        raise HTTPException(404, "Unknown dataset")
    path = find_real_file(DATASETS[dataset])
    if path is None:
        raise HTTPException(404, "Dataset CSV is missing from the data folder")
    return FileResponse(path, media_type="text/csv", filename=path.name)


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
