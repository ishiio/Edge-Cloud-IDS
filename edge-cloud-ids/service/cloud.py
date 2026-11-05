"""Cloud tier: higher-capacity model (XGBoost / DNN) behind a small HTTP API.

Run:  uvicorn service.cloud:app --port 8001
The gateway (service/app.py) calls POST /v1/classify only for flows its confidence gate forwards.
"""
from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ids.config import ARTIFACTS
from ids.models import CloudModel

app = FastAPI(title="Edge-Cloud IDS - cloud tier", version="1.0")
_models: dict = {}
_stats = dict(requests=0, flows=0, started=time.time())


def model_keys():
    return sorted(p.name for p in ARTIFACTS.iterdir()
                  if p.is_dir() and any((p / f).exists() for f in ("cloud.json", "cloud.joblib"))) if ARTIFACTS.exists() else []


def _model(key: str):
    if key not in _models:
        p = ARTIFACTS / key
        if key not in model_keys():
            raise HTTPException(404, f"no cloud model for dataset {key!r}")
        _models[key] = CloudModel.load(p)
    return _models[key]


class ClassifyRequest(BaseModel):
    dataset: str
    features: list[list[float]]


@app.on_event("startup")
def _warm():
    for key in model_keys():
        _model(key)


@app.get("/health")
def health():
    return dict(status="ok", datasets=model_keys(),
                **_stats)


@app.post("/v1/classify")
def classify(req: ClassifyRequest):
    t = time.perf_counter()
    X = np.asarray(req.features, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise HTTPException(422, "features must be a non-empty 2-D array")
    m = _model(req.dataset)
    labels = m.predict(X)
    _stats["requests"] += 1
    _stats["flows"] += len(X)
    return dict(labels=labels.tolist(), model=m.kind, inference_ms=(time.perf_counter() - t) * 1e3)
