"""Edge gateway + cloud engine (framework-agnostic so it is unit-testable without HTTP).

Implements Algorithm 1 of the paper for batches of flows:
  preprocess -> compact INT8 edge MLP -> temperature-scaled confidence -> gate (policy A/B)
  -> forwarded flows are classified by the cloud model, whose label replaces the edge label.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from ids import gate
from ids.config import ARTIFACTS, DEFAULT_DELTA_MS
from ids.models import CloudModel, EdgeModel, softmax


# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    key: str
    meta: dict
    pre: object
    edge: EdgeModel
    cloud: object
    eval: dict
    replay_raw: pd.DataFrame
    replay_y: np.ndarray
    results: dict
    _curve_cache: dict = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def benign_idx(self) -> int:
        return self.meta["benign_idx"]

    @classmethod
    def load(cls, key: str, root: Path = ARTIFACTS) -> "Bundle":
        d = root / key
        meta = json.loads((d / "meta.json").read_text())
        rp = joblib.load(d / "replay.joblib")
        z = np.load(d / "eval.npz")
        return cls(key=key, meta=meta, pre=joblib.load(d / "preprocessor.joblib"),
                   edge=EdgeModel.load(d / "edge_int8.npz"), cloud=CloudModel.load(d),
                   eval={k: z[k] for k in z.files}, replay_raw=rp["raw"], replay_y=rp["y"],
                   results=json.loads((d / "results.json").read_text()))


def available_datasets(root: Path = ARTIFACTS) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()) if root.exists() else []


# ---------------------------------------------------------------------------
class LocalCloud:
    """In-process cloud tier. The wide-area round trip is *simulated* (Delta), as in the paper."""
    kind = "local (simulated round trip)"

    def __init__(self, bundles: dict[str, Bundle]):
        self.bundles = bundles

    def classify(self, key: str, X_full: np.ndarray) -> tuple[np.ndarray, float | None]:
        return self.bundles[key].cloud.predict(X_full), None


class RemoteCloud:
    """HTTP client for the separately deployed cloud service (service/cloud.py)."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.kind = f"remote ({self.url})"
        self.client = httpx.Client(timeout=timeout)

    def classify(self, key: str, X_full: np.ndarray) -> tuple[np.ndarray, float | None]:
        t = time.perf_counter()
        r = self.client.post(f"{self.url}/v1/classify", json=dict(dataset=key, features=X_full.tolist()))
        r.raise_for_status()
        # measured HTTP + cloud-inference time, amortised over the flows in this batch
        per_flow = (time.perf_counter() - t) * 1e3 / max(len(X_full), 1)
        return np.asarray(r.json()["labels"], dtype=int), per_flow


# ---------------------------------------------------------------------------
@dataclass
class Config:
    dataset: str
    policy: str = "A"
    tau: float | None = None          # None -> use the validation-selected tau* for the policy
    delta_ms: float = DEFAULT_DELTA_MS


class Gateway:
    def __init__(self, bundles: dict[str, Bundle], cloud=None, history: int = 400):
        assert bundles, "no trained artifacts found - run `python -m ids.train` first"
        self.bundles = bundles
        self.cloud = cloud or LocalCloud(bundles)
        self.cfg = Config(dataset=next(iter(bundles)))
        self.lock = threading.RLock()
        self.recent: collections.deque = collections.deque(maxlen=history)
        self.alerts: collections.deque = collections.deque(maxlen=200)
        self.rng = np.random.default_rng(7)
        self._reset_stats()
        self._seq = 0
        self.last_error = None

    # -- config ----------------------------------------------------------
    @property
    def bundle(self) -> Bundle:
        return self.bundles[self.cfg.dataset]

    def effective_tau(self) -> float:
        if self.cfg.tau is not None:
            return self.cfg.tau
        return float(self.bundle.meta["tau_star"][self.cfg.policy])

    def set_config(self, dataset=None, policy=None, tau="__keep__", delta_ms=None) -> Config:
        with self.lock:
            if dataset is not None:
                if dataset not in self.bundles:
                    raise ValueError(f"unknown dataset {dataset!r}")
                if dataset != self.cfg.dataset:
                    self.cfg.dataset = dataset
                    self.cfg.tau = None
                    self._reset_stats()
            if policy is not None:
                if policy.upper() not in ("A", "B"):
                    raise ValueError("policy must be 'A' or 'B'")
                self.cfg.policy = policy.upper()
            if tau != "__keep__":
                if tau is not None and not (0.0 < tau <= 1.0):
                    raise ValueError("tau must be in (0, 1]")
                self.cfg.tau = tau
            if delta_ms is not None:
                if not (0 <= delta_ms <= 10000):
                    raise ValueError("delta_ms must be in [0, 10000]")
                self.cfg.delta_ms = float(delta_ms)
            return self.cfg

    # -- stats -----------------------------------------------------------
    def _reset_stats(self):
        self.stats = dict(total=0, forwarded=0, alerts=0, labelled=0, correct=0, correct_edge_only=0,
                          lat_sum=0.0, cloud_lat_sum=0.0, bytes_fwd=0, bytes_all=0,
                          started=time.time(), by_label=collections.Counter(),
                          fwd_by_label=collections.Counter())
        self.recent.clear()
        self.alerts.clear()

    def reset(self):
        with self.lock:
            self._reset_stats()

    # -- core: Algorithm 1 for a batch ------------------------------------
    def classify_frame(self, df: pd.DataFrame, truth: np.ndarray | None = None,
                       record: bool = True) -> list[dict]:
        with self.lock:
            b, cfg = self.bundle, self.cfg
            tau, policy, delta = self.effective_tau(), cfg.policy, cfg.delta_ms
        t0 = time.perf_counter()
        X = b.pre.transform_records(df.to_dict("records"))
        logits = b.edge.logits(b.pre.edge(X))
        probs = softmax(logits, b.meta["temperature"])
        conf, pe = gate.confidence(probs)
        fwd = gate.forward_mask(conf, pe, tau, policy, b.benign_idx)
        labels = pe.copy()
        rtt = None
        if fwd.any():
            cl, rtt = self.cloud.classify(b.key, X[fwd])
            labels[fwd] = cl
        compute_ms = (time.perf_counter() - t0) * 1e3

        le, lc = b.meta["le_ms"], b.meta["lc_ms"]
        fwd_lat = delta + (rtt if rtt is not None else lc)   # simulated WAN delta + (measured | profiled) cloud time
        n_feat = X.shape[1]
        out = []
        with self.lock:
            now = time.time()
            for i in range(len(df)):
                self._seq += 1
                lat = le + (fwd_lat if fwd[i] else 0.0)
                lab = b.classes[labels[i]]
                d = dict(id=self._seq, ts=now, label=lab, edge_label=b.classes[pe[i]],
                         confidence=float(conf[i]), forwarded=bool(fwd[i]),
                         decided_by="cloud" if fwd[i] else "edge",
                         alert=bool(labels[i] != b.benign_idx), latency_ms=float(lat),
                         cloud_only_latency_ms=float(delta + lc), edge_cloud_disagree=bool(fwd[i] and labels[i] != pe[i]),
                         top3=[dict(label=b.classes[j], p=float(probs[i, j])) for j in np.argsort(-probs[i])[:3]])
                if truth is not None:
                    d["truth"] = b.classes[int(truth[i])]
                    d["correct"] = bool(labels[i] == truth[i])
                out.append(d)
                if record:
                    s = self.stats
                    s["total"] += 1
                    s["forwarded"] += int(fwd[i])
                    s["alerts"] += int(d["alert"])
                    s["lat_sum"] += lat
                    s["cloud_lat_sum"] += delta + lc
                    s["bytes_fwd"] += int(fwd[i]) * n_feat * 4
                    s["bytes_all"] += n_feat * 4
                    s["by_label"][lab] += 1
                    if fwd[i]:
                        s["fwd_by_label"][lab] += 1
                    if truth is not None:
                        s["labelled"] += 1
                        s["correct"] += int(d["correct"])
                        s["correct_edge_only"] += int(pe[i] == truth[i])
                    self.recent.append(dict(id=d["id"], ts=now, conf=d["confidence"], fwd=d["forwarded"],
                                            alert=d["alert"], label=lab, truth=d.get("truth"),
                                            lat=d["latency_ms"]))
                    if d["alert"]:
                        self.alerts.appendleft(dict(d, compute_ms=compute_ms / len(df)))
        return out

    def classify_records(self, records: list[dict]) -> list[dict]:
        return self.classify_frame(pd.DataFrame(records))

    # -- replay ------------------------------------------------------------
    def sample(self, n: int = 1, scenario: str = "natural") -> tuple[pd.DataFrame, np.ndarray]:
        b = self.bundle
        y = b.replay_y
        if scenario == "benign":
            pool = np.flatnonzero(y == b.benign_idx)
        elif scenario == "attack":
            pool = np.flatnonzero(y != b.benign_idx)
        elif scenario in b.classes:
            pool = np.flatnonzero(y == b.classes.index(scenario))
        else:
            pool = np.arange(len(y))
        pick = self.rng.choice(pool, size=n, replace=len(pool) < n)
        return b.replay_raw.iloc[pick].reset_index(drop=True), y[pick]

    def snapshot(self) -> dict:
        with self.lock:
            s = self.stats
            n = max(s["total"], 1)
            f = s["forwarded"] / n
            b = self.bundle
            le, lc, delta = b.meta["le_ms"], b.meta["lc_ms"], self.cfg.delta_ms
            return dict(
                config=dict(dataset=self.cfg.dataset, policy=self.cfg.policy, tau=self.effective_tau(),
                            tau_is_auto=self.cfg.tau is None, delta_ms=delta,
                            tau_star=b.meta["tau_star"]),
                cloud_tier=self.cloud.kind,
                totals=dict(total=s["total"], forwarded=s["forwarded"], edge_decided=s["total"] - s["forwarded"],
                            alerts=s["alerts"], forwarded_pct=100 * f, traffic_reduction_pct=100 * (1 - f),
                            mean_latency_ms=s["lat_sum"] / n, cloud_only_latency_ms=s["cloud_lat_sum"] / n,
                            latency_reduction_pct=100 * (1 - s["lat_sum"] / max(s["cloud_lat_sum"], 1e-9)) if s["total"] else 0.0,
                            accuracy_pct=100 * s["correct"] / s["labelled"] if s["labelled"] else None,
                            edge_only_accuracy_pct=100 * s["correct_edge_only"] / s["labelled"] if s["labelled"] else None,
                            bytes_forwarded=s["bytes_fwd"], bytes_cloud_only=s["bytes_all"],
                            breakeven_fraction_pct=100 * gate.breakeven_fraction(le, lc, delta),
                            uptime_s=time.time() - s["started"]),
                last_error=self.last_error,
                by_label=dict(s["by_label"]), fwd_by_label=dict(s["fwd_by_label"]),
                recent=list(self.recent)[-250:], alerts=list(self.alerts)[:40])

    # -- operating curve over the held-out test set --------------------------
    def curve(self, dataset: str | None = None, delta_ms: float | None = None) -> dict:
        b = self.bundles[dataset or self.cfg.dataset]
        delta = self.cfg.delta_ms if delta_ms is None else delta_ms
        le, lc = b.meta["le_ms"], b.meta["lc_ms"]
        K = len(b.classes)
        e = b.eval
        grid = [round(t, 3) for t in list(np.linspace(0.30, 0.95, 27)) + [0.97, 0.99, 0.999]]
        if b.key not in b._curve_cache:
            cache = {}
            for p in "AB":
                cache[p] = [gate.evaluate_operating_point(e["y"], e["pred_edge"], e["conf"], e["pred_cloud"], t, p,
                                                          b.benign_idx, K, le, lc, 0.0) for t in grid]
            b._curve_cache[b.key] = cache
        cache = b._curve_cache[b.key]
        out = {}
        for p in "AB":
            pts = []
            for r in cache[p]:
                f = r["fwd"] / 100
                pts.append(dict(tau=r["tau"], f1=r["f1"], acc=r["acc"], fwd=r["fwd"],
                                latency_ms=gate.mean_latency(f, le, lc, delta),
                                beats_cloud_only=bool(f < gate.breakeven_fraction(le, lc, delta))))
            out[p] = pts
        edge_only = gate.metrics(e["y"], e["pred_edge"], K)
        cloud_only = gate.metrics(e["y"], e["pred_cloud"], K)
        return dict(dataset=b.key, delta_ms=delta, curves=out, edge_only_f1=edge_only["f1"],
                    cloud_only_f1=cloud_only["f1"], cloud_only_latency_ms=delta + lc, edge_latency_ms=le,
                    tau_star=b.meta["tau_star"], breakeven_fraction_pct=100 * gate.breakeven_fraction(le, lc, delta))
