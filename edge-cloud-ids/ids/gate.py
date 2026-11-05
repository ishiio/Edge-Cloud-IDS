"""Confidence gate, forwarding policies, latency model and threshold selection (paper Eqs. 1-6)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import EPSILON, TAU_GRID


def confidence(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (1): c_i = max_k p_ik, edge label = argmax_k p_ik (p already temperature-scaled)."""
    return probs.max(axis=1), probs.argmax(axis=1)


def forward_mask(conf: np.ndarray, pred_edge: np.ndarray, tau: float, policy: str,
                 benign_idx: int) -> np.ndarray:
    """Eq. (2) policy A: c < tau.   Eq. (3) policy B: c < tau  OR  predicted attack."""
    fwd = conf < tau
    if policy.upper() == "B":
        fwd = fwd | (pred_edge != benign_idx)
    return fwd


def final_labels(pred_edge, pred_cloud, fwd):
    return np.where(fwd, pred_cloud, pred_edge)


def mean_latency(f: float, le: float, lc: float, delta: float) -> float:
    """Eq. (4): L(tau) = Le + f (Delta + Lc)."""
    return le + f * (delta + lc)


def breakeven_fraction(le: float, lc: float, delta: float) -> float:
    """Eq. (5): gating beats Cloud-Only on average iff f < 1 - Le/(Delta+Lc)."""
    return 1.0 - le / (delta + lc)


def metrics(y, yhat, n_classes: int) -> dict:
    labels = np.arange(n_classes)
    kw = dict(average="macro", labels=labels, zero_division=0)
    return dict(
        acc=float(accuracy_score(y, yhat)) * 100,
        precision=float(precision_score(y, yhat, **kw)) * 100,
        recall=float(recall_score(y, yhat, **kw)) * 100,
        f1=float(f1_score(y, yhat, **kw)) * 100,
    )


def evaluate_operating_point(y, pred_edge, conf, pred_cloud, tau, policy, benign_idx, n_classes,
                             le, lc, delta) -> dict:
    fwd = forward_mask(conf, pred_edge, tau, policy, benign_idx)
    yhat = final_labels(pred_edge, pred_cloud, fwd)
    m = metrics(y, yhat, n_classes)
    f = float(fwd.mean())
    m.update(tau=float(tau), policy=policy.upper(), fwd=f * 100, traffic_reduction=(1 - f) * 100,
             latency_ms=mean_latency(f, le, lc, delta), cloud_only_latency_ms=delta + lc,
             beats_cloud_only=bool(f < breakeven_fraction(le, lc, delta)))
    return m


def sweep(y, pred_edge, conf, pred_cloud, policy, benign_idx, n_classes, le, lc, delta,
          grid=TAU_GRID) -> list[dict]:
    return [evaluate_operating_point(y, pred_edge, conf, pred_cloud, t, policy, benign_idx,
                                     n_classes, le, lc, delta) for t in grid]


def select_threshold(y_val, pred_edge_val, conf_val, pred_cloud_val, policy, benign_idx, n_classes,
                     grid=TAU_GRID, eps: float = EPSILON) -> dict:
    """Eq. (6): tau* = argmin_tau f(tau)  s.t.  macroF1(tau) >= macroF1_cloud - eps  (validation only)."""
    f1_cloud = metrics(y_val, pred_cloud_val, n_classes)["f1"] / 100
    rows = []
    for t in grid:
        fwd = forward_mask(conf_val, pred_edge_val, t, policy, benign_idx)
        f1 = metrics(y_val, final_labels(pred_edge_val, pred_cloud_val, fwd), n_classes)["f1"] / 100
        rows.append((t, float(fwd.mean()), f1))
    ok = [r for r in rows if r[2] >= f1_cloud - eps]
    if ok:
        t, f, f1 = min(ok, key=lambda r: (r[1], r[0]))
        feasible = True
    else:  # no grid point meets the tolerance: fall back to the best-quality one
        t, f, f1 = max(rows, key=lambda r: (r[2], -r[1]))
        feasible = False
    return dict(tau=t, val_fwd=f, val_f1=f1, val_f1_cloud=f1_cloud, feasible=feasible)
