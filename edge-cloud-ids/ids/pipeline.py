"""End-to-end training / evaluation pipeline following Sec. III-D of the paper.

(i) split train / val / test  (ii) fit preprocessing on the training partition only
(iii) train cloud + edge  (iv) calibrate Tc on validation  (v) select tau* per policy on validation
(vi) evaluate the selected operating point on the untouched test set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from . import gate
from .config import ARTIFACTS, DEFAULT_DELTA_MS, DatasetSpec, EDGE_QUANT, TAU_GRID
from .data import load_dataset
from .models import (EdgeModel, fit_temperature, softmax, time_single_flow, train_cloud,
                     train_edge_mlp)
from .preprocess import Preprocessor


def _log(msg, verbose=True):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(spec: DatasetSpec, seed: int = 0, n: int | None = None, save: bool = False,
        try_dnn: bool = True, verbose: bool = True, xgb_params: dict | None = None) -> dict:
    t_start = time.time()
    df, label_col, source = load_dataset(spec, seed=seed, n=n)
    classes = [c for c in spec.classes if c in set(df[label_col].unique())]
    cidx = {c: i for i, c in enumerate(classes)}
    y = df[label_col].map(cidx).to_numpy()
    K = len(classes)
    benign_idx = cidx[spec.benign]
    _log(f"{spec.title}: {len(df):,} de-duplicated flows, {K} classes, source={source}", verbose)

    feats = df.drop(columns=[label_col])
    # (i) stratified 80/20 train/test, 10% validation subset of the training part
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    fit_idx, va_idx = train_test_split(tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=seed)

    # (ii) preprocessing fitted on the training partition only
    pre = Preprocessor(spec.n_edge_features, drop_columns=spec.drop_columns, seed=seed)
    pre.fit(feats.iloc[fit_idx], y[fit_idx])
    Xf = pre.transform(feats.iloc[fit_idx])
    Xv = pre.transform(feats.iloc[va_idx])
    Xt = pre.transform(feats.iloc[te_idx])
    yf, yv, yt = y[fit_idx], y[va_idx], y[te_idx]
    _log(f"preprocessing: {Xf.shape[1]} full / {len(pre.edge_idx)} edge features; "
         f"dropped identifiers: {pre.dropped_ids}", verbose)

    # (iii) models
    t0 = time.time()
    fp32 = train_edge_mlp(pre.edge(Xf), yf, pre.edge(Xv), yv, K, seed=seed)
    int8 = fp32.quantize_int8(EDGE_QUANT)
    int8_pt = fp32.quantize_int8("per_tensor")        # paper's scheme, reported for comparison
    _log(f"edge MLP trained in {time.time()-t0:.1f}s  ({fp32.size_kb:.1f} KB fp32 / {int8.size_kb:.1f} KB int8)", verbose)
    t0 = time.time()
    cloud, cinfo = train_cloud(Xf, yf, Xv, yv, K, seed=seed, try_dnn=try_dnn, xgb_params=xgb_params)
    _log(f"cloud model: {cinfo}  ({time.time()-t0:.1f}s)", verbose)

    # (iv) temperature scaling on the *deployed* (INT8) edge logits, validation data only
    Tc = fit_temperature(int8.logits(pre.edge(Xv)), yv)

    def edge_outputs(X):
        lg = int8.logits(pre.edge(X))
        conf, pred = gate.confidence(softmax(lg, Tc))
        return conf, pred

    conf_v, pe_v = edge_outputs(Xv)
    pc_v = cloud.predict(Xv)
    conf_t, pe_t = edge_outputs(Xt)
    pc_t = cloud.predict(Xt)
    pe32_t = fp32.logits(pre.edge(Xt)).argmax(1)

    # (v) tau* per policy, validation only
    sel = {p: gate.select_threshold(yv, pe_v, conf_v, pc_v, p, benign_idx, K) for p in "AB"}

    # measured single-flow timings
    x1 = Xt[:1]
    le = time_single_flow(lambda a: int8.logits(pre.edge(a)), x1)
    lc = time_single_flow(cloud.predict_proba, x1, reps=120)
    delta = DEFAULT_DELTA_MS

    # (vi) test-set evaluation
    def row(name, yhat, fwd_pct, lat, size):
        m = gate.metrics(yt, yhat, K)
        m.update(method=name, fwd=fwd_pct, latency_ms=lat, size_kb=size)
        return m

    methods = {
        "edge_fp32": row("Edge-Only (FP32)", pe32_t, 0.0, le, fp32.size_kb),
        "edge_int8_pt": row("Edge-Only (INT8 per-tensor, paper)", int8_pt.logits(pre.edge(Xt)).argmax(1), 0.0, le, int8_pt.size_kb),
        "edge_int8": row("Edge-Only (INT8 per-channel)", int8.logits(pre.edge(Xt)).argmax(1), 0.0, le, int8.size_kb),
        "cloud_only": row("Cloud-Only", pc_t, 100.0, delta + lc, None),
    }
    for p in "AB":
        op = gate.evaluate_operating_point(yt, pe_t, conf_t, pc_t, sel[p]["tau"], p, benign_idx, K, le, lc, delta)
        methods[f"policy_{p.lower()}"] = dict(op, method=f"Edge-Cloud, Policy {p}", size_kb=int8.size_kb)

    sweeps = {p: gate.sweep(yt, pe_t, conf_t, pc_t, p, benign_idx, K, le, lc, delta) for p in "AB"}

    def per_class_recall(yhat):
        r = recall_score(yt, yhat, labels=np.arange(K), average=None, zero_division=0)
        return {classes[i]: float(r[i] * 100) for i in range(K)}

    fwdA = gate.forward_mask(conf_t, pe_t, sel["A"]["tau"], "A", benign_idx)
    per_class = {
        "edge_int8": per_class_recall(int8.logits(pre.edge(Xt)).argmax(1)),
        "cloud_only": per_class_recall(pc_t),
        "policy_a": per_class_recall(gate.final_labels(pe_t, pc_t, fwdA)),
    }
    class_counts = {classes[i]: int((yt == i).sum()) for i in range(K)}

    results = dict(
        dataset=spec.key, title=spec.title, source=source, seed=seed, n_flows=int(len(df)),
        n_classes=K, classes=classes, benign=spec.benign, n_test=int(len(te_idx)),
        n_full_features=int(Xf.shape[1]), n_edge_features=int(len(pre.edge_idx)),
        temperature=Tc, selection=sel, le_ms=le, lc_ms=lc, delta_ms=delta, cloud=cinfo,
        methods=methods, sweeps=sweeps, per_class_recall=per_class, test_class_counts=class_counts,
        preprocessing=pre.summary(), runtime_s=round(time.time() - t_start, 1),
    )
    _log(f"done in {results['runtime_s']}s | cloud F1 {methods['cloud_only']['f1']:.2f} | "
         f"A@{sel['A']['tau']}: F1 {methods['policy_a']['f1']:.2f}, fwd {methods['policy_a']['fwd']:.1f}% | "
         f"B@{sel['B']['tau']}: F1 {methods['policy_b']['f1']:.2f}, fwd {methods['policy_b']['fwd']:.1f}%", verbose)

    if save:
        out = ARTIFACTS / spec.key
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pre, out / "preprocessor.joblib")
        fp32.save(out / "edge_fp32.npz")
        int8.save(out / "edge_int8.npz")
        cloud.save(out)
        np.savez_compressed(out / "eval.npz", y=yt, pred_edge=pe_t, conf=conf_t, pred_cloud=pc_t,
                            pred_edge_fp32=pe32_t)
        # raw flows for live replay (through the full preprocessing path)
        rs = np.random.default_rng(seed).choice(len(te_idx), size=min(4000, len(te_idx)), replace=False)
        raw = feats.iloc[te_idx[rs]].reset_index(drop=True)
        joblib.dump(dict(raw=raw, y=yt[rs]), out / "replay.joblib")
        meta = dict(dataset=spec.key, title=spec.title, classes=classes, benign=spec.benign,
                    benign_idx=benign_idx, temperature=Tc, le_ms=le, lc_ms=lc,
                    tau_star={p: sel[p]["tau"] for p in "AB"}, source=source,
                    edge_size_kb_int8=int8.size_kb, edge_size_kb_fp32=fp32.size_kb,
                    cloud_kind=cloud.kind, n_full_features=int(Xf.shape[1]),
                    n_edge_features=int(len(pre.edge_idx)))
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        (out / "results.json").write_text(json.dumps(results, indent=2))
        _log(f"artifacts saved to {out}", verbose)
    return results
