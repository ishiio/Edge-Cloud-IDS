"""CLI: train deployable artifacts and/or run the multi-seed benchmark.

    python -m ids.train --dataset all              # train + save artifacts (seed 0)
    python -m ids.train --benchmark --seeds 5      # 5-seed study -> artifacts/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import pipeline
from .config import ARTIFACTS, DATASETS, SEEDS, TAU_GRID


def _agg(vals):
    a = np.asarray(vals, float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def aggregate(runs: list[dict]) -> dict:
    out = dict(dataset=runs[0]["dataset"], title=runs[0]["title"], source=runs[0]["source"],
               seeds=[r["seed"] for r in runs], n_flows=runs[0]["n_flows"], n_classes=runs[0]["n_classes"],
               methods={}, sweeps={"A": [], "B": []}, tau_star={}, temperature=_agg([r["temperature"] for r in runs]),
               le_ms=_agg([r["le_ms"] for r in runs]), lc_ms=_agg([r["lc_ms"] for r in runs]),
               delta_ms=runs[0]["delta_ms"], cloud_selected=[r["cloud"]["selected"] for r in runs])
    for key in runs[0]["methods"]:
        name = runs[0]["methods"][key]["method"]
        m = dict(method=name)
        for f in ["acc", "precision", "recall", "f1", "fwd", "latency_ms", "size_kb"]:
            vals = [r["methods"][key].get(f) for r in runs]
            if None in vals:
                continue
            m[f] = _agg(vals)
        out["methods"][key] = m
    for p in "AB":
        out["tau_star"][p] = _agg([r["selection"][p]["tau"] for r in runs])
        for i, t in enumerate(TAU_GRID):
            rows = [r["sweeps"][p][i] for r in runs]
            out["sweeps"][p].append(dict(tau=t, **{f: _agg([x[f] for x in rows])
                                                   for f in ["f1", "acc", "fwd", "latency_ms"]}))
    # paired difference Policy B - Policy A (supports the "no significant gain" finding)
    d = [r["methods"]["policy_b"]["f1"] - r["methods"]["policy_a"]["f1"] for r in runs]
    out["policy_b_minus_a_f1"] = _agg(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    ap.add_argument("--benchmark", action="store_true", help="multi-seed study instead of single artifact build")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=None, help="synthetic flows per dataset (default from spec)")
    ap.add_argument("--no-dnn", action="store_true", help="skip the DNN cloud candidate (faster)")
    a = ap.parse_args()
    keys = list(DATASETS) if a.dataset == "all" else [a.dataset]
    ARTIFACTS.mkdir(exist_ok=True)
    if not a.benchmark:
        for k in keys:
            pipeline.run(DATASETS[k], seed=0, n=a.n, save=True, try_dnn=not a.no_dnn)
        return
    bench = {}
    bpath = ARTIFACTS / "benchmark.json"
    if bpath.exists():
        bench = json.loads(bpath.read_text())
    for k in keys:
        runs = [pipeline.run(DATASETS[k], seed=s, n=a.n, save=False, try_dnn=not a.no_dnn)
                for s in SEEDS[: a.seeds]]
        bench[k] = aggregate(runs)
        bench["generated"] = time.strftime("%Y-%m-%d %H:%M")
        bpath.write_text(json.dumps(bench, indent=2))
        print(f"benchmark for {k} written to {bpath}")


if __name__ == "__main__":
    main()
