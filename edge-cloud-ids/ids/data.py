"""Data sources: a structured synthetic generator and a loader for the real CSVs.

The paper uses recorded flows from Edge-IIoTset and TON_IoT. User-supplied CSVs are bundled in this distribution;
``load_dataset`` uses a real CSV from ``data/`` when present and otherwise falls back to
``generate_synthetic``. The generator mimics the properties the paper's method depends on:
class imbalance with an attack-dominated mix, heavy-tailed counters, categorical protocol fields,
identifier / timestamp columns that must be dropped, near-duplicate features (|r|>0.95), constant
columns, missing values, duplicate rows, and overlapping attack families that create a
Bayes-error ceiling for every model (the "Cloud-Only ceiling is set by the data").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DatasetSpec

PROTOS = ["tcp", "udp", "icmp", "arp", "http", "mqtt"]
SERVICES = ["-", "dns", "http", "ssl", "ftp", "mqtt", "ssh"]
CONN_STATES = ["S0", "SF", "REJ", "RSTO", "OTH", "S1"]

N_LATENT = 26
N_NUMERIC = 38


def _world(spec: DatasetSpec):
    """Class geometry is fixed per dataset (independent of the sampling seed)."""
    w = np.random.default_rng(spec.world_seed)
    K = len(spec.classes)
    centers = w.normal(0, spec.sep, size=(K, spec.nsub, N_LATENT))
    # sparse, class-specific informative directions make boundaries axis-aligned-ish (tree friendly)
    for k in range(K):
        mask = w.random(N_LATENT) < 0.55
        centers[k, :, ~mask] *= 0.25
    amb = [w.normal(0, spec.sep, size=(2, N_LATENT)) for _ in spec.families]
    scale = w.uniform(0.6, 1.6, size=N_LATENT)
    col_src = w.integers(0, N_LATENT, size=N_NUMERIC)
    col_a = w.uniform(-0.5, 2.0, size=N_NUMERIC)
    col_b = w.uniform(0.5, 1.6, size=N_NUMERIC)
    heavy = w.random(N_NUMERIC) < 0.4
    return centers, amb, scale, col_src, col_a, col_b, heavy


def generate_synthetic(spec: DatasetSpec, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    n = n or spec.default_n
    rng = np.random.default_rng(1000 * spec.world_seed + seed)
    centers, amb, scale, col_src, col_a, col_b, heavy = _world(spec)
    K = len(spec.classes)
    pri = np.asarray(spec.priors, float)
    pri = pri / pri.sum()
    # guarantee every class has enough rows to stratify across splits
    counts = np.maximum((pri * n).astype(int), 60)
    y = np.repeat(np.arange(K), counts)
    rng.shuffle(y)
    N = len(y)

    fam_of = {k: fi for fi, fam in enumerate(spec.families) for k in fam}
    sub = rng.integers(0, spec.nsub, size=N)
    z = centers[y, sub] + rng.normal(0, spec.noise, size=(N, N_LATENT)) * scale
    # ambiguous mass: family members emit samples from a *shared* distribution
    in_fam = np.array([k in fam_of for k in y])
    use_amb = in_fam & (rng.random(N) < spec.overlap)
    for i in np.flatnonzero(use_amb):
        f = fam_of[y[i]]
        z[i] = amb[f][rng.integers(0, 2)] + rng.normal(0, spec.noise, N_LATENT) * scale
    # a little global confusion for every class
    flip = rng.random(N) < 0.012
    for i in np.flatnonzero(flip):
        k2 = rng.integers(0, K)
        z[i] = centers[k2, rng.integers(0, spec.nsub)] + rng.normal(0, spec.noise, N_LATENT) * scale

    cols: dict[str, np.ndarray] = {}
    names = []
    for j in range(N_NUMERIC):
        base = col_a[j] + col_b[j] * z[:, col_src[j]] + 0.15 * rng.normal(size=N)
        if heavy[j]:
            v = np.floor(np.exp(np.clip(base * 1.1 + 2.0, -5, 14)))          # heavy-tailed counters
            nm = f"cnt_{j:02d}"
        else:
            v = base
            nm = f"feat_{j:02d}"
        cols[nm] = v
        names.append(nm)
    # near-duplicate (|r|>0.95) features that the correlation filter must remove
    for j in range(5):
        src = names[j * 3]
        cols[f"{src}_dup"] = cols[src] * 1.01 + rng.normal(0, 1e-3, N) * (np.std(cols[src]) + 1e-9)
    cols["const_flag"] = np.zeros(N)
    cols["const_ver"] = np.full(N, 4.0)

    # categorical fields (derived from latent geometry + noise)
    p_idx = np.clip(((z[:, 0] + z[:, 1]) * 0.8 + rng.normal(0, 0.8, N) + 3).astype(int), 0, len(PROTOS) - 1)
    s_idx = np.clip(((z[:, 2] - z[:, 3]) * 0.7 + rng.normal(0, 0.8, N) + 3).astype(int), 0, len(SERVICES) - 1)
    c_idx = np.clip(((z[:, 4] + z[:, 5]) * 0.6 + rng.normal(0, 0.9, N) + 2).astype(int), 0, len(CONN_STATES) - 1)
    cols["proto"] = np.asarray(PROTOS)[p_idx]
    cols["service"] = np.asarray(SERVICES)[s_idx]
    cols["conn_state"] = np.asarray(CONN_STATES)[c_idx]

    # identifiers: class-correlated bursts would leak the label if not dropped
    t0 = 1_700_000_000
    cols["ts"] = t0 + y * 3600 + rng.integers(0, 3600, N)
    cols["src_ip"] = [f"192.168.{(k * 7 + a) % 250}.{b}" for k, a, b in
                      zip(y, rng.integers(0, 40, N), rng.integers(1, 250, N))]
    cols["dst_ip"] = [f"10.0.0.{b}" for b in rng.integers(1, 30, N)]

    df = pd.DataFrame(cols)
    # missing values (~0.4%) in a few numeric columns
    for nm in names[:6]:
        m = rng.random(N) < 0.004
        df.loc[m, nm] = np.nan
    df[spec.label_column or "label"] = np.asarray(spec.classes)[y]
    # exact duplicate rows (~1.5%) that the loader de-duplicates
    dups = df.sample(frac=0.015, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
LABEL_COL_SYNTH = "label"


def find_real_file(spec: DatasetSpec) -> Path | None:
    d = DATA_DIR / spec.key
    if not d.exists():
        return None
    files = sorted(d.glob("*.csv"))
    return files[0] if files else None


def load_real(spec: DatasetSpec, path: Path | None = None) -> pd.DataFrame:
    path = path or find_real_file(spec)
    if path is None:
        raise FileNotFoundError(f"No CSV in data/{spec.key}/ - expected {spec.real_file_hint}")
    df = pd.read_csv(path, low_memory=False)
    if spec.label_column not in df.columns:
        raise KeyError(f"Label column {spec.label_column!r} missing in {path.name}")
    return df


def load_dataset(spec: DatasetSpec, seed: int = 0, n: int | None = None,
                 allow_synthetic: bool = True) -> tuple[pd.DataFrame, str, str]:
    """Return (de-duplicated DataFrame, label_column, source) with source in {real, synthetic}."""
    real = find_real_file(spec)
    if real is not None:
        df, label_col, source = load_real(spec, real), spec.label_column, "real"
        # keep only classes the spec knows; map names exactly
        df = df[df[label_col].isin(spec.classes)] if set(df[label_col].unique()) & set(spec.classes) else df
    elif allow_synthetic:
        df, label_col, source = generate_synthetic(spec, n=n, seed=seed), spec.label_column or "label", "synthetic"
    else:
        raise FileNotFoundError(spec.real_file_hint)
    # de-duplicate *before* splitting so identical flows can never straddle train/test
    df = df.drop_duplicates().reset_index(drop=True)
    return df, label_col, source
