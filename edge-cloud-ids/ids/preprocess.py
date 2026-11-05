"""Leak-free preprocessing (paper Sec. III-C).

Everything fitted here (category vocabularies, medians, log decisions, min-max statistics,
correlation filter, importance ranking) is learned on the training partition only and then applied
unchanged to validation / test / live flows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paper Sec. III-C: "identifiers (IP addresses, timestamps) are dropped to avoid memorising hosts".
# Matching is on the WHOLE column name (or a src/dst endpoint prefix), never on a loose token: the old
# token heuristic dropped real features such as `src_ip_bytes`, `mbtcp.unit_id` and `udp.time_delta`.
ID_NAME_RE = re.compile(
    r"^(ts|timestamp|time|datetime|frame\.time)$"                    # timestamps
    r"|^(src|dst|source|destination)[._]?(ip|host|addr|address|mac)$"  # endpoint identifiers
    r"|^ip\.(src|dst)(_host)?$|^(ip|mac)$|^flow[._]?id$",
    re.I)
MAX_ONEHOT = 12
MAX_UNIQUE_RATIO = 0.5   # a non-numeric column that is unique per row is an identifier, not a category


class Preprocessor:
    def __init__(self, n_edge_features: int, corr_threshold: float = 0.95,
                 drop_columns: list[str] | None = None, seed: int = 0):
        self.n_edge = n_edge_features
        self.corr_threshold = corr_threshold
        self.drop_columns = list(drop_columns or [])
        self.seed = seed
        self.fitted = False

    # -- helpers ---------------------------------------------------------
    def _is_identifier(self, name: str) -> bool:
        return bool(ID_NAME_RE.search(name.strip()))

    def _raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Numeric columns -> filled -> log -> scaled -> clipped, categoricals one-hot."""
        parts = []
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            v = np.where(np.isfinite(v), v, self.medians[c])
            if c in self.log_cols:
                v = np.log1p(np.clip(v, 0, None))
            lo, hi = self.mins[c], self.maxs[c]
            v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
            parts.append(v)
        for c in self.cat_cols:
            s = df[c].astype(str).to_numpy()
            for cat in self.cat_vocab[c]:
                parts.append((s == cat).astype(float))
        return np.column_stack(parts).astype(np.float32)

    # -- fit -------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "Preprocessor":
        df = df.drop(columns=[c for c in self.drop_columns if c in df.columns])
        self.num_cols, self.cat_cols, self.dropped_ids = [], [], []
        for c in df.columns:
            if self._is_identifier(c):
                self.dropped_ids.append(c)
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                self.num_cols.append(c)
            else:
                nun = df[c].nunique(dropna=True)
                if nun > MAX_UNIQUE_RATIO * len(df):   # unique-per-row text: identifier-like, drop
                    self.dropped_ids.append(c)
                elif nun <= 1:                         # constant
                    continue
                else:                                  # encoded: top-MAX_ONEHOT training categories
                    self.cat_cols.append(c)
        self.medians, self.log_cols, self.mins, self.maxs = {}, set(), {}, {}
        keep_num = []
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.nunique(dropna=True) <= 1:  # constant -> useless
                continue
            med = float(v.median())
            self.medians[c] = med
            vv = v.fillna(med).to_numpy(float)
            skew = pd.Series(vv).skew()
            if vv.min() >= 0 and (skew > 3.0):
                self.log_cols.add(c)
                vv = np.log1p(vv)
            self.mins[c], self.maxs[c] = float(vv.min()), float(vv.max() if vv.max() > vv.min() else vv.min() + 1)
            keep_num.append(c)
        self.num_cols = keep_num
        self.cat_vocab = {}
        for c in self.cat_cols:
            vc = df[c].astype(str).value_counts()
            self.cat_vocab[c] = list(vc.index[:MAX_ONEHOT])
        names = list(self.num_cols) + [f"{c}={cat}" for c in self.cat_cols for cat in self.cat_vocab[c]]
        X = self._raw_matrix(df)

        # correlation filter |r| > threshold (greedy, keeps the earlier column)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr)
        keep = []
        for j in range(X.shape[1]):
            if all(abs(corr[j, k]) <= self.corr_threshold for k in keep):
                keep.append(j)
        self.keep_idx = np.array(keep)
        self.feature_names = [names[j] for j in keep]
        Xk = X[:, self.keep_idx]

        # tree-based importance for the compact edge subset (training data only)
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(Xk), size=min(20000, len(Xk)), replace=False)
        rf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=2, random_state=self.seed)
        rf.fit(Xk[sub], np.asarray(y)[sub])
        order = np.argsort(-rf.feature_importances_)
        k = min(self.n_edge, Xk.shape[1])
        self.edge_idx = np.sort(order[:k])
        self.importances = rf.feature_importances_
        self.edge_feature_names = [self.feature_names[j] for j in self.edge_idx]
        self.input_columns = list(self.num_cols) + list(self.cat_cols)
        self.fitted = True
        return self

    # -- transform -------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Full (cloud) feature matrix."""
        assert self.fitted
        return self._raw_matrix(df)[:, self.keep_idx]

    def edge(self, X_full: np.ndarray) -> np.ndarray:
        return X_full[:, self.edge_idx]

    def transform_records(self, records: list[dict]) -> np.ndarray:
        df = pd.DataFrame(records)
        for c in self.input_columns:
            if c not in df.columns:
                df[c] = np.nan if c in self.num_cols else ""
        return self.transform(df)

    def summary(self) -> dict:
        return dict(
            n_input_columns=len(self.input_columns),
            dropped_identifier_columns=self.dropped_ids,
            log_transformed=sorted(self.log_cols),
            n_full_features=len(self.feature_names),
            n_edge_features=len(self.edge_idx),
            full_features=self.feature_names,
            edge_features=self.edge_feature_names,
            corr_threshold=self.corr_threshold,
        )
