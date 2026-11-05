"""Edge MLP (64-32, INT8 PTQ), cloud models (XGBoost / deep DNN) and temperature scaling."""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from .config import DNN_HIDDEN, EDGE_HIDDEN, XGB_PARAMS


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
class EdgeModel:
    """Pure-NumPy inference for the compact MLP so it can run on a gateway without sklearn."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray],
                 int8: dict | None = None):
        self.biases = [b.astype(np.float32) for b in biases]
        self.int8 = int8
        if int8 is not None:  # simulated INT8: stored as integers, dequantized for the matmul
            self.weights = [q.astype(np.float32) * np.asarray(s, np.float32) for q, s in zip(int8["q"], int8["scale"])]
        else:
            self.weights = [w.astype(np.float32) for w in weights]

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h

    @property
    def n_weights(self) -> int:
        return int(sum(w.size for w in self.weights))

    @property
    def size_kb(self) -> float:
        nb = sum(b.size for b in self.biases) * 4
        if self.int8 is not None:
            return (sum(q.size for q in self.int8["q"]) + nb + 4 * sum(np.size(x) for x in self.int8["scale"])) / 1000.0
        return (self.n_weights * 4 + nb) / 1000.0

    def quantize_int8(self, mode: str = "per_channel") -> "EdgeModel":
        """Post-training symmetric INT8 quantization of the weight matrices (simulated).

        per_tensor  : one scale per matrix - the paper's scheme. A few outlier weights set the scale and
                      round a quarter of all other weights to zero, which is what makes the paper's INT8
                      model unstable on TON_IoT (F1 95 -> 78 on the real data).
        per_channel : one scale per output neuron (column) - same 8-bit storage, +<1 KB of scales, stable.
        """
        assert mode in ("per_tensor", "per_channel")
        qs, scales = [], []
        for W in self.weights:
            if mode == "per_tensor":
                s = float(np.abs(W).max() / 127.0) or 1.0
            else:
                s = np.abs(W).max(axis=0) / 127.0
                s = np.where(s == 0, 1.0, s).astype(np.float32)
            qs.append(np.clip(np.round(W / s), -127, 127).astype(np.int8))
            scales.append(s)
        return EdgeModel(self.weights, self.biases, int8=dict(q=qs, scale=scales, mode=mode))

    # -- persistence -----------------------------------------------------
    def save(self, path):
        d = {f"b{i}": b for i, b in enumerate(self.biases)}
        if self.int8 is not None:
            d.update({f"q{i}": q for i, q in enumerate(self.int8["q"])})
            d.update({f"s{i}": np.asarray(x, np.float32) for i, x in enumerate(self.int8["scale"])})
            d["mode"] = np.array(self.int8.get("mode", "per_tensor"))
            d["kind"] = np.array("int8")
        else:
            d.update({f"w{i}": w for i, w in enumerate(self.weights)})
            d["kind"] = np.array("fp32")
        np.savez(path, **d)

    @classmethod
    def load(cls, path) -> "EdgeModel":
        z = np.load(path, allow_pickle=False)
        n = sum(1 for k in z.files if k.startswith("b"))
        biases = [z[f"b{i}"] for i in range(n)]
        if str(z["kind"]) == "int8":
            scale = [z[f"s{i}"] for i in range(n)] if "s0" in z.files else [np.float32(x) for x in z["scale"]]
            mode = str(z["mode"]) if "mode" in z.files else "per_tensor"
            return cls([], biases, int8=dict(q=[z[f"q{i}"] for i in range(n)], scale=scale, mode=mode))
        return cls([z[f"w{i}"] for i in range(n)], biases)


def train_edge_mlp(Xtr, ytr, Xva, yva, n_classes: int, seed: int = 0,
                   max_epochs: int = 120, patience: int = 10) -> EdgeModel:
    """64-32 ReLU MLP, Adam, batch 512, early stopping on validation loss (paper Sec. IV-A)."""
    clf = MLPClassifier(hidden_layer_sizes=EDGE_HIDDEN, activation="relu", solver="adam",
                        batch_size=512, learning_rate_init=2e-3, random_state=seed)
    classes = np.arange(n_classes)
    best, best_loss, bad = None, np.inf, 0
    for _ in range(max_epochs):
        clf.partial_fit(Xtr, ytr, classes=classes)
        p = np.clip(clf.predict_proba(Xva), 1e-9, 1)
        loss = float(-np.log(p[np.arange(len(yva)), yva]).mean())
        if loss < best_loss - 1e-4:
            best_loss, bad = loss, 0
            best = (copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_))
        else:
            bad += 1
            if bad >= patience:
                break
    return EdgeModel(best[0], best[1])


# ---------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on validation data (Guo et al. 2017)."""
    def nll(T):
        p = np.clip(softmax(logits, T), 1e-12, 1)
        return -np.log(p[np.arange(len(y)), y]).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


# ---------------------------------------------------------------------------
class CloudModel:
    """Wraps XGBoost or the deep DNN behind one predict_proba interface."""

    def __init__(self, kind: str, model):
        self.kind, self.model = kind, model

    def save(self, directory):
        """Use XGBoost model IO, not a platform-dependent memory snapshot."""
        from pathlib import Path
        import joblib
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.kind == "xgboost":
            self.model.save_model(directory / "cloud.json")
            (directory / "cloud.joblib").unlink(missing_ok=True)
        else:
            joblib.dump(self, directory / "cloud.joblib")
            (directory / "cloud.json").unlink(missing_ok=True)

    @classmethod
    def load(cls, directory):
        from pathlib import Path
        import joblib
        directory = Path(directory)
        if (directory / "cloud.json").exists():
            from xgboost import XGBClassifier
            model = XGBClassifier(n_jobs=2)
            model.load_model(directory / "cloud.json")
            return cls("xgboost", model)
        # Compatibility with DNN models and older locally trained artifacts.
        return joblib.load(directory / "cloud.joblib")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)


def train_cloud(Xtr, ytr, Xva, yva, n_classes: int, seed: int = 0, try_dnn: bool = True,
                n_jobs: int = 2, xgb_params: dict | None = None) -> tuple[CloudModel, dict]:
    """Train XGBoost (250 trees, depth 8) and optionally a 256-128-64 DNN; keep the better on val F1."""
    from xgboost import XGBClassifier
    params = dict(XGB_PARAMS)
    params.update(xgb_params or {})
    t0 = time.time()
    xgb = XGBClassifier(**params, learning_rate=0.1, tree_method="hist", n_jobs=n_jobs,
                        objective="multi:softprob", num_class=n_classes, random_state=seed,
                        eval_metric="mlogloss", verbosity=0)
    xgb.fit(Xtr, ytr)
    f1_x = f1_score(yva, xgb.predict(Xva), average="macro", labels=np.arange(n_classes), zero_division=0)
    info = dict(xgb_val_f1=float(f1_x), xgb_secs=round(time.time() - t0, 1))
    best = CloudModel("xgboost", xgb)
    if try_dnn:
        t0 = time.time()
        dnn = MLPClassifier(hidden_layer_sizes=DNN_HIDDEN, batch_size=512, early_stopping=True,
                            n_iter_no_change=8, max_iter=80, random_state=seed)
        dnn.fit(Xtr, ytr)
        f1_d = f1_score(yva, dnn.predict(Xva), average="macro", labels=np.arange(n_classes), zero_division=0)
        info.update(dnn_val_f1=float(f1_d), dnn_secs=round(time.time() - t0, 1))
        if f1_d > f1_x:
            best = CloudModel("dnn", dnn)
    info["selected"] = best.kind
    return best, info


def time_single_flow(fn, x: np.ndarray, reps: int = 400) -> float:
    """Median single-flow inference time in ms."""
    fn(x)  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))
