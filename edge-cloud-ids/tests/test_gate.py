"""Unit tests for the paper's equations and the leakage-safe pipeline."""
import numpy as np
import pytest

from ids import gate
from ids.config import TAU_GRID
from ids.models import EdgeModel, fit_temperature, softmax


def test_confidence_and_policies():
    probs = np.array([[.95, .05, 0], [.40, .35, .25], [.2, .7, .1], [.6, .3, .1]])
    conf, pred = gate.confidence(probs)
    assert np.allclose(conf, [.95, .40, .7, .6]) and list(pred) == [0, 0, 1, 0]
    a = gate.forward_mask(conf, pred, 0.65, "A", benign_idx=0)
    b = gate.forward_mask(conf, pred, 0.65, "B", benign_idx=0)
    assert list(a) == [False, True, False, True]            # Eq. (2): c < tau
    assert list(b) == [False, True, True, True]             # Eq. (3): also predicted attacks
    assert (b | a).tolist() == b.tolist()                   # B is a superset of A


def test_latency_and_breakeven():
    le, lc, d = 0.01, 1.13, 50.0
    assert gate.mean_latency(0, le, lc, d) == le
    assert gate.mean_latency(1, le, lc, d) == pytest.approx(le + d + lc)
    f_star = gate.breakeven_fraction(le, lc, d)
    # at the break-even fraction the mean latency equals Cloud-Only (d + lc)
    assert gate.mean_latency(f_star, le, lc, d) == pytest.approx(d + lc)
    # paper: forwarding ~everything is no better than Cloud-Only
    assert gate.mean_latency(0.9999, le, lc, d) > gate.mean_latency(0.5, le, lc, d)


def test_final_labels_cloud_replaces_edge():
    pe, pc, fwd = np.array([0, 1, 2]), np.array([2, 2, 2]), np.array([True, False, True])
    assert gate.final_labels(pe, pc, fwd).tolist() == [2, 1, 2]


def test_threshold_selection_is_minimal_feasible():
    rng = np.random.default_rng(0)
    n, K = 4000, 4
    y = rng.integers(0, K, n)
    conf = rng.uniform(0.3, 1, n)
    pred_e = np.where(rng.random(n) < conf, y, rng.integers(0, K, n))   # accuracy grows with confidence
    pred_c = y.copy()                                                  # perfect cloud
    sel = gate.select_threshold(y, pred_e, conf, pred_c, "A", 0, K)
    assert sel["feasible"] and sel["tau"] in TAU_GRID
    assert sel["val_f1"] >= sel["val_f1_cloud"] - 0.005 - 1e-12
    # no feasible grid point with a smaller forwarded share
    for t in TAU_GRID:
        f = gate.forward_mask(conf, pred_e, t, "A", 0).mean()
        f1 = gate.metrics(y, gate.final_labels(pred_e, pred_c, gate.forward_mask(conf, pred_e, t, "A", 0)), K)["f1"] / 100
        if f < sel["val_fwd"] - 1e-12:
            assert f1 < sel["val_f1_cloud"] - 0.005


def test_macro_f1_is_unweighted_mean():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 0, 0, 1, 0, 2])
    m = gate.metrics(y, p, 3)
    f1s = [2 * .8 * 1 / 1.8, 2 * 1 * .5 / 1.5, 1.0]          # per-class F1 worked out by hand
    assert m["f1"] == pytest.approx(100 * np.mean(f1s))
    assert m["acc"] == pytest.approx(100 * 6 / 7)


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 4, (3000, 5))
    y = np.where(rng.random(3000) < 0.5, z.argmax(1), rng.integers(0, 5, 3000))  # overconfident logits
    T = fit_temperature(z, y)
    assert T > 1.5
    assert softmax(z, T).max(1).mean() < softmax(z, 1.0).max(1).mean()


def test_int8_quantization_roundtrip_small_error():
    rng = np.random.default_rng(2)
    W = [rng.normal(0, .3, (10, 8)), rng.normal(0, .3, (8, 3))]
    b = [np.zeros(8), np.zeros(3)]
    m = EdgeModel(W, b)
    q = m.quantize_int8()
    X = rng.normal(size=(200, 10))
    assert q.size_kb < m.size_kb / 2   # int8 weights + fp32 scales/biases (tiny model: scale overhead)
    assert np.abs(q.logits(X) - m.logits(X)).max() < 0.1


def test_per_channel_int8_survives_outlier_weights():
    """Regression for the paper's INT8 instability: one outlier weight must not zero out a whole matrix."""
    rng = np.random.default_rng(3)
    W1 = rng.normal(0, .3, (12, 16)); W1[0, 0] = 25.0           # outlier like the real TON_IoT layer (|w|max 19.8)
    W2 = rng.normal(0, .3, (16, 4))
    m = EdgeModel([W1, W2], [np.zeros(16), np.zeros(4)])
    X = rng.normal(size=(500, 12))
    ref = m.logits(X)
    err_pt = np.abs(m.quantize_int8("per_tensor").logits(X) - ref).mean()
    err_pc = np.abs(m.quantize_int8("per_channel").logits(X) - ref).mean()
    assert err_pc < err_pt / 3
    assert m.quantize_int8("per_channel").size_kb < m.size_kb / 2.5


def test_int8_roundtrip_save_load(tmp_path):
    rng = np.random.default_rng(4)
    m = EdgeModel([rng.normal(size=(6, 5)), rng.normal(size=(5, 3))], [np.zeros(5), np.zeros(3)])
    for mode in ("per_tensor", "per_channel"):
        q = m.quantize_int8(mode); q.save(tmp_path / f"{mode}.npz")
        X = rng.normal(size=(20, 6))
        assert np.allclose(EdgeModel.load(tmp_path / f"{mode}.npz").logits(X), q.logits(X), atol=1e-5)


def test_identifier_detection_keeps_real_features():
    """Regression: token matching used to drop src_ip_bytes, mbtcp.unit_id, udp.time_delta."""
    from ids.preprocess import Preprocessor
    p = Preprocessor(10)
    for keep in ["src_ip_bytes", "dst_ip_bytes", "mbtcp.unit_id", "mbtcp.trans_id", "udp.time_delta",
                 "src_port", "dst_port", "http_uri", "duration"]:
        assert not p._is_identifier(keep), keep
    for drop in ["src_ip", "dst_ip", "ts", "frame.time", "ip.src_host", "ip.dst_host"]:
        assert p._is_identifier(drop), drop


def test_preprocessing_fits_on_train_only():
    import pandas as pd
    from ids.preprocess import Preprocessor
    rng = np.random.default_rng(5)
    tr = pd.DataFrame(dict(a=rng.uniform(0, 1, 500), b=rng.choice(["x", "y"], 500)))
    y = rng.integers(0, 2, 500)
    pre = Preprocessor(2, seed=0).fit(tr, y)
    te = pd.DataFrame(dict(a=[100.0, -5.0], b=["x", "never_seen"]))      # out-of-range numeric, unseen category
    X = pre.transform(te)
    assert np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1      # clipped with TRAIN min/max, no refit
