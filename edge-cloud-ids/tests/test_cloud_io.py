"""Regression: portable cloud models must not restore an XGBoost pickle."""
import joblib
import numpy as np
import pytest
from ids.config import ARTIFACTS
from ids.models import CloudModel


@pytest.mark.parametrize("key", ["edge_iiotset", "ton_iot"])
def test_portable_cloud_roundtrip(key, tmp_path, monkeypatch):
    folder = ARTIFACTS / key
    pre = joblib.load(folder / "preprocessor.joblib")
    replay = joblib.load(folder / "replay.joblib")
    X = pre.transform(replay["raw"].iloc[:100])
    original = CloudModel.load(folder)
    original.save(tmp_path)
    assert (tmp_path / "cloud.json").exists()
    # A stale/broken pickle must never be read when the portable model exists.
    (tmp_path / "cloud.joblib").write_bytes(b"corrupted legacy snapshot")
    def reject_pickle(*args, **kwargs):
        raise AssertionError("XGBoost must use native model IO")
    monkeypatch.setattr(joblib, "load", reject_pickle)
    loaded = CloudModel.load(tmp_path)
    np.testing.assert_array_equal(original.predict_proba(X), loaded.predict_proba(X))


@pytest.mark.parametrize("key", ["edge_iiotset", "ton_iot"])
def test_split_cloud_endpoint(key):
    from fastapi.testclient import TestClient
    from service.cloud import app
    folder = ARTIFACTS / key
    pre = joblib.load(folder / "preprocessor.joblib")
    replay = joblib.load(folder / "replay.joblib")
    X = pre.transform(replay["raw"].iloc[:5])
    with TestClient(app) as client:
        assert key in client.get("/health").json()["datasets"]
        response = client.post("/v1/classify", json={"dataset": key, "features": X.tolist()})
        assert response.status_code == 200
        assert response.json()["labels"] == CloudModel.load(folder).predict(X).tolist()
