"""API tests against the trained artifacts (skipped if `python -m ids.train` has not been run)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ids.config import ARTIFACTS

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "ton_iot" / "meta.json").exists(), reason="no artifacts")


@pytest.fixture(scope="module")
def client():
    from service.app import app
    return TestClient(app)


def test_health_and_state(client):
    assert client.get("/api/health").json()["status"] == "ok"
    s = client.get("/api/state").json()
    assert {"config", "totals", "datasets", "recent"} <= set(s)


def test_config_validation(client):
    assert client.post("/api/config", json={"policy": "Z"}).status_code == 422
    assert client.post("/api/config", json={"tau": 1.5}).status_code == 422
    assert client.post("/api/config", json={"dataset": "nope"}).status_code == 422
    r = client.post("/api/config", json={"dataset": "ton_iot", "policy": "a", "auto_tau": True, "delta_ms": 50})
    assert r.status_code == 200 and r.json()["config"]["policy"] == "A"


def test_gate_semantics_end_to_end(client):
    client.post("/api/config", json={"dataset": "ton_iot", "policy": "A", "tau": 0.99})
    client.post("/api/reset", json={})
    flows = client.get("/api/sample?n=40").json()["flows"]
    recs = [f["features"] for f in flows]
    a = client.post("/api/classify", json={"records": recs}).json()["results"]
    for r in a:                                       # Policy A: forwarded iff c < tau
        assert r["forwarded"] == (r["confidence"] < 0.99)
        assert r["decided_by"] == ("cloud" if r["forwarded"] else "edge")
        if not r["forwarded"]:
            assert r["label"] == r["edge_label"]
    client.post("/api/config", json={"policy": "B"})
    b = client.post("/api/classify", json={"records": recs}).json()["results"]
    for r in b:                                       # Policy B: also every predicted attack
        assert r["forwarded"] == (r["confidence"] < 0.99 or r["edge_label"] != "normal")
    assert sum(r["forwarded"] for r in b) >= sum(r["forwarded"] for r in a)


def test_latency_matches_eq4(client):
    client.post("/api/config", json={"dataset": "ton_iot", "policy": "A", "tau": 0.9, "delta_ms": 100})
    d = client.get("/api/state").json()["datasets"]["ton_iot"]
    recs = [f["features"] for f in client.get("/api/sample?n=30").json()["flows"]]
    for r in client.post("/api/classify", json={"records": recs}).json()["results"]:
        exp = d["le_ms"] + (100 + d["lc_ms"] if r["forwarded"] else 0)
        assert r["latency_ms"] == pytest.approx(exp)
    client.post("/api/config", json={"delta_ms": 50, "auto_tau": True})


def test_bad_inputs(client):
    assert client.post("/api/classify", json={"records": []}).status_code == 422
    assert client.post("/api/classify-csv", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 422
    assert client.post("/api/classify-csv", files={"file": ("x.csv", b"\x00\x01garbage", "text/csv")}).status_code == 422
    assert client.post("/api/stream", json={"running": False, "scenario": "bogus"}).status_code == 422


def test_csv_roundtrip(client):
    import pandas as pd, io
    client.post("/api/config", json={"dataset": "ton_iot"})
    recs = [f["features"] for f in client.get("/api/sample?n=25").json()["flows"]]
    buf = io.BytesIO(pd.DataFrame(recs).to_csv(index=False).encode())
    r = client.post("/api/classify-csv", files={"file": ("f.csv", buf, "text/csv")})
    assert r.status_code == 200 and r.json()["n"] == 25


def test_curve_and_results(client):
    c = client.get("/api/curve?dataset=ton_iot&delta_ms=50").json()
    assert set(c["curves"]) == {"A", "B"}
    fw = [p["fwd"] for p in c["curves"]["A"]]
    assert fw == sorted(fw)                           # forwarded share is monotone in tau
    assert client.get("/api/results/ton_iot").json()["selection"]["A"]["tau"] > 0
    assert client.get("/api/results/xx").status_code == 404


def test_benchmark_source_matches_deployed_models(client, tmp_path, monkeypatch):
    import json
    import service.app as app_module
    from ids.config import ARTIFACTS
    for key in ("edge_iiotset", "ton_iot"):
        folder = tmp_path / key
        folder.mkdir()
        (folder / "meta.json").write_text((ARTIFACTS / key / "meta.json").read_text())
    # An incompatible benchmark must never be displayed as current model results.
    (tmp_path / "benchmark.json").write_text(json.dumps({
        key: {"source": "incompatible-source"} for key in ("edge_iiotset", "ton_iot")
    }))
    monkeypatch.setattr(app_module, "ARTIFACTS", tmp_path)
    assert client.get("/api/benchmark").json()["available"] is False
