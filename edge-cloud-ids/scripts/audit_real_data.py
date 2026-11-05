"""Integrity audit of the deployed models on the real CSVs (seed 0, same split as ids.pipeline).

Checks: paper row counts, target-leak columns, train/test index overlap, exact feature-vector overlap
between train and test (inflates scores if large), single-feature predictiveness, and metrics restricted to
test flows whose feature vector never appears in training. Writes artifacts/audit.json.
    python scripts/audit_real_data.py
"""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from ids.config import DATASETS, ARTIFACTS
from ids.data import find_real_file
from ids.models import CloudModel, EdgeModel, softmax
from ids import gate

PAPER_ROWS = {"edge_iiotset": 156_986, "ton_iot": 190_474}
out = {}
for key, spec in DATASETS.items():
    d = ARTIFACTS / key
    raw = pd.read_csv(find_real_file(spec), low_memory=False)
    df = raw.drop_duplicates().reset_index(drop=True)
    meta = json.loads((d / "meta.json").read_text())
    classes = meta["classes"]; K = len(classes)
    y = df[spec.label_column].map({c: i for i, c in enumerate(classes)}).to_numpy()
    F = df.drop(columns=[spec.label_column])
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    fit, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=0)
    pre = joblib.load(d / "preprocessor.joblib")
    cloud = CloudModel.load(d)
    edge = EdgeModel.load(d / "edge_int8.npz")
    Xtr, Xte = pre.transform(F.iloc[tr]), pre.transform(F.iloc[te])
    r = dict(raw_rows=len(raw), unique_rows=len(df), paper_unique_rows=PAPER_ROWS[key],
             rows_match_paper=len(df) == PAPER_ROWS[key], n_train=len(tr), n_test=len(te),
             split_disjoint=len(set(tr) & set(te)) == 0, classes_in_test=int(len(set(y[te]))),
             label_columns_in_features=[c for c in pre.input_columns if c.lower() in ("label", "attack_label", spec.label_column.lower())],
             dropped_by_config=spec.drop_columns, auto_dropped=pre.dropped_ids,
             n_model_input_columns=len(pre.input_columns), n_full_features=int(Xtr.shape[1]),
             n_edge_features=len(pre.edge_idx))
    # exact feature-vector overlap between train and test
    h = lambda X: [hashlib.blake2b(row.tobytes(), digest_size=8).digest() for row in X]
    trh = {}
    for hh, yy in zip(h(Xtr), y[tr]):
        trh.setdefault(hh, set()).add(yy)
    teh = h(Xte)
    seen = np.array([hh in trh for hh in teh])
    r["test_flows_with_identical_feature_vector_in_train_pct"] = float(100 * seen.mean())
    r["unique_feature_vectors_pct_of_unique_rows"] = float(100 * len(set(h(pre.transform(F)))) / len(df))
    # metrics: all test vs novel-only
    pc = cloud.predict(Xte)
    lg = edge.logits(pre.edge(Xte)); probs = softmax(lg, meta["temperature"]); conf, pe = gate.confidence(probs)
    fwd = gate.forward_mask(conf, pe, meta["tau_star"]["A"], "A", meta["benign_idx"])
    pa = gate.final_labels(pe, pc, fwd)
    def m(mask):
        return {n: dict(acc=100 * accuracy_score(y[te][mask], p[mask]), f1=100 * f1_score(y[te][mask], p[mask], average="macro", labels=np.arange(K), zero_division=0))
                for n, p in [("edge_int8", pe), ("cloud", pc), ("policy_a", pa)]}
    r["metrics_all_test"] = m(np.ones(len(te), bool))
    r["n_novel_test"] = int((~seen).sum())
    r["metrics_novel_only"] = m(~seen) if (~seen).sum() > 200 else None
    # single-feature leakage screen: best depth-3 tree on one feature at a time (macro-F1 on held-out test)
    sub = np.random.default_rng(0).choice(len(tr), size=min(40000, len(tr)), replace=False)
    best = []
    for j, nme in enumerate(pre.feature_names):
        t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr[sub][:, [j]], y[tr][sub])
        best.append((float(100 * f1_score(y[te], t.predict(Xte[:, [j]]), average="macro", zero_division=0)), nme))
    best.sort(reverse=True)
    r["top_single_feature_macro_f1"] = [dict(feature=n, f1=f) for f, n in best[:5]]
    r["majority_class_baseline_acc"] = float(100 * np.bincount(y[te]).max() / len(te))
    out[key] = r
    print(key, json.dumps({k: v for k, v in r.items() if k not in ("dropped_by_config",)}, indent=1)[:3000], flush=True)
(ARTIFACTS / "audit.json").write_text(json.dumps(out, indent=2))
