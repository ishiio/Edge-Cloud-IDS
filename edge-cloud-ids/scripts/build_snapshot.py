"""Export an honest, portable read-only view of the bundled datasets and results."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ids.config import DATASETS, ARTIFACTS, ROOT
from ids.data import find_real_file
from service.app import state, gateway

from service.app import benchmark, audit
payload = {'state': state(), 'catalog': {}, 'results': {}, 'curves': {}, 'schemas': {}, 'benchmark': benchmark(), 'audit': audit()}
for key, spec in DATASETS.items():
    p = find_real_file(spec)
    if p is None:
        continue
    df = pd.read_csv(p, low_memory=False)
    clean = df.drop_duplicates()
    rows = df.groupby(spec.label_column, sort=False).head(5)
    ranks = rows.groupby(spec.label_column, sort=False).cumcount().to_list()
    rows = rows.iloc[sorted(range(len(rows)), key=lambda i: ranks[i])]
    preferred = [spec.label_column, 'proto', 'service', 'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'tcp.len', 'tcp.flags', 'ip.proto', 'http.request.method', 'mqtt.topic', 'arp.opcode']
    columns = [c for c in preferred if c in df.columns]
    columns += [c for c in df.columns if c not in columns]
    b = gateway().bundles[key]
    payload['catalog'][key] = dict(key=key, title=spec.title, filename=p.name, path=str(p.relative_to(ROOT)),
        rows=len(df), unique_rows=len(clean), duplicates=len(df)-len(clean), columns=columns,
        label=spec.label_column, benign=spec.benign, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        class_counts={str(k): int(v) for k,v in df[spec.label_column].value_counts().items()},
        preview=json.loads(rows[columns].to_json(orient='records')),
        fields=[dict(name=c, dtype=str(df[c].dtype), missing=int(df[c].isna().sum())) for c in columns],
        source='User-uploaded real CSV', evaluation_seed=b.results['seed'],
        model_source=b.meta['source'], policy_a_f1=b.results['methods']['policy_a']['f1'],
        forwarded=b.results['methods']['policy_a']['fwd'])
    payload['results'][key] = b.results
    payload['schemas'][key] = dict(dataset=key, input_columns=b.pre.input_columns, edge_features=b.pre.edge_feature_names, classes=b.classes, benign=spec.benign)
    for delta in [20,50,100]:
        payload['curves'][key+':'+str(delta)] = gateway().curve(key, delta)
web=ROOT/'web'
(web/'snapshot.json').write_text(json.dumps(payload, allow_nan=False))
(web/'snapshot.js').write_text('window.IDS_SNAPSHOT = '+json.dumps(payload, allow_nan=False).replace('</','<\\/')+';')
print('Snapshot generated from real CSVs and trained artifacts.')
