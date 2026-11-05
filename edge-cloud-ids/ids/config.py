"""Dataset specifications and global constants taken from the paper (Sec. III-IV)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

# --- Paper constants -------------------------------------------------------
TAU_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]   # Sec. IV-A
EPSILON = 0.005                                          # Eq. (6) tolerance
DELTAS_MS = [20.0, 50.0, 100.0]                          # simulated round trips
DEFAULT_DELTA_MS = 50.0
SEEDS = [0, 1, 2, 3, 4]
EDGE_HIDDEN = (64, 32)
EDGE_QUANT = "per_channel"   # deployed INT8 scheme; the paper's per-tensor scheme is also reported (edge_int8_pt)
XGB_PARAMS = dict(n_estimators=250, max_depth=8)
DNN_HIDDEN = (256, 128, 64)


@dataclass
class DatasetSpec:
    key: str
    title: str
    classes: list[str]
    benign: str
    priors: list[float]
    n_edge_features: int
    default_n: int
    # synthetic-generator knobs (ignored when real CSVs are used)
    families: list[list[int]] = field(default_factory=list)  # classes that overlap
    overlap: float = 0.1       # share of a family member's mass that is ambiguous
    sep: float = 1.0
    noise: float = 1.0
    nsub: int = 4
    world_seed: int = 0
    # real-data loading
    label_column: str = ""
    drop_columns: list[str] = field(default_factory=list)
    real_file_hint: str = ""

    @property
    def benign_idx(self) -> int:
        return self.classes.index(self.benign)


EDGE_IIOTSET = DatasetSpec(
    key="edge_iiotset",
    title="Edge-IIoTset",
    classes=["Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Vulnerability_scanner",
             "DDoS_TCP", "Password", "Port_Scanning", "DDoS_HTTP", "Uploading", "Backdoor",
             "XSS", "Ransomware", "MITM", "Fingerprinting"],
    benign="Normal",
    priors=[0.14, 0.074, 0.064, 0.033, 0.07, 0.065, 0.065, 0.045, 0.065, 0.065, 0.065,
            0.065, 0.065, 0.004, 0.003],
    n_edge_features=27,
    default_n=60000,
    # classes whose traffic overlaps heavily in the ML version of the dataset
    families=[[4, 6, 9, 11, 12], [5, 8], [10, 3]],
    overlap=0.30, sep=2.3, noise=1.0, nsub=4, world_seed=11,
    label_column="Attack_type",
    drop_columns=["Attack_label", "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                  "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                  "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options", "tcp.payload",
                  "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"],
    real_file_hint="ML-EdgeIIoT-dataset.csv (Kaggle: mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)",
)

TON_IOT = DatasetSpec(
    key="ton_iot",
    title="TON_IoT",
    classes=["normal", "backdoor", "ddos", "dos", "injection", "mitm", "password",
             "ransomware", "scanning", "xss"],
    benign="normal",
    priors=[0.13, 0.05, 0.18, 0.11, 0.07, 0.01, 0.17, 0.04, 0.14, 0.10],
    n_edge_features=30,
    default_n=70000,
    families=[[4, 9], [6, 3]],
    overlap=0.02, sep=2.0, noise=1.0, nsub=6, world_seed=23,
    label_column="type",
    drop_columns=["label", "ts", "src_ip", "dst_ip"],
    real_file_hint="Train_Test_Network.csv (TON_IoT network dataset, UNSW)",
)

DATASETS: dict[str, DatasetSpec] = {EDGE_IIOTSET.key: EDGE_IIOTSET, TON_IOT.key: TON_IOT}
