"""Factor registry: every tested candidate is appended, pass or fail.
The line count is M -- the honest number of tries behind any 'discovery'.

config_fingerprint(): short hash of every frozen threshold plus gate order.
Records with different fingerprints were produced under different rules and
must never be pooled, compared, or counted into the same M.
"""
import hashlib
import json
import os

_FINGERPRINT_KEYS = [
    "horizon", "lag", "min_names", "is_fraction", "embargo_bars",
    "periods_per_year", "cost_bps_per_side", "gates", "gate_order",
    "short_circuit",
]


def config_fingerprint(cfg, gate_order=None):
    """12-hex-char fingerprint of the frozen validation rules."""
    snap = {k: cfg.get(k) for k in _FINGERPRINT_KEYS}
    if gate_order is not None:
        snap["gate_order"] = list(gate_order)
    blob = json.dumps(snap, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def append(path, record):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def count(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)
