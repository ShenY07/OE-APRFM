"""Explicit validity and completeness checks for E6 records."""

import hashlib, json
import numpy as np

PROTOCOL = "e6_periodic_be_2026_09_10"
IMPLEMENTATION = "factorized_svd_rhs_v1"
KEYS = (
    "epsilon",
    "dt",
    "seed",
    "J",
    "Ncoef",
    "Nrow",
    "nq",
    "rcond",
    "initialization",
)


def digest(record):
    return hashlib.sha256(
        json.dumps({k: record[k] for k in KEYS}, sort_keys=True).encode()
    ).hexdigest()


def validate(path):
    r = json.loads(path.read_text())
    if not (
        r.get("valid_for_e6") is True
        and r.get("protocol_id") == PROTOCOL
        and r.get("implementation_id") == IMPLEMENTATION
    ):
        return False
    if r.get("config_hash") != digest(r):
        return False
    try:
        with np.load(path.parent / "fields.npz") as d:
            if hashlib.sha256(d["parameters"].tobytes()).hexdigest() != r.get(
                "feature_hash"
            ):
                return False
            for t in (0.02, 0.1, 0.2):
                step = round(t / r["dt"])
                for name in ("rho", "q", "f", "coefficients"):
                    if not np.all(np.isfinite(d[f"{name}_{step}"])):
                        return False
    except (OSError, ValueError, KeyError):
        return False
    return len(r["history"]) == round(0.2 / r["dt"]) + 1
