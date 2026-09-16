"""Fixed-space, single-seed loss-metric audit; does not change frozen results."""
import json
from pathlib import Path
from run_p7_transient_pilot import build, reference, errors
import numpy as np


def main():
    rng = np.random.default_rng(934)
    x = rng.uniform(0, 1, 4096)
    v = rng.uniform(-1, 1, 4096)
    t = rng.uniform(0, .1, 4096)
    rows = []
    for eps in (1, .001):
        ref = reference(eps, 256, 16, 800)
        for iw, bw, average in ((1, 1, True), (10, 1, True),
                                (1, 10, True), (10, 10, True),
                                (100, 100, True), (1, 1, False)):
            f, record = build(eps, 128, 11, 'oe_ap', initial_weight=iw,
                              boundary_weight=bw, block_average=average)
            record['held_out'] = {
                'initial_rms': float(np.sqrt(np.mean(f(0*x, x, v)**2))),
                'left_rms': float(np.sqrt(np.mean((f(t, 0*x, abs(v))-1)**2))),
                'right_rms': float(np.sqrt(np.mean(f(t, 0*x+1, -abs(v))**2))),
            }
            record['snapshots'] = errors(f, ref, eps)
            rows.append(record)
            print(json.dumps(record), flush=True)
    out = Path(__file__).resolve().parents[1] / 'results/p7_transient_pilot/loss_metric_audit_seed11.json'
    out.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
