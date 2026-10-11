"""Rerun archived P3 configurations with explicit residual row accounting."""

import json
from pathlib import Path
import jax
import experiments.run_p3_oe_aprfm as runner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review_audit_20261010/p3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = runner.StreamingQR.add
    counts = []

    def counted(self, A, b, **kwargs):
        counts.append(int(A.shape[0]))
        return original(self, A, b, **kwargs)

    runner.StreamingQR.add = counted
    records = []
    for eps in (1e-3, 1.0):
        for seed in (11, 23, 37):
            counts.clear()
            print(f"START P3 eps={eps} seed={seed}", flush=True)
            r = runner.run(
                eps,
                seed,
                OUT / "raw",
                "p3",
                partitions=(1, 1, 1),
                features=128,
                collocation=(16, 16, 16),
                rcond=1e-12,
            )
            # Four sides, 512 physical inflow samples each, batches of 256.
            boundary = sum(counts[:8])
            interior = sum(counts[8:])
            assert boundary == 2048 and interior == 11760
            r.update(
                assembly_batch_rows=counts.copy(),
                block_rows=dict(
                    macro=interior // 5,
                    even_sum=interior // 5,
                    odd_sum=interior // 5,
                    even_difference=interior // 5,
                    odd_difference=interior // 5,
                    boundary=boundary,
                ),
            )
            records.append(r)
            (OUT / "audit.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(r), flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
