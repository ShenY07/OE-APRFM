"""Fresh E6 trajectories, independent Fourier references and archive checks."""

import json
from pathlib import Path
import numpy as np
import experiments.run_e6_periodic as runner
import experiments.reference_e6 as reference
from experiments.e6_protocol import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/checks/parity/e6"
ARCHIVE = (
    ROOT
    / ".release-backup/20261009/original-worktree/results/e6_periodic/factorized_check"
)


def main():
    runner.ROOT = OUT
    reference.ROOT = OUT / "results/e6_periodic"
    records = []
    for eps in (1.0, 1e-3):
        runner.run(eps, 0.002, 11, 128, 64, "review")
        name = f"s11_eps{eps:.0e}_dt2e-03_J128_q64"
        directory = reference.ROOT / "review" / name
        assert validate(directory / "result.json")
        with (
            np.load(directory / "fields.npz") as d,
            np.load(ARCHIVE / name / "fields.npz") as old,
        ):
            differences = {
                k: float(np.max(np.abs(d[k] - old[k])))
                for k in d.files
                if k in old.files
            }
            assert "coefficients_0" not in d.files
            np.testing.assert_array_equal(
                d["f_0"],
                np.broadcast_to(
                    (1 + 0.2 * np.cos(2 * np.pi * d["x"]))[:, None],
                    d["f_0"].shape,
                ),
            )
        records.append(
            dict(
                epsilon=eps,
                archive_max_abs_by_field=differences,
                t0_is_analytic=True,
                t0_coefficients_exist=False,
            )
        )
        (OUT / "audit.json").write_text(json.dumps(records, indent=2))
    reference.main()


if __name__ == "__main__":
    main()
