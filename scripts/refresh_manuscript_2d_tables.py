"""Serial, fresh-process reruns for the P3/P4 entries in results/table.md."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/table_refresh_20261010"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}",
        JAX_ENABLE_X64="true",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    specs = []
    for problem in ("p3", "p4"):
        for representation, features in (
            ("four_component", 128),
            ("legacy_two_component", 256),
        ):
            for epsilon in (1.0, 0.001):
                for seed in (11, 23, 37):
                    specs.append(
                        (problem, representation, features, epsilon, seed)
                    )
    for features in (32, 64):
        for seed in (11, 23, 37):
            specs.append(("p3", "four_component", features, 0.001, seed))
    manifest = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        python=sys.version,
        platform=platform.platform(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        environment={
            k: env[k]
            for k in (
                "PYTHONPATH",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        protocol="physical_inflow_trace_midpoint_evaluation_serial_fresh_process_v1",
        timing_definition="feature_seconds + assembly_seconds + solve_seconds; excludes evaluation and process startup",
        timing_limit="Runs are serialized by this launcher; the shared host is not exclusively reserved.",
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "scripts/run_p3_oe_aprfm.py",
                *sorted((ROOT / "src").rglob("*.py")),
            ]
        },
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    statuses = []
    for problem, representation, J, eps, seed in specs:
        name = f"{problem}_{representation}_J{J}_eps{eps:.0e}_s{seed}"
        dest = OUT / "raw" / name
        command = [
            sys.executable,
            "scripts/run_p3_oe_aprfm.py",
            "--problem",
            problem,
            "--angular-representation",
            representation,
            "--epsilon",
            str(eps),
            "--seed",
            str(seed),
            "--features",
            str(J),
            "--partitions",
            "1",
            "1",
            "1",
            "--collocation",
            "16",
            "16",
            "16",
            "--rcond",
            "1e-12",
            "--output-dir",
            str(dest),
        ]
        record = dest / f"{problem}_oe_aprfm_eps_{eps:.0e}_seed_{seed}.json"
        status = dict(
            name=name, command=command, record=str(record.relative_to(ROOT))
        )
        print(f"START {len(statuses)+1}/{len(specs)} {name}", flush=True)
        if record.exists():
            status["state"] = "existing"
        else:
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "process.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            status["returncode"] = result.returncode
            status["state"] = (
                "completed"
                if result.returncode == 0 and record.exists()
                else "failed"
            )
        statuses.append(status)
        (OUT / "status.json").write_text(json.dumps(statuses, indent=2))
        if status["state"] == "failed":
            raise RuntimeError(f"Failed: {name}; see process.log")
        r = json.loads(record.read_text())
        print(
            f"DONE rows={r['num_rows']} Ef={r['relative_l2_f']:.10e} Tcomp={sum(r[k] for k in ('feature_seconds','assembly_seconds','solve_seconds')):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
