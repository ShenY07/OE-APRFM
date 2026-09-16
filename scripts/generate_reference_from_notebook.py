"""Generate P2 references with the solver defined in parity_si_1d.ipynb."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_banded

from configuration.p2_heterogeneous_1d import get_config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/numerical/parity_si_1d.ipynb"


def notebook_solver():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][6]["source"])
    namespace = {
        "np": np,
        "solve_banded": solve_banded,
        "perf_counter": perf_counter,
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace["solve_parity_si_1d"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--level", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--max-iter", type=int, default=2**14)
    parser.add_argument("--dsa-relaxation", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/references"))
    args = parser.parse_args()
    grids = {"A": (512, 256), "B": (1024, 512), "C": (2048, 1024), "D": (4096, 2048)}
    grid = grids[args.level]
    config = get_config(args.epsilon)
    config.model.grid_sizes = grid
    result = notebook_solver()(
        config,
        tol=args.tol,
        max_iter=args.max_iter,
        report_every=100,
        dsa_relaxation=args.dsa_relaxation,
    )
    if not result["converged"]:
        raise RuntimeError(
            f"notebook reference did not converge: final difference={result['final_difference']:.6e}"
        )
    velocity = np.concatenate((-result["velocity"][::-1], result["velocity"]))
    weights = np.concatenate((result["weights"][::-1], result["weights"]))
    distribution = np.concatenate(
        (result["f_negative"][:, ::-1], result["f_positive"]), axis=1
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"p2_parity_ref_eps_{args.epsilon:.0e}_level_{args.level}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        x=result["x"], velocity=velocity, weights=weights,
        f=distribution, r=result["r"], j=result["j"], rho=result["rho"],
    )
    metadata = {
        "problem": "p2", "epsilon": args.epsilon, "level": args.level,
        "grid": grid, "iterations": result["iterations"],
        "converged": result["converged"],
        "final_difference": result["final_difference"],
        "runtime_seconds": result["runtime_seconds"], "tolerance": args.tol,
        "dsa_relaxation": args.dsa_relaxation,
        "solver_source": str(NOTEBOOK.relative_to(ROOT)),
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
