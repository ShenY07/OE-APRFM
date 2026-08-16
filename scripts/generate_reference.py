"""Generate one A/B parity reference level for P2 or P5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from configuration.p2_heterogeneous_1d import get_config as p2_config
from configuration.p5_heterogeneous_2d import get_config as p5_config
from numerical.parity_reference import solve_parity_gmres_2d, solve_parity_si_dsa_1d


GRIDS = {
    "p2": {"A": (512, 256), "B": (1024, 512)},
    "p5": {"A": (32, 32, 8), "B": (64, 64, 16)},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("p2", "p5"), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--level", choices=("A", "B"), required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/references"))
    args = parser.parse_args()
    grid = GRIDS[args.problem][args.level]
    if args.problem == "p2":
        result = solve_parity_si_dsa_1d(p2_config(args.epsilon), grid=grid)
    else:
        result = solve_parity_gmres_2d(
            p5_config(args.epsilon), grid=grid, max_iter=args.max_iter
        )
    if not result["converged"]:
        raise RuntimeError(f"{args.problem} level {args.level} did not converge")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.problem}_parity_ref_eps_{args.epsilon:.0e}_level_{args.level}"
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    metadata.update(problem=args.problem, epsilon=args.epsilon, level=args.level, grid=grid)
    np.savez_compressed(args.output_dir / f"{stem}.npz", **arrays)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
