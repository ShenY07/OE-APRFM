"""Run one indexed P1 tuning candidate."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from run_p1_oe_aprfm import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    combinations = list(itertools.product(((1, 1), (2, 1)), (0.5, 1.0, 2.0), (32, 64)))
    partitions, scale, features = combinations[args.index]
    run(args.epsilon, 11, Path("results/tuning/p1"), partitions=partitions, features=features, scale=scale, tag=f"t{args.index:03d}")


if __name__ == "__main__":
    main()
