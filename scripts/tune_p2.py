"""Coarse-to-fine P2 OE-RFM tuning with machine-readable results."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from run_p2_oe_aprfm import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tuning/p2"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    partitions = ((1, 1), (2, 1), (2, 2), (2, 4), (2, 8), (4, 2))
    scales = (0.5, 1.0, 2.0)
    features = (64,) if args.quick else (64, 96)
    weights = (None, (1.0, 1.0, 1.0, 1.0), (4.0, 1.0, 1.0, 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    combinations = list(itertools.product(partitions, scales, features, weights))
    if args.index is not None:
        combinations = [combinations[args.index]]
    for local_index, (part, scale, width, block) in enumerate(combinations):
        index = args.index if args.index is not None else local_index
        tag = f"t{index:03d}"
        record = run(args.epsilon, args.seed, args.output_dir, Path("results/references"), partitions=part, features=width, scale=scale, block_weights=block, tag=tag)
        records.append(record)
        if args.index is None:
            Path(args.output_dir / f"ranking_eps_{args.epsilon:.0e}.json").write_text(json.dumps(sorted(records, key=lambda item: item["relative_l2_f"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
