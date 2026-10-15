"""Select a method and forward its experiment options to the runner."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def runner_path(method, problem):
    if method == "oe_apnn":
        if problem == "p6":
            raise ValueError("OE-APNN supports P1–P5")
        return ROOT / "src/baselines/oe_apnn/train.py"
    if method == "mm_aprfm":
        if problem == "p5":
            raise ValueError("MM-APRFM supports P1–P4 and P6")
        dimension = "1d" if problem in ("p1", "p2", "p6") else "2d"
        return ROOT / f"src/baselines/mm_oerfm/run_{dimension}.py"
    group = problem if problem in ("p1", "p2") else "p3"
    if problem == "p6":
        group = "p1"
    return ROOT / f"experiments/run_{group}_oe_aprfm.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("oe_aprfm", "oe_apnn", "mm_aprfm"),
        default="oe_aprfm",
    )
    parser.add_argument(
        "--problem",
        choices=tuple(f"p{i}" for i in range(1, 7)),
        default="p1",
    )
    parser.add_argument("--runner-help", action="store_true")
    args, options = parser.parse_known_args()
    try:
        runner = runner_path(args.method, args.problem)
    except ValueError as error:
        parser.error(str(error))
    command = [sys.executable, str(runner)]
    if args.method != "oe_aprfm" or args.problem != "p2":
        command += ["--problem", args.problem]
    if args.method == "oe_aprfm" and args.problem == "p5" and "--p5-definition" not in options:
        command += ["--p5-definition", "square_hole"]
    command += ["--help"] if args.runner_help else options
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
