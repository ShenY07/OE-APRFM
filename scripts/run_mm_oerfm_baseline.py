"""List, validate, or execute the vendored MM-OERFM notebook baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "mm_oerfm"
MANIFEST = BASELINE / "baseline.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def check(manifest: dict) -> None:
    required = (
        "constraints/continuous1d.py",
        "constraints/continuous2d.py",
        "modules/func_space.py",
        "modules/solution.py",
        "modules/generator.py",
        "solver/least_square.py",
        "numerical/micro_macro.ipynb",
        "numerical/SI_1d.ipynb",
        "numerical/DSA_SI_1d.ipynb",
    )
    source = BASELINE / "original" / "src"
    missing = [name for name in required if not (source / name).is_file()]
    for item in manifest["experiments"].values():
        for key in ("notebook", "configuration"):
            if not (BASELINE / item[key]).is_file():
                missing.append(item[key])
    if not (BASELINE / "original/test/rte_1d_ex1.ipynb").is_file():
        missing.append("original/test/rte_1d_ex1.ipynb")
    if missing:
        raise SystemExit("Missing MM-OERFM files:\n" + "\n".join(missing))
    print("MM-OERFM baseline files: OK")


def execute(name: str, manifest: dict) -> None:
    item = manifest["experiments"][name]
    notebook = BASELINE / item["notebook"]
    output = ROOT / manifest["result_root"] / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source = str(BASELINE / "original" / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("JAX_ENABLE_X64", "true")
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        f"{name}.ipynb",
        "--output-dir",
        str(output),
        "--ExecutePreprocessor.timeout=-1",
    ]
    subprocess.run(command, cwd=BASELINE / "original", env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.list:
        for name, item in manifest["experiments"].items():
            target = item["comparison_target"] or "supplementary"
            print(f"{name:8s} -> {target:13s} | {item['status']}")
    if args.check:
        check(manifest)
    if args.execute:
        if args.experiment not in manifest["experiments"]:
            parser.error("--execute requires a valid --experiment")
        check(manifest)
        execute(args.experiment, manifest)
    if not (args.list or args.check or args.execute):
        parser.print_help()


if __name__ == "__main__":
    main()
