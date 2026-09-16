"""Sequentially regenerate every retained 1-D OE-APRFM result with unit-L2 rows."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(script: str, *args: object) -> None:
    command = [PYTHON, str(ROOT / "scripts" / script), *map(str, args)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def p1(output: str, epsilon: float, seed: int, *, features=64,
       collocation=None, tag="", variant="full") -> None:
    args = ["--epsilon", epsilon, "--seed", seed, "--output-dir", output,
            "--partitions", 1, 1, "--features", features, "--scale", 1,
            "--rcond", 1e-12, "--variant", variant]
    if collocation:
        args += ["--collocation", *collocation]
    if tag:
        args += ["--tag", tag]
    run("run_p1_oe_aprfm.py", *args)


def main() -> None:
    seeds = (11, 23, 37)
    epsilons = (1.0, 1e-3, 1e-6)

    if not os.environ.get("OE_QUEUE_RESUME"):
        # Main Table 2.
        for epsilon in epsilons:
            for seed in seeds:
                p1("results/epsilon_scan", epsilon, seed,
                   collocation=(32, 32), tag="epsscan")

        # Main Table 4 structural comparison (non-OE baselines already use L2).
        for epsilon in epsilons:
            for variant in ("full", "oe_original", "full_angular"):
                p1("results/ablation", epsilon, 11, collocation=(32, 32),
                   tag=variant, variant=variant)

    # Main Table 5, retained seeds only.
    for epsilon in (1.0, 1e-3):
        for seed in seeds:
            run("run_p2_oe_aprfm.py", "--epsilon", epsilon, "--seed", seed,
                "--output-dir", "results/consistency/p2", "--reference-dir",
                "results/references_notebook", "--partitions", 2, 4, "--features", 64,
                "--scale", 1, "--rcond", 1e-6, "--tag", "fixed")

    # Main Table 6.
    for epsilon in epsilons:
        for seed in seeds:
            args = ["--epsilon", epsilon, "--problem", "p6", "--seed", seed,
                    "--output-dir", "results/requirement_2026_08_20/p6_raw",
                    "--partitions", 1, 1, "--features", 64, "--scale", 1,
                    "--rcond", 1e-12, "--collocation", 32, 32,
                    "--tag", "fixed"]
            run("run_p1_oe_aprfm.py", *args)

    # Main/Supplement accuracy-cost OE-APRFM rows.
    for features in (32, 64, 128):
        for seed in seeds:
            dirname = f"oe_p1_{'seed'+str(seed)+'_' if seed != 11 else ''}J{features}"
            p1(f"results/requirement_2026_08_20/efficiency_raw/{dirname}",
               1e-3, seed, features=features, collocation=(32, 32),
               tag="efficiency")

    # Recommended 1-D collocation refinement.
    for label, size in ((2, 14), (4, 19), (8, 27), (12, 33)):
        for seed in seeds:
            p1("results/collocation_sufficiency", 1e-3, seed, features=128,
               collocation=(size, size), tag=f"oversampling_{label}")

    # Recommended supplementary feature-resolution sweep.
    feature_sizes = {32: 14, 64: 19, 128: 27, 256: 38}
    for epsilon in epsilons:
        for features, size in feature_sizes.items():
            for seed in seeds:
                p1("results/feature_convergence", epsilon, seed,
                   features=features, collocation=(size, size),
                   tag=f"jconv8_J{features}")

    # Recommended parity feature-budget sweep.
    parity_collocation = {16: (11, 28), 32: (16, 33),
                          64: (32, 32), 128: (64, 33)}
    parity_dir = "results/requirement_2026_08_20/parity_budget_raw"
    for features, collocation in parity_collocation.items():
        for variant in ("full", "full_angular"):
            for seed in seeds:
                p1(parity_dir, 1e-3, seed, features=features,
                   collocation=collocation,
                   tag=f"paritybudget_J{features}_{variant}", variant=variant)

    subprocess.run([PYTHON, str(ROOT / "scripts" / "build_frozen_manuscript_tables.py")],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
