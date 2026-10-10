#!/usr/bin/env python3
"""Retrain hard-parity OE-APNN, three seeds per GPU on cuda:1 and cuda:2."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 11, 17)
PROTOCOL = "oe-apnn-p1-p5-v2-hard-parity"


def groups():
    # Short cost runs first, then the full P1--P5 baseline.
    result = []
    for steps in (500, 2000, 5000):
        for problem in ("p1", "p3"):
            result.append(dict(suite="cost", problem=problem, epsilon=0.001, steps=steps))
    for problem in ("p1", "p2", "p3", "p4", "p5"):
        for eps in (1., .001):
            result.append(dict(suite="baseline", problem=problem, epsilon=eps, steps=20000))
    for group in result:
        group["id"] = f"{group['suite']}/{group['problem'].upper()}/eps_{group['epsilon']:.0e}/steps_{group['steps']}"
    return result


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/oe_apnn_parity_20261010")
    parser.add_argument("--devices", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = groups()
    source_paths = list((ROOT / "baselines/oe-apnn").glob("*.py")) + [Path(__file__), ROOT / "scripts/report_oe_apnn_retrain.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest = dict(protocol=PROTOCOL, seeds=SEEDS, devices=args.devices,
                    max_parallel_per_device=3, interior_samples=4096, boundary_samples=1024,
                    quadrature_points=16, dtype="float64", source_sha256=hashes, groups=plan)
    old_manifest = out / "manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError("Manifest changed; use a new output directory to avoid mixing runs")
    atomic_json(old_manifest, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    import torch
    for device in args.devices:
        torch.empty(1, device=f"cuda:{device}")
    # CPU reference generation is separate from the GPU neural-network training.
    for eps in (1., .001):
        reference = ROOT / f"results/references/p2_parity_ref_eps_{eps:.0e}_level_B"
        if not reference.with_suffix(".npz").exists():
            raise RuntimeError(f"Missing P2 reference: {reference}.npz")
        if not json.loads(reference.with_suffix(".json").read_text())["converged"]:
            raise RuntimeError(f"Unconverged P2 reference: {reference}")
    for p in source_paths:
        target = out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    for name in ("table.md", "figure.md"):
        target = out / "original" / name
        if not target.exists() and (ROOT / "results" / name).exists():
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / "results" / name, target)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(out / "mplconfig"), PYTHONUNBUFFERED="1")
    state = dict(status="running", pid=os.getpid(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), groups={})
    state_lock = threading.Lock()
    stop = threading.Event()

    def set_state(group, value):
        with state_lock:
            state["groups"][group["id"]] = value
            atomic_json(out / "status.json", state)

    def run_lane(device, lane):
        for group in lane:
            if stop.is_set():
                return
            processes = []
            try:
                for seed in SEEDS:
                    folder = out / "runs" / group["id"] / f"seed_{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    metrics = folder / "metrics.json"
                    if metrics.exists():
                        m = json.loads(metrics.read_text())
                        if (m.get("protocol") == PROTOCOL and m.get("evaluation_status") == "complete"
                                and m["steps"] == group["steps"] and m["seed"] == seed):
                            continue
                        raise RuntimeError(f"Incompatible existing metrics: {metrics}")
                    command = [sys.executable, str(ROOT / "baselines/oe-apnn/train.py"),
                               "--problem", group["problem"], "--epsilon", str(group["epsilon"]),
                               "--steps", str(group["steps"]), "--seed", str(seed), "--device", str(device),
                               "--require-cuda", "--interior-samples", "4096", "--boundary-samples", "1024",
                               "--quadrature-points", "16", "--output-dir", str(folder)]
                    atomic_json(folder / "launch.json", dict(command=command, device=device, seed=seed))
                    log = (folder / "train.log").open("w")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    processes.append((process, log, seed))
                set_state(group, dict(status="running", device=device, pids=[p.pid for p, _, _ in processes]))
                print(f"cuda:{device}: {group['id']} started", flush=True)
                while any(p.poll() is None for p, _, _ in processes):
                    if any(p.poll() not in (None, 0) for p, _, _ in processes):
                        raise RuntimeError(f"Training failed: {group['id']}; inspect train.log")
                    time.sleep(2)
                if any(p.returncode != 0 for p, _, _ in processes):
                    raise RuntimeError(f"Training failed: {group['id']}")
                set_state(group, dict(status="complete", device=device))
                # Serialize report writes across the two GPU lanes.
                with state_lock:
                    subprocess.run([sys.executable, str(ROOT / "scripts/report_oe_apnn_retrain.py"),
                                    "--output-dir", str(out)], cwd=ROOT, env=env, check=True)
                print(f"cuda:{device}: {group['id']} complete", flush=True)
            except BaseException as exc:
                stop.set()
                for p, _, _ in processes:
                    if p.poll() is None:
                        p.terminate()
                for p, _, _ in processes:
                    p.wait()
                set_state(group, dict(status="failed", device=device, error=str(exc)))
                raise
            finally:
                for _, log, _ in processes:
                    log.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, device, plan[i::2]) for i, device in enumerate(args.devices)]
            for future in as_completed(futures):
                future.result()
        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        atomic_json(out / "status.json", state)


if __name__ == "__main__":
    main()
