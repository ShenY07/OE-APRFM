#!/usr/bin/env python3
"""Evaluate imported OE-APNN r/j checkpoints on the common reference grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
APNN = ROOT / "baselines/oe-apnn"
sys.path.insert(0, str(APNN))
from ap_eqn import build_solver
from config import get_config
from networks import build_networks


def reference(problem, epsilon):
    if problem == "P1":
        path = ROOT / "results/references/p1_exact.npz"
    elif problem == "P2":
        path = ROOT / f"results/references/p2_parity_ref_eps_{epsilon:.0e}_level_B.npz"
    elif problem in ("P3", "P4"):
        path = ROOT / f"results/references/{problem.lower()}_exact.npz"
    else:
        x = np.linspace(-1.0, 1.0, 65)
        y = np.linspace(-1.0, 1.0, 65)
        theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        rho = np.exp(-xx - yy)
        mask = (np.abs(xx) > 1/3) | (np.abs(yy) > 1/3)
        return dict(x=x, y=y, theta=theta, f=np.broadcast_to(rho[..., None], (65, 65, 64)), rho=rho, mask=mask)
    if not path.exists():
        if problem == "P2":
            raise FileNotFoundError(f"generate the locked P2 reference first: {path}")
        if problem == "P1":
            x=np.linspace(0.0,1.0,257); velocity=np.linspace(-1.0,1.0,64)
            rho=1.0-x; return dict(x=x,velocity=velocity,f=np.broadcast_to(rho[:,None],(257,64)),rho=rho,mask=np.ones(257,bool))
        x=np.linspace(-1.0,1.0,65); y=x.copy(); theta=np.linspace(0.0,2*np.pi,64,endpoint=False)
        xx,yy=np.meshgrid(x,y,indexing="ij")
        rho=np.exp(-xx-yy) if problem=="P3" else 1/(1+xx**2+yy**2)
        mask=np.ones_like(rho,bool) if problem=="P3" else xx**2+yy**2>=0.25
        return dict(x=x,y=y,theta=theta,f=np.broadcast_to(rho[...,None],(65,65,64)),rho=rho,mask=mask)
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    result["mask"] = result.get("domain_mask", np.ones(result["rho"].shape, bool)).astype(bool)
    return result


def evaluate(run_dir: Path, batch_size=4096, device_id=0):
    metadata = json.loads((run_dir / "metrics.json").read_text())
    problem, epsilon, seed = metadata["problem"], float(metadata["epsilon"]), int(metadata["seed"])
    torch.set_default_dtype(torch.float64)
    config = get_config(problem.lower(), epsilon)
    config.model.device_ids = [device_id]
    solver = build_solver(config)
    input_size = 2 if problem in ("P1", "P2") else 4
    nets = {name: net.to(solver.device).double() for name, net in build_networks(config).items()}
    for name, net in nets.items():
        net.load_state_dict(torch.load(run_dir / f"{name}.pt", map_location=solver.device, weights_only=True))
        net.eval()
    ref = reference(problem, epsilon)
    if input_size == 2:
        xx, vv = np.meshgrid(ref["x"], ref["velocity"], indexing="ij")
        flat = (xx.ravel(), vv.ravel())
    else:
        xx, yy, tt = np.meshgrid(ref["x"], ref["y"], ref["theta"], indexing="ij")
        flat = (xx.ravel(), yy.ravel(), tt.ravel())
    started = time.perf_counter()
    blocks = []
    with torch.no_grad():
        for begin in range(0, flat[0].size, batch_size):
            args = [torch.as_tensor(a[begin:begin+batch_size], device=solver.device).reshape(-1, 1) for a in flat]
            blocks.append(solver.reconstruct(nets, *args).cpu().numpy().ravel())
    reconstruction_factor = 1.0
    field = reconstruction_factor * np.concatenate(blocks).reshape(ref["f"].shape)
    if input_size == 2:
        if "weights" in ref:
            rho = 0.5 * (field @ ref["weights"])
        else:
            rho = np.trapezoid(field, ref["velocity"], axis=1) / 2.0
        phase_mask = np.ones(field.shape, bool)
        axes = dict(x=ref["x"], velocity=ref["velocity"])
    else:
        rho = field.mean(axis=2)
        phase_mask = np.broadcast_to(ref["mask"][..., None], field.shape)
        axes = dict(x=ref["x"], y=ref["y"], theta=ref["theta"], mask=ref["mask"])
    mask = ref["mask"]
    ef = float(np.linalg.norm((field-ref["f"])[phase_mask]) / np.linalg.norm(ref["f"][phase_mask]))
    er = float(np.linalg.norm((rho-ref["rho"])[mask]) / np.linalg.norm(ref["rho"][mask]))
    inference = time.perf_counter() - started
    np.savez_compressed(run_dir / "solution.npz", **axes, f=field, rho=rho, reference_f=ref["f"], reference_rho=ref["rho"], error_f=np.abs(field-ref["f"]), error_rho=np.abs(rho-ref["rho"]))
    metadata.update(E_f=ef, E_rho=er, evaluation_status="complete", inference_time_s=inference, reconstruction_factor=reconstruction_factor)
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"problem":problem,"epsilon":epsilon,"seed":seed,"E_f":ef,"E_rho":er,"inference_time_s":inference}), flush=True)


def mark_incompatible(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    metadata = json.loads(metrics_path.read_text())
    problem = metadata["problem"]
    reasons = {
        "P1": "archive normalization differs from the current P1 definition",
        "P2": "archive boundary/solution definition differs from the current P2 reference",
        "P5": "archive solution is P3-like and does not match the current heterogeneous P5",
    }
    metadata.update(E_f=None, E_rho=None, evaluation_status="incompatible", compatibility_note=reasons[problem])
    metadata.pop("inference_time_s", None)
    metadata.pop("reconstruction_factor", None)
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "solution.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mark-incompatible", action="store_true")
    args = parser.parse_args()
    if args.mark_incompatible:
        mark_incompatible(args.run_dir)
    else:
        evaluate(args.run_dir, args.batch_size, args.device)
