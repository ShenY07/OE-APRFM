"""Generate reproducible solution, reference, and error visualizations."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-experiment-figures")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from configuration.p5_heterogeneous_2d import get_config as p5_config


ROOT = Path("results")
FIGURES = ROOT / "figures"
REFERENCES = ROOT / "references"


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def representative_p1(epsilon):
    records = []
    for path in (ROOT / "raw").glob(f"p1_oe_aprfm_eps_{epsilon:.0e}_seed_*.json"):
        records.append((path, json.loads(path.read_text())))
    median = np.median([record["relative_l2_f"] for _, record in records])
    path, record = min(records, key=lambda item: abs(item[1]["relative_l2_f"] - median))
    return path.with_suffix(".npz"), record


def plot_p1():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, epsilon in enumerate((1.0, 1.0e-16)):
        path, record = representative_p1(epsilon)
        with np.load(path) as data:
            x, v, numerical, exact = data["x"], data["v"], data["f"], data["exact"]
        extent = (v[0], v[-1], x[0], x[-1])
        error = np.abs(numerical - exact)
        images = (
            axes[row, 0].imshow(exact, origin="lower", extent=extent, aspect="auto"),
            axes[row, 1].imshow(numerical, origin="lower", extent=extent, aspect="auto"),
            axes[row, 2].imshow(error, origin="lower", extent=extent, aspect="auto", cmap="magma"),
        )
        axes[row, 0].set_ylabel(fr"$x$ ($\varepsilon={epsilon:.0e}$)")
        for axis, title, image in zip(axes[row], ("exact", "OE-APRFM", "absolute error"), images):
            axis.set_xlabel("v")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, shrink=0.82)
        axes[row, 2].text(0.02, 0.04, fr"$E_f={record['relative_l2_f']:.2e}$", transform=axes[row, 2].transAxes, color="white")
    save(fig, "p1_solution_and_error.png")


def plot_p2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, epsilon in enumerate((1.0, 1.0e-3)):
        prefix = REFERENCES / f"p2_parity_ref_eps_{epsilon:.0e}_level_"
        with np.load(f"{prefix}A.npz") as source:
            a = {key: source[key] for key in source.files}
        with np.load(f"{prefix}B.npz") as source:
            b = {key: source[key] for key in source.files}
        rho_a = np.interp(b["x"], a["x"], a["rho"])
        axes[row, 0].plot(b["x"], b["rho"], label="level B")
        axes[row, 0].plot(b["x"], rho_a, "--", label="level A interpolated")
        axes[row, 0].set(ylabel=fr"$\rho$ ($\varepsilon={epsilon:.0e}$)", title="reference density")
        axes[row, 1].semilogy(b["x"], np.abs(b["rho"] - rho_a) + 1e-18)
        axes[row, 1].set(title="A/B density difference", ylabel="absolute difference")
        for axis in axes[row]:
            axis.set_xlabel("x")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    save(fig, "p2_reference_and_refinement_error.png")


def plot_exact_2d():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, problem in zip(axes, ("p3", "p4")):
        with np.load(REFERENCES / f"{problem}_exact.npz") as data:
            x, y, rho = data["x"], data["y"], data["rho"]
        image = axis.imshow(rho.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="viridis")
        axis.set(xlabel="x", ylabel="y", title=f"{problem.upper()} exact density")
        fig.colorbar(image, ax=axis)
    save(fig, "p3_p4_exact_references.png")


def plot_p5_coefficients():
    config = p5_config(1.0e-3)
    x = np.linspace(0.0, 1.0, 401)
    y = np.linspace(0.0, 1.0, 401)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    scattering = np.asarray(config.model.coeff.scattering(xx, yy))
    absorption = np.asarray(config.model.coeff.absorption(xx, yy))
    source = np.asarray(config.model.source(xx, yy, 0.0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, field, title in zip(axes, (scattering, absorption, source), (r"$\sigma_s$", r"$\sigma_a$", "smooth internal source")):
        image = axis.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        axis.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(image, ax=axis)
    save(fig, "p5_smoothed_problem_fields.png")


def plot_p5_reference():
    datasets = []
    for epsilon in (1.0, 1.0e-1):
        prefix = REFERENCES / f"p5_parity_ref_eps_{epsilon:.0e}_level_"
        with np.load(f"{prefix}A.npz") as source:
            a = {key: source[key] for key in source.files}
        with np.load(f"{prefix}B.npz") as source:
            b = {key: source[key] for key in source.files}
        mesh = np.stack(np.meshgrid(b["x"], b["y"], indexing="ij"), axis=-1)
        rho_a = RegularGridInterpolator(
            (a["x"], a["y"]), a["rho"], bounds_error=False, fill_value=None
        )(mesh)
        datasets.append((epsilon, b, rho_a))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    for row, (epsilon, b, rho_a) in enumerate(datasets):
        difference = np.abs(b["rho"] - rho_a)
        fields = (b["rho"], rho_a, difference)
        titles = ("level B reference", "level A interpolated", "absolute A/B difference")
        for axis, field, title in zip(axes[row], fields, titles):
            image = axis.imshow(
                field.T, origin="lower",
                extent=(b["x"][0], b["x"][-1], b["y"][0], b["y"][-1]),
                cmap="magma" if "difference" in title else "viridis",
            )
            axis.set(xlabel="x", ylabel="y", title=title + fr" ($\varepsilon={epsilon:g}$)")
            fig.colorbar(image, ax=axis)
    save(fig, "p5_reference_and_refinement_error.png")

    epsilon, b, rho_a = datasets[-1]
    center_y = int(np.argmin(np.abs(b["y"])))
    channel_y = np.asarray(0.35 * b["x"])
    channel_indices = np.abs(b["y"][None, :] - channel_y[:, None]).argmin(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(b["x"], b["rho"][:, center_y], label="level B")
    axes[0].plot(b["x"], rho_a[:, center_y], "--", label="level A")
    axes[0].set(title=fr"horizontal disk cut ($\varepsilon={epsilon:g}$)", xlabel="x", ylabel=r"$\rho(x,0)$")
    axes[1].plot(b["x"], b["rho"][np.arange(len(b["x"])), channel_indices], label="level B")
    axes[1].plot(b["x"], rho_a[np.arange(len(b["x"])), channel_indices], "--", label="level A")
    axes[1].set(title=fr"oblique channel cut ($\varepsilon={epsilon:g}$)", xlabel="x", ylabel=r"$\rho(x,0.35x)$")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    save(fig, "p5_reference_line_cuts.png")


if __name__ == "__main__":
    plot_p1()
    plot_p2()
    plot_exact_2d()
    plot_p5_coefficients()
    plot_p5_reference()
