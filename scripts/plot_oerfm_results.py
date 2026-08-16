"""Plot OE-APRFM solutions and error fields from completed seed sweeps."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-oerfm")
import matplotlib.pyplot as plt
import numpy as np


def representative(problem, epsilon, root=Path("results/raw")):
    records=[]
    for path in root.glob(f"{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_*.json"):
        records.append((path,json.loads(path.read_text())))
    median=np.median([item[1]["relative_l2_rho"] for item in records])
    return min(records,key=lambda item:abs(item[1]["relative_l2_rho"]-median))


def plot_p2(root=Path("results/raw"), suffix=""):
    fig,axes=plt.subplots(2,3,figsize=(12,6.8),constrained_layout=True)
    for row,epsilon in enumerate((1.0,1e-3)):
        path,record=representative("p2",epsilon,root)
        with np.load(path.with_suffix(".npz")) as data:
            x=data["x"]; rho=data["rho"]; reference=data["reference_rho"]; error=data["error_rho"]
        axes[row,0].plot(x,reference,label="parity reference"); axes[row,0].plot(x,rho,"--",label="OE-APRFM")
        axes[row,1].semilogy(x,error+1e-18)
        axes[row,2].plot(x,rho-reference)
        axes[row,0].set_ylabel(fr"$\rho$ ($\varepsilon={epsilon:.0e}$)")
        axes[row,1].set_ylabel("absolute error"); axes[row,2].set_ylabel("signed error")
        for axis,title in zip(axes[row],("solution comparison","absolute error distribution","signed error")):
            axis.set(xlabel="x",title=title); axis.grid(alpha=.25)
        axes[row,2].text(.03,.07,fr"$E_f={record['relative_l2_f']:.2e}$\n$E_\rho={record['relative_l2_rho']:.2e}$",transform=axes[row,2].transAxes)
    axes[0,0].legend(frameon=False)
    output=Path(f"results/figures/p2_oerfm_solution_and_error{suffix}.png"); output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=300); plt.close(fig); print(output)


def plot_2d(problem, root=Path("results/raw"), suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    for row, epsilon in enumerate((1.0, 1.0e-3)):
        path, record = representative(problem, epsilon, root)
        with np.load(path.with_suffix(".npz")) as data:
            x, y = data["x"], data["y"]
            rho, reference, error = data["rho"], data["reference_rho"], data["error_rho"]
            mask = data["mask"].astype(bool)
        fields = tuple(np.where(mask, field, np.nan) for field in (reference, rho, error))
        for axis, field, title in zip(axes[row], fields, ("reference density", "OE-APRFM density", "absolute error")):
            image = axis.imshow(field.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="magma" if "error" in title else "viridis")
            axis.set(xlabel="x", ylabel="y", title=title + fr" ($\varepsilon={epsilon:.0e}$)")
            fig.colorbar(image, ax=axis)
        axes[row,2].text(.03,.06,fr"$E_f={record['relative_l2_f']:.2e}$\n$E_\rho={record['relative_l2_rho']:.2e}$",transform=axes[row,2].transAxes,color="white")
    output=Path(f"results/figures/{problem}_oerfm_solution_and_error{suffix}.png"); fig.savefig(output,dpi=300); plt.close(fig); print(output)


if __name__=="__main__":
    plot_p2()
    plot_p2(Path("results/final"), "_tuned")
    plot_2d("p3")
    plot_2d("p4")
    plot_2d("p3", Path("results/final"), "_tuned")
    plot_2d("p4", Path("results/final"), "_tuned")
    plot_2d("p5", Path("results/final"), "_tuned")
