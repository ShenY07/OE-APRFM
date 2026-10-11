"""Plot a saved density field or OE-APNN training history."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_file(source, output):
    with np.load(source, allow_pickle=False) as data:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        if "total_loss" in data:
            for key in (
                "total_loss",
                "loss_macro",
                "loss_even",
                "loss_odd",
                "loss_boundary",
            ):
                if key in data:
                    ax.semilogy(data["step"], data[key], label=key)
            ax.set(xlabel="Training step", ylabel="Loss")
            ax.legend()
        elif "rho" in data and data["rho"].ndim == 2:
            rho = data["rho"]
            if "mask" in data:
                rho = np.where(data["mask"], rho, np.nan)
            field = ax.pcolormesh(
                data["x"], data["y"], rho.T, shading="auto", cmap="viridis"
            )
            fig.colorbar(field, ax=ax, label="Density")
            ax.set(xlabel="x", ylabel="y", aspect="equal")
        elif "rho" in data:
            ax.plot(data["x"], data["rho"], label="Prediction")
            ax.set(xlabel="x", ylabel="Density")
        else:
            plt.close(fig)
            raise ValueError("Expected density fields or training history")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_file(args.source, args.output)


if __name__ == "__main__":
    main()
