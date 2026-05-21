import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np


def contourf(kn: str, method: str = "aprfm"):
    file = f"../data/{method}1d_solutions_{kn:.1e}.npz"
    results = np.load(file)
    X, V = results["X"], results["V"]
    F, F_hat = results["F"], results["F_hat"]

    font_format = {"family": "Times New Roman", "size": 20}

    plt.figure(figsize=(24, 6))
    ax1 = plt.subplot(1, 3, 1)
    plt.contourf(X, V, F, 100, cmap="plasma")
    plt.xticks(fontproperties="Times New Roman", size=12)
    plt.yticks(fontproperties="Times New Roman", size=12)
    plt.xlabel(r"$x$", font_format)
    plt.ylabel(r"$v$", font_format)
    plt.colorbar()
    plt.title("Reference solution", font_format)
    ax1.spines["bottom"].set_linewidth(2)
    ax1.spines["left"].set_linewidth(2)
    ax1.spines["right"].set_linewidth(2)
    ax1.spines["top"].set_linewidth(2)

    ax2 = plt.subplot(1, 3, 2)
    plt.contourf(X, V, F_hat, 100, cmap="plasma")
    plt.xticks(fontproperties="Times New Roman", size=12)
    plt.yticks(fontproperties="Times New Roman", size=12)
    plt.xlabel(r"$x$", font_format)
    plt.ylabel(r"$v$", font_format)
    plt.colorbar()
    plt.title("RFM solution", font_format)
    ax2.spines["bottom"].set_linewidth(2)
    ax2.spines["left"].set_linewidth(2)
    ax2.spines["right"].set_linewidth(2)
    ax2.spines["top"].set_linewidth(2)

    ax3 = plt.subplot(1, 3, 3)
    plt.contourf(X, V, np.abs(F - F_hat), 100, cmap="plasma")
    plt.xticks(fontproperties="Times New Roman", size=12)
    plt.yticks(fontproperties="Times New Roman", size=12)
    plt.xlabel(r"$x$", font_format)
    plt.ylabel(r"$v$", font_format)
    plt.colorbar(format="%.0e")
    plt.title("Error", font_format)
    ax3.spines["bottom"].set_linewidth(2)
    ax3.spines["left"].set_linewidth(2)
    ax3.spines["right"].set_linewidth(2)
    ax3.spines["top"].set_linewidth(2)

    plt.savefig(f"./{method}1d_{kn:.1e}_contourf.png", dpi=300)
    plt.show()
    plt.close()

    return f"Saved the plot as {method}1d_{kn:.1e}.png"


def surface(kn: str, method: str = "aprfm"):
    file = f"../data/{method}1d_solutions_{kn:.1e}.npz"
    results = np.load(file)
    X, V = results["X"], results["V"]
    F, F_hat = results["F"], results["F_hat"]

    font_format = {"family": "Times New Roman", "size": 20}
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        X, V, F, cmap="plasma", linewidth=0, antialiased=False
    )
    ax1.set_xlabel(r"$x$", font_format)
    ax1.set_ylabel(r"$v$", font_format)
    ax1.zaxis.set_major_locator(LinearLocator(5))
    ax1.zaxis.set_major_formatter("{x:.01f}")
    ax1.set_title("Reference solution", font_format)
    ax1.spines["bottom"].set_linewidth(2)
    ax1.spines["left"].set_linewidth(2)
    ax1.spines["right"].set_linewidth(2)
    ax1.spines["top"].set_linewidth(2)
    fig.colorbar(surf1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        X, V, F_hat, cmap="plasma", linewidth=0, antialiased=False
    )
    ax2.set_xlabel(r"$x$", font_format)
    ax2.set_ylabel(r"$v$", font_format)
    ax2.set_title("APRFM solution", font_format)
    ax2.zaxis.set_major_locator(LinearLocator(5))
    ax2.zaxis.set_major_formatter("{x:.01f}")
    ax2.spines["bottom"].set_linewidth(2)
    ax2.spines["left"].set_linewidth(2)
    ax2.spines["right"].set_linewidth(2)
    ax2.spines["top"].set_linewidth(2)
    fig.colorbar(surf2, shrink=0.5, aspect=5)

    plt.savefig(f"./{method}_{kn:.1e}_surface.png", dpi=300)
    plt.show()
    plt.close()

    return f"Saved the plot as {method}_{kn:.1e}_surface.png"
