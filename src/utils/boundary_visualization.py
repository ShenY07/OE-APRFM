"""Provide boundary visualization functionality for the utils layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

import matplotlib.pyplot as plt


def plotBC(xv_bcl, xv_bcr, vmap_approx_fn):
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.plot(
        xv_bcl[:, 1],
        vmap_approx_fn(xv_bcl),
        "--",
        color="#94A3C0",
        markersize=10,
        linewidth=2,
        label="x = 0, v > 0",
    )
    plt.plot(
        xv_bcr[:, 1],
        vmap_approx_fn(xv_bcr),
        "-",
        color="#725E79",
        label="x = 1, v < 0",
    )
    plt.ylim(ymin=-0.1, ymax=1.1)
    plt.xlabel(r"$v$", fontsize=16)
    plt.ylabel(r"$f (x, v)$", fontsize=16)
    plt.grid()
    plt.legend()
    # line width of four axises
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    plt.savefig("rte_approx_BC.png", dpi=300)
    plt.show()
    plt.close()
