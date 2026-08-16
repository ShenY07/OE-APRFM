"""Save exact P1/P3/P4 solutions on their independent evaluation grids."""

from pathlib import Path

import numpy as np

from configuration.p1_manufactured_1d import get_config as p1_config
from configuration.p3_manufactured_2d import get_config as p3_config
from configuration.p4_circular_hole_2d import get_config as p4_config


def main() -> None:
    output = Path("results/references")
    output.mkdir(parents=True, exist_ok=True)
    p1 = p1_config()
    nx, nv = map(int, p1.protocol.evaluation_grid)
    x = np.linspace(*p1.mesh.domain.x, nx)
    v = np.linspace(-1.0, 1.0, nv)
    f = 1.0 - x[:, None] + np.zeros((1, nv))
    np.savez_compressed(output / "p1_exact.npz", x=x, velocity=v, f=f, rho=1.0 - x)

    for problem, config in (("p3", p3_config()), ("p4", p4_config())):
        nx, ny, na = map(int, config.protocol.evaluation_grid)
        x = np.linspace(*config.mesh.domain.x, nx)
        y = np.linspace(*config.mesh.domain.y, ny)
        theta = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        if problem == "p3":
            rho = np.exp(-xx - yy)
            mask = np.ones_like(rho, dtype=bool)
        else:
            rho = 1.0 / (1.0 + xx**2 + yy**2)
            mask = xx**2 + yy**2 >= 0.5**2
            rho = np.where(mask, rho, np.nan)
        f = np.broadcast_to(rho[:, :, None], (nx, ny, na)).copy()
        np.savez_compressed(output / f"{problem}_exact.npz", x=x, y=y, theta=theta, f=f, rho=rho, domain_mask=mask)
        print(output / f"{problem}_exact.npz")


if __name__ == "__main__":
    main()
