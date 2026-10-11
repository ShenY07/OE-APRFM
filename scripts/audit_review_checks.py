"""Independent angular-patch/operator checks and archived field arithmetic."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from oeaprfm.features.function_space_v2 import FeatureSpaceXYV
from oeaprfm.features.partition_of_unity import psi_b
from oeaprfm.features.reconstruct_quadrant import OEField2D
from oeaprfm.constraints.oe_quadrant_2d import (
    OEBoundary2D,
    OEInterior2D,
)
from oeaprfm.utils.quadrature import leggauss

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review_audit_20261010"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(11)
    common = dict(
        domain={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "theta": (0.0, np.pi / 2)},
        strides={"x": 2.0, "y": 2.0, "theta": np.pi / 4},
        Jn={"j": 4, "r": 4},
        scale=1.0,
        init_rng=key,
        kn=0.001,
        activation=jnp.tanh,
    )
    c = jnp.asarray(np.random.default_rng(73).normal(size=32))
    constructor = OEField2D(**common, coefficients=c)
    boundary = OEBoundary2D(**common)
    interior = OEInterior2D(
        **common,
        num_quads=16,
        coeff_fns={
            "scattering": lambda x, y: 1 + 0.1 * y,
            "absorption": lambda x, y: 0.1 + 0 * x,
        },
    )
    field = FeatureSpaceXYV(
        domain=common["domain"],
        strides=common["strides"],
        Jn=4,
        scale=1.0,
        activation=jnp.tanh,
    )
    point = (jnp.array([0.2]), jnp.array([-0.3]), jnp.array([0.7]))
    params = field.init(key, *point)

    def components(x, y, theta):
        feats = lambda t: field.apply(
            params, jnp.atleast_1d(x), jnp.atleast_1d(y), jnp.atleast_1d(t)
        ).ravel()
        p, m = feats(theta), feats(theta + jnp.pi)
        p2, m2 = feats(-theta), feats(jnp.pi - theta)
        return jnp.array(
            [
                0.5 * (p + m) @ c[8:16],
                0.5 * (p - m) @ c[:8],
                0.5 * (p2 + m2) @ c[24:32],
                0.5 * (p2 - m2) @ c[16:24],
            ]
        )

    def values(x, y, theta):
        r1, j1, r2, j2 = components(x, y, theta)
        dx = jax.jacfwd(components, 0)(x, y, theta)
        dy = jax.jacfwd(components, 1)(x, y, theta)
        return jnp.array(
            [
                r1,
                j1,
                r2,
                j2,
                jnp.cos(theta) * dx[0] + jnp.sin(theta) * dy[0],
                jnp.cos(theta) * dx[1] + jnp.sin(theta) * dy[1],
                jnp.cos(theta) * dx[2] - jnp.sin(theta) * dy[2],
                jnp.cos(theta) * dx[3] - jnp.sin(theta) * dy[3],
            ]
        )

    q, w = leggauss(16, interval=(0.0, jnp.pi / 2))
    q, w = q.reshape(-1), w.reshape(-1)
    avg = (
        np.asarray(jax.vmap(lambda t: values(0.2, -0.3, t))(q)).T
        @ np.asarray(w)
        * 2
        / np.pi
    )
    errors = []
    for t in (0.1, 0.7, 1.3):
        r1, j1, r2, j2, dr1, dj1, dr2, dj2 = np.asarray(values(0.2, -0.3, t))
        total = 0.97 + 0.001**2 * 0.1
        ar, ad = avg[0] + avg[2], avg[5] + avg[7]
        expected = np.array(
            [
                ad + 0.1 * ar,
                0.001**2 * (dj1 + dj2 - ad) + total * (r1 + r2 - ar),
                dr1 + dr2 + total * (j1 + j2),
                0.001**2 * (dj1 - dj2) + total * (r1 - r2),
                dr1 - dr2 + total * (j1 - j2),
            ]
        )
        actual = np.asarray(
            interior.apply({}, point[0], point[1], jnp.array([t]))
        ) @ np.asarray(c)
        errors.append(float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)
    trace_errors = []
    for t in np.linspace(0.03, 2 * np.pi - 0.03, 25):
        args = (point[0], point[1], jnp.array([t]))
        physical = float(jnp.squeeze(constructor.apply({}, *args)))
        trace = float(
            jnp.squeeze(boundary.apply({}, *args, method=boundary.trace) @ c)
        )
        trace_errors.append(abs(physical - trace))
    assert max(trace_errors) < 1e-12
    theta = jnp.linspace(0.001, 2 * jnp.pi - 0.001, 1000)

    def windows(t):
        fold = jnp.arctan2(jnp.abs(jnp.sin(t)), jnp.abs(jnp.cos(t)))
        raw = psi_b(
            (fold[:, None] - jnp.array([np.pi / 8, 3 * np.pi / 8]))
            / (np.pi / 8)
        )
        return raw / jnp.sum(raw, axis=1, keepdims=True)

    window_error = float(
        jnp.max(jnp.abs(windows(theta) - windows(theta + jnp.pi)))
    )
    np.testing.assert_allclose(
        windows(theta), windows(theta + jnp.pi), atol=1e-13
    )
    result = dict(
        angular_partitions=[1, 1, 2],
        operator_max_abs_error=max(errors),
        boundary_constructor_max_abs_error=max(trace_errors),
        window_antipodal_max_abs_error=window_error,
    )
    (OUT / "angular_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    archive = ROOT / ".release-backup/20261009/original-worktree/results"
    checked = []
    for sub in (
        "p34_representation_comparison",
        "requirement_2026_08_20/parity_budget_raw",
    ):
        for path in sorted((archive / sub).glob("*.json")):
            r = json.loads(path.read_text())
            if not path.with_suffix(".npz").exists():
                continue
            with np.load(path.with_suffix(".npz")) as d:
                f = d["f"]
                exact = d["exact"] if "exact" in d else d["reference_f"]
                mask = (
                    np.broadcast_to(d["mask"][..., None], f.shape)
                    if "mask" in d
                    else np.ones(f.shape, dtype=bool)
                )
                ef = float(
                    np.linalg.norm((f - exact)[mask])
                    / np.linalg.norm(exact[mask])
                )
            checked.append(
                dict(
                    source=str(path.relative_to(ROOT)),
                    reported_Ef=r["relative_l2_f"],
                    recomputed_Ef=ef,
                    delta=abs(ef - r["relative_l2_f"]),
                )
            )
    (OUT / "archive_field_checks.json").write_text(
        json.dumps(checked, indent=2)
    )
    print(
        "Archived field checks:",
        len(checked),
        "max delta",
        max(r["delta"] for r in checked),
        flush=True,
    )


if __name__ == "__main__":
    main()
