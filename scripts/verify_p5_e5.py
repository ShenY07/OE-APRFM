"""Resolve P5 provenance from exact source hashes; audit E5 acceptance scope."""

import ast, csv, hashlib, importlib.util, json
from pathlib import Path
from scripts.source_compat import rename_source
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from oeaprfm.problems.p5_heterogeneous_2d import get_config
from oeaprfm.features.sampling import Sample2D
from oeaprfm.features.reconstruct_quadrant import OEField2D
import oeaprfm.features.function_space_v2 as fs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p5_e5_verification_20261010"
ARCH = ROOT / ".release-backup/20261009/original-worktree"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


oldrunner = OUT / "source/historical_run_p3_oe_aprfm.py"
oldbdy = OUT / "source/historical_continuous_2d_odd_even_v2.py"
records = [
    json.loads(p.read_text())
    for p in sorted(
        (ARCH / "results/comparison_repairs_2026_09_10").glob("p5_*/*.json")
    )
]
assert len(records) == 3
for r in records:
    assert r["source_sha256"]["scripts/run_p3_oe_aprfm.py"] == sha(oldrunner)
    assert r["source_sha256"][
        "src/constraints/continuous_2d_odd_even_v2.py"
    ] == sha(oldbdy)
    assert r["num_rows"] == 58096 and r["partitions"] == [1, 1, 2]


def trace_ast(p):
    tree = ast.parse(rename_source(p.read_text()))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OEBoundary2D"
    )
    return ast.dump(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "trace"
        ),
        include_attributes=False,
    )


assert trace_ast(oldbdy) == trace_ast(
    ROOT / "src/oeaprfm/constraints/oe_quadrant_2d.py"
)
spec = importlib.util.spec_from_file_location("p5_historical_boundary", oldbdy)
module = importlib.util.module_from_spec(spec)
exec(
    compile(rename_source(oldbdy.read_text()), str(oldbdy), "exec"),
    module.__dict__,
)
config = get_config(0.001)
domain = dict(config.mesh.domain)
sample = Sample2D(
    domain,
    {"interior": (32, 32, 16), "boundary": (32, 32, 16)},
    config.model.sample_mode,
)
coords = [
    sample.pts_left,
    sample.pts_right,
    sample.pts_lower,
    sample.pts_upper,
]
ni = len(sample.pts_int[0])
counts = dict(
    macro=ni, even=2 * ni, odd=2 * ni, boundary=sum(len(p[0]) for p in coords)
)
assert counts == dict(macro=10800, even=21600, odd=21600, boundary=4096)
assert sum(counts.values()) == 58096
for side, pts in enumerate(coords):
    theta = np.asarray(pts[2]).ravel()
    inward = [np.cos(theta), -np.cos(theta), np.sin(theta), -np.sin(theta)][
        side
    ]
    assert np.all(inward > 0)
fs.seedX = 11
common = dict(
    domain=domain,
    strides={"x": 1.0, "y": 1.0, "theta": np.pi / 4},
    Jn={"j": 128, "r": 128},
    scale=0.25,
    init_rng=jax.random.key(11),
    kn=0.001,
    activation=jnp.tanh,
)
boundary = module.OEBoundary2D(**common)
trace = jax.jit(
    jax.vmap(
        lambda x, y, t: boundary.apply({}, x, y, t, method=boundary.trace)
    )
)
rng = np.random.default_rng(20261010)
c = jnp.asarray(rng.normal(size=1024))
constructor = OEField2D(**common, coefficients=c)
field = jax.jit(jax.vmap(lambda x, y, t: constructor.apply({}, x, y, t)))
comparisons = []
for side, pts in zip(("left", "right", "bottom", "top"), coords):
    rows = np.asarray(trace(*pts))
    values = np.asarray(field(*pts)).ravel()
    expected = rows @ np.asarray(c)
    delta = float(np.max(np.abs(values - expected)))
    assert delta < 1e-11
    comparisons.append(
        dict(
            side=side,
            samples=len(values),
            row_shape=list(rows.shape),
            max_abs_difference=delta,
        )
    )
# Verify reference derivative ghost rules on a small grid, independent of OE RF code.
p = ROOT / "src/oeaprfm/reference/parity_reference.py"
tree = ast.parse(p.read_text())
solver = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "solve_gmres_2d"
)
functions = [
    n
    for n in solver.body
    if isinstance(n, ast.FunctionDef)
    and n.name in ("derivative_x", "derivative_y")
]
angles = np.array([0.2, 1.0, 2.0, 2.8, 3.4, 4.0, 5.0, 5.8])
nx, ny = 5, 6
x = (np.arange(nx) + 0.5) / nx
y = (np.arange(ny) + 0.5) / ny
scope = dict(
    np=np,
    x=x,
    y=y,
    vx=np.cos(angles),
    vy=np.sin(angles),
    dx=1 / nx,
    dy=1 / ny,
    boundary=config.model.bdy_cond,
)
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(p), "exec"), scope
)
f = rng.normal(size=(nx, ny, len(angles)))
dx = scope["derivative_x"](f)
dy = scope["derivative_y"](f)
for k, t in enumerate(angles):
    if np.cos(t) > 0:
        ghost = f[1, :, k] - 2 / nx * dx[0, :, k]
        value = (ghost + f[0, :, k]) / 2
        target = config.model.bdy_cond.f_l(y)
    else:
        ghost = 2 / nx * dx[-1, :, k] + f[-2, :, k]
        value = (ghost + f[-1, :, k]) / 2
        target = config.model.bdy_cond.f_r(y)
    np.testing.assert_allclose(value, target, atol=1e-12)
    if np.sin(t) > 0:
        ghost = f[:, 1, k] - 2 / ny * dy[:, 0, k]
        value = (ghost + f[:, 0, k]) / 2
        target = config.model.bdy_cond.f_b(x)
    else:
        ghost = 2 / ny * dy[:, -1, k] + f[:, -2, k]
        value = (ghost + f[:, -1, k]) / 2
        target = config.model.bdy_cond.f_t(x)
    np.testing.assert_allclose(value, target, atol=1e-12)
with (ARCH / "results/e5_mixed/physical_metrics.csv").open() as f:
    e5 = list(csv.DictReader(f))
failed = [
    dict(
        method=r["method"],
        seed=int(r["seed"]),
        angular_budget=int(r["Nang_positive"]),
        ratio=float(r["main_grid_reference_ratio_Ef"]),
    )
    for r in e5
    if r["main_grid_reference_pass_Ef"] == "False"
]
assert len(e5) == 24
result = dict(
    p5=dict(
        historical_source_hashes_verified_for_cutoffs=[
            r["rcond"] for r in records
        ],
        historical_runner_sha256=sha(oldrunner),
        historical_boundary_sha256=sha(oldbdy),
        historical_trace_ast_identical_to_current=True,
        actual_boundary_call="trace(x,y,physical_angle); one scalar row per inward physical velocity; no RHS duplication",
        misleading_legacy_metadata="independent_pair_traces_v2",
        residual_blocks=counts,
        total_rows=sum(counts.values()),
        hypothetical_double_row_total=5 * ni + 2 * counts["boundary"],
        boundary_constructor_checks=comparisons,
        reference_inflow_ghost_rules="passed for all four sides and eight directions",
        reference_solver_sha256=sha(p),
        conclusion="The archived P5 sweep uses physical inflow trace assembly. Prior classification as an unverified paired-boundary solve was incorrect. No boundary-protocol rerun is required.",
    ),
    e5=dict(
        runs=24,
        criterion="reference refinement discrepancy < 0.1 * discrepancy of method from archived reference, on main evaluation grid",
        main_grid_Ef_failed=len(failed),
        failures=failed,
        scope="Budget-dependence diagnostic relative to archived transport reference; excluded from main accuracy claims and equal-accuracy speedup claims. Failure is insufficient evidence, not proof the reference is wrong.",
    ),
)
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
