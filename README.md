# OE-APRFM

Research code for **Mitigating Ill-Conditioning in Asymptotic Preserving Random
Feature Methods for Multiscale Radiative Transfer Equations via Parity
Embedding**.

**Yan Shen** · University of Science and Technology of China

Contact: sheny2607@mail.ustc.edu.cn

OE-APRFM uses even/odd parity features, angular projection and rescaled
transport residuals to build a linear least-squares problem. The repository
contains steady 1D/2D solvers, manufactured and transport benchmarks,
micro–macro and neural-network comparisons, and a 1D time-dependent extension.

## Installation

Run commands from the repository root. Python **3.10** is the tested version;
the default setup uses CPU execution and double precision.

```bash
git clone https://github.com/ShenY07/OE-APRFM.git
cd OE-APRFM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

For tests, install `requirements-dev.txt`. The optional OE-APNN comparison
requires `requirements-apnn.txt` (PyTorch); it is not required by OE-APRFM.

## Quick start

Run the 1D manufactured problem with an exact reference:

```bash
python experiments/run_p1_oe_aprfm.py --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/quickstart
```

The runner writes a JSON record with errors, dimensions, solver diagnostics
and timing, plus an NPZ file with the numerical and exact fields.

Run the periodic time-dependent problem with implicit backward Euler:

```bash
python experiments/run_e6_periodic.py --epsilon 1 0.001 --seed 11 \
  --dt 0.002 --batch quickstart
python experiments/reference_e6.py
```

E6 saves coefficients and trajectories under `results/e6_periodic/quickstart/`.
The reference script evaluates both continuous-time Fourier evolution and a
same-step backward-Euler reference. Their errors measure different things;
see [numerical methods and limitations](docs/numerical_methods.md).

## Experiments and comparisons

| Problem / diagnostic | Entry point |
| --- | --- |
| P1: 1D manufactured solution; P6: angular-dependent manufactured solution | `experiments/run_p1_oe_aprfm.py --problem p1` / `--problem p6` |
| P2: heterogeneous 1D transport | `experiments/run_p2_oe_aprfm.py` |
| P3: 2D manufactured solution; P4: circular hole; P5: heterogeneous boundary-driven transport | `experiments/run_p3_oe_aprfm.py --problem p3` / `p4` / `p5` |
| E1: fixed-space minimum-gain diagnostic | `experiments/diagnose_oe_gain.py` |
| E3: steady slab diffusion limit | `experiments/run_e3_slab.py` |
| E5: mixed-scale slab | `experiments/run_e5_mixed.py` |
| E6: periodic transient transport | `experiments/run_e6_periodic.py` |
| Matched micro–macro random-feature comparisons | `experiments/run_mm_aprfm_1d.py`, `experiments/run_mm_aprfm_2d.py`, `experiments/run_e6_mm.py` |
| Direct RFM and deterministic transport | `experiments/run_rfm_1d.py`, `experiments/run_oe_si_dsa.py` |
| OE-APNN | `baselines/oe_apnn/train.py` |

P labels identify problem configurations; E labels identify experiment groups.
Full commands, reference-data prerequisites and the prescribed 20-trajectory
E6 study are in [the reproduction guide](docs/reproduction.md).

## Repository layout

```text
src/oeaprfm/         Numerical library
  problems/         Problem definitions and experiment defaults
  constraints/      Transport and parity residuals
  features/         Random features, sampling, and reconstruction
  geometry/         Spatial and angular meshes
  solvers/          Dense least squares and streaming QR
  reference/        Deterministic transport reference solvers
  utils/            Quadrature, coordinates, and timing
experiments/        Numerical experiment and comparison entry points
scripts/            Postprocessing, paper collection, and validation tools
tests/              Regression and consistency tests
baselines/          Comparison implementations and third-party originals
results/            Generated runs and selected experiment records
docs/               Reproduction guide and numerical assumptions
archive/            Historical source and original publication snapshots
```

Project Python code uses four-space indentation and 79-column Black formatting,
following `src/oeaprfm/constraints/oe_constraints_2d.py` (formerly
`continuous_2d_odd_even.py`). Function, class, and module names are at most
20 characters. Names distinguish the formulation, role, and dimension:
`OEInterior2D` assembles odd-even interior constraints, `OEBoundary2D` handles
inflow boundaries, and `OEField2D` reconstructs the field. `HalfOE` denotes
unprojected half-range features; `QuadOE` aliases distinguish four-component
implementations. Comments explain numerical choices, conventions, and shapes.

Imports and script paths use the new names listed in
[`scripts/name_changes.json`](scripts/name_changes.json) and
[`scripts/layout_changes.json`](scripts/layout_changes.json). Update external
callers using that mapping. Archived source and recorded hashes retain their
original names; source audits translate names in memory before comparing
executable syntax. Third-party originals and historical archives keep their
original style.

This organization follows the separation of library and experiments used in
[the JSC manifold-MOR companion repository](https://github.com/pbuchfink/manifold-mor-wave)
and the source/example/test/documentation separation in
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl). It is a repository
convention, not a universal journal submission requirement.

## Manuscript results (10 October 2026)

Manuscript assets under `paper/` are not distributed in this repository.
That directory is reserved for optional local files and is ignored by Git.
Experiment records are described in [the result inventory](results/README.md).

The 1D unprojected control now extends positive-half-range features consistently
with its inflow rows. Both controls preserve parity; this is a comparison of
basis constructions. The 2D runner uses physical-direction inflow traces and
records the corrected boundary-implementation label. The P3/P4 rerun launcher
is `scripts/refresh_2d_tables.py`. Unused candidate constraint
modules and defective historical reproduction entry points are excluded from
the production source tree.

The [angular-domain comparison](results/angular_domain_comparison/README.md)
adds twelve paired runs with the same OE trial space and weighted objective.
It measures the cost of removing symmetry-redundant residual rows at preserved
accuracy, rather than a search for minimum cost at a prescribed tolerance.
Per-seed records, configuration hashes and the aggregate table are included;
raw matrices and fields can be regenerated with the campaign launcher.

OE-APNN now enforces even/odd parity in its network outputs; old checkpoints
are not results for this revised protocol. See [the baseline guide](baselines/oe_apnn/README.md)
for the retraining commands. Mixed-scale transport differences remain relative
to an archived reference and do not establish fine accuracy rankings. CPU
archive timings, shared-host CPU reruns, and concurrent GPU training do not
establish controlled speedup factors.

Large raw fields, model checkpoints, historical audit inputs, and exploratory
studies are not included. This release is
not a self-contained archive of every historical experiment. Run the documented
experiments to generate new data; a successful quick start is not an accuracy
or speedup claim.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests --ignore=tests/test_apnn.py
```

To include the optional neural baseline:

```bash
python -m pip install -r requirements-apnn.txt
python -m pytest -q tests
```

The release checks are recorded in [validation notes](docs/validation.md).
Core CPU tests and formatting are also configured in GitHub Actions.

## Patent disclosure and licensing status

The authors declare that a patent application related to the methodology
presented in this work has been filed and published under publication number
**CN122263556A**. The patent application is currently pending examination.

This disclosure records the status supplied by the authors for this release;
it should not be read as a continuously updated patent-status record.

No open-source license is granted in this release. Public availability of the
source code is not itself an open-source license. For copyright or patent
licensing inquiries, contact the author above. Third-party attribution is
recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); no new license is
assigned to those components by this release.
