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
export PYTHONPATH="$PWD/src:$PWD/scripts"
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
python scripts/run_p1_oe_aprfm.py --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/quickstart
```

The runner writes a JSON record with errors, dimensions, solver diagnostics
and timing, plus an NPZ file with the numerical and exact fields.

Run the periodic time-dependent problem with implicit backward Euler:

```bash
python scripts/run_e6_periodic.py --epsilon 1 0.001 --seed 11 \
  --dt 0.002 --batch quickstart
python scripts/reference_e6.py
```

E6 saves coefficients and trajectories under `results/e6_periodic/quickstart/`.
The reference script evaluates both continuous-time Fourier evolution and a
same-step backward-Euler reference. Their errors measure different things;
see [numerical methods and limitations](docs/numerical_methods.md).

## Experiments and comparisons

| Problem / diagnostic | Entry point |
| --- | --- |
| P1: 1D manufactured solution; P6: angular-dependent manufactured solution | `scripts/run_p1_oe_aprfm.py --problem p1` / `--problem p6` |
| P2: heterogeneous 1D transport | `scripts/run_p2_oe_aprfm.py` |
| P3: 2D manufactured solution; P4: circular hole; P5: heterogeneous boundary-driven transport | `scripts/run_p3_oe_aprfm.py --problem p3` / `p4` / `p5` |
| E1: fixed-space minimum-gain diagnostic | `scripts/diagnose_oe_minimum_gain.py` |
| E3: steady slab diffusion limit | `scripts/run_e3_slab.py` |
| E5: mixed-scale slab | `scripts/run_e5_mixed.py` |
| E6: periodic transient transport | `scripts/run_e6_periodic.py` |
| Matched micro–macro random-feature comparisons | `scripts/run_mm_aprfm_1d.py`, `scripts/run_mm_aprfm_2d.py`, `scripts/run_e6_mm.py` |
| Direct RFM and deterministic transport | `scripts/run_rfm_1d.py`, `scripts/run_oe_si_dsa.py` |
| OE-APNN | `baselines/oe-apnn/train.py` |

P labels identify problem configurations; E labels identify experiment groups.
Full commands, reference-data prerequisites and the prescribed 20-trajectory
E6 study are in [the reproduction guide](docs/reproduction.md).

## Repository layout

```text
src/configuration/   Problem definitions and experiment defaults
src/constraints/     Transport, parity and micro–macro residuals
src/modules/         Random features, partition of unity and assembly
src/solver/          Least squares and streaming QR
src/numerical/       Deterministic reference solvers
scripts/             Experiment runners, references and diagnostics
baselines/           Comparison implementations and attribution
test/                Regression and consistency tests
docs/                Reproduction guide and numerical assumptions
results/             Selected manuscript tables, figures and summary data; raw runs generated locally
```

## Manuscript results (10 October 2026)

The current [25 tables](results/table.md), [figure index](results/figure.md),
and [PDF bundle](results/JSC_all_figures_20261010.zip) are versioned alongside
selected summary data. Panels are exported separately as square vector PDFs;
E6 legends use one row for three entries and two rows for four entries.
See [the result inventory](results/README.md) for the included artifacts and
which historical inputs are not bundled.

The 1D unprojected control now extends positive-half-range features consistently
with its inflow rows. Both controls preserve parity; this is a comparison of
basis constructions. The 2D runner uses physical-direction inflow traces and
records the corrected boundary-implementation label. The P3/P4 rerun launcher
is `scripts/refresh_manuscript_2d_tables.py`. Unused candidate constraint
modules and defective historical reproduction entry points are excluded from
the production source tree.

OE-APNN now enforces even/odd parity in its network outputs; old checkpoints
are not results for this revised protocol. See [the baseline guide](baselines/oe-apnn/README.md)
for the retraining commands. Mixed-scale transport differences remain relative
to an archived reference and do not establish fine accuracy rankings. CPU
archive timings, shared-host CPU reruns, and concurrent GPU training do not
establish controlled speedup factors.

Large raw fields, model checkpoints, local audit scripts that require the
historical backup, and exploratory studies are not included. This release is
not a self-contained archive of every historical experiment. Run the documented
experiments to generate new data; a successful quick start is not an accuracy
or speedup claim.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q test --ignore=test/test_oe_apnn_baseline.py
```

To include the optional neural baseline:

```bash
python -m pip install -r requirements-apnn.txt
python -m pytest -q test
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
