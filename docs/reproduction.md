# Reproducing the experiments

Use the environment and exports in the root [README](../README.md). Run all
commands from the repository root. Steady RF runs use JAX float64; E6 uses
NumPy/SciPy float64. Default examples use one CPU BLAS thread. GPU runs require
a separate environment and should not be mixed into CPU timing comparisons.

## Problem labels and manuscript experiment groups

P1–P6 name configurations; E1–E6 name experiment groups. They are not two
interchangeable numbering schemes.

| Group | Purpose | Reproduction entry |
| --- | --- | --- |
| E1 | Fixed-space minimum gain | `diagnose_oe_minimum_gain.py` |
| E2 | Manufactured accuracy and parity ablation | P1/P3 runners; P1 `--variant` |
| E3 | Steady slab diffusion limit | `run_e3_slab.py` |
| E4 | P5 cutoff sensitivity | P3 runner with `--problem p5 --rcond ...` |
| E5 | Mixed-scale slab, OE/MM | `run_e5_mixed.py` |
| E6 | Periodic time-dependent transport | `run_e6_periodic.py` |

P2, P4 and P6 provide additional heterogeneous, geometry and angular tests.
The historical P7 inflow-driven prototype is not E6 and is not included in
this source release. Archived tuning/reporting scripts remain in Git history.

## Manufactured problems: no downloaded data

```bash
python scripts/run_p1_oe_aprfm.py --problem p1 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/p1
python scripts/run_p1_oe_aprfm.py --problem p6 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/p6
python scripts/run_p3_oe_aprfm.py --problem p3 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/p3
python scripts/run_p3_oe_aprfm.py --problem p4 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/p4
```

These are executable starting configurations, not a promise to regenerate
all manuscript tables at these small budgets. Use each runner's `--help` to
set partitions, collocation counts, feature counts and SVD cutoff. The P1
parity comparison uses `--variant full` and `--variant full_angular` at matched
budgets. `oe_original` is the unprojected parity-residual control. Record every
changed parameter rather than interpreting two differently configured runs
as a single-factor comparison.

For steady RF seed studies, repeat the same configuration with seeds
**11, 23, 37** and report the median and range. Parameter values and timing
components are saved in each JSON record. Optional OE-APNN has its own saved
training protocol; its seeds/budgets must not silently be treated as matched RF
settings.

## P2 and P5: generate references first

P2 uses an independently computed transport reference:

```bash
python scripts/generate_reference.py --problem p2 --epsilon 0.001 --level A
python scripts/generate_reference.py --problem p2 --epsilon 0.001 --level B
python scripts/run_p2_oe_aprfm.py --epsilon 0.001 --seed 11 \
  --reference-level B --output-dir results/p2
```

P5 is the current **boundary-driven** heterogeneous configuration. Its
reference directory must match the runner:

```bash
python scripts/generate_reference.py --problem p5 --epsilon 0.001 --level A
python scripts/generate_reference.py --problem p5 --epsilon 0.001 --level B
python scripts/run_p3_oe_aprfm.py --problem p5 --epsilon 0.001 --seed 11 \
  --rcond 1e-6 --output-dir results/p5
```

P5 additionally supports reference level C. P2 supports generator levels A/B.
The P5 `--p5-constant-inflow` option must be supplied to both the generator and
the RF runner for that variant. Refine the reference before interpreting a
method's error; convergence of the reference solver alone is insufficient.
P5 references and larger 2D systems can require substantial time and memory.

## E1, E3 and E5

```bash
python scripts/diagnose_oe_minimum_gain.py
python scripts/run_e3_slab.py
python scripts/check_e3_transport.py
python scripts/report_e3_slab.py
```

The E3 report requires all 18 default runs (six epsilon values and three
seeds). Its diffusion differences are distinct from transport-reference
errors. E1 is a finite-space diagnostic, not a proof of a continuous
stability lower bound.

For one E5 pair, first generate its experiment-specific references:

```bash
python scripts/run_e5_mixed.py --method references
python scripts/run_e5_mixed.py --method oe --seed 11 --angles 32
python scripts/run_e5_mixed.py --method mm --seed 11 --angles 32
```

`--method all` runs references followed by all 24 preset method/seed/angular
combinations. Inspect failure records and reference refinements; completion
alone does not establish accuracy or a speed advantage.

## Full E6 study (20 OE trajectories)

The equation is

```text
epsilon^2 f_t + epsilon v f_x = <f> - f,
x in (0,1), v in [-1,1], t in [0,0.2],
f(0,x,v) = 1 + 0.2 cos(2 pi x), with periodic spatial boundaries.
```

Each field uses 128 random features (256 coefficients total). Features are
drawn once per seed and reused in time; the initial right-hand side is
analytic. The main step is 0.002. The following batches must retain their
names because the summary distinguishes main, time, feature and quadrature
checks.

```bash
for seed in 11 23 37; do
  python scripts/run_e6_periodic.py --epsilon 1 0.1 0.01 0.001 \
    --seed "$seed" --dt 0.002 --batch factorized_check
done
for dt in 0.004 0.001; do
  python scripts/run_e6_periodic.py --epsilon 1 0.001 \
    --seed 11 --dt "$dt" --batch time_check
done
python scripts/run_e6_periodic.py --epsilon 1 0.001 --seed 11 \
  --dt 0.002 --features 256 --batch feature_check
python scripts/run_e6_periodic.py --epsilon 1 0.001 --seed 11 \
  --dt 0.002 --quadrature 128 --batch quadrature_check
python scripts/reference_e6.py
python scripts/finalize_e6.py
```

The report is `results/e6_periodic/REPORT.md`. CSVs record continuous-time and
same-step BE errors, reference acceptance, independent test-grid refinement,
feature/quadrature checks, mass drift, periodic residuals and sampled minima.
Quick-start trajectories use a separate batch and are excluded from the
20-run final report.

After the main OE study and reference comparison, run the six MM trajectories:

```bash
python scripts/run_e6_mm.py
```

The MM report is `results/e6_mm_comparison/REPORT.md`. This runner relies on
the OE reference files and the three main OE seeds; it is not a standalone
quick-start command. It does not reproduce the separate historical MM
test-grid-refinement audit.

## Reporting checklist

- Save the Git commit, Python/dependency versions, CPU/GPU model, thread
  counts and the complete run command alongside the generated data.
- Use independent evaluation points. Report both distribution and density
  errors; include current/flux when relevant. Do not divide by a zero true
  current at the initial time.
- Report failed/nonconverged runs and reference-limited errors. Reference
  refinement is an empirical check, not a rigorous error bound.
- Keep seeds, coefficient counts, residual row counts and discretization
  parameters visible. Equal coefficient budgets need not imply equal cost.
- Separate assembly/factorization, solve/advance and evaluation costs. Repeat
  timings on the same hardware and compare accuracy as well as runtime.
- Distinguish reproducible source code from the historical data archive.
  Large archived tables and plots are not shipped or automatically rebuilt
  by this release.
