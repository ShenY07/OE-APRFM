# OE-APNN baseline

This directory implements the neural-network counterpart of OE-APRFM. Both
the 1D and 2D solvers have exactly two unknowns: the even parity `r` and the
scaled odd parity `j`, with

```text
f(x,v) = r(x,v) + epsilon j(x,v),
r(x,-v) = r(x,v),  j(x,-v) = -j(x,v).
```

The three interior residuals use the same order and scaling as OE-APRFM:

```text
<v dot grad j> + sigma_a <r>                         = <q_even>,
epsilon^2 (v dot grad j - <v dot grad j>)
  + (sigma_s + epsilon^2 sigma_a) (r - <r>)
                                                      = epsilon^2(q_even-<q_even>),
(sigma_s + epsilon^2 sigma_a) j + v dot grad r       = epsilon q_odd.
```

Raw networks R and J are projected at every solver call:
`r(x,v)=(R(x,v)+R(x,-v))/2`, `j(x,v)=(J(x,v)-J(x,-v))/2`.
In 2D both velocity components are reversed simultaneously. Interior
derivatives, quadrature, inflow and full-angle evaluation all use these
projected fields. This is protocol `oe-apnn-p1-p5-v2-hard-parity`; checkpoints
from v1 must be retrained even though their state-dict shapes still match.

The networks are trained on representative velocities: `v in [0,1]`
in 1D and the upper semicircle obtained from `theta in [0,pi/2]` and
`pi-theta` in 2D. Boundary and evaluation values use the trained network
outputs at the requested direction, `f=r+epsilon*j`.  Model keys and input
scaling are compatible with the three-seed archive.
P5 in the archived protocol is the square-hole manufactured problem with exact
solution `exp(-x-y)`.

Small runs:

```bash
python baselines/oe-apnn/train.py --problem p1 --steps 1 \
  --interior-samples 16 --boundary-samples 16 --quadrature-points 4
python baselines/oe-apnn/train.py --problem p3 --steps 1 \
  --interior-samples 8 --boundary-samples 8 --quadrature-points 4
```

The locked protocol uses float64, seeds 7/11/17, 20,000 Adam steps, unit loss
weights, 4,096 interior phase samples, 1,024 stratified inflow samples, and a
fixed independent physical validation batch every 200 steps. Each run saves
the lowest-validation-loss `r.pt` and `j.pt`, `config.json`, `history.npz`,
`metrics.json`, and (unless skipped) `solution.npz`.

Default output layout:
`results/baselines/oe_apnn/P{k}/eps_{1e0|1e-3}/seed_{seed}`.

The hard-parity retraining campaign uses a separate output directory and
the same 4096 interior / 1024 boundary phase samples for every budget:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 python3 scripts/generate_reference.py --problem p2 --epsilon 1 --level B
PYTHONPATH=src JAX_ENABLE_X64=1 python3 scripts/generate_reference.py --problem p2 --epsilon 0.001 --level B
python3 scripts/retrain_oe_apnn.py --devices 1 2
```

Each GPU runs the three seeds of one configuration concurrently. The campaign
contains 30 baseline runs (P1--P5, two epsilon values, 20000 steps) and 18 cost
runs (P1/P3, epsilon 0.001, 500/2000/5000 steps). CUDA is required; no CPU
fallback is allowed. Completed runs with a matching source manifest can be
reused when restarting the runner; interrupted runs restart from initialization.
`status.json` records process IDs and group status. Source snapshots, launch
commands, checkpoints and per-run logs are retained under
`results/oe_apnn_parity_20261010`.

After each three-seed group, `scripts/report_oe_apnn_retrain.py` updates
`summary.csv`, `seedwise.csv`, `REPORT.md`, PDF/PNG loss and field figures,
and the OE-APNN entries in `results/table.md` and `results/figure.md` when
those manuscript files exist. Original manuscript files are backed up first.
Only complete three-seed groups enter summary statistics. The new phase-sample
budget is 5120; this is not the scalar residual count. CUDA timings include
concurrent resource sharing and cannot establish a same-hardware speedup
against historical CPU timings for other methods.
