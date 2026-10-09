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

The networks are trained directly on representative velocities: `v in [0,1]`
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
