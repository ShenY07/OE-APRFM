# P7 transient pilot: measured results

Exploratory NumPy space-time tanh random-feature implementation of the projected/rescaled odd-even equations. This is an extension prototype, not the frozen steady solver or a reproduction of the authors' APNN implementation.

Case: zero initial data, left inflow 1, right inflow 0, x in [0,1], t in [0,0.1]. Source: https://arxiv.org/html/2306.15381v4 (Problem 1, Case I); epsilon=1 and 0.01 are extensions.

J=128 per OE field; 256 coefficients for every method; seeds 11,23,37. Same physical interior/boundary/initial samples. OE AP has an additional 256 macro rows (5376 vs 5120); this is a coefficient-budget-controlled structural comparison, not identical residual cost. Direct RFM uses an unconstrained full-angular space and therefore also changes approximation structure.

Reference: backward Euler and staggered parity finite volumes. Coarse 128 cells/8 positive angles/400 steps; fine 256/16/800. Angular nodes are Gauss–Legendre on [0,1]. Refinement compares both density and full kinetic fields on the coarse grid; it is an empirical discrepancy check, not a rigorous error bound.

| epsilon | method | t | E_f median | E_rho median [min,max] | reference check |
|---|---|---|---|---|---|
| 1 | oe_ap | 0.05 | 1.234 | 1.536 [1.345,1.719] | True |
| 1 | oe_ap | 0.1 | 0.7854 | 0.706 [0.6904,1.047] | False |
| 1 | oe_unprojected | 0.05 | 1.125 | 1.273 [1.14,1.367] | True |
| 1 | oe_unprojected | 0.1 | 0.7792 | 0.6441 [0.596,0.8455] | False |
| 1 | direct | 0.05 | 0.9868 | 0.9391 [0.9388,1.125] | True |
| 1 | direct | 0.1 | 0.7154 | 0.5343 [0.514,0.5847] | False |
| 0.01 | oe_ap | 0.05 | 0.0396 | 0.03592 [0.0304,0.03936] | True |
| 0.01 | oe_ap | 0.1 | 0.02841 | 0.01566 [0.01563,0.01928] | True |
| 0.01 | oe_unprojected | 0.05 | 1.265 | 1.264 [0.9023,1.977] | True |
| 0.01 | oe_unprojected | 0.1 | 1.829 | 1.829 [0.7718,2.594] | True |
| 0.01 | direct | 0.05 | 1.966 | 1.966 [1.868,2.245] | True |
| 0.01 | direct | 0.1 | 2.728 | 2.726 [1.031,4.337] | True |
| 0.001 | oe_ap | 0.05 | 0.03956 | 0.0384 [0.03033,0.04272] | True |
| 0.001 | oe_ap | 0.1 | 0.03071 | 0.0183 [0.01731,0.0255] | True |
| 0.001 | oe_unprojected | 0.05 | 2.65 | 2.65 [2.273,16.26] | True |
| 0.001 | oe_unprojected | 0.1 | 5.336 | 5.336 [2.785,20.39] | True |
| 0.001 | direct | 0.05 | 15.38 | 15.38 [7.074,60.49] | True |
| 0.001 | direct | 0.1 | 37.62 | 37.62 [12.59,70.82] | True |

Interpretation: projected/rescaled OE retains percent-level accuracy in the two small-epsilon regimes; the two controls deteriorate substantially. The kinetic regime epsilon=1 fails accuracy requirements for all three global spaces, and its kinetic reference refinement is insufficient for an OE error report under the 10% rule. It must not be advertised as a successful all-regime benchmark.

Timing is exploratory: one solve per seed, BLAS threads fixed to one, a small OE library warm-up only, no three-repeat same-seed timing protocol. No speedup claim against the frozen S3 measurements, a deterministic solver, or published APNN is supported.

Validation: analytical feature derivatives checked by finite differences; exact parity checked; small-epsilon reference checked against an independent heat-equation Fourier series; discrete mass balance checked. Initial and inflow conditions are soft RF constraints and the incompatible space-time corner is excluded.

Next publication gate: improve/resolve kinetic-front and initial-corner approximation, independently refine space/time/angle, validate residuals on held-out points, and reproduce external APNN and deterministic baseline under one timing protocol. No current advantage over MM-APNN is established.

Reproduce: `python3 test/test_p7_transient_pilot.py`; `python3 scripts/run_p7_transient_pilot.py --features 128 --nx 128 --nt 400 --output results/p7_transient_pilot/three_seed_J128`; `python3 scripts/summarize_p7_transient_pilot.py`.
