# P6 angular manufactured: OE vs MM

f=1+0.25 sin(2 pi x)+epsilon v 0.25 cos(pi x). Both methods use 128 total coefficients, one spatial/angular patch, tanh scale 1, rcond=1e-12, seeds 11/23/37, CPU float64. MM allocates 64 features to rho and 64 to g; OE allocates 64 to r and 64 to j.

This is a coefficient-budget comparison, not identical sampling: OE uses 2944 residual rows, MM 3008. Both evaluate the same 257x128 grid. f/rho errors follow the existing unweighted convention; flux/g use physical spatial trapezoidal and normalized angular trapezoidal weights. Different training angular quadratures and feature spaces are retained. No matched-time or speedup claim is made: OE is archived, MM newly computed.

Source conversion was derived from epsilon*v*f_x=rho-f+epsilon^2*q. With a=.25 cos(pi x), MM RHS is [a_prime/3, v*(rho_prime+a)+epsilon*(v^2-1/3)*a_prime, 0]. Both inflow boundaries use exact f at the physical signed velocities. P1/P2 source paths remain unchanged.

| epsilon | method | E_f | E_rho | relative scaled flux | relative g |
|---|---|---|---|---|---|
| 1 | OE-APRFM | 2.17292e-05 | 5.89530e-06 | 6.27632e-05 | 2.07517e-04 |
| 1 | MM-APRFM | 1.57914e-05 | 3.42765e-06 | 6.82384e-06 | 1.40625e-04 |
| 0.001 | OE-APRFM | 1.63840e-05 | 1.59314e-05 | 1.62021e-04 | 5.03673e-02 |
| 0.001 | MM-APRFM | 4.81956e-07 | 4.82016e-07 | 3.34212e-06 | 8.44222e-05 |
| 1e-06 | OE-APRFM | 1.63647e-05 | 1.59055e-05 | 1.60906e-04 | 5.03940e+01 |
| 1e-06 | MM-APRFM | 4.85325e-07 | 4.85326e-07 | 3.47425e-06 | 8.52192e-05 |

All seedwise errors and ranges are retained. This P6 solution is angular-dependent, but its even part is exactly isotropic and its microscopic field is linear in v; it is not a general angular-anisotropy benchmark. Conclusions are restricted to this case and frozen equal-allocation budget.

Reproduction: run scripts/run_mm_aprfm_1d.py with --problem p6 --features 64 --epsilon {1,1e-3,1e-6} --seed {11,23,37} --output-dir results/p6_mm_comparison/mm, then python3 scripts/summarize_p6_mm_comparison.py. Original frozen tables have not been overwritten.
