# Domain decomposition in 1-D and 2-D OE-APRFM

The main experiment fixes 64 random features in every spatial patch. Thus increasing the patch count changes both the spatial feature distribution and the total model capacity. All cases use seed 11, scale 1, and rcond 1e-12. The 2-D angular base domain is `[0,pi/2]`.

## 1-D P1

| epsilon | patches | total features | E_f | condition number |
|---:|---:|---:|---:|---:|
| 1 | 1 | 64 | 5.2207e-7 | 7.8773e8 |
| 1 | 2 | 128 | 9.6769e-7 | 1.7767e9 |
| 1 | 4 | 256 | 7.4170e-5 | 1.2929e11 |
| 1e-3 | 1 | 64 | 3.7512e-7 | 1.1216e10 |
| 1e-3 | 2 | 128 | 1.3919e-6 | 3.9146e10 |
| 1e-3 | 4 | 256 | 2.9944e-4 | 1.4067e13 |

The exact solution is globally linear in x. One global patch already represents it nearly exactly. Extra patches add redundant local bases and worsen conditioning, so they do not improve accuracy.

## 2-D P3

| epsilon | patches | total features | E_f | E_rho | condition number |
|---:|---:|---:|---:|---:|---:|
| 1 | 1x1 | 64 | 7.0327e-2 | 5.3815e-2 | 4.5101e4 |
| 1 | 2x1 | 128 | 5.4046e-2 | 2.5717e-2 | 4.3618e4 |
| 1 | 2x2 | 256 | 5.0792e-2 | 3.4957e-2 | 8.4630e4 |
| 1e-3 | 1x1 | 64 | 3.0456e-2 | 2.7202e-2 | 1.0859e5 |
| 1e-3 | 2x1 | 128 | 4.3118e-2 | 4.2169e-2 | 1.6813e5 |
| 1e-3 | 2x2 | 256 | 3.3725e-2 | 3.3650e-2 | 3.0521e5 |

For epsilon=1, adding localized features reduces E_f from 7.03e-2 to 5.08e-2, while the best density error occurs at 2x1. For epsilon=1e-3, the one-patch model is best. Hence decomposition can help the more kinetic solution, but is not automatically beneficial in the smooth diffusion regime.

## Separating localization from capacity

The fixed-total-feature ablation uses 256 total features for every 2-D partition. In that experiment, 1x1 is much more accurate than 2x1 or 2x2. Therefore the improvement from 1x1/64 to 2x2/64-per-patch at epsilon=1 cannot be attributed purely to localization: increased total capacity contributes. At equal total capacity, this smooth P3 solution favors a global feature distribution. A stronger decomposition benefit should be expected for nonsmooth coefficients, localized layers, or obstacle geometries.
