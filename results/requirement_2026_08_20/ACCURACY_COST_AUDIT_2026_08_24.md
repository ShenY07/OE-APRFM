# Accuracy--Cost audit (2026-08-24)

## 1D deterministic OE-$S_N$--Krylov

The previously reported sequence was not comparable: the four JSON files
mixed SI/DSA and Krylov stopping criteria, used different iteration limits,
and were sorted by runtime rather than resolution.  In addition, the old 1D
angular rule was an equally spaced trapezoidal rule despite the $S_N$ label.

All four levels were rerun with:

- the same matrix-free Krylov solver;
- Gauss--Legendre angular quadrature;
- relative tolerance $10^{-12}$;
- the same exact manufactured reference and direct on-grid error evaluation;
- a recorded true relative residual below $10^{-12}$.

| $N_x\times N_{\rm ang}$ | iterations | true residual | $E_f$ | $E_\rho$ | time (s) |
|---:|---:|---:|---:|---:|---:|
| $32\times16$ | 62 | $5.71\times10^{-15}$ | $1.95\times10^{-11}$ | $1.95\times10^{-11}$ | 0.08 |
| $64\times32$ | 115 | $8.62\times10^{-14}$ | $1.51\times10^{-10}$ | $1.51\times10^{-10}$ | 0.19 |
| $128\times64$ | 216 | $1.32\times10^{-14}$ | $9.80\times10^{-12}$ | $9.80\times10^{-12}$ | 0.94 |
| $256\times128$ | 401 | $9.86\times10^{-14}$ | $1.44\times10^{-11}$ | $1.44\times10^{-11}$ | 17.65 |

The exact solution is $f=1-x$, independent of angle.  Diamond difference
represents this linear solution exactly, so all four errors are at the
algebraic/roundoff floor.  Their residual non-monotonicity must not be
interpreted as spatial or angular convergence.

## 2D OE-APRFM

The earlier efficiency sweep was confirmed to use the legacy two-component
space.  It is now excluded from the table builder.  The replacement sweep uses
the final normalized four-component $(j_1,r_1,j_2,r_2)$ space, three seeds
(11, 23, 37), $16\times16\times16$ collocation, and matched total coefficient
budgets.  Per-component budgets $J=32,64,128$ correspond to
$N_{\rm coef}=128,256,512$.

| $J$ | $N_{\rm coef}$ | median $E_f$ | median $E_\rho$ | median compute time (s) |
|---:|---:|---:|---:|---:|
| 32 | 128 | $3.48\times10^{-2}$ | $1.94\times10^{-2}$ | 39.67 |
| 64 | 256 | $1.03\times10^{-2}$ | $5.81\times10^{-3}$ | 43.34 |
| 128 | 512 | $1.16\times10^{-3}$ | $4.86\times10^{-4}$ | 59.06 |

During this audit, the angular patch window was found to lack angular
partition-of-unity normalization.  The implementation now divides each angular
window by the sum over all angular windows.  Consequently, one angular patch
reduces exactly to the original global four-component space, while multiple
patches remain a genuine overlapping angular decomposition.

The manuscript label is `tab:accuracy-cost`; it is stored locally as
`tables_frozen/table_7.tex` even if it appears as Table 9 after manuscript
assembly.
