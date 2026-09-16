# Boundary-driven P5 certified status (2026-08-23)

## Frozen problem and OE-APRFM configuration

- $D=(0,1)^2$, $\sigma_s=1+x_2$, $\sigma_a=0.1$, $Q=0$, and diffuse inflow $f_{\rm in}=1+0.2x_2$.
- Corrected four-component odd-even representation and quadrant reconstruction.
- Identical settings for $\varepsilon=1$ and $10^{-3}$: partition $1\times1\times2$, $J=128$ per component and patch, $N_{\rm coef}=1024$, collocation $32\times32\times16$, $N_{\rm row}=58096$, scale $0.25$, rcond $10^{-6}$, boundary weight 10, unit-L2 row scaling.
- Angular errors use a 64-point periodic midpoint rule, avoiding finite quadrature weight at the four measure-zero seams of the quadrant-wise reconstruction.

## Three-seed OE-APRFM results against level-C reference

| $\varepsilon$ | median $E_f$ | median $E_\rho$ | median corr$(\rho_h,\rho_{\rm ref})$ | min $f_h$ | min $\rho_h$ |
|---:|---:|---:|---:|---:|---:|
| $1$ | $4.2343\times10^{-2}$ | $4.7643\times10^{-3}$ | 0.999498 | 0.7567 | 0.9845 |
| $10^{-3}$ | $3.6239\times10^{-2}$ | $4.6022\times10^{-3}$ | 0.999088 | 0.8246 | 0.9959 |

Seeds are 11, 23, and 37. Both regimes pass $E_f<0.05$, $E_\rho<0.05$, correlation $>0.995$, and positivity checks.

## Deterministic A/B/C certification

| $\varepsilon$ | level | grid | iterations | final residual | solve time (s) | $\delta_f$ from previous | $\delta_\rho$ from previous |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| $1$ | A | $32\times32\times8$ | 244 | $9.88\times10^{-10}$ | 3.38 | -- | -- |
| | B | $64\times64\times16$ | 536 | $9.86\times10^{-10}$ | 260.04 | $1.65\times10^{-3}$ | $1.58\times10^{-4}$ |
| | C | $96\times96\times24$ | 776 | $9.90\times10^{-10}$ | 76.81 | $6.32\times10^{-4}$ | $4.39\times10^{-5}$ |
| $10^{-3}$ | A | $32\times32\times8$ | 448 | $9.99\times10^{-10}$ | 11.83 | -- | -- |
| | B | $64\times64\times16$ | 3470 | $9.98\times10^{-10}$ | 540.58 | $4.93\times10^{-5}$ | $4.93\times10^{-5}$ |
| | C | $96\times96\times24$ | 7880 | $1.00\times10^{-9}$ | 906.29 | $1.09\times10^{-5}$ | $1.09\times10^{-5}$ |

For both fields and regimes, $\delta^{BC}<0.1E^{\rm OE}$; therefore both $E_f$ and $E_\rho$ are certified for reporting.
