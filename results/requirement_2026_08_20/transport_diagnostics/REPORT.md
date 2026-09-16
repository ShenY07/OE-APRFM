# Archived transport diagnostics

These are postprocessed saved fields, not new solves. P1/P3 in S3 have identically zero exact flux and g=(f-rho)/epsilon; only absolute L2 errors are meaningful. Their relative errors are blank, not zero.

Flux is <v f>/epsilon (a two-component vector in 2D). All angular weights are normalized. rho is recomputed from f with the same rule. Spatial L2 norms use physical integration weights, not raw unweighted array norms. P6 uses nonzero manufactured truth and permits relative errors.

Learned methods use their archived evaluation grids (P1 257x128; P3 65x65x64); deterministic methods retain native cell/angle grids with saved Gauss weights. No interpolation is introduced. Tiny absolute errors near roundoff should not be ranked; a publication comparison requires common evaluation and quadrature-refinement checks.

All S3 metadata are matched by problem/method/budget/seed; legacy two-component OE P3 runs are excluded. Source paths and SHA256 are recorded in seedwise.csv. No P6 external-method result is inferred.

Even/odd g-error components are also reported. Orthogonality is checked seedwise: ||error_g||^2=||error_g_even||^2+||error_g_odd||^2. Stable odd-flux accuracy does not imply stable accuracy of the entire scaled microscopic field.

| Problem | epsilon | Method | Budget | abs flux L2 | abs g L2 | relative flux | relative g | even g error | odd g error |
|---|---|---|---|---|---|---|---|---|---|
| p1 | 0.001 | MM-APRFM | J=128 | 3.4821e-11 | 6.0313e-11 | N/A | N/A | 6.3182e-14 | 6.0313e-11 |
| p1 | 0.001 | MM-APRFM | J=32 | 2.2846e-11 | 3.9568e-11 | N/A | N/A | 5.5886e-14 | 3.9568e-11 |
| p1 | 0.001 | MM-APRFM | J=64 | 3.4953e-11 | 6.0530e-11 | N/A | N/A | 4.4854e-14 | 6.0530e-11 |
| p1 | 0.001 | OE-$S_N$-Krylov | 128x64 | 6.0249e-12 | 1.0441e-11 | N/A | N/A | 1.5604e-13 | 1.0440e-11 |
| p1 | 0.001 | OE-$S_N$-Krylov | 256x128 | 9.6773e-12 | 1.6783e-11 | N/A | N/A | 3.9733e-13 | 1.6778e-11 |
| p1 | 0.001 | OE-$S_N$-Krylov | 32x16 | 1.2135e-11 | 2.1025e-11 | N/A | N/A | 1.8776e-13 | 2.1024e-11 |
| p1 | 0.001 | OE-$S_N$-Krylov | 64x32 | 9.4807e-11 | 1.6424e-10 | N/A | N/A | 1.2733e-12 | 1.6423e-10 |
| p1 | 0.001 | OE-APNN | 2000 steps | 3.2067e+00 | 6.6762e+00 | N/A | N/A | 3.1812e+00 | 5.8695e+00 |
| p1 | 0.001 | OE-APNN | 500 steps | 4.0284e+00 | 9.5325e+00 | N/A | N/A | 5.4549e+00 | 7.8175e+00 |
| p1 | 0.001 | OE-APNN | 5000 steps | 1.2376e+00 | 2.3841e+00 | N/A | N/A | 7.8008e-01 | 2.2528e+00 |
| p1 | 0.001 | OE-APRFM | J=128 | 1.6128e-12 | 8.6082e-10 | N/A | N/A | 8.6062e-10 | 1.8429e-11 |
| p1 | 0.001 | OE-APRFM | J=32 | 7.8029e-06 | 2.7354e-02 | N/A | N/A | 2.7354e-02 | 1.4096e-04 |
| p1 | 0.001 | OE-APRFM | J=64 | 1.1114e-09 | 5.8766e-06 | N/A | N/A | 5.8745e-06 | 1.5566e-07 |
| p3 | 0.001 | MM-APRFM | J=128 | 2.2240e-07 | 3.1479e-07 | N/A | N/A | 1.0393e-08 | 3.1462e-07 |
| p3 | 0.001 | MM-APRFM | J=256 | 2.5354e-07 | 3.5863e-07 | N/A | N/A | 3.8410e-09 | 3.5858e-07 |
| p3 | 0.001 | MM-APRFM | J=64 | 2.9398e-05 | 4.3959e-05 | N/A | N/A | 1.1712e-05 | 4.2370e-05 |
| p3 | 0.001 | OE-$S_N$-Krylov | 12x12x16 | 5.6601e-03 | 8.0200e-03 | N/A | N/A | 4.9666e-04 | 8.0046e-03 |
| p3 | 0.001 | OE-$S_N$-Krylov | 16x16x16 | 2.7407e-03 | 3.8814e-03 | N/A | N/A | 2.0744e-04 | 3.8759e-03 |
| p3 | 0.001 | OE-$S_N$-Krylov | 24x24x16 | 1.0005e-03 | 1.4162e-03 | N/A | N/A | 6.0914e-05 | 1.4149e-03 |
| p3 | 0.001 | OE-$S_N$-Krylov | 32x32x16 | 5.0141e-04 | 7.0956e-04 | N/A | N/A | 2.5617e-05 | 7.0910e-04 |
| p3 | 0.001 | OE-APNN | 2000 steps | 7.1730e+01 | 1.4644e+02 | N/A | N/A | 8.1556e+01 | 1.1559e+02 |
| p3 | 0.001 | OE-APNN | 500 steps | 1.9115e+02 | 3.2339e+02 | N/A | N/A | 1.6062e+02 | 2.8068e+02 |
| p3 | 0.001 | OE-APNN | 5000 steps | 3.3297e+01 | 6.3086e+01 | N/A | N/A | 3.5746e+01 | 5.1981e+01 |
| p3 | 0.001 | OE-APRFM | J=128 | 4.9465e-03 | 3.8005e+00 | N/A | N/A | 3.8005e+00 | 1.5255e-02 |
| p3 | 0.001 | OE-APRFM | J=32 | 9.1920e-02 | 1.0473e+02 | N/A | N/A | 1.0473e+02 | 1.9898e-01 |
| p3 | 0.001 | OE-APRFM | J=64 | 3.9747e-02 | 3.0452e+01 | N/A | N/A | 3.0452e+01 | 1.0495e-01 |
| p6 | 1e-06 | OE-APRFM | J=64 | 9.4827e-06 | 5.1436e+00 | 1.6091e-04 | 5.0394e+01 | 5.1436e+00 | 1.0563e-04 |
| p6 | 0.001 | OE-APRFM | J=64 | 9.5484e-06 | 5.1409e-03 | 1.6202e-04 | 5.0367e-02 | 5.1398e-03 | 1.0463e-04 |
| p6 | 1.0 | OE-APRFM | J=64 | 3.6988e-06 | 2.1181e-05 | 6.2763e-05 | 2.0752e-04 | 1.7713e-05 | 1.1614e-05 |

Missing field archives: 0

Reproduce: `python3 scripts/build_transport_diagnostics.py`.
