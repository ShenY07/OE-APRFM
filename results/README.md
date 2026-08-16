# Current numerical results

This directory contains only the latest accepted experiment artifacts.

- `consistency/p2`: five-seed fixed-parameter P2 results for epsilon 1 and 1e-3.
- `quadrant/final`: current P3-P5 OE-APRFM results using the independent angular domain `[0, pi/2]`.
- `domain_decomposition/1d_fixed_local`: current 1-D decomposition study with 64 features per patch.
- `quadrant/domain_decomposition_fixed_local`: current 2-D decomposition study with 64 features per patch.
- `figures/latest`: current publication figures. Reference, OE-RFM solution,
  and absolute error are separate title-free files at 320 dpi.
- `tables`: current summary tables and analysis.
- `references`: accepted exact/P2 parity references needed by the current runs.
- `tables/external_comparison.md`: unified external comparison schema for
  OE-APRFM, MM-APRFM, RFM, OE-APNN, and traditional solvers.

P2 files carrying `level_A` and `level_B` are internal reference-grid checks.
They are not independent manuscript experiments; the accepted fine-grid field
is used only as the numerical reference for method errors.

Obsolete half-circle 2-D results, tuning trials, preliminary checks, scheduler logs,
duplicate tables, and superseded figures were permanently removed on
2026-08-13.

Figure-name abbreviations: `e0` means epsilon=1, `e3` means epsilon=1e-3,
`ref`/`sol`/`err` mean reference/solution/absolute error, `rho2d` is a
density plane, `f3d` is a phase-space surface, `th45` is theta=pi/4, and
`dd` denotes the domain-decomposition experiment.
