# Controlled angular-domain reduction

Run from the repository root:

```
python3 scripts/run_angular_domain_campaign.py
OPENBLAS_NUM_THREADS=1 python3 scripts/report_angular_domain_comparison.py
```

The launcher executes 12 serial fresh-process runs (P1/P3, seeds 11/23/37,
reduced/expanded), CPU float64, one BLAS thread. It overwrites this campaign's
outputs when rerun. The current production operators, boundary traces and
reconstruction are shared. Total coefficients: 128; epsilon: 0.001;
rcond: 1e-8; no damping; row L2 normalization followed by column L2 equilibration
and dense gelsd in BOTH dimensions. This intentionally uses the same dense
solver for both controls, rather than the production 2D streaming QR.

1D nominal collocation 32x16 gives 30x16 interior points. The full orbit
adds their negative velocities. Macro/even residuals repeat and odd residuals
change sign with their RHS. There are 64 unchanged boundary rows; reduced
interior blocks contain 480 rows each, versus 960 each in the expanded run.
Assembly quadrature order is 8. Evaluation: 257x128, the production grid.

2D nominal collocation 8x8x8 gives 6x6x4 interior points. Each point carries
five parity equations (one macro, two even, two odd); the full-orbit control
evaluates the same local-coordinate blocks at all four reflected angular
locations. It folds each physical angle before evaluating BOTH the operator
and source, preserving the four-component trial space and its reconstruction.
There are 512 unchanged physical inflow rows. Reduced macro/even/odd counts
are 144/288/288, expanded counts 576/1152/1152. Assembly quadrature order is
32. Evaluation: 65x65x64, with periodic angular midpoints. Both problems use
their exact manufactured references.

Inverse square-root orbit multiplicities are applied to interior rows AFTER
row normalization. The tests check A^T A, A^T b, b^T b, effective ranks,
evaluation coordinates, and reconstructed fields for all six pairs.
This is a redundancy-removal experiment within the same OE space. It is not
a comparison to an independent full-angle direct-f method, nor evidence for
universal accuracy gains. The main-study budgets and existing parity ablation
are unchanged. Boundary rows are not duplicated, so total rows do not scale
exactly by two or four.

Assembly timings include first-call execution/compilation and exclude feature
setup; dense solve timings include column equilibration and SVD. They are
single timings per seed, summarized by medians on a shared host. The large
solve-stage reduction is not a corresponding end-to-end speedup. Per-seed
results, full evaluation fields, weighted systems, exact commands and source
hashes are retained in this directory; numerical outputs are not rounded until
rendering the table. summary.json contains all equivalence diagnostics.

## Published files

This directory publishes result.json for each of the twelve runs, manifest.json,
summary.json, table.tex and this protocol description. Raw system.npz and field
NPZ files and process logs remain local. To recompute the equivalence checks,
run the campaign first, then the report command. The published summary includes
the checks from the completed campaign; no rerun is needed to read those results.
