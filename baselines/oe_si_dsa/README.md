# OE-SI-DSA baseline

This baseline is reserved for source iteration with diffusion synthetic
acceleration applied after odd-even decomposition. It is distinct from the
existing directional transport sweep in `numerical/parity_reference.py`, which
is recorded as SI-DSA.

Fair comparisons use the same spatial grid, angular quadrature, tolerance and
maximum iteration count for SI-DSA and OE-SI-DSA. Every run records convergence
status; a point reaching the iteration limit is retained as a non-converged
result rather than omitted.

Output data are written to `results/baselines/oe_si_dsa/` and summarized in
`results/tables/oe_si_dsa_summary.csv`.
