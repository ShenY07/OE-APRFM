# Deterministic odd-even transport comparison

The implementations are in `src/numerical/parity_reference.py`; the executable
entry is `scripts/run_oe_si_dsa.py`. Use `--help` to select the problem, grid,
iteration tolerance, iteration limit and 1D solver (automatic, DSA or Krylov).

```bash
python scripts/run_oe_si_dsa.py --problem p1 --epsilon 0.001
```

Results are written to `results/baselines/oe_si_dsa/`. Inspect the saved solver
and convergence diagnostics rather than inferring the algorithm from the
directory name. Grid refinement and matched tolerances are required for a
meaningful comparison with random-feature methods.
