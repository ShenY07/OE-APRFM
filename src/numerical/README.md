# Deterministic transport references

`parity_reference.py` provides 1D and 2D odd-even transport reference solvers.
Use `scripts/generate_reference.py` or `scripts/run_oe_si_dsa.py` from the
repository root with `PYTHONPATH=src:scripts`.

A converged linear or iterative solve does not establish reference accuracy:
refine spatial and angular resolution independently before interpreting an RF
error. The separate E6 Fourier references are in `scripts/reference_e6.py`.
