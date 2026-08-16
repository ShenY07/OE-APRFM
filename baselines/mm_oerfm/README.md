# MM-OERFM comparison baseline

This directory vendors the micro-macro OERFM implementation from
`/home/sheny-y23/rte` for controlled comparison with OE-OERFM.

## Layout

- `original/src/`: original MM-OERFM implementation and configurations.
- `original/test/`: original executable experiment notebooks, without figures.
- `original/src/numerical/`: original 1D micro-macro, SI, and DSA-SI notebooks.
- `original/src/data/`: original 1D reference arrays required by the notebooks.
- `baseline.json`: method metadata and experiment correspondence.
- `../../scripts/run_mm_oerfm_baseline.py`: isolated notebook launcher.

The vendored modules are not merged into the repository's top-level `src/`
tree. The launcher gives `original/src` import priority only inside the MM
subprocess, preventing modules such as `constraints`, `modules`, and `solver`
from resolving to the OE-OERFM implementations.

## Commands

```bash
python scripts/run_mm_oerfm_baseline.py --list
python scripts/run_mm_oerfm_baseline.py --check
python scripts/run_mm_oerfm_baseline.py --experiment ex1_1d --execute
```

Executed notebooks are written to `results/baselines/mm_oerfm/notebooks/`.
The original notebooks contain their own numerical settings; matched P1--P5
comparison runs must use the settings recorded in `baseline.json` and the same
reference/evaluation grids as OE-OERFM.

## 1D components

| Component | Vendored path |
|---|---|
| Micro-macro constraints | `original/src/constraints/continuous1d.py` |
| Random feature spaces | `original/src/modules/func_space.py` |
| MM solution constructor | `original/src/modules/solution.py` |
| 1D collocation sampler | `original/src/modules/generator.py` |
| Least-squares solver | `original/src/solver/least_square.py` |
| Experiment configurations | `original/src/configuration/rte1d_settings*.py` |
| MM-APRFM experiments | `original/test/ap_rte_1d_ex{1,2,3}.ipynb` |
| RFM experiment | `original/test/rte_1d_ex1.ipynb` |
| Deterministic solvers | `original/src/numerical/{SI_1d,DSA_SI_1d}.ipynb` |
| Micro-macro numerical notebook | `original/src/numerical/micro_macro.ipynb` |
| Matched P1/P2 runner | `../../scripts/run_mm_aprfm_1d.py` |
