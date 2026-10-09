# Source-release validation

The release source tree was checked on 2026-10-09 with Python 3.10, CPU JAX
float64, NumPy/SciPy, and one BLAS thread. The checks used the available
Python environment with the pinned runtime versions, not a newly downloaded
virtual environment. Timings collected during these checks are not isolated
performance measurements.

| Check | Outcome |
| --- | --- |
| Existing regression suite, including optional OE-APNN tests | 25 passed |
| Python compilation | Passed for source, runners, baselines and tests |
| CLI imports / `--help` | 12 entry points passed |
| Black formatting | Passed for maintained OE source, runners, tests and OE-APNN |
| Formatting equivalence | All 75 formatted/checked Python files retained identical ASTs |
| README P1 example | Completed; distribution relative L2 error approximately 8.38e-5 |
| P3 example, 32 features per field/patch | Completed; distribution relative L2 error approximately 2.19e-2 |
| Matched MM P1 example | Completed |
| E6 quick start and independent references | Both epsilon values completed |
| Full prescribed E6 study | 20 OE trajectories completed |
| E6 constant-state checks | 20/20 passed |
| E6 independent test-grid refinement checks | 960/960 passed |
| E6 noninitial reference-acceptance checks | 360/360 passed |
| E6 MM comparison | Six trajectories and summary completed |
| Reference generator CLI | P2/P5/constant-inflow output paths checked with stubbed numerical solvers; unsupported P2 level rejected |

The E6 full study was run with the quick-start batch also present, verifying
that the final report selects only prescribed study batches. Formatting was
checked by comparing Python abstract syntax trees before and after formatting;
third-party vendored Python source was left unformatted.

Large P2/P5 transport-reference solves and all historical benchmark sweeps
were not repeated for this source-tree cleanup. Example execution is not an
accuracy claim for every configuration. The checked-in CI workflow runs the
core CPU tests and format check; the optional PyTorch tests were run locally.
A successful local check does not imply that a hosted CI run has completed.
