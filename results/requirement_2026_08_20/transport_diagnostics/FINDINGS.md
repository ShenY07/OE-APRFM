# Interpretation of additional indicators

Computed from 71 archived solution fields (all available S3 runs plus nine
P6 runs), aggregated into 29 problem/epsilon/method/budget groups. No solver
was rerun, no original error or timing measurement was changed. Exact source
paths and content hashes are in seedwise.csv.

## Model size at target accuracy

S3a now includes N_tau, selected independently of the minimum-time budget.
The qualification rule remains BOTH median E_f and E_rho <= tau. Separate
E_f and E_rho thresholds are fully tabulated in
`../tables_frozen/target_model_size_by_metric.csv`.

For P3 at E_rho <= 1e-3, the minimum tested qualifying model sizes are
9216 (deterministic), 128 (MM), 512 (OE); APNN does not qualify. Thus OE has
an 18x smaller representation than the tested qualifying deterministic grid,
but takes much longer and is not smaller than MM. Under the joint 1e-2
criterion, deterministic/OE sizes are 2304/512 (4.5x). These are restricted
to tested budgets; no global minimality or memory savings are claimed.

## Flux and microscopic errors

P1/P3 have angle-independent exact f, hence exact scaled flux and microscopic
g=(f-<f>)/epsilon are identically zero. Relative errors would divide by zero
and are omitted. Absolute errors measure spurious angular content.

P3, epsilon=1e-3, largest reported budget per method, three-seed medians
for randomized methods:

| Method | Model size | Absolute scaled-flux L2 error | Absolute g L2 error |
|---|---:|---:|---:|
| MM-APRFM | 512 | 2.5354e-7 | 3.5863e-7 |
| OE-APRFM | 512 | 4.9465e-3 | 3.8005 |
| OE-APNN | 25730 | 3.3297e1 | 6.3086e1 |
| OE-SN-Krylov | 16384 | 5.0141e-4 | 7.0956e-4 |

OE suppresses spurious scaled flux much better than APNN in these archived
runs, but MM remains stronger. Deterministic norms use native cell/angular
grids; their comparison is descriptive and needs common-grid or refinement
validation before making a precise performance-ratio claim.

## P6: strong flux accuracy, a limitation in the full microscopic field

| epsilon | Relative scaled-flux error | Relative g error |
|---|---:|---:|
| 1 | 6.2763e-5 | 2.0752e-4 |
| 1e-3 | 1.6202e-4 | 5.0367e-2 |
| 1e-6 | 1.6091e-4 | 5.0394e1 |

At epsilon=1e-6 the median absolute even-g error is 5.1436 while the odd-g
error is 1.0563e-4. At epsilon=1e-3 they are 5.1398e-3 and 1.0463e-4.
The nearly epsilon-independent unscaled even angular error (~5.14e-6)
is amplified by division by epsilon. This is consistent with accurate odd
flux and accurate full f, but does not support epsilon-uniform relative
accuracy of the full g. It is not a contradiction of a stability statement
controlling f alone. No external-method superiority can be inferred from
P6 because matching external P6 runs are absent.

Recommended manuscript claim remains comparison-specific. The added flux
indicator strengthens the comparison to OE-APNN; it does not overturn the
MM comparison. Report g as a limitation/diagnostic, not selectively omit it
because it is unfavorable.

Validation: analytical slab moments and 2D vector moments; correct omission
of relative errors for zero truth; angular-opposite pairing and weight
symmetry; orthogonal even/odd error decomposition for every archive; complete
S3 source matching. Archived fields share the methods' original evaluation
protocol; new angular quadrature convergence has not been performed.
