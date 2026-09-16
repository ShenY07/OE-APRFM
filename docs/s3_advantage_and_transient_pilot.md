# S3 evidence and transient extension (2026-09-08)

S3 supports a comparison-specific claim, not overall superiority of OE-APRFM.
At the largest reported 2D budgets, OE-APRFM (512 coefficients) has 17.98x
smaller kinetic error and 24.00x smaller density error than OE-APNN (25,730
trainable parameters, 5,000 steps), with 2.94x smaller reported computation
time. These are distinct approximation/training budgets, not a matched-error
speedup. Model coefficient counts are not memory measurements.

MM-APRFM is both faster and more accurate than OE-APRFM in this S3 2D sweep.
The deterministic method is also a strong baseline. A new derived metric
cannot reverse these observations. The 1D linear manufactured problem is
exactly represented by diamond difference and is unsuitable for demonstrating
an accuracy-cost advantage over that discretization.

The added `table_S3a.tex` compares the smallest **observed** aggregate runtime
among budgets attaining BOTH E_f and E_rho <= a common target (1e-2,1e-3,1e-4).
It includes every method and every target, and marks unattained targets with
dashes. It does not interpolate, infer new errors, or imply all seeds attain
the threshold. The original S3 measurements remain unchanged.
Generate with `python3 scripts/build_s3_target_accuracy.py`; the main frozen
table builder also regenerates the companion table. This is a descriptive
secondary summary of the existing sweep, not a prospectively selected metric.

Useful prospective measures for a final new benchmark:

- Error across epsilon with approximation space, sampling and cutoff frozen.
- Best measured cost at predeclared common error targets, counting setup,
  feature evaluation, assembly and solve/training; report failures as well.
- Seedwise success rate, median and range; timing repetitions separate from
  random-seed repetitions.
- Projection/rescaling and parity ablations, explicitly reporting both
  coefficient and residual dimensions.

Avoid inverse-error-per-parameter scores or ad hoc products of error and time:
they obscure the accuracy-cost tradeoff and have no intrinsic fairness claim.
Meshless geometry handling and direct linear solves are method properties;
they require corresponding experiments to support performance claims.

The new P7 prototype has completed 27 runs: three methods, three epsilon
values (1,1e-2,1e-3), three seeds (11,23,37), at 256 coefficients.
At t=.1, median density errors of projected/rescaled OE are 70.60%, 1.566%,
1.830%, respectively. At epsilon=1e-3 the controls have errors of 533.56%
(unprojected OE) and 3761.89% (direct RFM). These results support a small-epsilon
structural benefit in this prototype, not superiority over MM-APNN, author
parity-APNN, or deterministic AP discretizations. The kinetic-regime failure
is retained in the report and plot. P7 is used because P6 already names the
angular-dependent manufactured test.

See `results/p7_transient_pilot/three_seed_J128/REPORT.md` for assumptions,
reference checks and limitations. Pilot results are not inserted as frozen
manuscript accuracy results. The current paper's steady stability analysis
does not automatically cover the time-dependent extension.
