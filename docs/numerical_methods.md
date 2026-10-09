# Numerical methods and scope

## Steady OE-APRFM

The reconstruction is `f = r + epsilon*j`, with even `r` and odd `j` in
velocity. Random features and their spatial derivatives define the trial
space. Angular projection separates the macro residual from the even
microscopic residual; the scaled odd equation completes the system. Boundary
conditions are imposed as least-squares rows. The configuration and runner
select the actual parity representation and partition-of-unity version.

Row scaling, block weighting, column equilibration and SVD truncation affect
the discrete objective. Inspect the runner and saved metadata for the exact
settings. These operations improve numerical conditioning but are not a
proof of epsilon-uniform stability or approximation accuracy. The 2D runner
uses batched assembly and streaming QR to reduce storage requirements.

## E6 time-dependent extension

Let `P` denote angular averaging and `d_t u = (u_new-u_old)/dt`. At each step,
the three residuals are

```text
P(d_t r) + P(v * d_x j_new),
epsilon^2 * ((I-P)(d_t r) + (I-P)(v * d_x j_new)) + (I-P)r_new,
epsilon^2 * d_t j + v * d_x r_new + j_new.
```

All spatial and relaxation terms use the new time level: this is **implicit
backward Euler**, first order in time. Evaluating the old solution on the
right-hand side does not make the method explicit. A fixed spatial/angular
feature basis and fixed step give one weighted system whose SVD factors are
reused for all steps.

The E6 objective has five blocks: macro, even microscopic, odd, periodic even
trace and periodic odd trace. Rows are normalized and quadrature weights in
each block sum to one. Columns are equilibrated and singular values below
`1e-12` times the largest are discarded. The main configuration has 4,288
rows and 256 coefficients. Periodicity is a soft residual, not an exact
constraint; the initial state enters the first right-hand side analytically.

There is no explicit CFL check in this implementation. Implicit handling of
stiff relaxation does not by itself prove unconditional stability of the
complete RF/least-squares propagation. Accuracy still requires time
refinement. No general long-time power bound, exact positivity or exact mass
conservation is asserted. The small-epsilon initial current layer is not
claimed to be resolved at the default time step.

## Independent references and interpretation

Manufactured P1/P3/P4 problems are evaluated against analytic fields.
Nonmanufactured transport problems require independently refined numerical
references. E6 uses Fourier-mode angular evolution with a matrix exponential
for continuous time and a separate backward-Euler reference at the same step.

- Continuous-time error includes RF, algebraic and time-discretization error.
- Same-step BE error removes the common time-discretization contribution; it
  cannot be presented as total error against the continuous equation.
- Difference from the diffusion solution includes the finite-epsilon
  asymptotic difference. Continuous and BE diffusion curves are distinct.

Track mass drift, periodic residuals, constant-state errors and sampled
minima, together with feature, quadrature and test-grid refinement. Small
observed values support the tested cases only. They do not replace a
stability theorem or certify positivity between evaluation points.
