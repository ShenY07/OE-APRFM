# P5 pilot validation audit — 2026-08-23

## Outcome

The requested smooth P5 definition was tested without changing its physics:
`sigma_s=1+x2`, `sigma_a=0`, positive epsilon-independent source, vacuum
inflow, and epsilon values 1 and 1e-3.  The diffusive pilot passes its density
error target, but the kinetic pilot does not.  Consequently B/C reference
refinement and the formal three-seed run remain gated.

## Verified implementation corrections

- The V2 interior, boundary and reconstruction paths now use the same smooth
  overlapping partition of unity.
- The two antipodal-pair inflow equations are assembled independently.
- Streaming least squares now implements the protocol's unit-L2 row scaling,
  rather than infinity-norm scaling.
- A constant-solution check reconstructs `f=1` within `[0.990,1.002]`.
- A linear manufactured check reconstructs `f=1+x` with relative error
  `1.68e-3`, verifying the source sign and odd/even RHS decomposition.

## Frozen pilot results

With 2x2x1 patches, 64 features per component and patch, 32x32x16 uniform
collocation, scale 1, rcond 1e-12 and seed 11:

| epsilon | Ef | Erho | min rho_h | corr(rho_h,rho_ref) | decision |
|---:|---:|---:|---:|---:|---|
| 1 | 1.71 | 1.50 | -1.14 | 0.105 | reject |
| 1e-3 | 2.33e-2 | 1.47e-2 | -1.63e-2 | 0.9997 | error passes; slight negativity |

The independent level-A references converged to relative residuals 9.71e-10
and 2.41e-10, respectively, and are positive.

## Rejected diagnostic hypotheses

- Boundary weights 10 and 100 did not improve the kinetic error.
- Feature scale 3 reduced conditioning but left Erho above one.
- Truncation values 1e-6 and 1e-4 left Erho above one.
- A global one-patch space at the same 1024-coefficient budget failed.
- Increasing to 128 features per patch (2048 coefficients) failed.
- Uniform 4x4x1 patches with 32 features per patch (2048 coefficients) still
  gave `Erho=1.30` and a negative density.

These checks show that the failure is not a reference, sign, boundary-row,
partition or simple capacity defect.  The requested O(1) kinetic-source case
is not reportable with the current OE-APRFM space under the stated constraints.
Producing a positive O(1e-2) result would require a substantive method or
problem change (for example a source-aware deterministic lifting, positivity
constraints, absorption, or a different kinetic formulation), all outside
the frozen P5 specification.
