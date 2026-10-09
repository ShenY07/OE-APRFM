# Third-party attribution

## Micro–macro random-feature baseline

`baselines/mm_oerfm/original/` retains the implementation attributed to
**Keke Wu**, *Asymptotic Preserving Random Feature Method for Multiscale
Radiative Transfer Equations*. The original author and affiliation are kept
in that directory's README.

The code was present in this repository's earlier `rte/` tree and subsequently
vendored into the comparison directory. It is isolated from the OE modules
by the matched-baseline runners. Historical notebooks and data remain in Git
history; the current release retains Python source used by the comparisons.

No separate upstream LICENSE was present in the supplied baseline snapshot.
This release does not assign a new license to that code or certify permission
for its reuse. Contact the relevant rights holder concerning licensing.

## Dependencies

JAX, Flax, NumPy, SciPy, Matplotlib, ml-collections, pytest and optional PyTorch
are installed as dependencies and retain their respective licenses. Their
source distributions are not bundled here.
