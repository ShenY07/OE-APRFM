# Micro–macro random-feature comparison

`original/src/` contains the baseline implementation attributed to Keke Wu;
see [the original attribution](original/README.md) and
[third-party notices](../../THIRD_PARTY_NOTICES.md). Historical notebooks and
reference arrays are not part of this source release.

Run from the repository root, in separate processes so that the OE and MM
modules with matching package names do not collide:

```bash
python scripts/run_mm_aprfm_1d.py --problem p1 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/mm_quickstart
python scripts/run_mm_aprfm_2d.py --problem p3 --epsilon 0.001 --seed 11 \
  --features 32 --output-dir results/mm_quickstart
```

These commands demonstrate execution, not matched-accuracy timing comparisons.
The 1D runner also supports P2/P6; the 2D runner supports P3/P4. P2 requires
reference arrays; see [the reproduction guide](../../docs/reproduction.md).
The time-dependent MM extension is in `scripts/run_e6_mm.py`.
