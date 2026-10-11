# Historical code: excluded from production experiments

This directory preserves evidence needed to explain archived results. Do not
add it to the production PYTHONPATH or use its output in manuscript tables.
Current experiment commands are in `training/README.md`; current figure/table
checks remain in `scripts/`.

- `historical_diagnostics/`: the old signed-velocity P1 reconstruction and
  scripts that deliberately reproduce the invalid parity or paired-trace P3
  systems. These are for historical diagnosis only. The archived P1 runner
  imports its own historical constructor, not the production constructor.
- `unused_constraints/`: three candidate 2D variants with no callers in the
  current source, scripts or tests. They are archived unchanged; being unused
  does not establish that every equation in them is wrong.
- `cleanup_manifest.json`: original locations and hashes, archive locations
  and hashes. Diagnostic entry files have adjusted imports/paths and an
  explicit execution guard; the unused constraint files are byte-identical.

Historical diagnostic scripts target the import layout of their corresponding
revision. Run them in a matching historical checkout, with their explicit
`OE_RUN_HISTORICAL_DIAGNOSTICS=1` guard enabled. They are not entry points for
the reorganized `oeaprfm` package. Keep their outputs separate from current
experiments and manuscript assets.

P5's exact-hash historical source evidence under
`results/checks/geometry/source/` is retained. P5 used the correct
physical inflow trace; it must not be confused with the defective historical
P3 paired-trace reproduction. Original result JSON/NPZ files are not deleted
or relabelled. The active two-component 2D comparison is also retained.

Redundant manuscript ZIP files are kept locally and excluded from Git.
