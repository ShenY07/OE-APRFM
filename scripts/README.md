# Postprocessing and validation

Experiment execution lives in [`experiments/`](../experiments/README.md).
This directory contains result summaries, manuscript updates, and audits.
Run tools from the repository root with `PYTHONPATH=src:.`.

- `report_*.py` aggregates completed runs; `transport_moments.py` and
  `e5_field_diagnostics.py` compute physical diagnostics.
- `refresh_2d_tables.py` runs the prescribed P3/P4 campaign;
  `update_2d_tables.py` updates the corresponding manuscript rows.
- `refresh_figures.py` renders the historical E1–E6 panels;
  `paper_paths.py` uses optional local manuscript mappings, falling back to
  the original output paths when no manuscript manifest is present.
- `check_paper.py` can validate an independently supplied local `paper/`
  snapshot and manifest. It is not part of source-only CI. Manuscript update
  and audit tools require their local input files; these are not bundled.
- `audit_*.py`, `verify_p5_e5.py`, and `validate_tables.py` inspect archived
  evidence. Several require local files under `.release-backup/` and dated
  `results/` directories; they are not fresh-clone reproduction commands.
- `name_changes.json` and `layout_changes.json` record API and path migrations.
  `source_compat.py` applies them in memory during historical source checks.

Some campaign and audit scripts execute at module level. Use their documented
commands rather than importing them as a library. Reusable numerical code
belongs in `src/oeaprfm/`.
