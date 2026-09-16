# Frozen manuscript tables

`all_available_tables.tex` contains thirteen available tables: eight main tables plus supplementary Tables S1, S3, the derived S3a accuracy-target comparison, S4, and S5. The separate `supplement_tables.tex` file is retained for manuscript assembly. The 2D OE-APRFM accuracy--cost sweep uses the final normalized four-component formulation. Legacy two-component efficiency files remain archived but are excluded. P5 errors are certified against level-C deterministic references.

The S3a companion is derived from the existing S3 aggregate CSV without new measurements. Regenerate it with `python3 scripts/build_s3_target_accuracy.py`. P7 transient pilot results remain separate in `results/p7_transient_pilot/three_seed_J128/REPORT.md`.
