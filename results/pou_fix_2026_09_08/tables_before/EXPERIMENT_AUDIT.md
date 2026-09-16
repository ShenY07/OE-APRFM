# Final table experiment audit

## Completed additions

- P1/P3 accuracy--cost sweep: MM-APRFM, OE-APNN, and OE-APRFM; three budgets and seeds 11, 23, 37.
- Efficiency statistics use median error and median solve/train time. Independent evaluation time is stored separately and excluded from the frontier.
- P4 assembled-matrix audit: the current 256-feature rerun has 152736 rows and 512 columns.
- P5 reference level C: 96x96 spatial cells, 24 angular nodes per quadrant, 1614 GMRES iterations, relative residual 9.95e-10.
- P5 B/C discrepancies: 2.48e-2 for the kinetic field and 2.01e-3 for density.

## P5 blocker

The new Gaussian-source P5 was tested with the former formal budget (four spatial patches, J=96, 5952 rows) and with an enriched budget (J=192, 16160 rows), including feature scales 1 and 5. The density errors remained larger than one. These runs are retained under `results/requirement_2026_08_20/p5_new` as failure evidence and are excluded from the manuscript tables. A formulation-specific resolution/weight calibration is required before a three-seed P5 result is scientifically reportable.

## Remaining statistical limitation

Efficiency timings have three independent seed realizations and are summarized by their median. Each seed/configuration was timed once; the stricter optional protocol of three repeated wall-clock runs per seed has not been performed.
