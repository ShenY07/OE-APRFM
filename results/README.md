# Manuscript result snapshot — 10 October 2026

- `table.md`: 26 manuscript tables, with protocol and interpretation notes.
- `figure.md`: figure inclusion paths and captions. The two OE-APNN budget
  panels at the end are supplementary training diagnostics, optional in the
  manuscript; the main comparison uses their final-budget table entries.
- `figure/new/*.pdf` and `oe_apnn_parity_20261010/figures/*.pdf`: 31 separately
  exported square panels, retaining their original names.
- `JSC_all_figures_20261010.zip`: the same 31 PDFs, with paths relative to this
  directory matching the figure index.
- `oe_apnn_parity_20261010/{seedwise,summary}.csv`: revised baseline summaries.
- `angular_domain_comparison/`: twelve per-seed result records, the execution
  manifest, equivalence checks, summary, and the new table's LaTeX source.
  See its README for the matching rule and regeneration commands.
- `published_data/`: exact plotted line arrays, layout information, and
  retained source/run provenance indexes. Paths into `.release-backup` describe
  historical local inputs; those inputs are not bundled or downloadable from
  these indexes. They must not be treated as runnable commands on a fresh clone.

Raw NPZ fields, neural checkpoints, local backup archives and cache files are
not versioned. The historical archive is not fully reproduced by this snapshot.
Generate new reference-dependent runs using the [reproduction guide](../docs/reproduction.md).
The published experiment runners do not require the local backup. Historical
figure-refresh and audit scripts that require it are deliberately excluded.

Read the figure and table qualifications: E5 reports differences from an
archived reference whose acceptance checks did not pass for every case;
heterogeneous timing environments do not justify controlled speedup claims.
The PDF bundle is a result snapshot, not evidence of reference certification.
