# Experiment outputs

New experiment records, raw fields, reference solutions, logs, and checkpoints
belong here. The runners retain their existing output schemas and default
paths so saved experiments remain traceable.

The collected manuscript is in [`../paper/`](../paper/README.md): numbered
PDF panels, LaTeX figure/table fragments, plotted arrays, and a file manifest.
Use `python -m scripts.check_paper` from the repository root to verify it.

Versioned experiment records include:

- `oe_apnn_parity_20261010/{seedwise,summary}.csv`: revised neural-baseline
  summaries. Its selected PDF panels are now in `../paper/figures/`.
- `angular_domain_comparison/`: twelve per-seed records, execution manifest,
  equivalence checks, summary, and table source. See its README for commands.

Other dated folders are local run or audit outputs. Their dates and recorded
paths are preserved as provenance, rather than renamed to imply a new run.
Raw NPZ fields, checkpoints, process logs, caches, and local backup archives
are not versioned. Do not place generated data in `src/` or `experiments/`.

For new studies, separate the problem, configuration, and seed in the output
path, and save both machine-readable metrics and the corresponding run
configuration. Collect reviewed manuscript assets under `paper/`; keep source
records and raw fields here. Reference-dependent runs and their limitations
are described in [the reproduction guide](../docs/reproduction.md).
