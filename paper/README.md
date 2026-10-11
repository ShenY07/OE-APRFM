# Manuscript materials

This directory collects the 10 October 2026 manuscript snapshot.

| Path | Contents |
| --- | --- |
| `figures.tex` | Figure environments, captions, and labels |
| `tables.tex` | 26 tables and their interpretation notes |
| `figures/` | 31 vector PDF panels, numbered in figure order |
| `figures.zip` | The same PDF panels, with paths relative to this directory |
| `data/` | Plotted arrays, layouts, and source/run provenance indexes |
| `manifest.json` | Original paths, collected paths, and SHA-256 hashes |

Use `fig01.pdf` for a single-panel figure and `fig02a.pdf`, `fig02b.pdf`, etc.
for separately exported panels within one figure environment. The manifest
maps these short names to the original descriptive names. Compile figure and
table fragments from this directory, or configure the manuscript's graphics
path accordingly. These fragments are not a complete article source.

From the repository root:

```bash
python -m scripts.check_paper
python -m scripts.check_paper --bundle
```

The checker validates all 38 collected files before packaging. PDF bytes,
plotted data, captions, labels, and table values were retained during the
reorganization; figure inclusion paths were updated. A hash mismatch after
regenerating an asset requires reviewing the new snapshot and updating its
manifest, rather than treating it as the original published artifact.

Provenance JSON files retain historical paths and hashes. The path migration
is recorded separately in `../scripts/layout_changes.json`. Some provenance
entries point to local archives that are not distributed. The unchanged
original PDF bundle is retained as `../archive/paper_20261010.zip`.

Raw experiment fields and checkpoints belong in `../results/`. Selected
baseline CSV summaries and angular-domain result records remain there too;
see [the result inventory](../results/README.md). The two neural-training
figures are supplementary diagnostics. E5 retains its archived-reference
qualification, and heterogeneous timing environments do not establish
controlled speedups.
