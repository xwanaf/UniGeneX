# Cell2location modifications used by UniGeneX

This directory contains the importable `cell2location` Python package used for
the UniGeneX glioma/Visium tutorial. The upstream project is distributed under
the included Apache License 2.0.

- Upstream repository: <https://github.com/BayraktarLab/cell2location>
- Upstream version: `0.1.5`
- Upstream commit: `c01046d34216dbed23951a339d8f08c5499a6fc4`
- UniGeneX variant: `0.1.5+unigenex.gbmv1`
- Frozen from the manuscript working copy used to generate Figures 6C-D.

## Method citation

Kleshchevnikov et al. *Cell2location maps fine-grained cell types in spatial
transcriptomics.* Nature Biotechnology 40, 661–671 (2022).
<https://doi.org/10.1038/s41587-021-01139-4>

## Model changes

The spatial model can use an observed background-frequency profile calculated
from the spatial count matrix. For each spot, counts are normalized to 10,000;
the mean normalized value of each gene initializes the background profile.
When `use_bg=True`, the gene-specific additive component has a log-normal prior
and is scaled by this profile.

The model also accepts optional nuclei-derived cell counts per location. The
manuscript Figure 6 runs did not provide nuclei counts and instead used an
expected abundance of 20 cells per Visium spot.

The UniGeneX copy removes development-only NumPy debug-file writes and restores
the upstream gamma background branch when `use_bg=False`. These cleanups do not
change the `use_bg=True` model used in the tutorial.

## Manuscript settings

- `use_bg=True`
- total-count threshold: `1,000`
- expected cells per location: `20`
- detection alpha: `200`
- spatial training epochs: `30,000`
- posterior samples: `1,000`
