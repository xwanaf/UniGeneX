# Pulmonary fibrosis workflow

This workflow reproduces UniGeneX manuscript Figures 3B–H, 4A, 4D, and 4E.
Large inputs and optional precomputed mappings are distributed through the
[UniGeneX PF Zenodo record](https://doi.org/10.5281/zenodo.22489310).

## Setup

Extract the downloaded archives into one directory. Every archive contains
paths beginning with `PF/`, so the directories merge when extracted to the
same destination.

```bash
export UNIGENEX_DATA_DIR=/path/to/unigenex_data
```

The final input layout is:

```text
PF/
├── UGE_atlas.h5ad
├── obs_concat_atlas.csv
├── selected_index/
├── spatial_annotation/
│   ├── Reference_data/
│   ├── fitted_reference_nn_normal/
│   ├── fitted_reference_nn_disease/
│   ├── query_UGE/
│   ├── query_raw/
│   └── example_rerun_results/       # downloaded or generated
└── sp_map_to_sc_Epi/
    ├── UGE_atlas_sub.h5ad
    ├── UGE_sp_sub.h5ad
    ├── histology/
    └── figure4_rerun_results/       # downloaded or generated
```

See `data_manifest.tsv` for every input and its archive.

## Mapping

The precomputed-mappings archive lets notebooks 03 and 04 run without
repeating atlas mapping. To regenerate those mappings, activate the UniGeneX
environment and run the launchers from the repository root:

```bash
bash Real_data_PF/transformer_map_to_atlas.sh
bash Real_data_PF/transformer_map_to_epithelial_atlas.sh
```

Both launchers read `UNIGENEX_DATA_DIR`. An explicit data root may instead be
passed as the first argument:

```bash
bash Real_data_PF/transformer_map_to_atlas.sh /path/to/unigenex_data
```

The first launcher uses released normal and disease reference indexes. The
second creates 30-neighbor and 300-neighbor mappings for Figures 4A and 4D.

## Notebook order

1. `01_atlas_and_markers.ipynb` — Figures 3B–D.
2. `02_prepare_spatial_references.ipynb` — reference-selection provenance.
3. `03_spatial_annotation_results.ipynb` — Figures 3E–H.
4. `04_epithelial_trajectory_and_spatial_pseudotime.ipynb` — Figures 4A and 4D.
5. `05_spatial_pseudotime_histology.ipynb` — Figure 4E.

Notebook 02 documents the original reference-selection procedure. Because its
historical random seed was not recorded, exact mapping uses the released
references and their matching indexes.

Generated figures and compact tables are written below `outputs/`. Set
`UNIGENEX_OUTPUT_DIR` to use another output directory. Notebook 05 reads the
compact pseudotime table created by notebook 04.
