#!/usr/bin/env bash
set -euo pipefail

tutorial_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# User settings: change the data directory and Python executable if needed.
gbm_dir="/path/to/unigenex_data/GBM"
python_bin="python"

CUDA_VISIBLE_DEVICES=0 "$python_bin" "$tutorial_dir/deconvolve_visium.py" \
    --signatures "$gbm_dir/precomputed/reference/UGE_cell_state_signatures.csv" \
    --spatial \
        "$gbm_dir/visium/AT15-BRA-4-FO-C2-S27/spatial.h5ad" \
        "$gbm_dir/visium/AT10-BRA-5-FO-1_2/spatial.h5ad" \
    --output \
        "$gbm_dir/precomputed/cell2location/AT15-BRA-4-FO-C2-S27/cell_abundance_q05.csv" \
        "$gbm_dir/precomputed/cell2location/AT10-BRA-5-FO-1_2/cell_abundance_q05.csv" \
    --use-bg
