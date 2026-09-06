#!/usr/bin/env bash
set -euo pipefail

# Set UNIGENEX_REPRO_DATA to the extracted reproducibility-data directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${UNIGENEX_REPRO_DATA:-${SCRIPT_DIR}/../data}"
TRAINING_OUTPUT="${DATA_ROOT}/Training_output"

MAPPER="${SCRIPT_DIR}/transformer_map_to_atlas.py"
ATLAS_PATH="${TRAINING_OUTPUT}/atlas_1e3_maskp5_processed.h5ad"
QUERY_PATH="${TRAINING_OUTPUT}/Testdata_UGE.h5ad"
FITTED_NN_PATH="${TRAINING_OUTPUT}/NN_atlas_1e3_maskp5"
OUTPUT_PATH="${TRAINING_OUTPUT}/NN_mapped_results"
N_NEIGHBORS=30
PYTHON_BIN="${PYTHON:-python}"

# Map the query data to the reference atlas
"$PYTHON_BIN" "$MAPPER" \
    --atlas-path "$ATLAS_PATH" \
    --query-path "$QUERY_PATH" \
    --atlas-index-path "$FITTED_NN_PATH" \
    --output-path "$OUTPUT_PATH" \
    --atlas-label-column "ann_finest_level" \
    --n-neighbors "$N_NEIGHBORS" \
    --recompute-pca

echo "Mapping completed."
echo "Output: $OUTPUT_PATH/query_to_atlas_mapping.h5ad"
