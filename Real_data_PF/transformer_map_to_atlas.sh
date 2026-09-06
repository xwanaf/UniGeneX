#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${UNIGENEX_DATA_DIR:-}}"

if [[ -z "${DATA_ROOT}" ]]; then
    echo "Usage: $0 /path/to/data-root" >&2
    echo "Or set UNIGENEX_DATA_DIR to the directory containing PF/." >&2
    exit 2
fi

PF_DATA="${DATA_ROOT}/PF/spatial_annotation"
MAPPER="${SCRIPT_DIR}/../05_UGE_celltype_annotation/transformer_map_to_atlas.py"
OUTPUT="${PF_DATA}/example_rerun_results/ct_anno_release_v1"
PYTHON_BIN="${PYTHON:-python}"

# Figure 3F: normal sample mapped to the normal reference
"${PYTHON_BIN}" "${MAPPER}" \
    --atlas-path "${PF_DATA}/Reference_data/atlas_Normal.h5ad" \
    --query-path "${PF_DATA}/query_UGE/adata_inte_VUHD116A.h5ad" \
    --atlas-index-path "${PF_DATA}/fitted_reference_nn_normal" \
    --output-path "${OUTPUT}/VUHD116A_Normal" \
    --atlas-label-column ct_anno \
    --use-existing-index
# Figure 3G: PF sample mapped to the disease reference
"${PYTHON_BIN}" "${MAPPER}" \
    --atlas-path "${PF_DATA}/Reference_data/atlas_Disease.h5ad" \
    --query-path "${PF_DATA}/query_UGE/adata_inte_VUILD107MA.h5ad" \
    --atlas-index-path "${PF_DATA}/fitted_reference_nn_disease" \
    --output-path "${OUTPUT}/VUILD107MA_Disease" \
    --atlas-label-column ct_anno \
    --use-existing-index
# Figure 3H: PF sample mapped to the disease reference
"${PYTHON_BIN}" "${MAPPER}" \
    --atlas-path "${PF_DATA}/Reference_data/atlas_Disease.h5ad" \
    --query-path "${PF_DATA}/query_UGE/adata_inte_VUILD106MA.h5ad" \
    --atlas-index-path "${PF_DATA}/fitted_reference_nn_disease" \
    --output-path "${OUTPUT}/VUILD106MA_Disease" \
    --atlas-label-column ct_anno \
    --use-existing-index
echo "Completed the three PF spatial mappings."
echo "Results: ${OUTPUT}"
