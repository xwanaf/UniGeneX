#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${UNIGENEX_DATA_DIR:-}}"

if [[ -z "${DATA_ROOT}" ]]; then
    echo "Usage: $0 /path/to/data-root" >&2
    echo "Or set UNIGENEX_DATA_DIR to the directory containing PF/." >&2
    exit 2
fi

PF_DATA="${DATA_ROOT}/PF/sp_map_to_sc_Epi"
MAPPER="${SCRIPT_DIR}/../05_UGE_celltype_annotation/transformer_map_to_atlas.py"
OUTPUT="${PF_DATA}/figure4_rerun_results"
PYTHON_BIN="${PYTHON:-python}"

# Figure 4A: map spatial epithelial cells with 30 atlas neighbors
"${PYTHON_BIN}" "${MAPPER}" \
    --atlas-path "${PF_DATA}/UGE_atlas_sub.h5ad" \
    --query-path "${PF_DATA}/UGE_sp_sub.h5ad" \
    --atlas-index-path "${OUTPUT}/knn30" \
    --output-path "${OUTPUT}/knn30" \
    --atlas-label-column ct_anno \
    --keep-unlabeled-atlas \
    --n-neighbors 30 \
    --recompute-pca
# Figure 4D: use 300 atlas neighbors for pseudotime transfer
"${PYTHON_BIN}" "${MAPPER}" \
    --atlas-path "${PF_DATA}/UGE_atlas_sub.h5ad" \
    --query-path "${PF_DATA}/UGE_sp_sub.h5ad" \
    --atlas-index-path "${OUTPUT}/knn300" \
    --output-path "${OUTPUT}/knn300" \
    --atlas-label-column ct_anno \
    --keep-unlabeled-atlas \
    --n-neighbors 300 \
    --recompute-pca
echo "Completed the Figure 4 epithelial trajectory mappings."
echo "Results: ${OUTPUT}"
