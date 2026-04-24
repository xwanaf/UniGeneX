#!/bin/bash 

set -o pipefail
set -exu



python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/atlas_1e3_maskp5_processed.h5ad \
--adata_inte_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/Testdata_UGE.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/NN_atlas_1e3_maskp5 \
--recompute_pca \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/NN_testdata_mapped_results \
--atlas_assign_label_col ann_finest_level



