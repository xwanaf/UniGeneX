#!/bin/bash 

set -o pipefail
set -exu



python transformer_parquet.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/ \
--gene_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--traingene_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--vocab_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--data_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/log_scale \
--trainset_list_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/HLCA_data_files_sl.npy \
--save_folder Training_input \
--skip_check_umap \
--check_ct_col ann_finest_level



python transformer_parquet_valid.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/ \
--gene_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--traingene_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--vocab_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--data_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/log_scale \
--trainset_list_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/HLCA_data_files_sl.npy \
--save_folder Training_input \
--subsample_frac 0.07