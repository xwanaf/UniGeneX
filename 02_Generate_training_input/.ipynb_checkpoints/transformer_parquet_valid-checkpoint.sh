#!/bin/bash 

set -o pipefail
set -exu

python transformer_parquet_valid.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/ \
--gene_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain_1k \
--traingene_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain_1k \
--tissue '' \
--data_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain/log_scale \
--save_folder pretrain_1k \
--subsample_frac 0.08




# python transformer_parquet_valid.py \
# --base_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/ \
# --gene_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain \
# --traingene_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain \
# --tissue '' \
# --data_path /import/home2/xwanaf/Img2Expr/data/Lymph_node/pretrain/log_scale \
# --save_folder pretrain \
# --subsample_frac 0.08


