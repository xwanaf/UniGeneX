#!/bin/bash 

set -o pipefail
set -exu


python filter.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--data_folder raw \
--save_folder log_scale 

python hvg.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--hvg_batch_key donor_id \
--data_folder log_scale \
--save_folder hvg10k \
--n_top_genes 10000 

python hvg.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--hvg_batch_key donor_id \
--data_folder log_scale \
--save_folder hvg5k \
--n_top_genes 5000 

python hvg.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--hvg_batch_key donor_id \
--data_folder log_scale \
--save_folder hvg6k \
--n_top_genes 6000 

python hvg.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--tissue '' \
--hvg_batch_key donor_id \
--data_folder log_scale \
--save_folder hvg7k \
--n_top_genes 7000 



# ####################################################
# # make parquet
# ####################################################
# python transformer_inference_parquet.py \
# --base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
# --tissue '' \
# --data_folder log_scale \
# --save_folder '' 



