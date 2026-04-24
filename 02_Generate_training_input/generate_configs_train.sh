#!/bin/bash 
set -o pipefail
set -exu



base_path="/import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility"



python generate_configs_train.py \
--config_temp_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/config_templete_train.yaml \
--save_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_input \
--save_config_name config_train \
--vocab_path $base_path/default_census_vocab.json \
--kld_weight 1e-3 \
--common_dec_gene_len 1703 \
--CellTypeMapping_df_paths $base_path/Training_input/CellTypeMapping_df_valid.csv \
--data_source $base_path/Training_input/cls_prefix_data.parquet \
--test_out_of_sample_data_source $base_path/Training_input/cls_prefix_data_valid.parquet \
--common_dec_genes_path $base_path/pretrain_data_train_genes.npy \
--max_epochs 30 \
--gpus 0,1,2,3 \
--devices 4


