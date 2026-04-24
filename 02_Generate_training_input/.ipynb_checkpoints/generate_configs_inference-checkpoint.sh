#!/bin/bash 
set -o pipefail
set -exu

base_path="/import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility"



python generate_configs_inference.py \
--config_temp_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/config_templete_inference.yaml \
--save_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_input \
--save_config_name config_inference \
--vocab_path $base_path/default_census_vocab.json \
--common_dec_gene_len 1703 \
--CellTypeMapping_df_paths $base_path/Training_input/CellTypeMapping_df.csv \
--data_source $base_path/Training_input/cls_prefix_data.parquet \
--test_out_of_sample_data_source $base_path/Training_input/cls_prefix_data.parquet \
--common_dec_genes_path $base_path/pretrain_data_train_genes.npy \
--gpus 0,1,2,3 \
--devices 1






# # declare -a arr=('VUILD107MA' 'VUILD96LA' 'VUHD116B' 'VUILD102MA' 'VUILD96MA' 'VUILD102LA' 'VUHD116A' 'VUILD48LA1' 'VUILD104MA2' 'VUILD48LA2' 'VUILD105MA1' 'VUHD113' 'VUILD104MA1' 'VUHD069' 'VUILD105MA2' 'VUHD095') 
# declare -a arr=('VUHD113' 'VUILD91LA' 'VUILD48LA2' 'VUHD069' 'VUHD038' 'VUILD102LA' 'VUILD48LA1' 'VUHD049' 'VUILD106MA' 'VUILD105MA1' 'VUILD107MA' 'VUHD116A' 'VUILD96MA' 'VUILD104MA1' 'VUILD102MA' 'VUHD116B' 'VUILD142MA' 'VUILD96LA' 'VUILD141MA' 'VUILD58MA' 'VUILD91MA' 'VUILD49LA' 'VUHD095' 'VUHD090' 'VUILD78LA' 'VUILD110LA' 'VUILD105MA2' 'VUILD115MA' 'VUILD78MA' 'VUILD104MA2')


# ##################### 
# # sagittal
# ##################### 
# for adata in "${arr[@]}"
# do

# echo ${adata%.h5ad}


# python generate_configs.py \
# --config_temp_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/configs/config_templete.yaml \
# --save_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/configs_pretrainV4 \
# --save_config_name config_${adata} \
# --vocab_path $base_path/pretrain_data/default_census_vocab.json \
# --common_dec_gene_len 1742 \
# --CellTypeMapping_df_paths $base_path/Vannan/${adata}_1e3/CellTypeMapping_df.csv \
# --data_source $base_path/Vannan/${adata}_1e3/cls_prefix_data.parquet \
# --test_out_of_sample_data_source $base_path/Vannan/${adata}_1e3/cls_prefix_data.parquet \
# --common_dec_genes_path $base_path/pretrain_data/pretrain_data_train_genes.npy \
# --gpus 0,1,2,3,4,5,6,7 \
# --devices 1

# done

