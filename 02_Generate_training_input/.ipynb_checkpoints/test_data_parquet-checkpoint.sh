#!/bin/bash 

set -o pipefail
set -exu




# declare -a values=("fraction_g100_r50;0.50;1." "fraction_g00_r50;0.50;0.")
# declare -a values=("fraction_g100_r10;0.10;1.")
# declare -a values=("fraction_g100_r01;0.01;1." "fraction_g100_r05;0.05;1." "fraction_g100_r25;0.25;1.")
# declare -a values=("fraction_g100_r05;0.05;1.")
declare -a values=("fraction_g00_r50;0.05;1.")
# declare -a values
# values[0]="fraction_g50_r50;0.50;0.5"
# values[1]="fraction_g70_r50;0.50;0.7"
# values[2]="fraction_g80_r50;0.50;0.8"
# values[3]="fraction_g90_r50;0.50;0.9"

for value in "${values[@]}"
do
    # turn e.g. 'domain.de;de;https' into
    # array ['domain.de', 'de', 'https']
    IFS=";" read -r -a params <<< "${value}"

    tissue="${params[0]}"
    fraction="${params[1]}"
    subsample_gene_frac="${params[2]}"
    echo "tissue : $tissue"
    echo "fraction : $fraction"
    echo "subsample_gene_frac : $subsample_gene_frac"
    
    
#     python filter_hvg.py \
#     --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
#     --tissue $tissue \
#     --data_folder raw \
#     --save_folder log_scale \
#     --filter_cells_min_genes 3 \
#     --do_not_take_raw 


#     python hvg.py \
#     --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
#     --tissue $tissue \
#     --hvg_batch_key donor_id \
#     --data_folder log_scale \
#     --save_folder hvg10k \
#     --n_top_genes 10000 




    ####################################################
    # make parquet
    ####################################################
    python transformer_inference_parquet.py \
    --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
    --tissue $tissue \
    --data_folder log_scale \
    --save_path /import/macyang_home2/xwanaf/Img2Expr/data/lung_HLCAbench/subsample


done








# # # tissue='fraction_g50_r50'
# # # fraction=0.50
# # # subsample_gene_frac=0.5

# tissue='fraction_g60_r50'
# fraction=0.50
# subsample_gene_frac=0.6


# # ####################################################
# # # subsample
# # ####################################################
# # python subsample_test_adata.py \
# # --fraction $fraction \
# # --save_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample \
# # --tissue $tissue \
# # --subsample_gene_frac $subsample_gene_frac 



# ####################################################
# # preprocess
# ####################################################
# # python sep.py \
# # --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
# # --tissue $tissue \
# # --data_folder '' \
# # --data_file_name raw_test.h5ad \
# # --split_col data_file_names \
# # --save_folder raw 



# python filter_hvg.py \
# --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
# --tissue $tissue \
# --data_folder raw \
# --save_folder log_scale \
# --filter_cells_min_genes 3 \
# --do_not_take_raw 


# python hvg.py \
# --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
# --tissue $tissue \
# --hvg_batch_key donor_id \
# --data_folder log_scale \
# --save_folder hvg10k \
# --n_top_genes 10000 




# ####################################################
# # make parquet
# ####################################################
# python transformer_inference_parquet.py \
# --base_path /home/share/xwanaf/Img2Expr/data/lung_HLCAbench/pretrain/test_data_subsample/ \
# --tissue $tissue \
# --data_folder log_scale \
# --save_folder '' 



