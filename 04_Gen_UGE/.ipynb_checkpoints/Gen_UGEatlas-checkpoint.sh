#!/bin/bash 
set -o pipefail
set -exu

# python /home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/Reproducibility/Validation/04_Gen_UGE/Gen_UGE.py \
# --config /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_input/config_inference.yaml \
# -s /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/Inference_HLCA_KL1e-3_maskp5 \
# --pretrain_root /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/Generation_HLCA_KL1e-3_maskp5 \
# --eval_batch_size 256 \
# --subsample 1000000 \
# --custom_ckptpath \
# --ckpt_filename "reproduce_epoch30.ckpt" 



python /home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/Reproducibility/Validation/04_Gen_UGE/UGE_to_adata.py \
--base_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/ \
--tissue 'Training_input' \
--save_folder 'Training_output' \
--traingene_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility \
--transformer_out_path /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/Inference_HLCA_KL1e-3_maskp5 \
--save_atlas_name atlas_1e3_maskp5.h5ad


# python /home/maxwan/mind-vis/train_scGPT_src/Inference/transformer_atlas_preprocess_leiden.py \
# --base_path /project/spatomics/maxwan/gene_pert/lung_HLCAall_addIPF_data/pretrain_data/ \
# --tissue '' \
# --save_folder '' \
# --save_atlas_name atlas_2e3_maskp5.h5ad \
# --level1_res p1 \
# --level2_res 2p

