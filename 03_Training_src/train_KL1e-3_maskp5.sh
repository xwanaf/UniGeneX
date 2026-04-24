#!/bin/bash 
set -o pipefail
set -exu


python /home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/Reproducibility/Validation/03_Training_src/train_UniGeneX.py \
--config /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_input/config_train.yaml \
-s /import/home2/xwanaf/Img2Expr/data/Benchmarking/reproducibility/Training_output/Generation_HLCA_KL1e-3_maskp5 \
--num_workers 28 
# --pin_memory 
