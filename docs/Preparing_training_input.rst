
Prepare traning input files
========================

After preprocessing the Atlas data, we will build vocabulary for credible gene set and intersect each dataset's hvgs with credible gene set and map to vocabulary to get the input for training transformer. Becauese the large scale of training data, we will build parquet format of training data for training.


Mapping to Credible Gene Set
----------------------------


Please run the following from 02_Generate_training_input and modify path of input and output. 

.. code-block:: 

   ./transformer_parquet.sh


command:


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


Output:


This step will output the following files 

CellTypeMapping_df.csv: training data
cls_prefix_data.parquet: traing data parquet 

CellTypeMapping_df_valid.csv

cls_prefix_data_valid.parquet
config_inference.yaml
config_train.yaml
generate_configs_log.log
log_scale_train_genes_pca
obs_concat.csv
obs_concat_valid.csv
transformer_parquet_log.log
transformer_parquet_valid_log.log
