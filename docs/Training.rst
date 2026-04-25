
Preparing Training Input
========================

After preprocessing the Atlas data, we construct a vocabulary for the **credible gene set**. Each dataset's Highly Variable Genes (HVGs) are intersected with this credible gene set and mapped to the vocabulary to create the transformer input. 

Due to the large scale of the training data, we convert the processed files into **Parquet** format to ensure efficient data loading during training.

Preparing .parquet data
----------------------------

You can execute the entire workflow from the ``./02_Generate_training_input`` directory using the provided shell script:


.. code-block:: 

   ./transformer_parquet.sh


The shell script sequentially calls two main functions:


.. code-block:: bash

   python transformer_parquet.py \
       --base_path /path/to/reproducibility/ \
       --gene_path /path/to/reproducibility \
       --traingene_path /path/to/reproducibility \
       --vocab_path /path/to/reproducibility \
       --tissue '' \
       --data_path /path/to/reproducibility/log_scale \
       --trainset_list_path /path/to/reproducibility/HLCA_data_files_sl.npy \
       --save_folder Training_input \
       --skip_check_umap \
       --check_ct_col ann_finest_level

**Parameter Descriptions:**

* ``--base_path``: The root project directory.
* ``--gene_path``: Directory for Highly Variable Genes (HVGs) results, generated in :doc:`Atlas_data_preprocessing`.

* ``--traingene_path``: Path to the ``.npy`` file defining the **credible gene set**, as generated in the :doc:`Atlas_data_preprocessing` section.
* ``--vocab_path``: Directory where the generated gene vocabulary will be saved.
* ``--tissue``: The specific subfolder. The full path to the parent of save path will be ``base_path / tissue``.
* ``--data_path``: Path to the log-normalized data, generated in :doc:`Atlas_data_preprocessing`.
* ``--trainset_list_path``: 
*Optional.* Path to an ``.npy`` file specifying a subset of datasets for training.

* ``--save_folder``: The destination directory where the output parquet and related files will be stored.
* ``--check_ct_col``: The metadata column in ``adata.obs`` representing cell type annotations. This column will be used for visualization purposes to verify that the UGE correctly aligns with known cell types during training.
* ``--skip_check_umap``: 
*Flag.* If set, skips the time-consuming UMAP verification of the intersection between dataset HVGs and the credible gene set.


Output Files
------------

Successful execution will generate the following files in your ``--save_folder``:

* **Training Data**:
    * ``cls_prefix_data.parquet``: Main training data in parquet format.
    * ``CellTypeMapping_df.csv``: Mapping of cell type labels to numeric IDs.
    * ``obs_concat.csv``: Concatenated metadata for the training set.



.. code-block:: bash

   python transformer_parquet_valid.py \
       --base_path /path/to/reproducibility/ \
       --gene_path /path/to/reproducibility \
       --traingene_path /path/to/reproducibility \
       --vocab_path /path/to/reproducibility \
       --tissue '' \
       --data_path /path/to/reproducibility/log_scale \
       --trainset_list_path /path/to/reproducibility/HLCA_data_files_sl.npy \
       --save_folder Training_input \
       --subsample_frac 0.07

**Parameter Descriptions**

.. note::
   This script functions is almost the same as ``transformer_parquet.py``, but is specifically designed to prepare a validation subset. This subset is used for UMAP visualization to monitor UGE quality during training.

``--base_path``, ``--gene_path``, ``--traingene_path``, ``--vocab_path``, ``--tissue``, ``--data_path``, ``--trainset_list_path``, ``--save_folder``
    These parameters function identically to those described in the :doc:`transformer_parquet` section.

``--subsample_frac``
    The fraction of the training data (0.0 to 1.0) to be randomly sampled. This subsampled data is used to generate UMAP visualizations for monitoring UGE performance during the training process.
    
    
    
    

Preparing config.yaml file for training
----------------------------

Please download templete (``config_templete_train.yaml`` or ``config_templete_inference.yaml``) from `Zenodo <https://doi.org/10.5281/zenodo.19750491>`_ .

Run the following under ``./02_Generate_training_input`` directory to modify parameters in templete using the provided shell script:


.. code-block:: 

   ./generate_configs_train.sh
   ./generate_configs_inference.sh



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

