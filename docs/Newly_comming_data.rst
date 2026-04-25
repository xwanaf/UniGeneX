Zero-shot cell type annotation for newly comming data
========

After training is complete, the following steps generate UGE of trainnig data in ``.h5ad`` format. You can also directly download processed UGE atlas (``Training_output/atlas_1e3_maskp5_processed.h5ad``) from `Zenodo <https://doi.org/10.5281/zenodo.19750491>`_ .


Navigate to the ``./04_Gen_UGE`` directory and execute the following script.

.. code-block:: bash

   ./Gen_UGEatlas.sh 2>&1 | tee debug.log


The shell script calls the following:

.. code-block:: bash

   python Gen_UGE.py \
       --config /path/to/config_inference.yaml \
       -s /path/to/Inference_output_directory \
       --pretrain_root /path/to/Training_output_directory \
       --eval_batch_size 256 \
       --subsample 1000000 \
       --custom_ckptpath \
       --ckpt_filename "reproduce_epoch30.ckpt"

**Key Parameters:**
* ``--config``: Path to the YAML inference configuration file, as generated in the :doc:`Preparing_training_input` section.
* ``-s`` (or ``--save_dir``): The directory where the generated output will be stored.
* ``--pretrain_root``: The directory containing the saved training model checkpoints.
* ``--eval_batch_size``: The number of cells processed per batch during inference.
* ``--subsample``: The maximum number of cells to process (e.g., 1,000,000).
* ``--custom_ckptpath``: *Flag.* If set, the script uses the manual checkpoint path (``/pretrain_root/ckpt``). Otherwise, it defaults to the PyTorch Lightning path (``/pretrain_root/plainLogger/0.1/checkpoints``).
* ``--ckpt_filename``: The specific checkpoint filename to use for generating UGE.



Step 2: Convert UGE to AnnData
------------------------------

Once the UGE of training data are generated, use the following script to map them back to the original gene metadata and save the result as an integrated Atlas file.

.. code-block:: bash

   python UGE_to_adata.py \
       --base_path /path/to/reproducibility/ \
       --tissue 'Training_input' \
       --save_folder 'Training_output' \
       --traingene_path /path/to/reproducibility \
       --transformer_out_path /path/to/Inference_output_directory \
       --save_atlas_name atlas_1e3_maskp5.h5ad


**Key Parameters:**
* ``--base_path``, ``--tissue``: Used to construct the path (``base_path/tissue``) where the credible gene set (``pretrain_data_train_genes.npy``) and metadata (``obs_concat.csv``) are located.
* ``--traingene_path``: Explicit path to the directory containing ``pretrain_data_train_genes.npy``. If not specified, the script defaults to ``base_path/tissue``.
* ``--transformer_out_path``: Path to the directory containing the UGE output generated in Step 1.
* ``--save_atlas_name``: The filename for the final integrated ``.h5ad`` object.

