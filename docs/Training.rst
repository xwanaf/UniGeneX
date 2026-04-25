Training
========

Once the training input and configuration files have been prepared, you can initiate the model training using the following script. You can also directly download the whole output of this step (``Training_output``) from `Zenodo <https://doi.org/10.5281/zenodo.19750491>`_ .

Training Script
---------------
Navigate to the ``./03_Training_src`` directory and execute the training script. We recommend piping the output to ``tee`` to monitor the progress in real-time while saving a copy to a log file for debugging.

.. code-block:: bash

   ./train_KL1e-3_maskp5.sh 2>&1 | tee debug.log


The shell script calls the following:

.. code-block:: bash

   python /path/to/train_UniGeneX.py \
       --config /path/to/config_train.yaml \
       -s /path/to/Training_output/Generation_HLCA_KL1e-3_maskp5 \
       --num_workers 28


Execution Parameters
--------------------

``--config``
    The path to the YAML configuration file generated in :doc:`Preparing_training_input`). This file contains all hyperparameters and data paths.

``-s`` (or ``--save_dir``)
    The directory where model checkpoints, logs, figs output will be saved.

``--num_workers``
    The number of CPU worker processes to use for data loading. Higher values can speed up training but require more system memory.



Output Directory Structure
--------------------------

After starting the training, the save directory (``-s``) will contain the following files and folders:

.. code-block:: text

   ckpt/          # Model checkpoints (.ckpt) saved during training for resuming or inference.
   log.log        # Detailed text logs containing training progress and system messages.
   plainLogger/   # Structured logs (e.g., TensorBoard or CSV) for monitoring loss and metrics.
   sample_adata/  # Processed AnnData objects or metadata samples used during validation.
   sample_img/    # Visualization outputs, such as UGE UMAP plots, to monitor UGE quality.
