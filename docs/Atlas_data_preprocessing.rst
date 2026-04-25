
Atlas Data Preprocessing
========================

This section describes how to prepare the Human Lung Cell Atlas (HLCA) data for UniGeneX training and inference.

Preprocessing raw count data
--------------------

You can downloaded the raw count data from `Zenodo <https://doi.org/10.5281/zenodo.19751716>`_ and also refer to the original paper https://www.nature.com/articles/s41591-023-02327-2

The preprocessing workflow follows standard single-cell transcriptomics procedures, including filtering, normalization, log transformation, and highly variable gene (HVG) selection.


You can execute the entire workflow from the ``./01_preprocess_src`` directory using the provided shell script:

.. code-block:: bash

    ./preprocess_data.sh
    
    
The shell script sequentially calls two main functions:

1. Filtering and Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``filter.py`` to remove low-quality cells and perform log-scale normalization:

.. code-block:: bash

   python filter.py \
       --base_path /parent/path/to/raw \
       --tissue '' \
       --data_folder raw \
       --save_folder log_scale 

**Parameters:**

* ``--base_path``: The root directory to data.
* ``--tissue``: The specific subfolder. The full path to the data will be ``base_path / tissue``.
* ``--data_folder``: The folder containing raw ``.h5ad`` files. Whole input path: ``base_path / tissue / data_folder``.
* ``--save_folder``: The folder where processed data will be stored. Output path: ``base_path / tissue / save_folder``.


2. Highly Variable Gene (HVG) Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``hvg.py`` to identify HVGs for each dataset.

.. code-block:: bash

   python hvg.py \
       --base_path /path/to/reproducibility \
       --tissue '' \
       --hvg_batch_key donor_id \
       --data_folder log_scale \
       --save_folder hvg10k \
       --n_top_genes 10000

**Parameters:**

* ``--base_path``: Same as ``filter.py``.
* ``--tissue``: Same as ``filter.py``.
* ``--hvg_batch_key``: The column name in ``adata.obs`` used to account for batch effects during HVG selection (e.g., ``donor_id``).
* ``--data_folder``: Same as ``filter.py``.
* ``--save_folder``: Same as ``filter.py``.
* ``--n_top_genes``: The number of highly variable genes to retain (default is 10,000).




    
Construct credible gene set
--------------------

The following notebook demonstrates how to construct a **credible gene set**.

**Data Requirements**
You can download the required input files (``Xenium_panel.npy`` or ``Xenium_Vannan_panel.npy``) and the output (``pretrain_data_train_genes.npy``) from `Zenodo <https://doi.org/10.5281/zenodo.19750491>`_ .


.. toctree::
   :maxdepth: 1

   vignettes/select_genes_2k
