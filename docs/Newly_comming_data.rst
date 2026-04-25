Zero-shot cell type annotation for newly comming data
========

For new datasets at single-cell resolution—whether derived from dissociated single-cell sequencing or spatial transcriptomics—the same preprocessing pipeline can be followed to generate Parquet input files. Once the input is prepared, use ``Gen_UGE.py`` to generate the UGE. 

You can download the necessary files for this step (``Training_input_testdata.zip``) from `Zenodo (DOI: 10.5281/zenodo.19750491) <https://doi.org/10.5281/zenodo.19750491>`_ or you can directly download the processed UGE for the test data (``Training_output/Testdata_UGE.h5ad``).

These UGE can then be mapped onto the existing UGE atlas to achieve zero-shot cell type annotation by leveraging the high-resolution labels of the reference atlas.


Mapping New Data to the UGE Atlas
---------------------------------

Navigate to the ``./05_UGE_celltype_annotation`` directory and execute the mapping script:

.. code-block:: bash

   ./transformer_map_to_atlas.sh

The shell script executes the following Python command:

.. code-block:: bash

    python transformer_map_to_atlas.py \
        --atlas_path /path/to/atlas_1e3_maskp5_processed.h5ad \
        --adata_inte_path /path/to/Testdata_UGE.h5ad \
        --fitted_NN_path /path/to/NN_atlas_1e3_maskp5 \
        --recompute_pca \
        --save_nn_path /path/to/NN_testdata_mapped_results \
        --atlas_assign_label_col ann_finest_level

**Parameter Descriptions**

``--atlas_path``
    Path to the reference UGE atlas file (``.h5ad``).
``--adata_inte_path``
    Path to the UGE file of the new (query) dataset to be annotated.
``--fitted_NN_path``
    Path to save fitted Nearest Neighbor (NN) model for the reference atlas.
``--recompute_pca``
    *Flag.* If set, the script will recompute the PCA space for UGE atlas.
``--save_nn_path``
    The directory where the mapping results and the updated NN model will be saved.
``--atlas_assign_label_col``
    The metadata column in the reference atlas (e.g., ``ann_finest_level``) that will be used to assign labels to the new data.
    
    
Verification and Visualization
------------------------------

After completing the mapping and zero-shot annotation, you can verify and visualize the results using the provided validation notebook:

.. toctree::
   :maxdepth: 1

   vignettes/Testdata_mapped_results.ipynb