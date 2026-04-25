
Atlas Data Preprocessing
========================

This section describes how to prepare the Human Lung Cell Atlas (HLCA) data for UniGeneX training and inference.

Preprocessing raw count data
--------------------

You can downloaded the raw count data from `Zenodo (DOI: 10.5281/zenodo.19751716) <https://zenodo.org>`_ and also refer to the original paper https://www.nature.com/articles/s41591-023-02327-2


Run the following from ./01_preprocess_src. This step follows standard preprocessing steps for single-cell transcriptomics data: filter, normalize, log transformed and also select hvgs.

.. code-block:: bash
    ./preprocess_data.sh
    
    
Construct credible gene set
--------------------

The following notebook demonstrates how to construct the credible gene set by integrating Xenium gene panels and existing vocabulary files:

.. toctree::
   :maxdepth: 1

   _templates/select_genes_2k.ipynb
