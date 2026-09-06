Pulmonary fibrosis application
==============================

This tutorial applies UniGeneX to a pulmonary-fibrosis (PF) atlas and spatial
transcriptomics data. It reproduces the computational results in Figures
3B--H and 4A, 4D and 4E: atlas characterization, spatial cell-type transfer,
an epithelial disease trajectory and spatial pseudotime on registered
histology.

The general mapping interface is described in :doc:`Newly_comming_data`. This
page focuses on the PF-specific inputs, parameter choices and order of
analysis.

Data and setup
--------------

Download the modular archives from the
`UniGeneX PF Zenodo record <https://doi.org/10.5281/zenodo.22489310>`_ and
extract them into one directory. Set the data root to the directory that
contains ``PF/`` before opening a notebook.

.. code-block:: bash

   export UNIGENEX_DATA_DIR=/path/to/zenodo/extraction

The workflow uses four groups of inputs:

.. code-block:: text

   PF/
   ├── UGE_atlas.h5ad
   ├── obs_concat_atlas.csv
   ├── spatial_annotation/
   │   ├── Reference_data/
   │   ├── fitted_reference_nn_normal/
   │   ├── fitted_reference_nn_disease/
   │   ├── query_UGE/
   │   └── query_raw/
   ├── selected_index/
   └── sp_map_to_sc_Epi/
       └── histology/

Each notebook begins with its exact input and output contract. Large inputs
remain outside the Git repository, and generated results are written below
``outputs/`` unless ``UNIGENEX_OUTPUT_DIR`` is set.

Workflow
--------

.. list-table::
   :header-rows: 1
   :widths: 8 34 42 16

   * - Step
     - Notebook
     - Purpose
     - Figure
   * - 1
     - Atlas and cell-type markers
     - Visualize the PF UGE atlas, selected markers and epithelial-state
       proportions.
     - 3B--D
   * - 2
     - Prepare spatial references
     - Document the historical normal/PF reference-selection procedure.
     - provenance
   * - 3
     - Spatial annotation results
     - Transfer atlas labels to three Xenium samples and compare them with
       measured marker expression.
     - 3E--H
   * - 4
     - Epithelial trajectory and spatial pseudotime
     - Reconstruct the atlas diffusion trajectory and transfer its
       AT1-to--Aberrant Basaloid pseudotime to spatial cells.
     - 4A, 4D
   * - 5
     - Spatial pseudotime on histology
     - Overlay the transferred pseudotime on the registered H&E section.
     - 4E

Notebook 02 explains how the references were selected, but the historical
random subsampling seed was not recorded. Exact figure reproduction therefore
uses the released normal and PF references and their matching neighbor-index
caches.

Mapping modes
-------------

The mapping notebooks support two interfaces:

* For a large or unattended job, set ``UNIGENEX_DATA_DIR``, run the relevant
  shell launcher once and leave ``RUN_MAPPING = False`` in the notebook.
* For an interactive demonstration, set ``RUN_MAPPING = True``. The notebook
  then calls ``UniGeneX.mapping.AtlasMapper`` directly and saves the same
  compact mapping format.

Loading saved mappings is the recommended documentation path because it
separates the long mapping calculation from visualization and interpretation.

Spatial cell-type annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, run:

.. code-block:: bash

   bash Real_data_PF/transformer_map_to_atlas.sh

The launcher maps one unaffected sample to the normal reference and two PF
samples to the disease reference. All three mappings use 30 cosine-nearest
atlas cells. Notebook 03 aligns the saved labels to original Xenium barcodes,
retains labels with neighbor-vote support greater than 0.5 for display, and
uses measured marker expression only as a biological validation layer.

Epithelial trajectory and pseudotime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the epithelial launcher with the same data root:

.. code-block:: bash

   bash Real_data_PF/transformer_map_to_epithelial_atlas.sh

This launcher creates two mappings of the epithelial query:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Result
     - Neighbors
     - Use
   * - ``knn30``
     - 30
     - Local projection of spatial cells onto the atlas trajectory in Figure
       4A.
   * - ``knn300``
     - 300
     - Smoother connectivity-weighted pseudotime transfer in Figure 4D.

Notebook 04 calculates pseudotime from the atlas expression-derived graph;
tissue coordinates are not used in that calculation. It writes
``spatial_pseudotime.csv.gz``, the compact handoff to notebook 05. Notebook 05
then uses tissue coordinates only to render the pseudotime surface over the
registered H&E image.

Generated outputs
-----------------

By default, each analysis writes to its own directory:

.. code-block:: text

   outputs/
   ├── 01_atlas_and_markers/
   ├── 03_spatial_annotation_results/
   ├── 04_epithelial_trajectory_and_spatial_pseudotime/
   └── 05_spatial_pseudotime_histology/

The notebooks save manuscript panels together with compact CSV, TSV or NPZ
files that record the plotted data and important parameters. Input atlases,
mapping results and histology files are not overwritten.

Tutorial notebooks
------------------

Run the notebooks in the order shown below. Notebook 05 depends on the compact
table generated by notebook 04; the other visualization notebooks can use the
released inputs or mappings independently.

.. toctree::
   :maxdepth: 1
   :numbered:

   vignettes/PF/01_atlas_and_markers.ipynb
   vignettes/PF/02_prepare_spatial_references.ipynb
   vignettes/PF/03_spatial_annotation_results.ipynb
   vignettes/PF/04_epithelial_trajectory_and_spatial_pseudotime.ipynb
   vignettes/PF/05_spatial_pseudotime_histology.ipynb
