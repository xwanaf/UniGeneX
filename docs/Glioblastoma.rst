Glioblastoma application
========================

This tutorial shows how a UniGeneX universal gene expression (UGE) atlas can
serve as a single-cell reference for deconvolving low-resolution Visium data.
It reproduces the computational results in Figures 5B--C and 6C--D: the
integrated glioma UGE atlas, conserved tumor states, and their inferred spatial
distributions in two glioblastoma sections.

Figure 5A is a conceptual workflow and is not generated computationally.

Data and setup
--------------

Download the modular archives from the
`UniGeneX GBM Zenodo record <https://doi.org/10.5281/zenodo.22566537>`_.
Extract the required archives into the same directory, then set the data root
to the directory containing ``GBM/``.

.. code-block:: bash

   bash extract_gbm_data.sh /path/to/unigenex_data
   export UNIGENEX_DATA_DIR=/path/to/unigenex_data

.. list-table::
   :header-rows: 1
   :widths: 42 24 34

   * - Archive
     - Used by
     - Contents
   * - ``GBM_UGE_atlas_v1.0.0.zip``
     - Notebooks 01--02
     - UGE expression, annotations and atlas coordinates
   * - ``GBM_Visium_v1.0.0.zip``
     - Notebook 02
     - Two Visium sections with counts, coordinates and tissue images
   * - ``GBM_precomputed_cell2location_v1.0.0.zip``
     - Notebook 02
     - UGE signatures and precomputed spatial abundance estimates

The archives merge into one data tree:

.. code-block:: text

   GBM/
   ├── UGE_atlas.h5ad
   ├── visium/
   │   ├── AT15-BRA-4-FO-C2-S27/spatial.h5ad
   │   └── AT10-BRA-5-FO-1_2/spatial.h5ad
   └── precomputed/
       ├── reference/UGE_cell_state_signatures.csv
       └── cell2location/<sample>/cell_abundance_q05.csv

UGE atlas-guided Visium deconvolution
-------------------------------------

The analysis has four stages:

1. Select normal and tumor states from the glioma UGE atlas that are
   biologically plausible in the Visium tissue.
2. Fit one expression signature for each selected UGE state.
3. Match the Visium genes to those signatures and estimate state abundance at
   every sufficiently covered, in-tissue spot.
4. Visualize individual states or combine related states into the spatial
   programs shown in Figure 6C--D.

Reference selection is dataset-specific. An omitted population can be
absorbed into a similar state, whereas irrelevant or poorly represented states
can make the decomposition difficult to interpret. Notebook 02 therefore
keeps the GBM-specific ``prepare_uge_reference`` step in the tutorial instead
of presenting it as a general software function.

Cell2location functions
-----------------------

The reusable functions are defined in
``Real_data_GBM/deconvolve_visium.py``. Both model-fitting functions require a
CUDA GPU; CPU-only systems should use the released precomputed results.

Fit UGE cell-state signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit_cell_state_signatures`` fits the Cell2location regression model to a
prepared, count-like single-cell reference and returns a genes-by-states
signature table.

.. code-block:: python

   signatures = fit_cell_state_signatures(
       reference,
       labels_key="cell_state",
       output_path=signature_path,
       max_epochs=250,
       batch_size=2500,
       learning_rate=0.002,
       posterior_samples=1000,
       seed=0,
   )

.. list-table::
   :header-rows: 1
   :widths: 23 23 54

   * - Parameter
     - Figure 6 value
     - Meaning
   * - ``reference``
     - Prepared UGE reference
     - Count-like cells-by-genes ``AnnData`` after state and gene selection
   * - ``labels_key``
     - ``cell_state``
     - Reference ``.obs`` column defining the 19 signatures
   * - ``output_path``
     - Signature CSV
     - Optional destination for the reusable genes-by-states table
   * - ``max_epochs``
     - 250
     - Maximum optimization passes through the reference data
   * - ``batch_size``
     - 2,500
     - Reference cells per training minibatch
   * - ``learning_rate``
     - 0.002
     - Optimizer step size
   * - ``posterior_samples``
     - 1,000
     - Posterior draws used to summarize each signature
   * - ``seed``
     - 0
     - Random seed for initialization and posterior sampling

Prepare spatial expression
~~~~~~~~~~~~~~~~~~~~~~~~~~

``prepare_spatial`` is called internally by ``deconvolve_spatial``. It loads
one spatial ``AnnData``, intersects its genes with the signature table,
retains ``in_tissue`` spots, converts the expression matrix to integer sparse
counts, and filters low-coverage spots.

.. list-table::
   :header-rows: 1
   :widths: 23 23 54

   * - Parameter
     - Figure 6 value
     - Meaning
   * - ``spatial_path``
     - One Visium H5AD
     - Counts, spot coordinates, tissue image and spatial metadata
   * - ``signatures``
     - 1,867 genes by 19 states
     - UGE reference signatures; genes are matched by name
   * - ``minimum_counts``
     - 1,000
     - Minimum spot count after signature-gene matching

Deconvolve spatial data
~~~~~~~~~~~~~~~~~~~~~~~

``deconvolve_spatial`` fits the spatial Cell2location model and returns the
q05 abundance table: the conservative 5% posterior quantile for every state
at every retained spot.

.. code-block:: python

   abundance = deconvolve_spatial(
       spatial_path=spatial_path,
       signatures=signatures,
       output_path=abundance_path,
       minimum_counts=1000,
       cells_per_location=20,
       detection_alpha=200,
       use_bg=True,
       max_epochs=30000,
       posterior_samples=1000,
       seed=0,
   )

.. list-table::
   :header-rows: 1
   :widths: 23 23 54

   * - Parameter
     - Figure 6 value
     - Meaning
   * - ``spatial_path``
     - One Visium H5AD
     - Spatial expression input
   * - ``signatures``
     - UGE signature table
     - DataFrame with genes in rows and reference states in columns
   * - ``output_path``
     - Abundance CSV
     - Optional destination for the q05 spot-by-state table
   * - ``minimum_counts``
     - 1,000
     - Coverage threshold applied after gene matching
   * - ``cells_per_location``
     - 20
     - Prior mean number of cells represented by a Visium spot
   * - ``detection_alpha``
     - 200
     - Detection-efficiency prior concentration; larger values keep
       spot-level sensitivity closer to the sample mean
   * - ``use_bg``
     - ``True``
     - Enable the UniGeneX gene-specific background component
   * - ``max_epochs``
     - 30,000
     - Maximum spatial-model training epochs
   * - ``posterior_samples``
     - 1,000
     - Posterior draws used to calculate abundance summaries
   * - ``seed``
     - 0
     - Random seed for initialization and posterior sampling

The supplied Cell2location variant is version
``0.1.5+unigenex.gbmv1``. With ``use_bg=True``, it derives a gene-frequency
profile from the spatial counts and uses that profile to initialize an
additive background term. This is the modification used for the manuscript
Figure 6 analysis.

Precomputed results and GPU rerunning
-------------------------------------

Notebook 02 loads the Zenodo signatures and abundance tables by default:

.. code-block:: python

   RUN_REFERENCE = False
   RUN_DECONVOLUTION = False

This reproduces the spatial visualizations without fitting a model. To rerun
both model stages in the notebook, set both values to ``True`` and use a CUDA
GPU.

For a long spatial fit, first generate or download the signature CSV, edit the
clearly labeled ``gbm_dir`` and ``python_bin`` values at the top of
``Real_data_GBM/run_cell2location.sh``, change the GPU index if needed, and
run:

.. code-block:: bash

   bash Real_data_GBM/run_cell2location.sh

The shell runner performs the two Visium fits outside Jupyter; it does not
rebuild the UGE reference signatures. After it finishes, keep
``RUN_DECONVOLUTION = False`` and continue with the visualization section.

Outputs and figures
-------------------

The workflow writes the reusable signature table, one q05 abundance table per
Visium sample, overview grids for all 19 states, and the composite spatial
program maps corresponding to Figure 6C--D. Input H5AD files are not
overwritten.

Tutorial notebooks
------------------

The notebooks below contain saved outputs and are not executed by Read the
Docs. To run them, use the canonical copies and helper files in
``Real_data_GBM/``.

.. toctree::
   :maxdepth: 1
   :numbered:

   vignettes/GBM/01_glioma_uge_atlas.ipynb
   vignettes/GBM/02_uge_atlas_guided_visium.ipynb
