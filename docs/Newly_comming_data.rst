Mapping new data to the UGE atlas
=================================

UniGeneX maps a new single-cell or spatial-transcriptomics dataset to a
reference UGE atlas and transfers an atlas annotation to each query cell. The
public implementation is provided by ``UniGeneX.mapping.AtlasMapper`` in
``UniGeneX/UniGeneX/mapping.py``. The command-line program in
``UniGeneX/05_UGE_celltype_annotation/`` is a thin interface around the same
class.

The query must first be processed with the UniGeneX inference workflow to
generate an AnnData object containing UGE values. Atlas and query gene names
must be unique, and the query must contain the atlas genes used for mapping.

Example data
------------

The tutorial uses the processed UGE atlas, a held-out Human Lung Cell Atlas
query and its compact mapping result. Download ``Training_output.zip``,
``Training_input.zip`` and ``Training_input_testdata.zip`` from the
`UniGeneX Zenodo deposit <https://doi.org/10.5281/zenodo.19750490>`_. The
concept DOI resolves to the latest release; record ``19750491`` is the initial
version-specific release used by this tutorial.

After extracting the files, define the directory containing the three
folders before opening the notebook:

.. code-block:: bash

   export UNIGENEX_REPRO_DATA=/path/to/reproducibility

The main inputs are:

.. code-block:: text

   reproducibility/
   ├── Training_output/
   │   ├── atlas_1e3_maskp5_processed.h5ad
   │   ├── Testdata_UGE.h5ad
   │   ├── NN_atlas_1e3_maskp5/
   │   └── NN_mapped_results/query_to_atlas_mapping.h5ad
   ├── Training_input/obs_concat.csv
   └── Training_input_testdata/obs_concat.csv

Python API
----------

Load the atlas, query and a compatible fitted atlas index, then call
``map()``:

.. code-block:: python

   import scanpy as sc
   from UniGeneX.mapping import AtlasMapper

   atlas = sc.read_h5ad("/path/to/atlas_1e3_maskp5_processed.h5ad")
   query = sc.read_h5ad("/path/to/Testdata_UGE.h5ad")

   mapper = AtlasMapper(
       atlas,
       label_column="ann_finest_level",
       n_neighbors=30,
       keep_unlabeled_atlas=False,
   ).load_index("/path/to/NN_atlas_1e3_maskp5")

   mapping = mapper.map(query)
   mapping.write_h5ad("/path/to/NN_mapped_results/query_to_atlas_mapping.h5ad")

``AtlasMapper`` parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - Meaning
   * - ``atlas_adata``
     - Reference AnnData containing atlas UGE values and annotations.
   * - ``label_column``
     - Atlas ``obs`` column transferred to query cells by majority vote.
   * - ``n_neighbors``
     - Number of cosine-nearest atlas cells used for each query cell.
   * - ``keep_unlabeled_atlas``
     - If ``False``, atlas cells missing ``label_column`` are excluded. If
       ``True``, they remain in the mapping coordinates but do not contribute
       a label vote.

Index and mapping methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - Use
   * - ``fit(recompute_pca=False)``
     - Fit a cosine-neighbor index using the PCA already stored in the atlas.
       Set ``recompute_pca=True`` to calculate a new 30-component atlas PCA
       first.
   * - ``load_index(path)``
     - Load a previously fitted atlas PCA and neighbor index. Use this when
       mapping additional queries to the same released atlas.
   * - ``save_index(path)``
     - Save the fitted PCA arrays, neighbor model, indices and distances for
       reuse.
   * - ``map(query_adata)``
     - Align query genes, project query UGE values into atlas PCA space, find
       atlas neighbors and return the compact mapping result.

To construct a new atlas index instead of loading one:

.. code-block:: python

   mapper = AtlasMapper(
       atlas,
       label_column="ann_finest_level",
       n_neighbors=30,
   ).fit(recompute_pca=True)

   mapper.save_index("/path/to/new_atlas_index")
   mapping = mapper.map(query)

The fitted mapper can be reused for multiple query AnnData objects without
rebuilding the reference index.

Command-line interface
----------------------

For an unattended or long-running mapping job, edit the paths in the supplied
shell launcher and run it from the public repository:

.. code-block:: bash

   cd UniGeneX/05_UGE_celltype_annotation
   bash transformer_map_to_atlas.sh

The equivalent direct command for a released atlas index is:

.. code-block:: bash

   python transformer_map_to_atlas.py \
       --atlas-path /path/to/atlas_1e3_maskp5_processed.h5ad \
       --query-path /path/to/Testdata_UGE.h5ad \
       --atlas-index-path /path/to/NN_atlas_1e3_maskp5 \
       --output-path /path/to/NN_mapped_results \
       --atlas-label-column ann_finest_level \
       --n-neighbors 30 \
       --use-existing-index

Use ``--recompute-pca`` instead of ``--use-existing-index`` when a new atlas
PCA and index must be fitted. These two flags are mutually exclusive. If
neither flag is supplied, the mapper fits an index from the PCA already stored
in the atlas.

CLI parameters
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning
   * - ``--atlas-path``
     - Reference atlas AnnData file.
   * - ``--query-path``
     - Query UGE AnnData file to map.
   * - ``--atlas-index-path``
     - Directory from which the atlas index is loaded or to which a newly
       fitted index is saved.
   * - ``--output-path``
     - Directory for ``query_to_atlas_mapping.h5ad`` and the mapping log.
   * - ``--atlas-label-column``
     - Atlas ``obs`` annotation transferred to query cells.
   * - ``--n-neighbors``
     - Number of atlas neighbors; the default is 30.
   * - ``--use-existing-index``
     - Load the index from ``--atlas-index-path``.
   * - ``--recompute-pca``
     - Recalculate atlas PCA before fitting and saving a new index.
   * - ``--keep-unlabeled-atlas``
     - Keep atlas cells whose transfer label is missing.

Mapping result
--------------

``AtlasMapper.map()`` and the CLI both produce a compact AnnData object:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Location
     - Contents
   * - ``X``
     - Sparse query-to-atlas fuzzy connectivity matrix; rows are query cells
       and columns are atlas cells.
   * - ``obs``
     - Query metadata, ``predict_mapped_<label_column>`` and the corresponding
       ``_prob`` neighbor-vote confidence.
   * - ``var``
     - Atlas cell identifiers and the transferred annotation.
   * - ``obsm``
     - Nearest-atlas-cell indices and cosine distances for every query cell.
   * - ``uns``
     - Mapping parameters and a description of the matrix semantics.

The confidence value is the fraction of labeled nearest neighbors supporting
the majority-vote label. It is useful for filtering weak assignments, but it
is not a calibrated probability.

Tutorial: projection and evaluation
-----------------------------------

The following notebook loads or generates the mapping, validates cell
alignment, projects held-out query cells onto the atlas UMAP and evaluates the
transferred labels.

.. toctree::
   :maxdepth: 1

   vignettes/Testdata_mapped_results.ipynb

Real-data application
---------------------

For an end-to-end example using condition-specific lung references, three
Xenium samples and atlas-derived spatial pseudotime, continue to the
:doc:`Pulmonary_fibrosis` tutorial.
