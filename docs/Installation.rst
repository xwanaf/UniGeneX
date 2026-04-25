.. UniGeneX documentation master file, created by
   sphinx-quickstart on Fri Apr 24 19:33:14 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
======================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Create new conda environment
-------------------

.. code-block:: bash

    conda create -n unigenex_env python=3.10
    conda activate unigenex_env

Install Dependencies
--------------------
.. code-block:: bash
    pip install -r requirements.txt
The following step may take a while. Please refer to the `FlashAttention GitHub <https://github.com/dao-ailab/flash-attention>`_ for more details:

.. code-block:: bash

   pip install flash-attn==2.5.5 --no-build-isolation

If you are using Jupyter Notebook, you can add your environment as a kernel:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name unigenex_env --display-name "unigenex_env"
   
   
To exactly reproduce the results, please copy _highly_variable_genes.py from `Zenodo <https://doi.org/10.5281/zenodo.19751716>`_ and then copy the file to conda env path. 


This step is since modern Scanpy versions have updated their default settings, we provide a modified ``_highly_variable_genes.py`` file.
.. code-block:: bash

  SCANPY_DIR=$(python -c "import scanpy; print(scanpy.__path__[0])")
  echo $SCANPY_DIR
  cp /path/to/_highly_variable_genes.py <SCANPY_PATH>/preprocessing/_highly_variable_genes.py


.. warning::
   This step overwrites a core Scanpy file. We recommend performing this only inside a dedicated conda environment (e.g., ``unigenex_env``) to avoid affecting your other projects.
