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
