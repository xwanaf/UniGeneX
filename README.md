# UniGeneX

[Documentation Status](https://unigenex.readthedocs.io/en/latest/)

[DOI](https://doi.org)

**UniGeneX** is a single-cell foundation model designed to construct a comprehensive Universal Gene Expression (UGE) atlas to uncover underlying cell-state transitions and associated microenvironments in human diseases. The framework consists of two main stages—training and inference—and features three novel characteristics: it is **context-specific**, **interpretable**, and **actionable**. 

The UGE atlas serves as a comprehensive reference dataset that integrates multiple data types for various downstream analyses, such as the deconvolution of bulk RNA-seq and spatial transcriptomics data across various resolutions. By integrating a highly interpretable UGE atlas with spatial transcriptomics data, UniGeneX provides a powerful framework for deciphering cell-state transitions during disease progression and their associated microenvironmental changes.

![Figure 1](docs/_static/Fig1.png)  
*Figure 1: Overview of the UniGeneX framework and its applications in disease characterization.*

## 🛠️ Installation

### 1. Create and Activate the Environment
```bash
conda create -n unigenex_env python=3.10
conda activate unigenex_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The following step may take some time. Please refer to the [FlashAttention GitHub](https://github.com/dao-ailab/flash-attention) for technical details:

```bash
pip install flash-attn==2.5.5 --no-build-isolation
```

### 3. Ensure Reproducibility (Scanpy & Fonts)
To ensure reproducibility, download _highly_variable_genes.py from [Zenodo](https://doi.org/10.5281/zenodo.19751716). 

Because default settings in recent Scanpy versions have changed, replacing the `_highly_variable_genes.py` file in your environment is required to match our original results.

```bash
# 1. Locate your Scanpy installation path
SCANPY_DIR=$(python -c "import scanpy; print(scanpy.__path__[0])")
echo $SCANPY_DIR

# 2. Overwrite the default Scanpy file with the downloaded version
# (Replace '/path/to/' with the actual directory of your downloaded file)
cp /path/to/_highly_variable_genes.py $SCANPY_DIR/preprocessing/_highly_variable_genes.py
```

## 📖 Core Workflow

1.  **Preprocessing**: Process large-scale training data in raw count format, construct a credible gene set, and prepare `.parquet` files for the training pipeline.
2.  **Training**: Train the transformer-based model using ``03_Training_src/train_UniGeneX.py``. Construct the UGE atlas and assemble it into a unified ``.h5ad`` atlas.
3.  **Zero-Shot Annotation**: Map new datasets—either single-cell or spatial—to the reference atlas for automated cell-type labeling.
4.  **Downstream Analysis**: Deconvolution, cell-state transitions during disease progression, and so on.

## 📂 Data Availability

All intermediate input and output files are available on [Zenodo](https://doi.org/10.5281/zenodo.19751716) and [Zenodo](https://doi.org/10.5281/zenodo.19750491)


## 📊 Documentation

For full tutorials, vignettes, and parameter descriptions, please visit the:  
[**UniGeneX Documentation**](https://unigenex.readthedocs.io/en/latest/).
