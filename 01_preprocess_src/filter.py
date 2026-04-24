# build large-scale data in scBank format from a group of AnnData objects
# %%
import gc
import json
from pathlib import Path
import argparse
import shutil
import traceback
from typing import Dict, List, Optional
import warnings
import numpy as np
import os
import anndata
import glob

import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal
from scipy.sparse import issparse


import sys
sys.path.append('/home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/Reproducibility/Validation')
from UniGeneX.utils import *



import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--tissue",  
        type=str,
        default=None,
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--base_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--data_folder",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--save_folder",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--do_not_take_raw",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--do_not_log",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--filter_genes_min_cells",  
        type=int,
        default=3,
        help="Decoder type.",
    )
parser.add_argument(
        "--filter_cells_min_genes",  
        type=int,
        default=300,
        help="Decoder type.",
    )
args = parser.parse_args()


base_path = args.base_path + args.tissue 
data_path = Path(base_path) / args.data_folder
save_path = Path(base_path) / args.save_folder
save_path.mkdir(parents=True, exist_ok=True)
data_files = glob.glob(str(data_path) + '/*.h5ad')

loggings = configure_logging(str(Path(base_path) / 'filter_hvg_log'))


for i, f in enumerate(data_files):
    loggings.info(f'============================== {i}-th data ===============================')
    file_name = Path(f).stem
    fname = save_path / f'{file_name}.h5ad'
    fname2 = save_path / f'{file_name}.csv'
    if os.path.isfile(fname) or os.path.isfile(fname2):
        loggings.info(f'{file_name} exists')
    else:
        loggings.info(f'processing {file_name}')
        adata = sc.read(f)
        adata.obs['cell_id_raw'] = np.arange(adata.shape[0]).astype(str).tolist()



        if 'feature_name' in adata.var.columns:
            adata.var.reset_index(inplace = True)
            adata.var.set_index('feature_name', inplace = True)


        loggings.info(f'adata shape: {adata.shape}')
        try:
            loggings.info(f'adata X max: {adata.X.max()}')
        except:
            pass

        if not args.do_not_take_raw:
            if adata.raw is not None:
                loggings.info(f'raw is not None')
                try:
                    adata.X = adata.raw.X
                except Exception as e:
                    print(f"An error occurred: {e}")
                loggings.info(f'after take raw')
                loggings.info(f'adata shape: {adata.shape}')
                loggings.info(f'adata X max: {adata.X.max()}')
            else:
                loggings.info(f'raw is None')

        if adata.X.max() > 30:
            adata = adata[:, [not _.startswith('MT-') for _ in adata.var.index.tolist()]]
            loggings.info(f'after remove MT- genes, adata shape: {adata.shape}')

#             sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_cells(adata, min_genes=300)
            sc.pp.filter_genes(adata, min_cells=max(3 * int(adata.shape[0] / 1e4), 3))
            
            adata.raw = adata
            loggings.info(f'adata after filter shape: {adata.shape}')
            loggings.info(f'adata X max: {adata.X.max()}')

            if args.keep_raw:
                adata.layers['raw_counts'] = adata.X.copy()
                adata.layers['log_scale'] = adata.X
            if not args.do_not_log:
                sc.pp.normalize_total(adata, target_sum = 1e4, inplace=True)
                sc.pp.log1p(adata)
    #             adata.X = adata.layers['log_scale']
                loggings.info(f'adata log max: {adata.X.max()}')



        adata.write(save_path / f'{file_name}.h5ad')
        del adata



