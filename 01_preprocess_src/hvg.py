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
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
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
        "--n_top_genes",  
        type=int,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--ignore_batch_key",
        action="store_true",
        help="whether quantize final gene emb.",
    )
parser.add_argument(
        "--hvg_batch_key",  
        type=str,
        default='donor_id',
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
    
        f = data_path / f'{file_name}.h5ad'
        loggings.info(f'processing {f}')
        adata = sc.read(f)
        loggings.info(f'adata shape: {adata.shape}')
        loggings.info(f'adata X max: {adata.X.max()}')
        

        n_top_genes = min(args.n_top_genes, adata.shape[1])

        if adata.shape[1] > n_top_genes:
            if args.ignore_batch_key:
                hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
            else:
                if (args.hvg_batch_key in adata.obs.columns):
                    if (adata.obs[args.hvg_batch_key].unique().shape[0] > 1):
                        loggings.info(f'select hvg with batch_key {args.hvg_batch_key}')
                        try:
                            adata = adata[adata.obs[args.hvg_batch_key].isin(adata.obs[args.hvg_batch_key].value_counts()[(adata.obs[args.hvg_batch_key].value_counts() > 500)].index.tolist())]
                            hvg_df = sc.pp.highly_variable_genes(adata, batch_key = args.hvg_batch_key, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                        except:
                            loggings.info(f'failed to apply batch_key, select hvg from all cells')
                            hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                    else:
                        hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                        
                elif ('sample' in adata.obs.columns):
                    if (adata.obs['sample'].unique().shape[0] > 1):
                        loggings.info(f'select hvg with batch_key sample')
                        try:
                            adata = adata[adata.obs['sample'].isin(adata.obs['sample'].value_counts()[(adata.obs['sample'].value_counts() > 500)].index.tolist())]
                            hvg_df = sc.pp.highly_variable_genes(adata, batch_key = 'sample', min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                        except:
                            loggings.info(f'failed to apply batch_key, select hvg from all cells')
                            hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                    else:
                        hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                
                else:
                    hvg_df = sc.pp.highly_variable_genes(adata, batch_key = None, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes, inplace = False)
                
            hvg_df.index = adata.var.index.tolist()
#             sc.pl.highly_variable_genes(adata)
#             adata = adata[:, adata.var.highly_variable].copy()
#             loggings.info(f'adata after hvg shape: {adata.shape}')



#         adata.write(save_path / f'{file_name}.h5ad')
        hvg_df.to_csv(save_path / f'{file_name}.csv')
        del adata
    


