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

import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal
from scipy.sparse import issparse


import sys
# sys.path.append('/home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/lung')
# from utils import *
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
        "--transformer_out_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--save_atlas_name",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--traingene_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--vocab_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--not_pca_umap",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
args = parser.parse_args()


import glob
import logging
def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger




########################################
# define paths
########################################
data_path = Path(args.base_path) / args.tissue
data_path.mkdir(parents=True, exist_ok=True)

save_path = Path(args.base_path) / args.save_folder
save_path.mkdir(parents=True, exist_ok=True)


loggings = configure_logging(str(Path(save_path) / 'UGE_to_adata_log'))
loggings.info(f'save_path: {save_path}')



########################################
# define paths
########################################
if args.traingene_path is not None:
    common_dec_genes = np.load(Path(args.traingene_path) / 'pretrain_data_train_genes.npy')    
else:
    common_dec_genes = np.load(Path(data_path) / 'pretrain_data_train_genes.npy')
loggings.info(f'common_dec_genes len: {common_dec_genes.shape[0]}')
loggings.info(f'common_dec_genes first 3: {common_dec_genes[:3]}')



ckpt_path = Path(args.transformer_out_path)

import glob as gb

resfiles = gb.glob(str(ckpt_path / 'sample_adata') + '/*gene_out_adata*.h5ad')
for f in resfiles:
    adata_inte = sc.read(f)
    adata_inte.var.index = common_dec_genes
#         adata_tmp = sc.read_h5ad(f, backed = 'r')
            
loggings.info(f'loaded raw adata_inte shape: {adata_inte.shape}')
loggings.info(f'loaded raw adata_inte X max: {adata_inte.X.max()}')


loggings.info(f'load obs_concat ...')
save_parquet_path = Path(data_path)
obs_concat_save_name = 'obs_concat.csv'
obs_concat = pd.read_csv(save_parquet_path / obs_concat_save_name, index_col = 0)

# adata_inte_sub = subsample_data(adata_inte, 200000)
adata_inte_sub = adata_inte

adata_inte_sub.obs['dataset_cell_id'] = [f'{i}_{int(j)}' for i, j in zip(adata_inte_sub.obs['dataset_idx'], adata_inte_sub.obs['cell_id'])]
adata_inte_sub.obs.index = adata_inte_sub.obs['dataset_cell_id'].tolist()
adata_inte_sub = adata_inte_sub[~adata_inte_sub.obs.index.duplicated(keep='first')]


obs_concat_sub = obs_concat.loc[adata_inte_sub.obs.index]
# adata_inte_sub.obs = adata_inte_sub.obs.merge(obs_concat_sub, left_index = True, right_index = True, suffixes = (None, '_y')) 
# adata_inte_sub.obs['cell_type'] = adata_inte_sub.obs['cell_type'].fillna('unknown')
# adata_inte_sub.obs['dataset_idx'].value_counts()

atlas_adata = adata_inte_sub

if not args.not_pca_umap:
    loggings.info(f'start pca and umap ...')
    sc.tl.pca(atlas_adata, n_comps=30,use_highly_variable=False) #svd_solver='arpack', n_comps=10, use_highly_variable=False)
    sc.pp.neighbors(atlas_adata, metric='cosine', n_neighbors=30, n_pcs = 30)
    sc.tl.umap(atlas_adata, min_dist = 0.3, spread = 1, maxiter=100)

########################################
# save adata_inte_sub
########################################
save_adata_inte_path = Path(save_path) / args.save_atlas_name
loggings.info(f'save adata_inte_sub to {str(save_adata_inte_path)}')
atlas_adata.write(save_adata_inte_path)
obs_concat_sub.to_csv(Path(save_path) / 'obs_concat_atlas.csv')


