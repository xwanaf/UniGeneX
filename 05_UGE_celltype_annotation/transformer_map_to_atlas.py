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

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from UniGeneX.utils import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--atlas_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--adata_inte_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--fitted_NN_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--recompute_pca",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--use_existing_nn",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--save_nn_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--atlas_assign_label_col",  
        type=str,
        default=None,
        help="Decoder type.",
    )
args = parser.parse_args()

########################################
# define paths
########################################
Path(args.fitted_NN_path).mkdir(parents=True, exist_ok=True)
Path(args.save_nn_path).mkdir(parents=True, exist_ok=True)
loggings = configure_logging(str(Path(args.save_nn_path) / 'transformer_map_to_atlas_log'))
loggings.info(f'save_nn_path: {args.save_nn_path}')


########################################
# map to atlas
########################################
loggings.info(f' ============================== start map to atlas ============================== ')

atlas_adata = sc.read(args.atlas_path)
atlas_adata.obs[args.atlas_assign_label_col] = atlas_adata.obs[args.atlas_assign_label_col].astype('category')
index = [_ is not np.nan for _ in atlas_adata.obs[args.atlas_assign_label_col]]
loggings.info(f'loaded atlas_adata shape: {atlas_adata.shape}')
atlas_adata = atlas_adata[index]
loggings.info(f'after atlas_adata filter nan in atlas_assign_label_col: {atlas_adata.shape}')

adata_inte_sub = sc.read(args.adata_inte_path)

same_gene = np.array_equal(atlas_adata.var.index, adata_inte_sub.var.index)
if not same_gene:
    overlap_genes = np.array(list(set(atlas_adata.var.index) & set(adata_inte_sub.var.index)))
    loggings.info(f'{overlap_genes.shape[0]} overlap_genes')
    atlas_adata = atlas_adata[:, overlap_genes]
    adata_inte_sub = adata_inte_sub[:, overlap_genes]
    loggings.info(f'after overlap genes atlas_adata shape: {atlas_adata.shape}')
    loggings.info(f'after overlap genes adata_inte_sub shape: {adata_inte_sub.shape}')


########################################
# fit NN
########################################
import pickle
loggings.info(f'load fitted nn ...')
file_path = args.fitted_NN_path
pickle_file = Path(file_path) / 'knnpickle_file'
if args.use_existing_nn:
    loggings.info(f'loaded calculated X_pca')
    atlas_pcs = np.load(Path(file_path) / 'atlas_pcs.npy')
    pcs = np.load(Path(file_path) / 'pcs.npy')
    atlas_adata.obsm['X_pca'] = atlas_pcs
    atlas_adata.varm['PCs'] = pcs
    
    loggings.info(f'loaded calculated atlas neibor files')
    nn_ = pickle.load(open(Path(file_path) / 'knnpickle_file', 'rb'))
    neigh_dist_atlas = np.load(Path(file_path) / 'neigh_dist_atlas.npy')
    neigh_ind_atlas = np.load(Path(file_path) / 'neigh_ind_atlas.npy')
else:
    if args.recompute_pca:
        sc.tl.pca(atlas_adata, n_comps=30,use_highly_variable=False) #svd_solver='arpack', n_comps=10, use_highly_variable=False)
    else:
        assert 'PCs' in atlas_adata.varm.keys()
    atlas_pcs = atlas_adata.obsm['X_pca']
    pcs = atlas_adata.varm['PCs']
    np.save(Path(file_path) / 'atlas_pcs.npy', atlas_pcs)
    np.save(Path(file_path) / 'pcs.npy', pcs)
    
    

    loggings.info(f'refit nn ...')
    ds1 = atlas_pcs
    if isinstance(ds1, np.matrix):
        ds1 = np.asarray(ds1)


    nn_ = NearestNeighbors(n_neighbors=30, metric='cosine')
    nn_.fit(ds1)
    neigh_dist_atlas, neigh_ind_atlas = nn_.kneighbors(return_distance=True)

    np.save(Path(file_path) / 'neigh_dist_atlas.npy', neigh_dist_atlas)
    np.save(Path(file_path) / 'neigh_ind_atlas.npy', neigh_ind_atlas)

    import pickle
    knnPickle = open(Path(file_path) / 'knnpickle_file', 'wb') 
    pickle.dump(nn_, knnPickle)  
    knnPickle.close()

    # load the model from disk
    nn_ = pickle.load(open(Path(file_path) / 'knnpickle_file', 'rb'))



########################################
# fit NN
########################################

loggings.info(f'get mapped pc ...')
adata_inte =  sc.AnnData(adata_inte_sub.to_df())
atlas_adata_bg = sc.AnnData(atlas_adata.to_df())
atlas_adata_bg.varm['PCs'] = atlas_adata.varm['PCs']
atlas_adata_bg.obsm['X_pca'] = atlas_adata.obsm['X_pca']


# map_cell_num = 19246
join = 'outer'
recompute_pca = False


pcs = atlas_adata_bg.varm['PCs']
atlas_pcs = atlas_adata_bg.obsm['X_pca']
adata_inte.obsm['X_pca'] = (adata_inte.X - atlas_adata_bg.X.mean(0)) @ pcs
adata_inte.varm['PCs'] = pcs

loggings.info(f'nn fit mapped data ...')
mapped_pcs = adata_inte.obsm['X_pca']

ds1 = atlas_pcs
ds2 = mapped_pcs
if isinstance(ds1, np.matrix):
    ds1 = np.asarray(ds1)
if isinstance(ds2, np.matrix):
    ds2 = np.asarray(ds2)

neigh_dist, neigh_ind = nn_.kneighbors(ds2, return_distance=True)

loggings.info(f'nn finish fit mapped data, make adata_all')
adata_all = anndata.concat(
    [atlas_adata_bg, adata_inte], 
    axis=0, 
    join = join, 
    merge="same", 
    label = 'atlas_inte', 
    keys = ['atlas', 'mapped_data']
)

save_nn_path = args.save_nn_path
save_nn_path = Path(save_nn_path)
loggings.info(f'save nn results to {str(save_nn_path)}')
np.save(save_nn_path / 'neigh_dist.npy', neigh_dist)
np.save(save_nn_path / 'neigh_ind.npy', neigh_ind)
adata_all.write(save_nn_path / 'adata_all.h5ad')


########################################
# make adata all
########################################
loggings.info(f'make distances and connectivities to adata_all')
knn_distances = np.vstack((neigh_dist_atlas, neigh_dist))
knn_indices = np.vstack((neigh_ind_atlas, neigh_ind))
adata_all.obsp['distances'], adata_all.obsp['connectivities'] = sc.neighbors._compute_connectivities_umap(
    knn_indices,
    knn_distances,
    n_obs=adata_all.shape[0],
    n_neighbors=30,
)


########################################
# map atlas cell type to adata inte
########################################
loggings.info(f'map raw atlas cell_type col to mapped data')
assign_label_col = args.atlas_assign_label_col
adata_inte_add_col = f'predict_mapped_{assign_label_col}'
adata_inte_add_col_prob = f'{adata_inte_add_col}_prob'
    
rm_ct = [
     'unknown', 'Unknown',
]
    
ind_map = neigh_ind
# total_iterations = ind_map.shape[0]
# progress_bar = tqdm.tqdm(total=total_iterations)

nrb_celltype = atlas_adata.obs[assign_label_col].to_numpy()[ind_map.reshape(-1)]
cell_index = np.arange(ind_map.shape[0]).repeat(ind_map.shape[1])


nrb_celltype_df = pd.DataFrame(nrb_celltype, columns = ['nrb_celltype'])
nrb_celltype_df['cell_index'] = cell_index
nrb_celltype_df = nrb_celltype_df.groupby('cell_index')['nrb_celltype'].value_counts(normalize=True).unstack(fill_value=0)
mapped_ct = nrb_celltype_df.idxmax(1).values
prob = nrb_celltype_df.max(1).values

adata_inte_sub.obs[adata_inte_add_col] = mapped_ct 
adata_inte_sub.obs[adata_inte_add_col_prob] = prob 

# predict_mapped_ct = []
# predict_mapped_ct_prob = []
# for i in range(ind_map.shape[0]):
#     value_counts = atlas_adata.obs.iloc[ind_map[i],:][assign_label_col].value_counts()
# #     value_counts = value_counts[value_counts.index != 'unknown']
#     value_counts = value_counts[~value_counts.index.isin(rm_ct)]
#     mapped_ct = value_counts.idxmax()
#     prob = value_counts[0] / value_counts.sum()
#     predict_mapped_ct.append(mapped_ct)
#     predict_mapped_ct_prob.append(prob)
#     progress_bar.update(1)

# adata_inte_sub.obs[adata_inte_add_col] = predict_mapped_ct 
# adata_inte_sub.obs[adata_inte_add_col_prob] = predict_mapped_ct_prob 


loggings.info(f'save adata_inte_sub that added mapped cell type')
adata_inte_sub.write(save_nn_path / f'adata_inte_mapped.h5ad')


