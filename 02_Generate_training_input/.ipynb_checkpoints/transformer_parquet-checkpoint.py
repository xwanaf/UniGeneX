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
from UniGeneX import *
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
        "--gene_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--data_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--trainset_list_path",  
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
        "--save_folder",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--check_ct_col",  
        type=str,
        default='cell_type',
        help="Decoder type.",
    )
parser.add_argument(
    "--skip_check_umap",
    action="store_true",
    help="skip_check_umap",
)

args = parser.parse_args()

########################################
# define paths
########################################
base_path = Path(args.base_path) / args.tissue
base_path.mkdir(parents=True, exist_ok=True)

data_path = Path(args.data_path)


save_path = Path(base_path) / args.save_folder
save_path.mkdir(parents=True, exist_ok=True)
loggings = configure_logging(str(Path(save_path) / 'transformer_parquet_log'))
loggings.info(f'base_path: {base_path}')
loggings.info(f'data_path: {data_path}')
loggings.info(f'save_path: {save_path}')



########################################
# concatenate datas
########################################

data_files_h5ad = glob.glob(str(data_path) + '/*.h5ad')
loggings.info(f'All data files: {data_files_h5ad}')

if args.trainset_list_path is not None:
    HLCA_data_files_sl = np.load(args.trainset_list_path)
    data_files_selected = [Path(_).stem for _ in HLCA_data_files_sl]

    print(f'{HLCA_data_files_sl.shape[0]} training files in total')
else:
    data_files_selected = [Path(_).stem for _ in data_files_h5ad]
    




loggings.info(f'load adatas from: {str(data_path)}')
adata_list = []
data_file_names = []
for i, f in enumerate(data_files_h5ad):
    loggings.info(f'============================== {i}-th data ===============================')
    file_name = Path(f).stem
    if file_name in data_files_selected:
        data_file_names.append(file_name)
        loggings.info(f'processing {file_name}')
    #     f = data_path / f'{file}'

        adata = sc.read(f)

        adata_list.append(adata)
        loggings.info(f'adata X max {adata.X.max()}')
        loggings.info(adata.shape)
    
all_cell_num = np.array([_.shape[0] for _ in adata_list]).sum()
loggings.info(f'{all_cell_num} cells in total')

########################################
# make parquet
########################################

traingene_path = Path(args.traingene_path)
train_genes = np.load(traingene_path / 'pretrain_data_train_genes.npy')
loggings.info(f'train_genes length: {len(train_genes)}')


gene_path = Path(args.gene_path) / 'hvg10k'
save_path_log_scale = Path(data_path) / 'log_scale_train_genes'
# save_path_log_scale.mkdir(parents=True, exist_ok=True)
data_files = glob.glob(str(gene_path) + '/*.csv')
data_files_h5ad = glob.glob(str(gene_path) + '/*.h5ad')

# for i, f in enumerate(data_files + data_files_h5ad):
adata_list_train_genes = []
for i, (adata, file_name) in enumerate(zip(adata_list, data_file_names)):
    loggings.info(f'============================== {i}-th data ===============================')
#     file_name = Path(f).stem


    f = gene_path / f'{file_name}.csv'
    if not Path(f).is_file():
        genes = [_.upper() for _ in adata.var.index.tolist()]
    else:
        hvg_df = pd.read_csv(f, index_col = 0)
        genes = hvg_df.index[hvg_df['highly_variable']].tolist()
        genes = [_.upper() for _ in genes]
        
    

    loggings.info(f'processing {file_name}')
#         adata = sc.read(data_path / f'{file_name}.h5ad')
    loggings.info(f'adata shape: {adata.shape}')
    loggings.info(f'adata X max: {adata.X.max()}')


#     if 'feature_name' in adata.var.columns:
#         adata.var.reset_index(inplace = True)
#         adata.var.set_index('feature_name', inplace = True)

    
    genes = [_.split('_ENSG')[0] if '_ENSG' in _ else _ for _ in genes]
    adata.var.index = [_.split('_ENSG')[0] if '_ENSG' in _ else _ for _ in adata.var.index]
    adata.var.index = [_.upper() for _ in adata.var.index.tolist()]

    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'unknown'



    hvg_genes = genes
    selected_genes = set(train_genes) & set(hvg_genes)
    adata = adata[:, adata.var.index.isin(selected_genes)]

    loggings.info(f'after selcted train genes')
    loggings.info(f'adata shape: {adata.shape}')
    loggings.info(f'adata X max: {adata.X.max()}')
    adata_list_train_genes.append(adata)


#     adata.write(save_path / f'{file_name}.h5ad')
#     loggings.info(f'save data to {str(save_path)}')
#         del adata




###############
# load vocab
###############
if not os.path.exists(Path(args.vocab_path) / 'default_census_vocab.json'):
    loggings.info(f'build vocab ...')
    build_vocab(train_genes, Path(args.vocab_path))

loggings.info(f'load vocab ...')
special_tokens = ["<pad>", "<cls>", "<eoc>"]
pad_value = -2

vocab_path = Path(args.vocab_path) / 'default_census_vocab.json'
vocab = GeneVocab.from_file(Path(vocab_path))
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

for i, (adata, file_name) in enumerate(zip(adata_list_train_genes, data_file_names)):
    loggings.info(f'============================== {i}-th data {file_name} ===============================')
    genes = adata.var.index
    genes = [_.upper() for _ in genes]
     
        
    gene2idx = vocab.get_stoi()
    gene_in_vocab = [True if gene in list(gene2idx.keys()) else False for gene in genes]
    gene_not_in_vocab = np.array(genes)[~np.array(gene_in_vocab)]

    loggings.info(f'{gene_not_in_vocab.shape[0]} gene that is not in vocab')

###############
# obs_concat
###############
loggings.info(f'make obs_concat ...')
def add_batch_col(df, i):
    df = df.loc[:, ~df.columns.duplicated()]
    df['dataset_idx'] = [int(i)] * df.shape[0]
    df['cell_parquet_index'] = np.arange(df.shape[0]).astype(int).tolist()
    df['dataset_cell_id'] = [f'dataset_{i}_{j}' for i, j in zip(df['dataset_idx'], df['cell_parquet_index'])]
#     df.index = df['dataset_cell_id'].tolist()
    df = df.reset_index()
    df = df.set_index('dataset_cell_id')
    return df

df_list = [add_batch_col(adata.obs.copy(), i) for i, adata in enumerate(adata_list_train_genes)]
obs_concat = pd.concat(df_list, axis = 0)
loggings.info(f'obs_concat shape: {obs_concat.shape}')

# obs_concat['dataset_cell_id'] = [f'dataset_{i}_{j}' for i, j in zip(obs_concat['dataset_idx'], obs_concat['cell_id'])]
# obs_concat.index = obs_concat['dataset_cell_id'].tolist()
obs_concat['cell_type'] = obs_concat[args.check_ct_col]

obs_concat_save_name = 'obs_concat.csv'
loggings.info(f'save obs_concat to {str(save_path)}')
obs_concat.to_csv(save_path / obs_concat_save_name)


###############
# CellTypeMapping_df
###############
CellTypeMapping_df_save_name = 'CellTypeMapping_df.csv'

ct_col = 'cell_type'
cell_type_array = np.array(obs_concat[ct_col])
cell_type_class = np.unique(cell_type_array)
df_category = obs_concat[[ct_col]].astype('category').apply(lambda x: x.cat.codes)
CellTypeMapping_df = pd.DataFrame.from_dict(dict(enumerate(obs_concat[ct_col].astype('category').cat.categories)), orient='index', columns = ['cell_type'])
loggings.info(f'save CellTypeMapping_df to {str(save_path)}')
CellTypeMapping_df.to_csv(save_path / CellTypeMapping_df_save_name)

# parameters: mean and cell type index
cell_type_array_code = np.array(df_category[ct_col]) 


###############
# make_parquet_data_list
###############
raw_dataset = make_parquet_data_list(
    adata_list_train_genes,
    vocab = vocab,
    match_vocab_gene = True,
    loggings = loggings,
    pad_value = pad_value

)

raw_dataset = raw_dataset.add_column('celltype_code', cell_type_array_code.tolist())
cls_prefix_datatable = save_path / 'cls_prefix_data.parquet'

raw_dataset.to_parquet(str(cls_prefix_datatable))
loggings.info(f"Saved raw_dataset to {cls_prefix_datatable}.")





if not args.skip_check_umap:
    ########################################
    # check cell type
    ########################################
    loggings.info(f"check cell type ...")
    # data_path = Path(base_path) / 'log_scale_train_genes_inference'
    save_pca_path = Path(save_path) / 'log_scale_train_genes_pca'

    save_pca_path.mkdir(parents=True, exist_ok=True)
    # data_files = glob.glob(str(data_path) + '/*.h5ad')

    set_vis_default(font = 1.)
    # for i, f in enumerate(data_files):
    for i, (adata, file_name) in enumerate(zip(adata_list_train_genes, data_file_names)):
        loggings.info(f'============================== {i}-th data ===============================')

        sc.pp.pca(adata,svd_solver='arpack', n_comps=30, use_highly_variable=False)

        try:
            sc.pp.neighbors(adata, metric='cosine', n_neighbors=30, n_pcs = 30)
        except:
            del adata.uns
            sc.pp.neighbors(adata, metric='cosine', n_neighbors=30, n_pcs = 30)
        sc.tl.umap(adata, min_dist = 0.3, spread = 1, maxiter=100)


        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        sc.pl.umap(
                adata, color=args.check_ct_col, size=15, frameon=False, show=False, ax=axs#, legend_loc = 'on data' #'lower'#'on data'
            )
        axs.legend(bbox_to_anchor=(1., 1.0))

        plt.tight_layout()
        plt.savefig(save_pca_path / f"{file_name}.png", dpi=100)


    #     plt.show()
        plt.close()




