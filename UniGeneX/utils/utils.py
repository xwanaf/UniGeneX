import functools
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from typing import Dict, List, Optional, Tuple, Union
import sys

import numpy as np
import torch
import pandas as pd
from anndata import AnnData
import scib
from matplotlib import pyplot as plt
from matplotlib import axes
from IPython import get_ipython
from scipy.sparse import issparse

import tqdm
from datasets import Dataset, concatenate_datasets


import importlib




from .. import logger
from ..tokenizer import *

def subsample_data(adata, subsample_cell_num = 10000): 
    print(f'Subsampling ...')
    if isinstance(adata, list):
        subsample_index = np.random.choice(np.arange(adata[0].shape[0]), subsample_cell_num, replace = False)
        adata = [ad[subsample_index, :] for ad in adata]
    else:
        subsample_index = np.random.choice(np.arange(adata.shape[0]), subsample_cell_num, replace = False)
        adata = adata[subsample_index, :]
    return adata

    
# import PyComplexHeatmap as pch
def get_color_dict(adata, col, uns_key):
    c_dict = dict(zip(adata.obs[col].astype('category').unique().categories, adata.uns[uns_key]))
    return c_dict

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v
      
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    
    

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

# cortex without donor id

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_vis_default(font = 1.6, circle_marker = 1.):
    mpl.rcParams['font.family'] = 'Arial'
    plt.rcParams.update(plt.rcParamsDefault)
#     %matplotlib inline
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')

    sns.set_context('paper',font_scale=font)
    plt.rcParams["legend.markerscale"] = circle_marker
    mpl.rcParams['font.family'] = 'Arial'

    

def gene_list_overlap_heatmap(gene_lists, file_names = None, vmin = 0, vmax = None, return_file_name_reordered = False):
    set_vis_default(font = 1.2)


#     gene_lists = gene_list_5k
    if file_names is None:
        file_names = np.arange(len(gene_lists)).astype(str)
    overlap_counts = np.zeros((len(gene_lists), len(gene_lists)))

    # Find the overlap between each pair of adata objects
    for i, j in itertools.combinations(range(len(gene_lists)), 2):
        overlap = len(set(gene_lists[i]).intersection(gene_lists[j]))
        overlap_counts[i, j] = overlap
        overlap_counts[j, i] = overlap


    # fig, ax = plt.subplots(1,1, figsize = (14, 12))
    # sns.heatmap(overlap_counts, annot=False, fmt="g", cmap="crest")

    set_vis_default(font = 1.)
    clustermap = sns.clustermap(
        overlap_counts,
        figsize = (0.5 * len(gene_lists), 0.4 * len(gene_lists)),
        row_cluster=True, 
        col_cluster=True,
        vmin = vmin, vmax = vmax,
        cmap="crest"
    #     col_colors = col_colors
    )

    reorder_index_col = clustermap.dendrogram_col.reordered_ind
    reorder_index_row = clustermap.dendrogram_row.reordered_ind

    index_equal = np.array_equal(reorder_index_col, reorder_index_row)
    print(f'order of row and col are equal: {index_equal}')
    # clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_ymajorticklabels(), rotation=0)
    clustermap.ax_heatmap.set_yticklabels(np.array(file_names)[reorder_index_row], rotation=0)
    clustermap.ax_heatmap.set_xticklabels(np.array(file_names)[reorder_index_col], rotation=90)

    reordered_file_names = np.array(file_names)[reorder_index_row]

    # plt.xticks(np.arange(len(gene_lists)) + 0.5, np.array(file_names)[reorder_index_col], rotation=90)
    # plt.yticks(np.arange(len(gene_lists)) + 0.5, np.array(file_names)[reorder_index_row], rotation=0)
    # plt.xlabel("Adata Index")
    # plt.ylabel("Adata Index")
    plt.title("Overlap of Genes between Adata Objects")
    plt.show()
    plt.close()
    
    if return_file_name_reordered:
        return overlap_counts, reordered_file_names
    
    else:
        return overlap_counts



def make_parquet_data_list(
    adata_list, 
    vocab,
    gene_name_column = None,
    match_vocab_gene = False, 
    use_layer = None, 
    cell_type_column = None, 
    CellTypeMapping_df_save_name = None,
    loggings = None,
    pad_value = -2,
):
    
    raw_dataset_list = []

    for j, adata in enumerate(adata_list):
        print(f'Raw single cell adata shape: {adata.shape}')


        if gene_name_column is None:
            genes = adata.var.index
            genes = [_.upper() for _ in genes]
        else:
            genes = adata.var[gene_name_column]
            genes = [_.upper() for _ in genes]

        if match_vocab_gene:
            gene2idx = vocab.get_stoi()
            gene_in_vocab = [True if gene in list(gene2idx.keys()) else False for gene in genes]
            gene_not_in_vocab = np.array(genes)[~np.array(gene_in_vocab)]
            
            print(f'{gene_not_in_vocab.shape[0]} gene that is not in vocab')
            adata = adata[:, gene_in_vocab]
            genes = np.array(genes)[gene_in_vocab].tolist()
            


        if use_layer is None:
            all_counts = (
                adata.X.toarray()
                if issparse(adata.X)
                else adata.X
            )
        else:
            all_counts = (
                adata.layers[use_layer].toarray()
                if issparse(adata.layers[use_layer])
                else adata.layers[use_layer]
            )
        assert all_counts.max() < 100
        pad_column = np.array([pad_value] * all_counts.shape[0])[:, None] 
        all_counts = np.hstack([pad_column, all_counts])


        #     genes = adata.var.index.tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        gene_ids = np.array([vocab["<cls>"]] + gene_ids.tolist(), dtype=int)


        tokenized_data_ = tokenize_batch(
            all_counts,
            gene_ids,
            append_cls=False,  # append <cls> token at the beginning
            include_zero_gene=True,
        )

        def tokenize_(data, dataset_idx = 0):
            tokenized_data = {"id": [], "gene": [], "expr": [], 'dataset_idx': []}


            tokenized_data["id"] = list(range(len(data))) # list(range(len(data)))
            tokenized_data["dataset_idx"] = [dataset_idx] * len(data)
            total_iterations = len(data)
            with tqdm.tqdm(total=total_iterations, desc=f'Tokenize cells') as pbar:
                for i in range(len(data)):  # ~2s/100k cells
                    gene_ids, values = data[i]
                    tokenized_data["gene"].append(gene_ids.tolist())
                    tokenized_data["expr"].append(values.tolist())

                    pbar.update(1)
            return tokenized_data



        tokenized_data = tokenize_(tokenized_data_, dataset_idx = j)
        tokenized_data = Dataset.from_dict(tokenized_data)

        print(
            f"Dataset number of samples: {len(tokenized_data['gene'])}, "
            f"\n\t feature length: {len(tokenized_data['gene'][0])}"
        )

        raw_dataset_list.append(tokenized_data)

    print("merging dataset...")
    raw_dataset = concatenate_datasets(raw_dataset_list)
    print("done merging dataset")
    
    return raw_dataset
    
    
    
def gene_vocabulary():
    """
    Generate the gene name2id and id2name dictionaries.
    """
    pass


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def category_str2int(category_strs: List[str]) -> List[int]:
    set_category_strs = set(category_strs)
    name2id = {name: i for i, name in enumerate(set_category_strs)}
    return [name2id[name] for name in category_strs]


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_free_gpu():
    import subprocess
    import sys
    from io import StringIO
    import pandas as pd

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=memory.used,memory.free",
        ]
    ).decode("utf-8")
    gpu_df = pd.read_csv(
        StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Find free GPU{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"])
    )

    return idx


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def histogram(
    *data: List[np.ndarray],
    label: List[str] = ["train", "valid"],
    color: List[str] = ["blue", "red"],
    figsize: Tuple[int, int] = (9, 4),
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[str] = None,
) -> axes.Axes:
    """
    Plot histogram of the data.

    Args:
        data (List[np.ndarray]): The data to plot.
        label (List[str]): The label of the data.
        color (List[str]): The color of the data.
        figsize (Tuple[int, int]): The size of the figure.
        title (Optional[str]): The title of the figure.
        show (bool): Whether to show the figure.
        save (Optional[str]): The path to save the figure.

    Returns:
        axes.Axes: The axes of the figure.
    """
    # show histogram of the clipped values
    assert len(data) == len(label), "The number of data and labels must be equal."

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    max_value = max(np.max(data) for data in data)
    ax.hist(
        [d.flatten() for d in data],
        bins=np.arange(0, max_value + 1, 1) + 0.5 if max_value < 60 else 60,
        label=label,
        density=True,
        histtype="bar",
        linewidth=2,
        rwidth=0.85,
        color=color,
    )
    ax.legend()
    ax.set_xlabel("counts")
    ax.set_ylabel("density")

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax


def _indicate_col_name(adata: AnnData, promt_str: str) -> Optional[str]:
    """
    Indicate the column name of the data.

    Args:
        adata (AnnData): The AnnData object.
        promt_str (str): The prompt string.

    Returns:
        Optional[str]: The column name.
    """
    while True:
        col_name = input(promt_str)
        if col_name == "":
            col_name = None
            break
        elif col_name in adata.var.columns:
            break
        elif col_name in adata.obs.columns:
            break
        else:
            print(f"The column {col_name} is not in the data. " f"Please input again.")

    return col_name


def find_required_colums(
    adata: AnnData,
    id: str,
    configs_dir: Union[str, Path],
    update: bool = False,
) -> List[Optional[str]]:
    """
    Find the required columns in AnnData, including celltype column, str_celltype
    column, the gene name column, and the experimental batch key.

    This function asks the user to input the required column names if the first
    time loading the data. The names are saved in the config file and will be
    automatically loaded next time.

    Args:
        adata (AnnData): The AnnData object.
        id (str): The id of the AnnData object, will be used as the file name for
            saving the config file.
        configs_dir (Union[str, Path]): The directory of saved config files.
        update (bool): Whether to update the config file.

    Returns:
        List[Optional[str]]: The required columns, including celltype_col, str_celltype_col,
            gene_col, and batch_col.
    """
    if isinstance(configs_dir, str):
        configs_dir = Path(configs_dir)

    if not configs_dir.exists():
        configs_dir.mkdir()

    config_file = configs_dir / f"{id}.json"

    if not config_file.exists() or update:
        print(
            "The config file does not exist, this may be the first time "
            "loading the data. \nPlease input the required column names."
        )
        print(adata)
        celltype_col = _indicate_col_name(
            adata,
            "Please input the celltype column name (skip if not applicable): ",
        )
        str_celltype_col = _indicate_col_name(
            adata, "Please input the str_celltype column name: "
        )
        gene_col = _indicate_col_name(adata, "Please input the gene column name: ")
        batch_col = _indicate_col_name(adata, "Please input the batch column name: ")

        config = {
            "celltype_col": celltype_col,
            "str_celltype_col": str_celltype_col,
            "gene_col": gene_col,
            "batch_col": batch_col,
        }

        with open(config_file, "w") as f:
            json.dump(config, f)

    else:
        with open(config_file, "r") as f:
            config = json.load(f)

    return [
        config["celltype_col"],
        config["str_celltype_col"],
        config["gene_col"],
        config["batch_col"],
    ]


def tensorlist2tensor(tensorlist, pad_value):
    max_len = max(len(t) for t in tensorlist)
    dtype = tensorlist[0].dtype
    device = tensorlist[0].device
    tensor = torch.zeros(len(tensorlist), max_len, dtype=dtype, device=device)
    tensor.fill_(pad_value)
    for i, t in enumerate(tensorlist):
        tensor[i, : len(t)] = t
    return tensor


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


# wrapper to make sure all methods are called only on the main process
def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            return func(*args, **kwargs)

    return wrapper


# class wrapper to make sure all methods are called only on the main process
class MainProcessOnly:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        attr = getattr(self.obj, name)

        if callable(attr):
            attr = main_process_only(attr)

        return attr
