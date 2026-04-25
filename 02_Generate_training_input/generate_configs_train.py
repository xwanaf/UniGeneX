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
        "--save_path",  
        type=str,
        default=None,
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--save_config_name",  
        type=str,
        default=None,
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--config_temp_path",  
        type=str,
        default='/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/configs/config_templete.yaml',
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--vocab_path",  
        type=str,
        default=None,
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--kld_weight",  
        type=float,
        default=1e3,
#         choices=[None, 'LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm'],
        help="Decoder type.",
    )
parser.add_argument(
        "--common_dec_gene_len",  
        type=int,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--CellTypeMapping_df_paths",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--data_source",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--test_out_of_sample_data_source",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--common_dec_genes_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--max_epochs",  
        type=int,
        default=150,
        help="Decoder type.",
    )
parser.add_argument(
        "--gpus",  
        type=str,
        default='0,1,2,3,4,5,6,7',
        help="Decoder type.",
    )
parser.add_argument(
        "--devices",  
        type=int,
        default=2,
        help="Decoder type.",
    )

args = parser.parse_args()

########################################
# define paths
########################################
save_path = Path(args.save_path) 
save_path.mkdir(parents=True, exist_ok=True)
loggings = configure_logging(str(Path(save_path) / 'generate_configs_log'))
loggings.info(f"save_path: {str(save_path)}")

from omegaconf import OmegaConf
config_templete = OmegaConf.load(args.config_temp_path)


config_templete.model.params.vocab_path = args.vocab_path
config_templete.model.params.kld_weight = args.kld_weight


config_templete.model.params.common_dec_gene_len = args.common_dec_gene_len
config_templete.model.params.CellTypeMapping_df_paths = [args.CellTypeMapping_df_paths]
config_templete.data.data_source = args.data_source
config_templete.data.test_out_of_sample_data_source = args.test_out_of_sample_data_source
config_templete.data.common_dec_genes_path = args.common_dec_genes_path
config_templete.data.vocab_path = args.vocab_path
config_templete.lightning.trainer.gpus = args.gpus
config_templete.lightning.trainer.devices = args.devices
config_templete.lightning.trainer.max_epochs = args.max_epochs


OmegaConf.save(config_templete, save_path / f"{args.save_config_name}.yaml")
loggings.info(f"Configuration saved to {args.save_config_name}.yaml")