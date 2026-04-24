import argparse, os, sys, datetime, glob, importlib, csv
from typing import List, Tuple, Dict, Union, Optional
# from tqdm import tqdm
import tqdm
import pickle
from datetime import timedelta
from pathlib import Path
import copy
import json
import sys
sys.path.append('/home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/Reproducibility/Validation')


import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
import torch.distributed as dist
from datasets import Dataset, load_dataset, concatenate_datasets
torch.set_float32_matmul_precision('medium')

from packaging import version
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from torchvision.utils import make_grid
from torchvision import transforms
# from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf


print(0)
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import FSDPStrategy
# from pytorch_lightning.strategies import DDPFullyShardedStrategy as FSDPStrategy




from CustomLogger import plainLogger

#============================== first stage model ==============================#
# sys.path.insert(0, "../")
from UniGeneX.utils import *
from UniGeneX.model import FlashTransformerEncoderLayer_FiLM, FlashTransformerEncoderLayer
from UniGeneX.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_poisson, smooth_l1_loss
from UniGeneX.tokenizer import GeneVocab

from scipy.sparse import issparse
import pandas as pd
import scanpy as sc


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}



print('finish import')



def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)

# torch.autograd.set_detect_anomaly(True)

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = "debug"
set_seed(42)


parser = argparse.ArgumentParser()
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
# parser.add_argument(
#     "--scale_lr",
#     type=str2bool,
#     nargs="?",
#     const=True,
#     default=True,
#     help="scale base-lr by ngpu * batch_size * n_accumulate",
# )
parser.add_argument(
    "--scale_lr",
    action="store_true",
    help="scale base-lr by ngpu * batch_size * n_accumulate.",
)
#=========================================== path =========================================#
parser.add_argument(
    "--config",
    type=str,
    required=True,
    default='/home/xwanaf/generative_model/mind-vis/pretrains/ldm/label2img_pretrain/config.yaml',
    help='The name of the data source (currently support "scvi" datasets), or the '
    "path to the data file.",
)
parser.add_argument(
    "--pin_memory",
    action="store_true",
    help="scale base-lr by ngpu * batch_size * n_accumulate.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=224,
    help="The number of workers. Default is 224.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="The directory to save the trained model and the results.",
)
parser.add_argument(
    "--log_step_interval",
    type=int,
    default=50,
    help="The interval for log. Default is 10.",
)
parser.add_argument(
    "--scheduler_epochs",
    type=int,
    default=100,
    help="The number of epochs for setting scheduler. Default is 50.",
)


parser.add_argument('--device_num', type=int, help='device_num', default = None)

args = parser.parse_args()

import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union






#=========================================== path =========================================#
config = OmegaConf.load(args.config)
print(f'vocab path: {config.model.params.vocab_path}')
# init and save configs
lightning_config = config.pop("lightning", OmegaConf.create())
# merge trainer cli with config
trainer_config = lightning_config.get("trainer", OmegaConf.create())
# default to ddp
trainer_config["accelerator"] = 'gpu' # "GPU"
if 'strategy' not in trainer_config.keys():
    trainer_config["strategy"] = 'dp' # "ddp"
    
    
os.environ["CUDA_VISIBLE_DEVICES"] = trainer_config["gpus"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not "gpus" in trainer_config:
    del trainer_config["accelerator"]
    cpu = True
else:
    if trainer_config["strategy"] == 'dp':
        gpuinfo = trainer_config.pop("gpus")
        if args.gpus is not None:
            gpuinfo = args.gpus
        print(f"Running on GPUs {gpuinfo}")
        trainer_config["devices"] = gpuinfo
    # elif trainer_config["strategy"] == 'ddp':
        # trainer_config["devices"] = 8
    cpu = False

if args.device_num is not None:
    trainer_config["devices"] = args.device_num

if trainer_config["strategy"] == 'fsdp':
    layers = {
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            FlashTransformerEncoderLayer,
            FlashTransformerEncoderLayer_FiLM,
        }
    FSDP_strategy = FSDPStrategy(
        auto_wrap_policy=layers,
        activation_checkpointing_policy=layers  # enables activation checkpointing for the given layers
    )
    # trainer_config.pop('strategy')
    # trainer_config["strategy"] = strategy
    
trainer_opt = argparse.Namespace(**trainer_config)


#=========================================== model =========================================#
# for finetune 
if 'pretrain_root' in config.model.keys():
    model_files = os.listdir(config.model.pretrain_root)
    model_files = [file for file in model_files if file.endswith(".ckpt")]
    model_file = os.path.join(config.model.pretrain_root, model_files[0])
    config_path = os.path.join(config.model.pretrain_config) 
    model_config = OmegaConf.load(config_path)
    

    model = instantiate_from_config(model_config.model)
    pl_sd = torch.load(model_file, map_location="cpu")['state_dict']

    # missing, unexpected
    m, u = model.load_state_dict(pl_sd, strict=False)
    
    print(model)

else:
    model = instantiate_from_config(config.model)

#=========================================== logger & ckpt =========================================#
# trainer and callbacks
trainer_kwargs = dict()
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
nowname = now
save_dir = args.save_dir # '/home/xwanaf/generative_model/mind-vis/results/tmp'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass
ckpt_path = os.path.join(save_dir, 'ckpt')    
if not os.path.exists(ckpt_path):
    try:
        os.makedirs(ckpt_path)
    except:
        pass
sample_img_path = os.path.join(save_dir, 'sample_img')    
if not os.path.exists(sample_img_path):
    try:
        os.makedirs(sample_img_path)
    except:
        pass
sample_adata_path = os.path.join(save_dir, 'sample_adata')    
if not os.path.exists(sample_adata_path):
    try:
        os.makedirs(sample_adata_path)
    except:
        pass
    
    
# default logger configs
default_logger_cfgs = {
    "plain": {
        "target": "CustomLogger.plainLogger",
        "params": {
            "save_dir": save_dir,
            'log_step_interval': args.log_step_interval,
        }
    },
    "CSVLogger": {
        "target": "pytorch_lightning.loggers.CSVLogger",
        "params": {
            "name": "CSVLogger",
            "save_dir": save_dir,
        }
    },
}
default_logger_cfg = default_logger_cfgs["plain"]
if "logger" in lightning_config:
    logger_cfg = lightning_config.logger
else:
    logger_cfg = OmegaConf.create()
logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)


# modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
# specify which metric is used to determine best models
default_modelckpt_cfg = {
    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    "params": {
        "dirpath": ckpt_path,
        "filename": "{epoch:06}",
        "verbose": True,
        "save_last": True,
        "every_n_epochs": 5,
        "save_top_k": -1,  # <--- this is important!
        "enable_version_counter": False,
    }
}
if hasattr(model, "monitor"):
    print(f"Monitoring {model.monitor} as checkpoint metric.")
    default_modelckpt_cfg["params"]["monitor"] = model.monitor
    # default_modelckpt_cfg["params"]["save_top_k"] = 3

if "modelcheckpoint" in lightning_config:
    modelckpt_cfg = lightning_config.modelcheckpoint
else:
    modelckpt_cfg =  OmegaConf.create()
modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
trainer_kwargs["callbacks"] = [instantiate_from_config(modelckpt_cfg)]
# if version.parse(pl.__version__) < version.parse('1.4.0'):
    # trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)




#=========================================== dataset =========================================#
data_config = config.data
print(data_config)

#============ define vocab ============#
special_tokens = [data_config.pad_token, "<cls>", "<eoc>"]
pad_value = data_config.pad_value

vocab_path = data_config.vocab_path
vocab = GeneVocab.from_file(Path(vocab_path))
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
        

#======================== produce data ========================#
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import random
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler



def subsample_2list(list_1, list_2, ratio):
    combined_list = list(zip(list_1, list_2))
    combined_list_start = combined_list[0]
    combined_list = combined_list[1:]
    num_samples = int(len(combined_list) * ratio)
    subsample = random.sample(combined_list, num_samples)
    subsample = [combined_list_start] + subsample
    subsampled_list_1, subsampled_list_2 = zip(*subsample)
    return subsampled_list_1, subsampled_list_2
    
@dataclass
class DataCollator:
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    mask_ratio: float = 0.0


    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        

        if self.mask_ratio > 0:
            for example in examples:
                example['dec_gene'] = example['gene']
                example['dec_expr'] = example['expr']
                
                example['gene'], example['expr'] = subsample_2list(example['gene'].tolist(), example['expr'].tolist(), self.mask_ratio)
                example['gene'], example['expr'] = torch.Tensor(example['gene']).long(), torch.Tensor(example['expr'])
            max_length_dec = max(len(example["dec_gene"]) for example in examples)
        


        
        max_length = max(len(example["gene"]) for example in examples)
        
        
        # max_length = torch.max(torch.Tensor([len(example["gene"]) for example in examples]))
        # rank = torch.distributed.get_rank()
        # max_length = max_length.to(rank)
        # dist.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX) # Synchronize and find the maximum length across all GPUs
        # max_length = max_length.item()
    

        keys = examples[0].keys()
        
        # pad 
        padded_genes = []
        padded_expressions = []
        celltypes = []
        dataset_idxs = []
        ids = []
        
        padded_genes_dec = []
        padded_expressions_dec = []
        
        
        for i in range(len(examples)):
            id = examples[i]["id"]
            genes = examples[i]["gene"]
            expressions = examples[i]["expr"]
            if 'celltype_code' in keys:
                celltype = torch.Tensor(examples[i]["celltype_code"])
            if 'dataset_idx' in keys:
                dataset_idx = torch.Tensor(examples[i]["dataset_idx"])
            
            genes, expressions = self._pad(genes, expressions, max_length)  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_expressions.append(expressions)


            if 'celltype_code' in keys:
                celltypes.append(celltype)
            if 'dataset_idx' in keys:
                dataset_idxs.append(dataset_idx)
            ids.append(id)
            
            if self.mask_ratio > 0:
                genes = examples[i]["dec_gene"]
                expressions = examples[i]["dec_expr"]

                genes, expressions = self._pad(genes, expressions, max_length_dec)  # torch tensors of length _max_length
                padded_genes_dec.append(genes)
                padded_expressions_dec.append(expressions)
                
                
    
        padded_genes = torch.stack(padded_genes, dim=0)
        padded_expressions = torch.stack(padded_expressions, dim=0)
        if self.mask_ratio > 0:
            padded_genes_dec = torch.stack(padded_genes_dec, dim=0)
            padded_expressions_dec = torch.stack(padded_expressions_dec, dim=0)
        
        if 'celltype_code' in keys:
            celltypes = torch.stack(celltypes, dim=0)
        if 'dataset_idx' in keys:
            dataset_idxs = torch.stack(dataset_idxs, dim=0)
        ids = torch.stack(ids, dim=0)
        
        data_dict = {
                "id": ids,
                "gene": padded_genes,
                "expr": padded_expressions,
            }
        if self.mask_ratio > 0:
            data_dict['dec_gene'] = padded_genes_dec
            data_dict['dec_expr'] = padded_expressions_dec
            
            
        if 'celltype_code' in keys:
            data_dict['celltype_code'] = celltypes
        if 'dataset_idx' in keys:
            data_dict['dataset_idx'] = dataset_idxs
        return data_dict
    
    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
         
        gene_pad_tensor = torch.full((max_length - len(genes),), fill_value = self.pad_token_id, dtype=genes.dtype)
        genes = torch.cat([genes, gene_pad_tensor])
         
        expr_pad_tensor = torch.full((max_length - len(expressions),), fill_value = self.pad_value, dtype=expressions.dtype)
        expressions = torch.cat([expressions, expr_pad_tensor])

        return genes, expressions
    

    

def load_parquet_data(data_path):
    assert data_path.endswith(".parquet")
    cls_prefix_datatable = data_path
    cache_dir = Path(data_path).parent / "cache"

    print(f"Load dataset from {cls_prefix_datatable}")
        
    raw_dataset = load_dataset(
        "parquet",
        data_files=str(cls_prefix_datatable),
        split="train",
        cache_dir=str(cache_dir),
    )
    # raw_dataset keys: dict_keys(['id', 'genes', 'expressions', 'celltypes', 'celltype_code'])
    if ('gene' in raw_dataset[0].keys()) and ('expr' in raw_dataset[0].keys()):
        pass
    else:
        raw_dataset = raw_dataset.rename_column('genes', 'gene')
        raw_dataset = raw_dataset.rename_column('expressions', 'expr')

    print(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")

    return raw_dataset
    
train_dataset = load_parquet_data(data_config.data_source)
valid_dataset = load_parquet_data(data_config.test_out_of_sample_data_source)
# %% [markdown]
# # Data processing
# convert format to return torch.tensor
train_dataset = train_dataset.with_format("torch")
valid_dataset = valid_dataset.with_format("torch")

def remove_key(dataset, dataset_name, key):
    try:
        dataset = dataset.remove_columns(key)
        print(f'remove key {key} from {dataset_name} dataset')
    except:
        print(f'{dataset_name} dataset do not have key celltypes ')

    return dataset

train_dataset = remove_key(train_dataset, 'train', 'celltypes')
valid_dataset = remove_key(valid_dataset, 'valid out of sample', 'celltypes')

train_dataset = remove_key(train_dataset, 'train', 'file_name')
        
print(f"train set number of samples: {len(train_dataset)}, ")
print(f"valid set number of samples: {len(valid_dataset)}, ")

print(f"train set gene len: {len(train_dataset[0]['gene'])}, ")
print(f"valid set gene len: {len(valid_dataset[0]['gene'])}, ")

if 'test_data_source' in data_config.keys():
    if data_config.test_data_source is not None:
        test_dataset = load_parquet_data(data_config.test_data_source)
        test_dataset = test_dataset.with_format("torch")
        test_dataset = remove_key(test_dataset, 'test', 'celltypes')
        print(f"test set number of samples: {len(test_dataset)}, ")
        print(f"test set gene len: {len(test_dataset[0]['gene'])}, ")

#============= define common dec genes =============#
if data_config.valid_use_common_gene or datload_dataseta_config.test_use_common_gene:
    common_dec_genes = np.load(data_config['common_dec_genes_path']).tolist()
    print(f"will use common_dec_genes, common_dec_genes gene len: {len(common_dec_genes)}, ")
    common_dec_gene_ids = np.array(vocab(common_dec_genes), dtype=int)
    common_dec_gene_ids = np.array([vocab["<cls>"]] + common_dec_gene_ids.tolist())
    

    # model.register_buffer('common_dec_genes', torch.from_numpy(common_dec_gene_ids))
    model.common_dec_genes = torch.from_numpy(common_dec_gene_ids)
    model.valid_use_common_gene = data_config.valid_use_common_gene
    model.test_use_common_gene = data_config.test_use_common_gene

    if data_config.test_use_common_gene and (config.model.params.background_adata_paths is not None):
        test_genes_id = test_dataset[0]['gene'].tolist()
        test_genes = vocab.lookup_tokens(test_genes_id)[1:]
        overlap_genes = list(set(common_dec_genes) & set(test_genes))
        print(f'common_dec_genes and test_genes overlap {len(overlap_genes)} genes.')
        adata_reorder_index = torch.Tensor([test_genes.index(_) for _ in overlap_genes]).long()
        dec_adata_reorder_index = torch.Tensor([common_dec_genes.index(_) for _ in overlap_genes]).long()
        model.adata_reorder_index = adata_reorder_index
        model.dec_adata_reorder_index = dec_adata_reorder_index


    

else:
    model.common_dec_genes = None

collator_train = DataCollator(
    pad_token_id=vocab[data_config.pad_token],
    pad_value=data_config.pad_value,
    mask_ratio = data_config.mask_ratio
)
print(f'================================================')
print(f'Use mask; mask ratio: {data_config.mask_ratio}')
print(f'================================================')

collator_valid = DataCollator(
    pad_token_id=vocab[data_config.pad_token],
    pad_value=data_config.pad_value,
    mask_ratio = 0
)


train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, collate_fn=collator_train, shuffle=True, num_workers = args.num_workers, pin_memory=args.pin_memory)
valid_loader = DataLoader(valid_dataset, batch_size=data_config.eval_batch_size, collate_fn=collator_valid, shuffle=False, num_workers = args.num_workers, pin_memory=args.pin_memory)



val_dataloaders = valid_loader
if 'test_data_source' in data_config.keys():
    if data_config.test_data_source is not None:
        test_loader = DataLoader(test_dataset, batch_size=data_config.eval_batch_size, collate_fn=collator_valid, shuffle=False, num_workers = args.num_workers, pin_memory=args.pin_memory)
        val_dataloaders = [valid_loader, test_loader]

# for lr schedule
if trainer_config["strategy"] == 'dp':
    total_num_batches = len(train_loader) * args.scheduler_epochs
elif (trainer_config["strategy"] == 'ddp') or (trainer_config["strategy"] == 'fsdp'):
    total_num_batches = int(len(train_loader) * args.scheduler_epochs / trainer_config["devices"])
model.total_num_batches = total_num_batches
if model.scheduler_config is not None:
    model.scheduler_config.params.total_num_batches = total_num_batches
trainer_config.pop('strategy')
trainer_config.pop('gpus')



    


#=========================================== lr =========================================#
# configure learning rate
bs, base_lr = config.data.batch_size, config.model.base_learning_rate
if not cpu:
    try:
        ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
    except:
        ngpu = lightning_config.trainer.devices
else:
    ngpu = 1
if 'accumulate_grad_batches' in lightning_config.trainer:
    accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
else:
    accumulate_grad_batches = 1
print(f"accumulate_grad_batches = {accumulate_grad_batches}")
lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
if args.scale_lr:
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print(
        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
else:
    model.learning_rate = base_lr
    print("++++ NOT USING LR SCALING ++++")
    print(f"Setting learning rate to {model.learning_rate:.2e}")

    
    
#=========================================== make trainer =========================================#
print(trainer_opt, trainer_kwargs)
# print(f'FSDP_strategy: {FSDP_strategy}')
# trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs, strategy = FSDP_strategy)
trainer = Trainer(**trainer_config, **trainer_kwargs, strategy = FSDP_strategy)
# trainer = Trainer(**trainer_config, **trainer_kwargs)

print(f'finish initializing trainer, start fit')




#=========================================== fit model =========================================#
trainer.fit(model, train_loader, val_dataloaders=val_dataloaders)
    
