import os
import gc
import math
import copy
import glob
from pathlib import Path
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from typing import Dict, Mapping, Optional, Tuple, Any, Union, Callable, Iterable
import warnings
import json
from omegaconf import OmegaConf

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from tqdm import trange
from torchvision.utils import make_grid
# from pytorch_lightning.utilities.distributed import rank_zero_only



import sys
sys.path.append('/home/xwanaf/superpod/mind-vis/code')


from UniGeneX.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_poisson, smooth_l1_loss
from util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config


# from UniGeneX.preprocess import Preprocessor
from UniGeneX.tokenizer import GeneVocab, random_mask_value, tokenize_batch
from UniGeneX.utils import MainProcessOnly, ConfigWrapper, configure_logging
from UniGeneX import logger
# from diffusion.plotumap import plotsampledata


try:
    # from flash_attn.flash_attention import FlashMHA
    from UniGeneX.model.flash_attention import FlashMHA
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    


    
def cal_final_updim(output_dim, factor, kernel_size = 5, padding = 3):
    output_dim_ori = output_dim
    output_dim = int((output_dim + 2 * padding - kernel_size) / factor + 1)
    output_padding = output_dim_ori - ((output_dim - 1) * factor) + 2 * padding - kernel_size
    return output_dim, output_padding


class TransformerVAEModel_pl_integration(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        vocab_path: str = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        cell_emb_style: str = "cls",
        explicit_zero_prob: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        mvc_decoder_type: str = None, # ['LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm']
        nlayers_dec: int = 4,
        decode_gene: bool = False,
        L1_reg: bool = True,
        L1_reg_weight: float = 0.2,
        do_vae: bool = False,
        kld_weight: float = 0.00001,
        ckpt_path = None,
        scheduler_config=None,
        monitor = None,
        CellTypeMapping_df_paths=None,
        background_adata_paths=None,
        cell_emb_sample_iter_num=10000000,
        test_use_common_gene = False,
        valid_use_common_gene = False,
        common_dec_gene_len = 0,
        inference_mode = False,
        
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.vocab_path = vocab_path
        self.dropout = dropout
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.fast_transformer_backend = fast_transformer_backend
        self.pre_norm = pre_norm
        self.norm_scheme = "pre" if pre_norm else "post"
        self.mvc_decoder_type = mvc_decoder_type
        self.nlayers_dec = nlayers_dec
        self.decode_gene = decode_gene
        self.add_gene_loss = decode_gene
        self.pad_token = pad_token
        self.L1_reg = L1_reg
        self.L1_reg_weight = L1_reg_weight
        self.do_vae = do_vae
        self.kld_weight = kld_weight
        self.scheduler_config = scheduler_config
        self.use_scheduler = scheduler_config is not None
        self.cell_emb_sample_iter_num = cell_emb_sample_iter_num
        self.inference_mode = inference_mode

        self.adata_reorder_index = None
        self.dec_adata_reorder_index = None
        self.test_use_common_gene = test_use_common_gene
        self.valid_use_common_gene = valid_use_common_gene
        if common_dec_gene_len > 0:
            self.register_buffer('common_dec_genes', torch.zeros(common_dec_gene_len))
            print(f'initialize common_dec_genes with length: {common_dec_gene_len}')

        if self.use_scheduler:
            self.scheduler_config = scheduler_config
            self.warmup_ratio_or_step = scheduler_config.params.warmup_ratio_or_step
            # self.total_num_batches = total_num_batches

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        self.CellTypeMapping_df_paths = CellTypeMapping_df_paths
        if CellTypeMapping_df_paths is not None:
            self.CellTypeMapping_df = []
            for CellTypeMapping_df_path in CellTypeMapping_df_paths:
                self.CellTypeMapping_df.append(pd.read_csv(CellTypeMapping_df_path, index_col = 0))
        self.background_adata_paths = background_adata_paths
        if background_adata_paths is not None:
            self.background_adatas = []
            for background_adata_path in background_adata_paths:
                self.background_adatas.append(self.get_background_adata(background_adata_path))
        
    
            
        special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        vocab = GeneVocab.from_file(Path(self.vocab_path))
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        self.vocab = vocab
        self.ntoken = len(self.vocab) 
        self.norm_scheme = "pre" if pre_norm else "post"

        if monitor is not None:
            self.monitor = monitor
            
        
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
            
        self.build_network()
        print(f'#============================ Transformer ckpt_path: {ckpt_path} ============================#')
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        

    def get_background_adata(self, background_adata_path):
        background_adata = sc.read(background_adata_path)
        if background_adata.shape[0] > 30000:
#                 self.logger.log_info(f'Subsampling background_adata ...')
            subsample_index = np.random.choice(np.arange(background_adata.shape[0]), 10000, replace = False)
            background_adata = background_adata[subsample_index, :]
        return background_adata
            
            
    def build_network(self):

        #=================================== embedding ==================================#
        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(self.ntoken, self.d_model, padding_idx=self.vocab[self.pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        self.value_encoder = ContinuousValueEncoder(self.d_model, self.dropout)

        #=================================== encoder ==================================#
        if self.fast_transformer_backend == "linear":
            self.transformer_encoder = FastTransformerEncoderWrapper(
                self.d_model, self.nhead, self.d_hid, self.nlayers, self.dropout
            )
        elif self.fast_transformer_backend == "flash":
            encoder_layers = FlashTransformerEncoderLayer(
                self.d_model,
                self.nhead,
                self.d_hid,
                self.dropout,
                batch_first=True,
                norm_scheme=self.norm_scheme,
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        elif self.fast_transformer_backend == "normal":
            encoder_layers = TransformerEncoderLayer(
                self.d_model, self.nhead, self.d_hid, self.dropout, batch_first=True, norm_first = True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        #=================================== decode_gene ==================================#
        if self.decode_gene:
            self.decoder = ExprDecoder(
                self.d_model,
                explicit_zero_prob=self.explicit_zero_prob,
            )

        #=================================== do_vae ==================================#
        if self.do_vae:
            # moment layer
            self.fc_mu = nn.Linear(self.d_model, self.d_model)
            self.fc_var = nn.Linear(self.d_model, self.d_model)

            self.post_moment = nn.Linear(self.d_model, self.d_model)
        

        
        #=================================== mvc decode ==================================#
        # ['LinearConcat', 'Inner', 'TransformerConcat', 'TransformerFilm']
        if (self.mvc_decoder_type is None) or (self.mvc_decoder_type == 'Inner'):
            self.mvc_decoder = MVCDecoder(
                self.d_model,
                arch_style=self.mvc_decoder_style,
                explicit_zero_prob=self.explicit_zero_prob,
                use_batch_labels=self.use_batch_labels,
            )
        elif self.mvc_decoder_type == 'LinearConcat':
            self.mvc_decoder = MVCConcatDecoder(
                self.d_model,
                explicit_zero_prob=self.explicit_zero_prob,
                use_batch_labels=self.use_batch_labels,
            )
        elif self.mvc_decoder_type == 'TransformerConcat':
            self.MidLayer = MidLayer(2 * self.d_model, self.d_model, dropout=self.dropout, cell_emb_concat = True)
            if self.fast_transformer_backend == "flash":
                mvc_layers = FlashTransformerEncoderLayer_FiLM(
                            self.d_model,
                            self.nhead,
                            self.d_hid,
                            self.dropout,
                            batch_first=True,
                            norm_scheme=self.norm_scheme,
                        )
                self.mvc_decoder = FlashTransformerEncoder(mvc_layers, self.nlayers_dec)
            elif self.fast_transformer_backend == "normal":
                mvc_layers = TransformerEncoderLayer(
                    self.d_model, self.nhead, self.d_hid, self.dropout, batch_first=True, norm_first = False
                )
                self.mvc_decoder = TransformerEncoder(mvc_layers, self.nlayers_dec)
                
                
            self.mvc_expr_decoder = ExprDecoder(
                                                self.d_model,
                                                explicit_zero_prob=self.explicit_zero_prob,
                                            )
        
        self.init_weights()
        
        
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        # model_file = Path(path)
        model_files = os.listdir(os.path.join(path, 'plainLogger/0.1/checkpoints'))
        model_files = [file for file in model_files if file.endswith(".ckpt")]
        sort_index = np.argsort(np.array([int(_.split('=')[1].split('-')[0]) for _ in model_files]))
        model_files = np.array(model_files)[sort_index].tolist()
        model_file = os.path.join(path, 'plainLogger/0.1/checkpoints', model_files[-1])

        try:
            self.load_state_dict(torch.load(model_file)['state_dict'])
        except:
            logger.info(f"Load ckpt {model_file} on cuda failed, will load on cpu.")    
            self.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
        logger.info(f"Loading all model params from {model_file}")
        # try:
        #     self.load_state_dict(torch.load(model_file))
        #     logger.info(f"Loading all model params from {model_file}")
        # except:
        #     # only load params that are in the model and match the size
        #     model_dict = self.state_dict()
        #     pretrained_dict = torch.load(model_file)
        #     pretrained_dict = {
        #         k: v
        #         for k, v in pretrained_dict.items()
        #         if k in model_dict and v.shape == model_dict[k].shape
        #     }
        #     for k, v in pretrained_dict.items():
        #         logger.info(f"Loading params {k} with shape {v.shape}")
        #     model_dict.update(pretrained_dict)
        #     self.load_state_dict(model_dict)
        
        

        # sd = torch.load(path, map_location="cpu")
        # if "state_dict" in list(sd.keys()):
        #     sd = sd["state_dict"]
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict.".format(k))
        #             del sd[k]
        # try:
        #     self.load_state_dict(torch.load(model_file))
        #     logger.info(f"Loading all model params from {model_file}")
        # except:
        #     # only load params that are in the model and match the size
        #     model_dict = self.state_dict()
        #     pretrained_dict = torch.load(model_file)
        #     pretrained_dict = {
        #         k: v
        #         for k, v in pretrained_dict.items()
        #         if k in model_dict and v.shape == model_dict[k].shape
        #     }
        #     for k, v in pretrained_dict.items():
        #         logger.info(f"Loading params {k} with shape {v.shape}")
        #     model_dict.update(pretrained_dict)
        #     self.load_state_dict(model_dict)


        
        
    

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.transformer_encoder.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        
    @torch.no_grad()
    def encode(self, data_dict):
        input_gene_ids = data_dict["gene"]
        input_values = data_dict["expr"]
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
        src = input_gene_ids
        values = input_values
        src_key_padding_mask = src_key_padding_mask


        #======================== embed ===========================#
        src = self.encoder(src)  # (batch, seq_len, embsize)
        values = self.value_encoder(values)
        
        total_embs = src + values
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        
        #======================== get cell emb ===========================#
        cell_emb = self._get_cell_emb_from_layer(transformer_output)

        if self.do_vae:
            mu = self.fc_mu(cell_emb) # batch x d_model
            log_var = self.fc_var(cell_emb)
            cell_emb = self.reparameterize(mu, log_var)
            
        return cell_emb

    def set_common_gene(self, input_gene_list1, dec_gene_list2):
        overlap_genes = list(set(input_gene_list1) & set(dec_gene_list2))
        adata_reorder_index = torch.Tensor([input_gene_list1.index(_) for _ in overlap_genes]).long()
        dec_adata_reorder_index = torch.Tensor([dec_gene_list2.index(_) for _ in overlap_genes]).long()
        # self.adata_reorder_index = adata_reorder_index
        # self.dec_adata_reorder_index = dec_adata_reorder_index
        return adata_reorder_index, dec_adata_reorder_index
        
    @torch.no_grad()
    def decode(self, cell_emb, batch):
        print(f'cell_emb device: {cell_emb.device}; first_stage_model device: {self.device}; common_dec_genes device: {self.common_dec_genes.device}')
        data_dict = {k: v.to(self.device) for k, v in batch.items()}
        
    
        input_gene_ids = data_dict["gene"][:, 1:]
        if 'dec_gene' not in data_dict.keys():
            if self.test_use_common_gene:
                assert self.common_dec_genes.any()
                dec_gene_ids = self.common_dec_genes # 1d tensor with cls prefixed
                dec_gene_ids = torch.stack([dec_gene_ids] * input_gene_ids.shape[0]).long().to(self.device) # repeat to batch size   
            else:
                dec_gene_ids = data_dict["gene"]
        else:
            dec_gene_ids = data_dict["dec_gene"]
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])
        # print(f'src_key_padding_mask_dec: {src_key_padding_mask_dec.shape}; {src_key_padding_mask_dec[:, -10:]}')
        
        cur_gene_token_embs = self.encoder(dec_gene_ids[:, 1:])

        adata_reorder_index, dec_adata_reorder_index = self.set_common_gene(input_gene_ids[0].tolist(), dec_gene_ids[:, 1:][0].tolist()) 
        

        
        
        # data_dict = batch
        # if ('dec_gene' not in data_dict.keys()):
        #     dec_gene_ids = data_dict["gene"][:, 1:]
        # else:
        #     dec_gene_ids = data_dict["dec_gene"][:, 1:]
        # src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])

        # cur_gene_token_embs = self.encoder(dec_gene_ids)

        if self.do_vae:
            cell_emb = self.post_moment(cell_emb)
        #======================== decoder ===========================#
        if (self.mvc_decoder_type is None) or (self.mvc_decoder_type == 'Inner'):
            dec_output = self.mvc_decoder(
                cell_emb,
                cur_gene_token_embs,
            )
        elif self.mvc_decoder_type == 'LinearConcat':
            dec_output = self.mvc_decoder(
                cell_emb,
                cur_gene_token_embs,
            )
        elif self.mvc_decoder_type == 'TransformerConcat':
            src_key_padding_mask = src_key_padding_mask_dec[:, 1:] # dim: batch x gene_len 
            dec_output = self.MidLayer(cur_gene_token_embs, cell_emb)
            dec_output = self.mvc_decoder(dec_output, src_key_padding_mask = src_key_padding_mask)
            dec_output = self.mvc_expr_decoder(dec_output)
        
        output = {}
        output['cell_preds'] = dec_output["pred"]
        output['cell_emb'] = cell_emb
        output['celltypes'] = data_dict['celltype_code']
        output['adata_reorder_index'] = adata_reorder_index
        output['dec_adata_reorder_index'] = dec_adata_reorder_index
        

        return output

        

    def training_step(self, batch, batch_idx):
        self.train()
        loss, loss_dict, _ = self.p_losses(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        



        outputs = {'loss': loss}

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            outputs['lr'] = lr

        outputs.update(loss_dict)

        return outputs

    def on_train_batch_end(self, outputs, batch, batch_idx):
        keys = list(outputs.keys())
        dict_to_print = {}
        for k in keys:
            try:
                dict_to_print[k] = torch.mean(outputs[k]).item()
            except:
                dict_to_print[k] = outputs[k]
        self.logger.log_metrics_tofile(dict_to_print, step=self.global_step, epoch = self.current_epoch)


    
    def on_validation_epoch_start(self):
        print(f'cell_emb_sample_iter_num: {self.cell_emb_sample_iter_num}')
        
        self.sample_cells_out_of_sample = []
        self.sample_cell_embs_out_of_sample = []
        self.celltype_column_out_of_sample = []
        self.cell_id_out_of_sample = []
        self.dataset_idx_out_of_sample = []
    

        self.sample_cells_test = []
        self.sample_cell_embs_test = []
        self.celltype_column_test = []
        self.cell_id_test = []
        self.dataset_idx_test = []

        # if self.current_epoch % 10 == 0:
        #     print(f'save ckpt at epoch: {self.current_epoch}')
        #     self.trainer.save_checkpoint()
    
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):  
        self.eval()
        if batch_idx > self.cell_emb_sample_iter_num:
            return

        # output_dict possible keys: (
        #     gene_expr_zero_probs,
        #     gene_expr_output,
        #     cell_emb,
        #     cell_emb_for_diffusion,
        #     cell_emb_beforeDec,
        #     recon_output,
        #     recon_zero_probs
        # )

        if self.inference_mode:
            dataloader_idx = 1
        if dataloader_idx == 0: # out of sample
            
            _, loss_dict_no_ema, _ = self.p_losses(batch)
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            output_dict = self.p_losses_test(batch, dec_use_common_gene = self.valid_use_common_gene)
        elif dataloader_idx == 1: # out of sample
            output_dict = self.p_losses_test(batch, dec_use_common_gene = self.test_use_common_gene)
            
        cell_id, X_samples, X_sample_cell_emb, celltype_column = batch['id'], output_dict['recon_output'].detach(), output_dict['cell_emb'].detach(), batch['celltype_code'].detach()

        out_dict = {'cell_id': cell_id, 'sample_cells': X_samples, 'sample_cell_embs': X_sample_cell_emb, 'celltype_column': celltype_column}
        if 'dataset_idx' in batch.keys():
            out_dict['dataset_idx'] = batch['dataset_idx']
        else:
            out_dict['dataset_idx'] = torch.zeros(len(cell_id))
        return out_dict
        
    @torch.no_grad()
    def on_validation_batch_end(self, batch_parts, batch, batch_idx, dataloader_idx = 0): # batch_parts already combined the results of all gpus!! great
        if batch_idx > self.cell_emb_sample_iter_num:
            return
        if dataloader_idx == 0: # out_of_sample
            self.sample_cells_out_of_sample.extend(batch_parts['sample_cells'].tolist())
            self.sample_cell_embs_out_of_sample.extend(batch_parts['sample_cell_embs'].tolist())
            self.celltype_column_out_of_sample.extend(batch_parts['celltype_column'].tolist())
            self.cell_id_out_of_sample.extend(batch_parts['cell_id'].tolist()) 
            self.dataset_idx_out_of_sample.extend(batch_parts['dataset_idx'].tolist()) 
        if dataloader_idx == 1: # test
            self.sample_cells_test.extend(batch_parts['sample_cells'].tolist())
            self.sample_cell_embs_test.extend(batch_parts['sample_cell_embs'].tolist())
            self.celltype_column_test.extend(batch_parts['celltype_column'].tolist())
            self.cell_id_test.extend(batch_parts['cell_id'].tolist()) 
            self.dataset_idx_test.extend(batch_parts['dataset_idx'].tolist()) 
            
            

            
    @torch.no_grad()    
    def on_validation_epoch_end(self): 
        self.celltype_column_out_of_sample = torch.Tensor(self.celltype_column_out_of_sample).to(self.device)
        self.sample_cells_out_of_sample = torch.Tensor(self.sample_cells_out_of_sample).to(self.device)
        self.sample_cell_embs_out_of_sample = torch.Tensor(self.sample_cell_embs_out_of_sample).to(self.device)
        self.cell_id_out_of_sample = torch.Tensor(self.cell_id_out_of_sample).to(self.device)
        self.dataset_idx_out_of_sample = torch.Tensor(self.dataset_idx_out_of_sample).to(self.device)
        

        if len(self.CellTypeMapping_df) == 2:
            self.celltype_column_test = torch.Tensor(self.celltype_column_test).to(self.device)
            self.sample_cells_test = torch.Tensor(self.sample_cells_test).to(self.device)
            self.sample_cell_embs_test = torch.Tensor(self.sample_cell_embs_test).to(self.device)
            self.cell_id_test = torch.Tensor(self.cell_id_test).to(self.device)
            self.dataset_idx_test = torch.Tensor(self.dataset_idx_test).to(self.device)
        
    
        # world_size = self.trainer.num_processes
        # world_size = self.trainer.strategy.cluster_environment.world_size()
        # world_size = self.trainer.world_size()
        # current_rank = self.trainer.global_rank
        world_size = torch.distributed.get_world_size()
        print(f'world_size: {world_size}')

        def plot_cell_sample(cell_id, celltype_column, sample_cells, sample_cell_embs, CellTypeMapping_df, dataset_idx = None, suffix = None, background_adata = None, testDS = False):
            def gather_list(tensor_to_be_gathered):
                # gather the results from all the processes
                tensor_to_be_gathered_list = [torch.zeros_like(torch.Tensor(tensor_to_be_gathered)) for _ in range(world_size)]
                print(f'tensor_to_be_gathered_list len: {len(tensor_to_be_gathered_list)}')
                print(f'tensor_to_be_gathered_list 1st element shape: {tensor_to_be_gathered_list[0].shape}')
                torch.distributed.all_gather(tensor_to_be_gathered_list, tensor_to_be_gathered)
                print(f'tensor_to_be_gathered_list after gathering len: {len(tensor_to_be_gathered_list)}')
                # print(f'tensor after gathering values from 0,1: {tensor_to_be_gathered_list[0][0], tensor_to_be_gathered_list[1][0]}')
                cat_tensor = torch.cat(tensor_to_be_gathered_list).tolist()
                print(f'tensor_to_be_gathered_list after cating shape: {torch.cat(tensor_to_be_gathered_list).shape}')
                return cat_tensor

            
            # def gather_list(tensor_to_be_gathered):
            #     # gather the results from all the processes
            #     tensor_to_be_gathered_list = list(self.all_gather(tensor_to_be_gathered))
            #     print(f'# ======================= ------->>>>  tensor_to_be_gathered_list shape :{tensor_to_be_gathered_list.shape}')
            #     cat_tensor = torch.cat(tensor_to_be_gathered_list).tolist()
            #     return cat_tensor
                
            celltype_column = gather_list(celltype_column)
            sample_cells = gather_list(sample_cells)
            sample_cell_embs = gather_list(sample_cell_embs)
            cell_id = gather_list(cell_id)
            dataset_idx = gather_list(dataset_idx)
            
    
            celltypes = torch.Tensor(celltype_column).tolist()
            celtype_df_column = CellTypeMapping_df.columns[0]
            celltype_column = CellTypeMapping_df.iloc[celltypes][celtype_df_column].tolist()
            cell_id = torch.Tensor(cell_id).tolist()
            dataset_idx = [int(_) for _ in torch.Tensor(dataset_idx).tolist()]
        

                
            
            X_samples = torch.Tensor(sample_cells).cpu().numpy()
            sample_cells_adata = sc.AnnData(X = X_samples)
            sample_cells_adata.X = csr_matrix(sample_cells_adata.X)
            sample_cells_adata.obs['cell_type'] = celltype_column
            sample_cells_adata.obs['cell_id'] = cell_id
            sample_cells_adata.obs['dataset_idx'] = ['dataset_' + str(_) for _ in dataset_idx]
#             self.logger.save_h5ad(sample_cells_adata, f'{self.current_epoch}_sample_cell_adata_{suffix}')
#             self.logger.log_info(f'Writing {self.current_epoch}_sample_cell_adata.h5ad in {self.logger._save_dir}.')
            self.logger.log_info(f'sample_cell_adata shape: {sample_cells_adata.shape}.')
            
            X_sample_cell_emb = torch.Tensor(sample_cell_embs).cpu().numpy()
            sample_cell_emb_adata = sc.AnnData(X = X_sample_cell_emb)
            sample_cell_emb_adata.X = csr_matrix(sample_cell_emb_adata.X)
            sample_cell_emb_adata.obs['cell_type'] = celltype_column
            sample_cell_emb_adata.obs['cell_id'] = cell_id
            sample_cell_emb_adata.obs['dataset_idx'] = ['dataset_' + str(_) for _ in dataset_idx]
#             self.logger.save_h5ad(sample_cell_emb_adata, f'{self.current_epoch}_sample_cell_emb_adata_{suffix}')
#             self.logger.log_info(f'Writing {self.current_epoch}_sample_cell_emb_adata.h5ad in {self.logger._save_dir}.')
            self.logger.log_info(f'sample_cell_emb_adata shape: {sample_cell_emb_adata.shape}.')

            
            self.plotsampleumap_single(
                sample_adata = sample_cell_emb_adata,
                metric = 'euclidean',
                prefix = f'cell_emb_{suffix}',
                save_dir = self.logger._save_dir / 'sample_img',
                step = self.current_epoch
            )

            if testDS:
                if self.adata_reorder_index is not None:
                    background_adata = background_adata[:, self.adata_reorder_index.tolist()]
                    sample_cells_adata = sample_cells_adata[:, self.dec_adata_reorder_index.tolist()]
            else:
                background_adata = background_adata
            if background_adata is not None:
                sample_cells_adata.var = background_adata.var.copy()
    
                self.plotsampleumap(
                    sample_adata = sample_cells_adata,
                    background_adata = background_adata,
                    prefix = f'cell_preds_{suffix}'
                )
            else:
                self.plotsampleumap_single(
                    sample_adata = sample_cells_adata,
                    metric = 'cosine',
                    prefix = f'cell_preds_{suffix}',
                    save_dir = self.logger._save_dir / 'sample_img',
                    step = self.current_epoch
                )

            celltype_column.clear()
            sample_cells.clear()
            sample_cell_embs.clear()
            cell_id.clear()

            

        # out of sample
        
        plot_cell_sample(
            self.cell_id_out_of_sample,
            self.celltype_column_out_of_sample,
            self.sample_cells_out_of_sample,
            self.sample_cell_embs_out_of_sample,
            CellTypeMapping_df = self.CellTypeMapping_df[0],
            dataset_idx = self.dataset_idx_out_of_sample,  
            suffix = 'out_of_sample',
            background_adata = self.background_adatas[0] if self.background_adata_paths is not None else None,
            testDS = self.inference_mode
        )
    
        # test
        if len(self.CellTypeMapping_df) == 2:
            plot_cell_sample(
                self.cell_id_test,
                self.celltype_column_test,
                self.sample_cells_test,
                self.sample_cell_embs_test,
                CellTypeMapping_df = self.CellTypeMapping_df[1],
                dataset_idx = self.dataset_idx_test,  
                suffix = 'test',
                background_adata = self.background_adatas[1] if self.background_adata_paths is not None else None,
                testDS = True
            )
        

    def plotsampleumap_single(self, sample_adata, metric = 'cosine', prefix = '', save_dir = None, step = None):
        if sample_adata.shape[0] > 10000:
            subsample_index = np.random.choice(np.arange(sample_adata.shape[0]), 10000, replace = False)
            sample_adata = sample_adata[subsample_index, :]
        
        sc.pp.pca(sample_adata,svd_solver='arpack', n_comps=30, use_highly_variable=False)

        if metric == 'cosine':
            sc.pp.neighbors(sample_adata, metric='cosine', n_neighbors=30, n_pcs = 30)
        else:
            sc.pp.neighbors(sample_adata, n_neighbors=30, n_pcs = 30)
        sc.tl.umap(sample_adata, min_dist = 0.3, spread = 1, maxiter=100)

        with mpl.rc_context({'figure.figsize': [10, 10],
                         'axes.facecolor': 'white'}):
            sc.pl.umap(sample_adata, color=['cell_type'], size=15,
                       color_map = 'RdPu', ncols = 1, na_in_legend=False, legend_loc='on data',
                       legend_fontsize=10, return_fig = True)
            plt.tight_layout()
            if save_dir is not None:
                plt.savefig(save_dir / f"{step}_celltype_{prefix}.png") 
            plt.show()
            plt.close()

        with mpl.rc_context({'figure.figsize': [10, 10],
                         'axes.facecolor': 'white'}):
            sc.pl.umap(sample_adata, color=['dataset_idx'], size=15,
                       color_map = 'RdPu', ncols = 1, na_in_legend=False, legend_loc='on data',
                       legend_fontsize=10, return_fig = True)
            plt.tight_layout()
            if save_dir is not None:
                plt.savefig(save_dir / f"{step}_batch_{prefix}.png") 
            plt.show()
            plt.close()


        
    def plotsampleumap(self, sample_adata, background_adata, prefix = ''):            
        gen_sample_adata = sample_adata
        plotsampledata(gen_sample_adata,background_adata,scale=False, save_dir = self.logger._save_dir / 'sample_img', prefix = prefix, step = self.current_epoch)

        
        
    def p_losses_test(self, batch, dec_use_common_gene = True):
        assert not self.training
        data_dict = {k: v.to(self.device) for k, v in batch.items()}
        
    
        input_gene_ids = data_dict["gene"]
        if 'dec_gene' not in data_dict.keys():
            if dec_use_common_gene:
                assert self.common_dec_genes.any()
                dec_gene_ids = self.common_dec_genes # 1d tensor with cls prefixed
                dec_gene_ids = torch.stack([dec_gene_ids] * input_gene_ids.shape[0]).long().to(self.device) # repeat to batch size      
            else:
                dec_gene_ids = data_dict["gene"]
        else:
            dec_gene_ids = data_dict["dec_gene"]
            
        input_values = data_dict["expr"]

        

        
            
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])

        
        # with torch.cuda.amp.autocast(enabled=True):
        output_dict = self(
            input_gene_ids,
            dec_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            src_key_padding_mask_dec=src_key_padding_mask_dec
        )
        
    
        return output_dict

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        
        bs, seq_len = genes.shape

        gene_pad_tensor = torch.full((bs, max_length - seq_len), fill_value = self.vocab[self.pad_token], dtype=genes.dtype, device = genes.device)
        genes = torch.cat([genes, gene_pad_tensor], dim = -1)
         
        expr_pad_tensor = torch.full((bs, max_length - seq_len), fill_value = self.pad_value, dtype=expressions.dtype, device = genes.device)
        expressions = torch.cat([expressions, expr_pad_tensor], dim = -1)

        return genes, expressions


    
    def p_losses(self, batch):
        data_dict = {k: v.to(self.device) for k, v in batch.items()}
        
    
        input_gene_ids = data_dict["gene"]
        if 'dec_gene' not in data_dict.keys():
            dec_gene_ids = data_dict["gene"]
        else:
            dec_gene_ids = data_dict["dec_gene"]

            
        input_values = data_dict["expr"]
        if 'dec_expr' not in data_dict.keys():
            target_values = data_dict["expr"] #[:, 1:]
        else:
            target_values = data_dict["dec_expr"] #[:, 1:]
            

        
        # #====================== padding ======================#
        # bs, max_len = input_gene_ids.shape
        # max_length = torch.tensor(max_len).to(self.device)
        # if True: #self.training:
        #     print(max_length)
        # dist.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX) # Synchronize and find the maximum length across all GPUs
        # if True: #self.training:
        #     self.logger.log_info(f'======================= oringinal max_length at {torch.distributed.get_rank()} is {max_length} =======================')
        # max_length = max_length.item() + 1 # ensure padding
        # input_gene_ids, input_values = self._pad(input_gene_ids, input_values, max_length)
        # dec_gene_ids, target_values = self._pad(dec_gene_ids, target_values, max_length)
        # if True: #self.training:
        #     new_gene_shape = input_gene_ids.shape
        #     self.logger.log_info(f'data_dict[gene] shape: {new_gene_shape}')
        #     self.logger.log_info(f'======================= after combine all device, max_length at {torch.distributed.get_rank()} is {max_length} =======================')

        target_values = target_values[:, 1:]
        
        # print(f'input_gene_ids: {input_gene_ids.shape}; {input_gene_ids[:, -10:]}')    
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])

        # print(f'src_key_padding_mask after eq: {src_key_padding_mask.shape}; {src_key_padding_mask[:, -10:]}')
        
        mask = (~src_key_padding_mask)[:, 1:]
        mask_dec = (~src_key_padding_mask_dec)[:, 1:]
        criterion = criterion_neg_log_poisson
        criterion_gene = F.mse_loss
        if self.L1_reg:
            criterion_L1reg = torch.mean 
    
    
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # with torch.cuda.amp.autocast(enabled=True):
        output_dict = self(
            input_gene_ids,
            dec_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            src_key_padding_mask_dec=src_key_padding_mask_dec
        )
        output_values = output_dict["recon_output"]
        loss = loss_recon = criterion(output_values, target_values, mask_dec)
        loss_dict.update({f'{prefix}/recon': loss_recon})

        if self.L1_reg:
            loss_L1reg = self.L1_reg_weight * criterion_L1reg(torch.abs(output_values[mask_dec]))
            loss = loss + loss_L1reg
            loss_dict.update({f'{prefix}/L1_reg': loss_L1reg})

        if self.add_gene_loss:
            gene_expr_output = output_dict["gene_expr_output"]
            if 'dec_expr' in data_dict.keys():
                loss_geneloss = torch.tensor(0).to(self.device)
            else:
                loss_geneloss = criterion_gene(gene_expr_output[mask], input_values[:, 1:][mask])
            loss = loss + loss_geneloss
            loss_dict.update({f'{prefix}/geneloss': loss_geneloss})


        if self.do_vae:
            loss_kld = output_dict["kld_loss"]
            loss_kld = loss_kld.mean()
            loss = loss + loss_kld
            loss_dict.update({f'{prefix}/kld_loss': loss_kld})

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict, output_dict

    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(
        self,
        src: Tensor,
        dec_src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        src_key_padding_mask_dec: Tensor,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]

        Returns:
            dict of output Tensors.
        """
        #======================== embed ===========================#
        src = self.encoder(src)  # (batch, seq_len, embsize)
        dec_src = self.encoder(dec_src)
        values = self.value_encoder(values)
        self.cur_gene_token_embs = dec_src[:, 1:, :]
        
        total_embs = src + values
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        
        output = {}
        #======================== gene expr decode ===========================#
        if self.decode_gene:
            gene_expr_output = self.decoder(transformer_output)

            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=gene_expr_output["zero_probs"])
                gene_expr_output = bernoulli.sample() * gene_expr_output["pred"]
                output["gene_expr_output"] = gene_expr_output[:, 1:]
            else:
                output["gene_expr_output"] = gene_expr_output["pred"][:, 1:]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["gene_expr_zero_probs"] = gene_expr_output["zero_probs"]
                output["gene_expr_output"] = gene_expr_output[:, 1:]

        

        #============================================ decoder ===============================================#
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb
        output["cell_emb_for_diffusion"] = cell_emb
        output["cell_emb_beforeDec"] = cell_emb
        


        #<<<<<<<<<<<<<<<<<========== vae ==========>>>>>>>>>>>>>>>>>>>#
        if self.do_vae:
            mu = self.fc_mu(cell_emb) # batch x d_model
            log_var = self.fc_var(cell_emb)

    #         Computes the VAE loss function.
    #         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))
            output['kld_loss'] = self.kld_weight * kld_loss
            
            cell_emb = self.reparameterize(mu, log_var)
            output["cell_emb_for_diffusion"] = cell_emb

            cell_emb = self.post_moment(cell_emb)
            output["cell_emb_beforeDec"] = cell_emb
            


        
        #======================== decoder ===========================#
        if (self.mvc_decoder_type is None) or (self.mvc_decoder_type == 'Inner'):
            dec_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )
        elif self.mvc_decoder_type == 'LinearConcat':
            dec_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )
        elif self.mvc_decoder_type == 'TransformerConcat':
            src_key_padding_mask = src_key_padding_mask_dec[:, 1:] # dim: batch x gene_len 
            dec_output = self.MidLayer(self.cur_gene_token_embs, cell_emb)
            dec_output = self.mvc_decoder(dec_output, src_key_padding_mask = src_key_padding_mask)
            dec_output = self.mvc_expr_decoder(dec_output)
        
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=dec_output["zero_probs"])
            output["recon_output"] = bernoulli.sample() * dec_output["pred"]
        else:
            output["recon_output"] = dec_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["recon_zero_probs"] = dec_output["zero_probs"] 

        return output

    def configure_optimizers(self):
        lr = self.learning_rate
    
        params = list(self.parameters())
        
        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]

            return [opt], scheduler
            
        return opt




    
    
class MidLayer(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, decoder_d_model: int, dropout: float = 0.1, cell_emb_concat: bool = False):
        super().__init__()
        self.cell_emb_concat = cell_emb_concat
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, decoder_d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(decoder_d_model, decoder_d_model)
        self.norm = nn.LayerNorm(decoder_d_model)

    def forward(self, x: Tensor, cell_emb=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        if self.cell_emb_concat:
            cell_emb = cell_emb.unsqueeze(1)
            cell_emb = cell_emb.repeat(1,x.shape[1],1)
            x = torch.cat([cell_emb, x], dim = -1) # (batch, seq_len, 1024)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x) 

class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    
    
class FeatureWiseAffine(BaseModule):
    def __init__(self, num_features):
        super().__init__()
#         super(FeatureWiseAffine, self).__init__()
#         self.instance_norm = torch.nn.InstanceNorm1d(num_features)
        self.layer_norm = torch.nn.LayerNorm(num_features)

    def forward(self, x, scale, shift):
#         x = self.instance_norm(x)
        x = self.layer_norm(x)
        outputs = scale * x + shift
        return outputs
    

    
class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = in_channels,
                                              out_features = out_channels), 
                                    torch.nn.LeakyReLU(0.2)
                                ])
        
        self.scale_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = out_channels,
                                              out_features = out_channels), 
                                ])
        self.shift_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = out_channels,
                                              out_features = out_channels), 
                                ])

    def forward(self, x):
        outputs = self.signal_linear(x)
        scale, shift = self.scale_linear(outputs), self.shift_linear(outputs)
        return scale, shift
    
    
    
class TransformerEncoderLayer_FiLM(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """
    __constants__ = ['norm_first']

    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, 
                 norm_first: bool = False,
                 bias: bool = True, 
                 device=None, 
                 dtype=None,  
                 FiLM_in_channel=None, 
                 FiLM_out_channel=None,
                 no_attn_cell = True
                ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.no_attn_cell = no_attn_cell
        if self.no_attn_cell:
            FiLM_out_channel = FiLM_out_channel - 1
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.linear1_cell = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout_cell = Dropout(dropout)
        self.linear2_cell = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout2_cell = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
        
        
        self.featurewise_affine1 = FeatureWiseAffine(d_model)
        self.featurewise_affine2 = FeatureWiseAffine(d_model)

        self.films1 = FeatureWiseLinearModulation(
                                in_channels=FiLM_in_channel,
                                out_channels=FiLM_out_channel 
                            )
        self.films2 = FeatureWiseLinearModulation(
                                in_channels=FiLM_in_channel,
                                out_channels=FiLM_out_channel 
                            )

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            cell_emb: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )



        
        if self.no_attn_cell:
            src_key_padding_mask = src_key_padding_mask[:, 1:]
            cell_emb_extract = src[:, 0, :]
            x = src[:, 1:, :]
            if self.norm_first:
                # replace norm by film
                #==================== film ====================#
                scale1, shift1 = self.films1(cell_emb_extract) # batch x emb_dim --> batch x (gene_len + 1(cell_emb))
                scale1 = scale1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim 
                shift1 = shift1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine1(x, scale1, shift1)
                #==================== forward attn ====================#
                x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
                
                # replace norm by film
                #==================== film ====================#
                scale2, shift2 = self.films2(cell_emb_extract) # batch x emb_dim --> batch x (gene_len + 1(cell_emb))
                scale2 = scale2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim 
                shift2 = shift2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine2(x, scale2, shift2)
                #==================== forward ====================#
                cell_emb_extract = cell_emb_extract + self._ff_block_cell(self.norm1(cell_emb_extract))
                x = x + self._ff_block(x)
                x = torch.cat((cell_emb_extract.unsqueeze(1), gene_emb_extract), dim = 1)
                
            else:
                #==================== forward attn ====================#
                x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
                # replace norm by film
                #==================== film ====================#
                scale1, shift1 = self.films1(cell_emb_extract) # emb_dim --> (gene_len + 1(cell_emb))
                scale1 = scale1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim 
                shift1 = shift1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine1(x, scale1, shift1)
                #==================== forward ====================#
                cell_emb_extract = self.norm1(cell_emb_extract + self._ff_block_cell(cell_emb_extract))
                x = x + self._ff_block(x)
                
                # replace norm by film
                #==================== film ====================#
                scale2, shift2 = self.films2(cell_emb_extract) # emb_dim --> (gene_len + 1(cell_emb))
                scale2 = scale2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim 
                shift2 = shift2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x gene_len --> batch x gene_len x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine2(x, scale2, shift2)
                
                x = torch.cat((cell_emb_extract.unsqueeze(1), x), dim = 1)

        else:      
            x = src
            if self.norm_first:
                # replace norm by film
                #==================== film ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                scale1, shift1 = self.films1(cell_emb_extract) # batch x emb_dim --> batch x (gene_len + 1(cell_emb))
                scale1 = scale1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim 
                shift1 = shift1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine1(x, scale1, shift1)
                #==================== forward attn ====================#
                x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)

                # replace norm by film
                #==================== film ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                scale2, shift2 = self.films2(cell_emb_extract) # batch x emb_dim --> batch x (gene_len + 1(cell_emb))
                scale2 = scale2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim 
                shift2 = shift2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine2(x, scale2, shift2)
                #==================== forward ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                gene_emb_extract = x[:, 1:, :]
                cell_emb_extract = cell_emb_extract + self._ff_block_cell(cell_emb_extract)
                gene_emb_extract = gene_emb_extract + self._ff_block(gene_emb_extract)
                x = torch.cat((cell_emb_extract.unsqueeze(1), gene_emb_extract), dim = 1)


            else:
                #==================== forward attn ====================#
                x = x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
                # replace norm by film
                #==================== film ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                scale1, shift1 = self.films1(cell_emb_extract) # emb_dim --> (gene_len + 1(cell_emb))
                scale1 = scale1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim 
                shift1 = shift1.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine1(x, scale1, shift1)
                #==================== forward ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                gene_emb_extract = x[:, 1:, :]
                cell_emb_extract = cell_emb_extract + self._ff_block_cell(cell_emb_extract)
                gene_emb_extract = gene_emb_extract + self._ff_block(gene_emb_extract)
                x = torch.cat((cell_emb_extract.unsqueeze(1), gene_emb_extract), dim = 1)

                # replace norm by film
                #==================== film ====================#
                cell_emb_extract = x[:, 0, :] # batch x emb_dim
                scale2, shift2 = self.films2(cell_emb_extract) # emb_dim --> (gene_len + 1(cell_emb))
                scale2 = scale2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim 
                shift2 = shift2.unsqueeze(2).repeat(1, 1, x.shape[2]) # batch x (gene_len + 1) --> batch x (gene_len + 1) x emb_dim
                #==================== film norm ====================#
                x = self.featurewise_affine2(x, scale2, shift2)

#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
#             x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    # feed forward block
    def _ff_block_cell(self, x: Tensor) -> Tensor:
        x = self.linear2_cell(self.dropout_cell(self.activation(self.linear1_cell(x))))
        return self.dropout2_cell(x)
    
    
    
    
    

    
class FlashTransformerEncoderLayer_FiLM(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
        cond_norm=False,
        FeatureWiseAffine_num_features=None,
        FiLM_in_channel=None,
        FiLM_out_channel=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
#             batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        self.self_attn.batch_first = batch_first
        self.cond_norm = cond_norm
        
            
        if cond_norm:
            self.featurewise_affine1 = FeatureWiseAffine(FeatureWiseAffine_num_features)
            self.featurewise_affine2 = FeatureWiseAffine(FeatureWiseAffine_num_features)
            
            self.films1 = FeatureWiseLinearModulation(
                                    in_channels=FiLM_in_channel,
                                    out_channels=FiLM_out_channel 
                                )
            self.films2 = FeatureWiseLinearModulation(
                                    in_channels=FiLM_in_channel,
                                    out_channels=FiLM_out_channel 
                                )
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        cell_emb: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        # print(f'src_key_padding_mask: {src_key_padding_mask.shape}; {src_key_padding_mask[:, -10:]}')
        if not src_key_padding_mask.any().item():
        # if not src_key_padding_mask.sum().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask
            
            

        if self.cond_norm:
            if self.norm_scheme == "pre":
                # cell_emb dim : batch x d_model
                # self.films1(cell_emb) scale dim: batch x dec_d_model
                # src dim: batch x gene_len x dec_model
                scale1, shift1 = self.films1(cell_emb)
                scale1 = scale1.unsqueeze(1).repeat(1, src.shape[1], 1)
                shift1 = shift1.unsqueeze(1).repeat(1, src.shape[1], 1)
                src = self.featurewise_affine1(src, scale1, shift1) 
                src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
                src = src + self.dropout1(src2)
                scale2, shift2 = self.films2(cell_emb)
                scale2 = scale2.unsqueeze(1).repeat(1, src.shape[1], 1)
                shift2 = shift2.unsqueeze(1).repeat(1, src.shape[1], 1)
                src = self.featurewise_affine2(src, scale2, shift2)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
            else:
                src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
                src = src + self.dropout1(src2)
                scale1, shift1 = self.films1(cell_emb)
                scale1 = scale1.unsqueeze(1).repeat(1, src.shape[1], 1)
                shift1 = shift1.unsqueeze(1).repeat(1, src.shape[1], 1)
                src = self.featurewise_affine1(src, scale1, shift1) 
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                scale2, shift2 = self.films2(cell_emb)
                scale2 = scale2.unsqueeze(1).repeat(1, src.shape[1], 1)
                shift2 = shift2.unsqueeze(1).repeat(1, src.shape[1], 1)
                src = self.featurewise_affine2(src, scale2, shift2)
            
        else:
            if self.norm_scheme == "pre":
                src = self.norm1(src)
                src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
                src = src + self.dropout1(src2)
                src = self.norm2(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
            else:
                src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)

        return src

    
    
class FlashTransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None,
            cell_emb: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)


        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, cell_emb = cell_emb)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output
    
    
    
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

    
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
        
class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":
        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings  # (N,)
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len, embsize]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
#             batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        self.self_attn.batch_first = batch_first
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
        # if not src_key_padding_mask.sum().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

class DepthScalarDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hid: int = 128,
    ):
        super().__init__()
        d_in = d_model 
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.LeakyReLU(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the cell_emb, (batch, d_model)"""
        out_val = self.fc(x).squeeze(-1)  # (batch, seq_len)

        depth_scalar = torch.sigmoid(out_val)
        return depth_scalar # shape: 1D batch


class MVCConcatDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor, gene_name_emb: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        x = x.unsqueeze(1)
        x = x.repeat(1,gene_name_emb.shape[1],1)
        x = torch.cat([x, gene_name_emb], dim = -1) # (batch, seq_len, 1024)
        
        pred_value = self.fc(x).squeeze(-1)    # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.





class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)



    
class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




