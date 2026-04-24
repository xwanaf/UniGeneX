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
#======================== dataloader ========================#
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import random
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler



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


from scgpt_pcpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_poisson, smooth_l1_loss
from util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config

import scgpt_pcpt as scg
# from scgpt_pcpt.preprocess import Preprocessor
from scgpt_pcpt.tokenizer import GeneVocab, random_mask_value, tokenize_batch
from scgpt_pcpt.scbank import DataBank
from scgpt_pcpt.utils import MainProcessOnly, ConfigWrapper, configure_logging
from scgpt_pcpt import logger
# from diffusion.plotumap import plotsampledata


try:
    # from flash_attn.flash_attention import FlashMHA
    from scgpt_pcpt.model.flash_attention import FlashMHA
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    
class TransformerGeneCorr(pl.LightningModule):
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
        pre_norm: bool = False,
        kld_weight: float = 0.00001,
        ckpt_path = None,
        scheduler_config=None,
        monitor = None,
        common_dec_gene_len = 0,
        CellTypeMapping_df_paths=None,
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
        
        self.pre_norm = pre_norm
        self.norm_scheme = "pre" if pre_norm else "post"
        self.pad_token = pad_token
    
        self.kld_weight = kld_weight
        self.scheduler_config = scheduler_config
        self.use_scheduler = scheduler_config is not None
    
    
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
            
        
        self.build_network()
        

    def build_network(self):

        #=================================== embedding ==================================#
        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(self.ntoken, self.d_model, padding_idx=self.vocab[self.pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        self.value_encoder = ContinuousValueEncoder(self.d_model, self.dropout)


        encoder_layers = FlashTransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.d_hid,
            self.dropout,
            batch_first=True,
            norm_scheme=self.norm_scheme,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        #=================================== gene mean (\mu) and variance (\Sig_tilde) ==================================#
        self.fc_mu_gene = GeneStatsDecoder(self.d_model, self.d_model)
        self.fc_mid_gene = GeneStatsDecoder(self.d_model, self.d_model, self.d_model)
        self.fc_var_gene = GeneStatsDecoder(self.d_model, self.d_model)

        
        # self.fc_var_z = GeneStatsDecoder(self.d_model, self.d_model)
        # self.fc_var_z_post = GeneStatsDecoder(
        #     self.common_dec_genes.shape[0] - 1,
        #     self.common_dec_genes.shape[0] - 1,
        #     self.d_model
        # )
        # self.fc_var_gene = GeneStatsDecoder(self.d_model, self.d_model)

        #=================================== vae ==================================#
        self.fc_mu = nn.Linear(self.d_model, self.d_model)
        self.fc_var = nn.Linear(self.d_model, self.d_model)

        
        
        self.init_weights()
        
    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

#     def train_dataloader(self):
#         print(' ######################## ')
#         print('reset train dataloader')
#         print(' ######################## ')

#         sampler = RandomSampler(self.dataset, num_samples=len(self.dataset), replacement=False)
#         collator_train = DataCollator(
#             pad_token_id=vocab[self.pad_token],
#             pad_value=self.pad_value,
#             mask_ratio = 0.5
#         )
        
#         train_loader = DataLoader(self.dataset, sampler=sampler, batch_size=64, collate_fn=collator_train, shuffle=True, pin_memory=True)
        
#         return train_loader

    
    
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
        self.sample_cell_embs_out_of_sample = []
        self.celltype_column_out_of_sample = []
        self.cell_id_out_of_sample = []
        self.dataset_idx_out_of_sample = []
        

            
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):  
        self.eval()
        
        _, loss_dict_no_ema, output_dict = self.p_losses(batch)
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        if batch_idx == 0:
            #======================== recon/decoder ========================#
            mu = self.fc_mu_gene(self.common_gene_token_embs.unsqueeze(0)) # batch x seq_len x d_model
           
            V_common = self.common_gene_token_embs
           
        
            self.V_common = V_common
            self.mu = mu
            
        z = output_dict['cell_emb'].detach()
#         z = z + self.transfered_mu

        cell_id, celltype_column = batch['id'], batch['celltype_code'].detach()
        out_dict = {'cell_id': cell_id, 'celltype_column': celltype_column, 'sample_cell_embs': z}

            
        if 'dataset_idx' in batch.keys():
            out_dict['dataset_idx'] = batch['dataset_idx']
        else:
            out_dict['dataset_idx'] = torch.zeros(len(cell_id))
        # return {'cell_id': cell_id, 'celltype_column': celltype_column, 'attn_weights': attn_weights, 'sample_cells': X_samples, 'sample_cell_embs': X_sample_cell_emb}
        return out_dict



        
        
        
    @torch.no_grad()
    def on_validation_batch_end(self, batch_parts, batch, batch_idx, dataloader_idx = 0): # batch_parts already combined the results of all gpus!! great

        if dataloader_idx == 0: # out_of_sample
            self.cell_id_out_of_sample.extend(batch_parts['cell_id'].tolist())  
            self.celltype_column_out_of_sample.extend(batch_parts['celltype_column'].tolist())
            self.sample_cell_embs_out_of_sample.extend(batch_parts['sample_cell_embs'].tolist())
            self.dataset_idx_out_of_sample.extend(batch_parts['dataset_idx'].tolist()) 

            
    @torch.no_grad()    
    def on_validation_epoch_end(self): 
        
        ################################s
        gene_token_embs = self.V_common.detach().cpu().numpy()
        self.logger.save_array(gene_token_embs, 'gene_token_embs')
        
        mu = self.mu.detach().cpu().numpy()
        self.logger.save_array(mu, 'mu')

        ################################
        self.cell_id_out_of_sample = torch.Tensor(self.cell_id_out_of_sample).to(self.device)
        self.celltype_column_out_of_sample = torch.Tensor(self.celltype_column_out_of_sample).to(self.device)

    
        self.sample_cell_embs_out_of_sample = torch.Tensor(self.sample_cell_embs_out_of_sample).to(self.device)
        self.dataset_idx_out_of_sample = torch.Tensor(self.dataset_idx_out_of_sample).to(self.device)
        world_size = torch.distributed.get_world_size()
        print(f'world_size: {world_size}')
        
        cell_id = self.cell_id_out_of_sample
        celltype_column = self.celltype_column_out_of_sample
        sample_cell_embs = self.sample_cell_embs_out_of_sample
        CellTypeMapping_df = self.CellTypeMapping_df[0]
        dataset_idx = self.dataset_idx_out_of_sample




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


        celltype_column = gather_list(celltype_column)
        sample_cell_embs = gather_list(sample_cell_embs)
        cell_id = gather_list(cell_id)
        dataset_idx = gather_list(dataset_idx)

        celltypes = torch.Tensor(celltype_column).tolist()
        celtype_df_column = CellTypeMapping_df.columns[0]
        celltype_column = CellTypeMapping_df.iloc[celltypes][celtype_df_column].tolist()
        cell_id = torch.Tensor(cell_id).tolist()
        dataset_idx = [int(_) for _ in torch.Tensor(dataset_idx).tolist()]

        X_sample_cell_emb = torch.Tensor(sample_cell_embs).cpu().numpy()
        sample_cell_emb_adata = sc.AnnData(X = X_sample_cell_emb)
        sample_cell_emb_adata.X = csr_matrix(sample_cell_emb_adata.X)
        sample_cell_emb_adata.obs['cell_type'] = celltype_column
        sample_cell_emb_adata.obs['cell_id'] = cell_id
        sample_cell_emb_adata.obs['dataset_idx'] = ['dataset_' + str(_) for _ in dataset_idx]
        self.logger.save_h5ad(sample_cell_emb_adata, f'{self.current_epoch}_sample_cell_emb_adata')
        self.logger.log_info(f'Writing {self.current_epoch}_sample_cell_emb_adata.h5ad in {self.logger._save_dir}.')
        self.logger.log_info(f'sample_cell_emb_adata shape: {sample_cell_emb_adata.shape}.')
        
        self.plotsampleumap_single(
            sample_adata = sample_cell_emb_adata,
            metric = 'euclidean',
            prefix = f'cell_emb',
            save_dir = self.logger._save_dir / 'sample_img',
            step = self.current_epoch
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
            
        target_values = target_values[:, 1:]
        
        # print(f'input_gene_ids: {input_gene_ids.shape}; {input_gene_ids[:, -10:]}')    
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])

        # print(f'src_key_padding_mask after eq: {src_key_padding_mask.shape}; {src_key_padding_mask[:, -10:]}')
        
        mask = (~src_key_padding_mask)[:, 1:]
        mask_dec = (~src_key_padding_mask_dec)[:, 1:]
        
        ########################################
        ##### start go through network
        ########################################
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        self.common_gene_token_embs = self.encoder(self.common_dec_genes)[1:, :]
        
        output_dict = self(
            input_gene_ids,
            dec_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            src_key_padding_mask_dec=src_key_padding_mask_dec
        )
        # V, logV, cell_emb = output_dict['V'], output_dict['logV'], output_dict['cell_emb']
#         V, cell_emb = output_dict['V'], output_dict['cell_emb']

        V_loading = output_dict['V']
        V, logV, cell_emb = output_dict['V_norm'], output_dict['logV_norm'], output_dict['cell_emb']


        #======================== entropy ========================#
        V_mask = V * mask_dec.unsqueeze(2)
        logV_mask = logV * mask_dec.unsqueeze(2)
        entropy_loss = -1.0 * torch.sum(V_mask * logV_mask, dim = -1)[mask_dec]
        entropy_loss = entropy_loss.mean() * 1e-3
        loss = entropy_loss
        loss_dict.update({f'{prefix}/entropy': entropy_loss})

        # V_emb = (V_loading * mask_dec.unsqueeze(2)).sum(1) # batch x d_model
        # Vemb_entropy_loss = (-1) * -1.0 * torch.sum(F.softmax(V_emb) * F.log_softmax(V_emb), dim = -1) 
        # Vemb_entropy_loss = Vemb_entropy_loss.mean() * 1e-2
        # loss = loss + Vemb_entropy_loss 
        # loss_dict.update({f'{prefix}/Vemb_entropy': Vemb_entropy_loss})




        #======================== recon/decoder ========================#
        mu = self.fc_mu_gene(self.common_gene_token_embs.unsqueeze(0)) # batch x seq_len x d_model
        log_std_gene = self.fc_var_gene(self.common_gene_token_embs.unsqueeze(0))
        Sig_tilde_half_gene = torch.exp(0.5 * log_std_gene)
        Sig_tilde_half_gene = Sig_tilde_half_gene[:, dec_gene_ids[:, 1:][mask_dec]]

        V_Sig_tilde_half_z = (V_loading * cell_emb.unsqueeze(1)).sum(2)
        X_tilde = Sig_tilde_half_gene * V_Sig_tilde_half_z[mask_dec] + mu[:, dec_gene_ids[:, 1:][mask_dec]]

#         V_norm = F.normalize(V, p=2, dim=-1)
# #         cell_emb_norm = F.normalize(cell_emb, p=2, dim=-1)

# #         V_Sig_tilde_half_z = (V_norm * cell_emb_norm.unsqueeze(1)).sum(2)
#         V_Sig_tilde_half_z = (V_norm * cell_emb.unsqueeze(1)).sum(2)
#         X_tilde = V_Sig_tilde_half_z[mask_dec] + mu[:, dec_gene_ids[:, 1:][mask_dec]]

        lamb = torch.exp(X_tilde)
        Pois_var = torch.exp(target_values[mask_dec])
        loss_recon = (lamb - Pois_var * X_tilde).mean()
        loss = loss + loss_recon
        loss_dict.update({f'{prefix}/recon': loss_recon})


        #======================== vae ========================#
        loss_kld = output_dict["kld_loss"]
        loss_kld = loss_kld.mean()
        loss = loss + loss_kld
        loss_dict.update({f'{prefix}/kld_loss': loss_kld})

        return loss, loss_dict, output_dict


        
        
    def forward(
        self,
        src: Tensor,
        dec_src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        src_key_padding_mask_dec: Tensor,
    ) -> Mapping[str, Tensor]:
    
        src = self.encoder(src)  # (batch, seq_len, embsize)
        dec_src = self.encoder(dec_src)
        values = self.value_encoder(values)
        self.cur_gene_token_embs = dec_src[:, 1:, :]
        
        output = {}
        # output['V'] = F.softmax(self.cur_gene_token_embs, dim=-1) 
        # output['logV'] = F.log_softmax(self.cur_gene_token_embs, dim=-1)
#         output['V'] = self.cur_gene_token_embs

        V_out = self.fc_mid_gene(self.cur_gene_token_embs)
        # log_std_gene = self.fc_var_gene(self.common_gene_token_embs.unsqueeze(0))
        # Sig_tilde_half_gene = torch.exp(0.5 * log_std_gene)
        # V_out = Sig_tilde_half_gene.unsqueeze(-1) * V_out

#         V_out = F.softmax(V_out, dim=1)
        output['V'] = F.softmax(V_out, dim=-1)
        output['V_norm'] = F.softmax(V_out, dim=-1)
        output['logV_norm'] = F.log_softmax(V_out, dim=-1)


        
        
        total_embs = src + values
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        cell_emb = transformer_output[:, 0, :]
        mu = self.fc_mu(cell_emb) # batch x d_model
        log_var = self.fc_var(cell_emb)

#         Computes the VAE loss function.
#         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))
        output['kld_loss'] = self.kld_weight * kld_loss

        cell_emb = self.reparameterize(mu, log_var)
        output["cell_emb"] = cell_emb

        return output
        
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



class GeneStatsDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hid: int = 128,
        d_out: int = 1, 
    ):
        super().__init__()
        d_in = d_model 
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.LeakyReLU(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the cell_emb, (batch, d_model)"""
        out_val = self.fc(x).squeeze(-1)  # (batch, seq_len)

        return out_val # shape: 1D batch


