import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import warnings
from pathlib import Path
import json

import torch
import numpy as np
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

from tqdm import trange
from einops import rearrange

try:
    # from flash_attn.flash_attention import FlashMHA
    from scgpt_pcpt.model.flash_attention import FlashMHA
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    
import copy
from typing import Optional, Any, Union, Callable


import scgpt_pcpt as scg
# from scgpt_pcpt.preprocess import Preprocessor
from scgpt_pcpt.tokenizer import GeneVocab, random_mask_value, tokenize_batch
from scgpt_pcpt.scbank import DataBank
from scgpt_pcpt.utils import MainProcessOnly, ConfigWrapper, configure_logging
from scgpt_pcpt import logger



    
def cal_final_updim(output_dim, factor, kernel_size = 5, padding = 3):
    output_dim_ori = output_dim
    output_dim = int((output_dim + 2 * padding - kernel_size) / factor + 1)
    output_padding = output_dim_ori - ((output_dim - 1) * factor) + 2 * padding - kernel_size
    return output_dim, output_padding

class TransformerVAEModel(nn.Module):
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
        gene_len: int = 2001,
        decode_gene: bool = False,
        do_vq: bool = False,
        vq_emb_dim: int = 3,
        vq_n_emb: int = 1024,
        vq_loss_beta: float = 0.25,
        vq_downfactors: list = [3,3],
        do_vae: bool = False,
        kld_weight: float = 0.00025,
        no_attn_cell: bool = True,
        depth_scalar: bool = False,
        ckpt_path: str = None,
        use_batch_labels: bool = False,
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
        self.gene_len = gene_len
        self.decode_gene = decode_gene
        self.do_vq = do_vq
        self.vq_emb_dim = vq_emb_dim
        self.vq_n_emb = vq_n_emb
        self.vq_loss_beta = vq_loss_beta
        self.vq_downfactors = vq_downfactors
        self.do_vae = do_vae
        self.kld_weight = kld_weight
        self.no_attn_cell = no_attn_cell
        self.depth_scalar = depth_scalar
        self.ckpt_path = ckpt_path
        self.use_batch_labels = use_batch_labels
        
        
        
        if ckpt_path is not None:
            self.update_para_from_ckpt(ckpt_path) 
            
        special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        vocab = GeneVocab.from_file(Path(self.vocab_path))
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        self.vocab = vocab
        self.ntoken = len(self.vocab) 
        self.norm_scheme = "pre" if pre_norm else "post"
            
        
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
            
        self.build_network()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        
            
            
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
                self.d_model, self.nhead, self.d_hid, self.dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        #=================================== decode_gene ==================================#
        if self.decode_gene:
            self.decoder = ExprDecoder(
                self.d_model,
                explicit_zero_prob=self.explicit_zero_prob,
            )

        #=================================== do_vq ==================================#
        if self.do_vq:
            self.vq_layer = VectorQuantizer(self.vq_n_emb,
                                        self.vq_emb_dim,
                                        self.vq_loss_beta)
            
            self.vq_pre_conv = Conv1dWithInitialization(
                                            in_channels=1,
                                            out_channels=self.vq_emb_dim,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1
                                        )
            cal_factor = self.vq_downfactors
            output_dim = self.d_model
            output_padding = []
            LayerNorm_emb_dim = []
            LayerNorm_emb_dim.append(self.d_model)
            for fac in cal_factor:
                output_dim, out_padding = cal_final_updim(output_dim, fac, kernel_size = 5, padding = fac)
                LayerNorm_emb_dim.append(output_dim)
                output_padding.append(out_padding)

            self.quant_conv = torch.nn.Sequential(*[
                                        ResnetBlock_Down(d_model=self.d_model, in_channels=self.vq_emb_dim, downfactor = self.vq_downfactors[0], LayerNorm_emb_dim = LayerNorm_emb_dim[:2]),
                                        ResnetBlock_Down(d_model=self.d_model, in_channels=self.vq_emb_dim, downfactor = self.vq_downfactors[1], LayerNorm_emb_dim = LayerNorm_emb_dim[1:])
                                    ])
            self.post_quant_conv = torch.nn.Sequential(*[
                                        ResnetBlock_Up(d_model=self.d_model, in_channels=self.vq_emb_dim, upfactor = self.vq_downfactors[1], LayerNorm_emb_dim = LayerNorm_emb_dim[1:][::-1], output_padding = output_padding[1]),
                                        ResnetBlock_Up(d_model=self.d_model, in_channels=self.vq_emb_dim, upfactor = self.vq_downfactors[0], LayerNorm_emb_dim = LayerNorm_emb_dim[:2][::-1], output_padding = output_padding[0])
                                    ])
            self.vq_post_conv = Conv1dWithInitialization(
                                            in_channels=self.vq_emb_dim,
                                            out_channels=1,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1
                                        )
        
        #=================================== do_vae ==================================#
        if self.do_vae:
            self.fc_mu = nn.Linear(self.d_model, self.d_model)
            self.fc_var = nn.Linear(self.d_model, self.d_model)
        
        
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
                    self.d_model, self.nhead, self.d_hid, self.dropout, batch_first=True
                )
                self.mvc_decoder = TransformerEncoder(mvc_layers, self.nlayers_dec)
                
                
            self.mvc_expr_decoder = ExprDecoder(
                                                self.d_model,
                                                explicit_zero_prob=self.explicit_zero_prob,
                                            )
        elif self.mvc_decoder_type == 'TransformerFilm':
            if self.fast_transformer_backend == "flash":
                mvc_layers = FlashTransformerEncoderLayer_FiLM(
                            self.d_model,
                            self.nhead,
                            self.d_hid,
                            self.dropout,
                            batch_first=True,
                            norm_scheme=self.norm_scheme,
                            cond_norm=True,
                            FiLM_in_channel=self.d_model,
                            FiLM_out_channel=self.gene_len,

                        )
                self.mvc_decoder = FlashTransformerEncoder(mvc_layers, nlayers_dec)
            elif self.fast_transformer_backend == "normal":
                mvc_layers = TransformerEncoderLayer_FiLM(
                            self.d_model,
                            self.nhead,
                            self.d_hid,
                            self.dropout,
                            batch_first=True,
                            FiLM_in_channel=self.d_model,
                            FiLM_out_channel=self.gene_len,
                            no_attn_cell = self.no_attn_cell
                        )
#                 self.mvc_decoder = TransformerEncoder(mvc_layers, nlayers_dec)
                self.mvc_decoder = FlashTransformerEncoder(mvc_layers, self.nlayers_dec)
            self.mvc_expr_decoder = ExprDecoder(
                                                self.d_model,
                                                explicit_zero_prob=self.explicit_zero_prob,
                                            )
        if self.depth_scalar:
            self.depth_scalar_decoder = DepthScalarDecoder(self.d_model)
                

                
        self.init_weights()
        
        
            
    def init_from_ckpt(self, path):
        model_file = Path(path)
        try:
            self.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = self.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        
    def update_para_from_ckpt(self, path):
        model_dir = Path(path).parent
        model_config_file = model_dir / "args.json"
        model_file = Path(path) # model_dir / "best_model.pt"
        
        if hasattr(self, 'vocab'):
            if len(self.vocab) != len(json.load(open(model_dir / "vocab.json"))):
                logger.warning(
                    f"The vocabulary in the model directory to load ({model_dir}) does "
                    "not match the current vocabulary. "
                )
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        if hasattr(self, 'pad_token'):
            if self.pad_token != model_configs["pad_token"]:
                logger.warning(
                    f"The pad token in the model directory to load ({model_dir}) "
                    "does not match the current pad token. Be careful if this is not expected."
                )
        if hasattr(self, 'pad_value'):
            if self.pad_value != model_configs["pad_value"]:
                logger.warning(
                    f"The pad value in the model directory to load ({model_dir}) "
                    "does not match the current pad value. Be careful if this is not expected."
                )
        logger.info(
            f"Resume model from {model_file}, the model args will be overridden the "
            f"config {model_config_file}."
        )
        self.d_model = model_configs["embsize"]
        self.nhead = model_configs["nheads"]
        self.d_hid = model_configs["d_hid"]
        self.nlayers = model_configs["nlayers"]
        self.vocab_path = model_configs["vocab_path"]
        self.dropout = model_configs["dropout"]
        self.pad_token = model_configs["pad_token"]
        self.pad_value = model_configs["pad_value"]
        self.fast_transformer_backend = model_configs["fast_transformer_backend"]
        self.mvc_decoder_type = model_configs["mvc_decoder_type"]
        self.nlayers_dec = model_configs["nlayers_dec"]
        self.decode_gene = model_configs["add_gene_loss"]
        self.do_vq = model_configs["do_vq"]
        self.vq_emb_dim = model_configs["vq_emb_dim"]
        self.vq_n_emb = model_configs["vq_n_emb"]
        self.vq_loss_beta = model_configs["vq_loss_beta"]
        self.do_vae = model_configs["do_vae"]
        self.kld_weight = model_configs["kld_weight"]
        self.no_attn_cell = model_configs["Film_no_attn_cell"]
        # self.depth_scalar = model_configs["depth_scalar"]
        
  
        
        
    

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
        
        
    def encode(self, data_dict):
        """
        data_dict:
            {'gene': tensor([[60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             ...,
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089]]),
     'expr': tensor([[-2.0000,  0.0000,  1.7887,  ...,  0.0000,  3.1382,  0.0000],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  2.7163],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             ...,
             [-2.0000,  0.0000,  3.1623,  ...,  0.0000,  4.5163,  0.0000],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  3.7978,  3.4035],
             [-2.0000,  0.0000,  1.8141,  ...,  0.0000,  2.8361,  0.0000]]),
     'celltype_code': tensor([23, 38,  3, 11,  9,  1, 34, 23, 10, 35, 23,  6,  5,  7, 35, 15, 19, 12,
             28, 33,  4, 15, 33,  1, 38,  5,  0, 20,  3, 36, 38, 12]),
     'masked_gene': tensor([[60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             ...,
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089],
             [60695, 27658, 13001,  ...,  2809, 30374, 10089]]),
     'masked_expr': tensor([[-2.0000,  0.0000,  1.7887,  ...,  0.0000,  3.1382,  0.0000],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  2.7163],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
             ...,
             [-2.0000,  0.0000,  3.1623,  ...,  0.0000,  4.5163,  0.0000],
             [-2.0000,  0.0000,  0.0000,  ...,  0.0000,  3.7978,  3.4035],
             [-2.0000,  0.0000,  1.8141,  ...,  0.0000,  2.8361,  0.0000]])}
        """
        input_gene_ids = data_dict["masked_gene"]
        dec_gene_ids = data_dict["gene"]
        input_values = data_dict["masked_expr"]
        target_values = data_dict["expr"][:, 1:]
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])
        src = input_gene_ids
        dec_src = dec_gene_ids
        values = input_values
        src_key_padding_mask = src_key_padding_mask
        src_key_padding_mask_dec = src_key_padding_mask_dec


        #======================== embed ===========================#
        src = self.encoder(src)  # (batch, seq_len, embsize)
        dec_src = self.encoder(dec_src)
        values = self.value_encoder(values)
        self.cur_gene_token_embs = dec_src[:, 1:, :]
        
        total_embs = src + values
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        
        #======================== get cell emb ===========================#
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        return cell_emb
    
    def decode(self, cell_emb, batch):
        data_dict = batch
        dec_gene_ids = data_dict["gene"][:, 1:]
        src_key_padding_mask_dec = dec_gene_ids.eq(self.vocab[self.pad_token])

        cur_gene_token_embs = self.encoder(dec_gene_ids)

        
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
            dec_output = self.mvc_decoder(dec_output, src_key_padding_mask = src_key_padding_mask_dec)
            dec_output = self.mvc_expr_decoder(dec_output)
        elif self.mvc_decoder_type == 'TransformerFilm':
            if self.no_attn_cell:
                dec_input = torch.cat((cell_emb.unsqueeze(1), cur_gene_token_embs), dim = 1)
                dec_output = self.mvc_decoder(dec_input, src_key_padding_mask = src_key_padding_mask_dec) # batch x (gene_len + 1(cell_emb)) x emb_dim
                dec_output = dec_output[:, 1:, :] # remove cell_emb --> batch x gene_len x emb_dim
                dec_output = self.mvc_expr_decoder(dec_output)
            else:
                dec_input = torch.cat((cell_emb.unsqueeze(1), cur_gene_token_embs), dim = 1)
                dec_output = self.mvc_decoder(dec_input, src_key_padding_mask = src_key_padding_mask_dec) # batch x (gene_len + 1(cell_emb)) x emb_dim
                dec_output = dec_output[:, 1:, :] # remove cell_emb --> batch x gene_len x emb_dim
                dec_output = self.mvc_expr_decoder(dec_output)

        output = {}
        output['cell_preds'] = dec_output["pred"]
        output['cell_emb'] = cell_emb
        output['celltypes'] = data_dict['celltype_code']

#         for k, v in output.items():
#             if k == 'cell_preds':
#                 v = v.detach().cpu()

#                 v[v < 0] = 0
#                 v[v > 50] = 50
#             else:
#                 v = v.detach().cpu()
#             output[k] = v
        return output

        

    
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
        #<<<<<<<<<<<<<<<<<========== vq =========>>>>>>>>>>>>>>>>>>>#
        if self.do_vq:
            cell_emb = self.vq_pre_conv(cell_emb.unsqueeze(1))
            cell_emb = self.quant_conv(cell_emb)
            output["cell_emb_ze"] = cell_emb
            output["cell_emb_for_diffusion"] = cell_emb
            quantized_cell_emb, vq_loss = self.vq_layer(cell_emb)
            output['cell_emb_zq'] = quantized_cell_emb
            output['vq_loss'] = vq_loss
            quantized_cell_emb = self.post_quant_conv(quantized_cell_emb)
            quantized_cell_emb = self.vq_post_conv(quantized_cell_emb).squeeze(1)
            cell_emb = quantized_cell_emb
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
            output["cell_emb_beforeDec"] = cell_emb
            output["cell_emb_for_diffusion"] = cell_emb
            
            
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
            dec_output = self.mvc_decoder(dec_output, src_key_padding_mask = src_key_padding_mask_dec)
            dec_output = self.mvc_expr_decoder(dec_output)
        elif self.mvc_decoder_type == 'TransformerFilm':
            if self.no_attn_cell:
                dec_input = torch.cat((cell_emb.unsqueeze(1), self.cur_gene_token_embs), dim = 1)
                dec_output = self.mvc_decoder(dec_input, src_key_padding_mask = src_key_padding_mask_dec) # batch x (gene_len + 1(cell_emb)) x emb_dim
                dec_output = dec_output[:, 1:, :] # remove cell_emb --> batch x gene_len x emb_dim
                dec_output = self.mvc_expr_decoder(dec_output)
            else:
                dec_input = torch.cat((cell_emb.unsqueeze(1), self.cur_gene_token_embs), dim = 1)
                dec_output = self.mvc_decoder(dec_input, src_key_padding_mask = src_key_padding_mask_dec) # batch x (gene_len + 1(cell_emb)) x emb_dim
                dec_output = dec_output[:, 1:, :] # remove cell_emb --> batch x gene_len x emb_dim
                dec_output = self.mvc_expr_decoder(dec_output)

        if self.depth_scalar:
            depth_scalar = self.depth_scalar_decoder(cell_emb)
        else:
            depth_scalar = torch.ones(cell_emb.shape[0], device = cell_emb.device)
            
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=dec_output["zero_probs"])
            output["recon_output"] = bernoulli.sample() * dec_output["pred"] * depth_scalar.unsqueeze(1)
        else:
            output["recon_output"] = dec_output["pred"] * depth_scalar.unsqueeze(1)  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["recon_zero_probs"] = dec_output["zero_probs"] 

        return output

    
    
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

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
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
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
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


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

    
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor: 
        latents = rearrange(latents, 'b c h -> b h c').contiguous() # batch x vq_emb_dim(3) x emb_dim -> batch x emb_dim x vq_emb_dim(3) 
#         latents = latents.permute(0, 2, 1).contiguous() 
        latents_shape = latents.shape 
        flat_latents = latents.reshape(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # # batch x emb_dim
        
        
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        
        quantized_latents = rearrange(quantized_latents, 'b h c -> b c h').contiguous()

        return quantized_latents, vq_loss  # # batch x emb_dim
    

    
class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)

    
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)



class ResnetBlock_Down(nn.Module):
    def __init__(self, *, d_model, in_channels, downfactor, LayerNorm_emb_dim, conv_shortcut=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.LayerNorm([in_channels, LayerNorm_emb_dim[0]])
        self.down_conv1 = Conv1dWithInitialization(
                                     in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=5,
                                     stride=downfactor,
                                     padding=downfactor
        )

        self.norm2 = torch.nn.LayerNorm([in_channels, LayerNorm_emb_dim[1]])
        self.conv1 = Conv1dWithInitialization(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
        
        
        if self.use_conv_shortcut:
            self.residual_branch = torch.nn.Sequential(*[
                                        Conv1dWithInitialization(
                                            in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=1,
                                            stride=1
                                        ),
                                        Conv1dWithInitialization(
                                                 in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=5,
                                                 stride=downfactor,
                                                 padding=downfactor
                                        )
                                    ])


    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.down_conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        

        if self.use_conv_shortcut:
            x = self.residual_branch(x)

        return x+h
    
    
class ResnetBlock_Up(nn.Module):
    def __init__(self, *, d_model, in_channels, upfactor, LayerNorm_emb_dim, output_padding, conv_shortcut=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.LayerNorm([in_channels, LayerNorm_emb_dim[0]])
        self.up_conv1 = torch.nn.ConvTranspose1d(in_channels,
                                                 in_channels,
                                                 kernel_size=5,
                                                 stride=upfactor,
                                                 padding=upfactor,
                                                 output_padding = output_padding)

        self.norm2 = torch.nn.LayerNorm([in_channels, LayerNorm_emb_dim[1]])
        self.conv1 = Conv1dWithInitialization(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )


        
        if self.use_conv_shortcut:
            self.residual_branch = torch.nn.Sequential(*[
                                        Conv1dWithInitialization(
                                            in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=1,
                                            stride=1
                                        ),
                                        torch.nn.ConvTranspose1d(
                                                 in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=5,
                                                 stride=upfactor,
                                                 padding=upfactor,
                                                 output_padding = output_padding)
                                    ])


    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.up_conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        

        if self.use_conv_shortcut:
            x = self.residual_branch(x)

        return x+h
                                                  
                                                  
                                                
    


# class ResnetBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
#                  dropout, temb_channels=512):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut

#         self.norm1 = Normalize(in_channels)
#         self.conv1 = torch.nn.Conv2d(in_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if temb_channels > 0:
#             self.temb_proj = torch.nn.Linear(temb_channels,
#                                              out_channels)
#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv2d(out_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = torch.nn.Conv2d(in_channels,
#                                                      out_channels,
#                                                      kernel_size=3,
#                                                      stride=1,
#                                                      padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv2d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=1,
#                                                     padding=0)

#     def forward(self, x, temb):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         if temb is not None:
#             h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)

#         return x+h
