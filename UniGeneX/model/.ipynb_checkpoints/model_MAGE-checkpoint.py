import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import warnings

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder
from torch.distributions import Bernoulli
from tqdm import trange

try:
    from flash_attn.flash_attention import FlashMHA
    from .flash_layers import FlashscGPTLayer, FlashscGPTGenerator
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    
import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

# from .dsbn import DomainSpecificBatchNorm1d
# from .grad_reverse import grad_reverse

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

    
    
class HeMAE(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        decoder_d_model: int,
        nhead: int,
        d_hid: int,
        nlayers_enc: int = 12,
        nlayers_dec: int = 4,
        vocab: Any = None,
        pad_token: str = "<pad>",
        dropout: float = 0.5,
        pre_norm: bool = False,
        detach_gene_name_emb: bool = True,
        device=None, 
        dtype=None,
        do_mvc: bool = False,
        explicit_zero_prob: bool = False,
        PatchGAN: bool = False,
        gene_num_per_path: int = 500,
        do_vq: bool = False,
        vq_codebook_len: int = 256,
        vq_loss_beta: float = 0.25,
        do_film_cond_cell_emb: bool = False,
        gene_len: int = 2000,
        do_vae: bool = False,
        kld_weight: float = 0.00025,
        do_one_layer_ExprDec = False,
        ExprDecoder_nlayers: int = 2,
        ExprDecoder_d_hid: int = 0,
        do_gene_emb_for_dec: bool = False,
    ):
        super().__init__()
        self.norm_scheme = "pre" if pre_norm else "post"
        self.do_mvc = do_mvc
        self.explicit_zero_prob = explicit_zero_prob
        self.do_vq = do_vq
        self.do_film_cond_cell_emb = do_film_cond_cell_emb
        self.do_vae = do_vae
        self.kld_weight = kld_weight
        self.do_one_layer_ExprDec = do_one_layer_ExprDec
        self.do_gene_emb_for_dec = do_gene_emb_for_dec
        
        
        self.gene_name_emb = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token], do_detach = detach_gene_name_emb)
        if do_gene_emb_for_dec:
            # add dec gene_name_emb to learn gene-gene interaction
            self.gene_name_emb_dec = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token], do_detach = False)
        self.flag_encoder = nn.Embedding(2, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.decoder_embed = nn.Linear(d_model, decoder_d_model, bias=True)
        
        encoder_layers = FlashTransformerEncoderLayer_FiLM(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_enc)
        
        decoder_layers = FlashTransformerEncoderLayer_FiLM(
                    decoder_d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                    cond_norm=do_film_cond_cell_emb,
                    FeatureWiseAffine_num_features=gene_len,
                    FiLM_in_channel=d_model,
                    FiLM_out_channel=decoder_d_model,
                    
                )
        self.transformer_decoder = FlashTransformerEncoder(decoder_layers, nlayers_dec)
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.final_self_attn = FlashMHA(embed_dim=decoder_d_model, 
                                  num_heads=nhead, 
                                  attention_dropout=dropout,
                                  **factory_kwargs)
        if do_one_layer_ExprDec:
            self.pred_decoder = nn.Linear(decoder_d_model, 1, bias=True)
        else:
            if ExprDecoder_d_hid == 0:
                ExprDecoder_d_hid = decoder_d_model
            self.pred_decoder = ExprDecoder(
                                    decoder_d_model,
                                    explicit_zero_prob=explicit_zero_prob,
                                    ExprDecoder_nlayers=ExprDecoder_nlayers,
                                    ExprDecoder_d_hid = ExprDecoder_d_hid,
                                )
        
        if do_mvc:
            self.mvc_decoder = MVCConcatDecoder(
                d_model
            )
            
        if do_vq:
            self.vq_layer = VectorQuantizer(vq_codebook_len,
                                        decoder_d_model,
                                        vq_loss_beta)
        if do_vae:
            self.fc_mu = nn.Linear(decoder_d_model, decoder_d_model)
            self.fc_var = nn.Linear(decoder_d_model, decoder_d_model)
           
        
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
                
    def get_last_layer(self):
        return self.pred_decoder.fc[4].weight


    def embedding(
        self, 
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        gen_genes: Tensor,
    ):
        pcpt_token_embs = self.gene_name_emb(pcpt_genes)  # (batch, pcpt_len, embsize)
        pcpt_values = self.value_encoder(pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values


        gen_token_embs = self.gene_name_emb(gen_genes)  # (batch, gen_len, embsize)
        self.cur_gene_token_embs = torch.cat(
            [pcpt_token_embs, gen_token_embs], dim=1
        )
        gen_flags = self.flag_encoder(
            torch.tensor(1).to(pcpt_values.device)
        ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

        gen_total_embs = gen_token_embs + gen_flags
    
        return pcpt_total_embs, gen_total_embs
    
    def forward(
        self, 
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
    ):
        #=========================encode===========================#
        pcpt_emb, gen_emb = self.embedding(pcpt_genes, pcpt_values, gen_genes)
        pcpt_enc_output = self.transformer_encoder(pcpt_emb, src_key_padding_mask = pcpt_key_padding_mask)
        cell_emb = pcpt_enc_output[:,:1,:].squeeze(1)
        
        
        

        #=========================combine pcpt & gen, reorder===========================#
        all_genes = torch.cat([pcpt_genes, gen_genes], dim = 1)
        all_latent = torch.cat([pcpt_emb, gen_emb], dim = 1)
        all_mask = torch.cat([pcpt_key_padding_mask, gen_key_padding_mask], dim = 1)

        index = torch.argsort(all_genes, dim = -1).unsqueeze(-1).repeat(1, 1, all_latent.shape[2])
        reordered_latent = torch.gather(all_latent, dim = 1, index = index)
        latent = torch.cat([reordered_latent[:, -1:, :], reordered_latent[:, :-1, :]], dim = 1)
        
    
        index = torch.argsort(all_genes, dim = -1)
        if self.do_gene_emb_for_dec:
            reordered_genes = torch.gather(all_genes, dim = 1, index = index)
            gene_name = torch.cat([reordered_genes[:, -1:], reordered_genes[:, :-1]], dim = 1)
            gene_name_emb_dec = self.gene_name_emb_dec(gene_name)
            latent = latent + gene_name_emb_dec
            
        reordered_mask = torch.gather(all_mask, dim = 1, index = index)
        mask = torch.cat([reordered_mask[:, -1:], reordered_mask[:, :-1]], dim = 1)

        #=========================decode===========================#
        dec_input = self.decoder_embed(latent)
        if self.do_film_cond_cell_emb:
            dec_output = self.transformer_decoder(dec_input, src_key_padding_mask = mask, cell_emb = cell_emb)
        else:
            dec_output = self.transformer_decoder(dec_input, src_key_padding_mask = mask)
            
        
        
        
        
        output = {}
        final_gene_emb = self.final_self_attn(dec_output)[0][:,1:,:] # batch x gene_len x emb_dim
        
        if self.do_vae:
            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(final_gene_emb) # batch x d_model
            log_var = self.fc_var(final_gene_emb)

    #         Computes the VAE loss function.
    #         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))
            output['kld_loss'] = self.kld_weight * kld_loss
            
            final_gene_emb = self.reparameterize(mu, log_var)
            
            
        output['final_gene_emb'] = final_gene_emb
        if self.do_vq:
            quantized_inputs, vq_loss = self.vq_layer(final_gene_emb)
            output['quantized_final_gene_emb'] = quantized_inputs
            output['vq_loss'] = vq_loss
            final_gene_emb = quantized_inputs
        
        pred = self.pred_decoder(final_gene_emb)
        
        if self.do_one_layer_ExprDec:
            pred = pred
        else:
            if self.explicit_zero_prob:
                nonzeros = torch.bernoulli(pred['zero_probs']) 
                pred = pred['pred'] * nonzeros
            else:
                pred = pred['pred']

        output['pred'] = pred.squeeze(-1) # batch x gene_len 
        output['cell_emb'] = cell_emb # batch x emb_dim


        
        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                    cell_emb,
                    self.cur_gene_token_embs,
                )['pred']
            reordered_mvc_output = torch.gather(mvc_output, dim = 1, index = index)
            mvc_output = torch.cat([reordered_mvc_output[:, -1:], reordered_mvc_output[:, :-1]], dim = 1)
            mvc_output = mvc_output[:, 1:]

            output['mvc_output'] = mvc_output
       
        
        return output
    
class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    
    
class FeatureWiseAffine(BaseModule):
    def __init__(self, num_features):
        super(FeatureWiseAffine, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm1d(num_features)

    def forward(self, x, scale, shift):
        x = self.instance_norm(x)
        outputs = scale * x + shift
        return outputs
    

    
class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = in_channels,
                                              out_features = in_channels), 
                                    torch.nn.LeakyReLU(0.2)
                                ])
        
        self.scale_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = in_channels,
                                              out_features = out_channels), 
                                ])
        self.shift_linear = torch.nn.Sequential(*[
                                    nn.Linear(in_features = in_channels,
                                              out_features = out_channels), 
                                ])

    def forward(self, x):
        outputs = self.signal_linear(x)
        scale, shift = self.scale_linear(outputs), self.shift_linear(outputs)
        return scale, shift
    
    
    
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
#         latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape # batch x gene_len x emb_dim
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
        quantized_latents = quantized_latents.view(latents_shape)  # # batch x gene_len x emb_dim

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents, vq_loss  # # batch x gene_len x emb_dim
    
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




class TransformerAutoEncoderModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        decoder_d_model: int,
        nhead: int,
        d_hid: int,
        nlayers_enc: int = 12,
        nlayers_dec: int = 12,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        device=None, 
        dtype=None,
        do_mvc: bool = False,
        explicit_zero_prob: bool = False,
        PatchGAN: bool = False,
        gene_num_per_path: int = 500,
        pre_norm: bool = False,
        use_sim_decoder: bool = False,
        detach_gene_name_emb: bool = False,
        do_vq: bool = True,
        vq_codebook_len: int = 256,
        vq_loss_beta: float = 0.25,
        do_film_cond_cell_emb: bool = False,
        gene_len: int = 2000,
        do_vae: bool = False,
        kld_weight: float = 0.00025,
        do_one_layer_ExprDec = False,
        ExprDecoder_nlayers: int = 2,
        ExprDecoder_d_hid: int = 0,
        
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.do_mvc = do_mvc
        self.explicit_zero_prob = explicit_zero_prob
        self.do_vq = do_vq
        self.do_film_cond_cell_emb = do_film_cond_cell_emb
        self.do_vae = do_vae
        self.kld_weight = kld_weight
        self.do_one_layer_ExprDec = do_one_layer_ExprDec
        
  

        # TODO: add dropout in the GeneEncoder
        self.gene_name_emb = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token], do_detach = detach_gene_name_emb)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        
        
        encoder_layers = FlashTransformerEncoderLayer_FiLM(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_enc)
        
        decoder_layers = FlashTransformerEncoderLayer_FiLM(
                    decoder_d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                    cond_norm=do_film_cond_cell_emb,
                    FeatureWiseAffine_num_features=gene_len,
                    FiLM_in_channel=d_model,
                    FiLM_out_channel=decoder_d_model,
                )
        self.transformer_decoder = FlashTransformerEncoder(decoder_layers, nlayers_dec)
        
        
        if do_vq:
            self.vq_layer = VectorQuantizer(vq_codebook_len,
                                        decoder_d_model,
                                        vq_loss_beta)
        
        
        self.MidLayer = MidLayer(d_model, decoder_d_model, dropout=dropout)
        
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.final_self_attn = FlashMHA(embed_dim=decoder_d_model, 
                                  num_heads=nhead, 
                                  attention_dropout=dropout,
                                  **factory_kwargs)
        
        if do_one_layer_ExprDec:
            self.pred_decoder = nn.Linear(decoder_d_model, 1, bias=True)
        else:
            if ExprDecoder_d_hid == 0:
                ExprDecoder_d_hid = decoder_d_model
            self.pred_decoder = ExprDecoder(
                                    decoder_d_model,
                                    explicit_zero_prob=explicit_zero_prob,
                                    ExprDecoder_nlayers=ExprDecoder_nlayers,
                                    ExprDecoder_d_hid = ExprDecoder_d_hid,
                                )
        if do_mvc:
            self.mvc_decoder = MVCConcatDecoder(
                d_model
            )
            
        if do_vq:
            self.vq_layer = VectorQuantizer(vq_codebook_len,
                                        d_model,
                                        vq_loss_beta)
        if do_vae:
            self.fc_mu = nn.Linear(d_model, d_model)
            self.fc_var = nn.Linear(d_model, d_model)
            
            
        if n_cls > 1:
            if use_sim_decoder:
                self.cls_decoder = SimDecoder(d_model, n_cls, nlayers=nlayers_cls)
            else:
                self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)


        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.gene_name_emb.embedding.weight.data.uniform_(-initrange, initrange)

      
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
    
    
    def embedding(
        self, 
        src,
        values
    ):
        token_embs = self.gene_name_emb(src)  # (batch, pcpt_len, embsize)
        values_embs = self.value_encoder(values)  # (batch, pcpt_len, embsize)
        total_embs = token_embs + values_embs

        self.cur_gene_token_embs = token_embs
    
        return total_embs
    
    def forward(
        self, 
        src,
        values,
        src_key_padding_mask=None,
    ):
        output = {}
        #=========================encode===========================#
        total_embs = self.embedding(src, values)
        enc_output = self.transformer_encoder(total_embs, src_key_padding_mask = src_key_padding_mask)
        cell_emb = enc_output[:,:1,:].squeeze(1)
        
        
        #=========================shrinkage===========================#
        if self.do_vae:
            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(enc_output) # batch x gene_len x d_model
            log_var = self.fc_var(enc_output)

    #         Computes the VAE loss function.
    #         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))
            output['kld_loss'] = self.kld_weight * kld_loss
            
            enc_output = self.reparameterize(mu, log_var)
            
            
        output['enc_output'] = enc_output
        if self.do_vq:
            quantized_inputs, vq_loss = self.vq_layer(enc_output[:, 1:, :])
            output['quantized_final_gene_emb'] = quantized_inputs
            output['vq_loss'] = vq_loss
#             enc_output = quantized_inputs
            enc_output = torch.cat([cell_emb.unsqueeze(1), quantized_inputs], dim = 1)
            
            
        #=========================decode===========================#
        dec_input = self.MidLayer(enc_output)
        
        if self.do_film_cond_cell_emb:
            dec_output = self.transformer_decoder(dec_input, src_key_padding_mask = src_key_padding_mask, cell_emb = cell_emb)
        else:
            dec_output = self.transformer_decoder(dec_input, src_key_padding_mask = src_key_padding_mask)
            
            
        
        final_gene_emb = self.final_self_attn(dec_output)[0][:,1:,:] # batch x gene_len x emb_dim
      
        pred = self.pred_decoder(final_gene_emb)
        
        if self.do_one_layer_ExprDec:
            pred = pred
        else:
            if self.explicit_zero_prob:
                nonzeros = torch.bernoulli(pred['zero_probs']) 
                pred = pred['pred'] * nonzeros
            else:
                pred = pred['pred']

        output['pred'] = pred.squeeze(-1) # batch x gene_len 
        output['cell_emb'] = cell_emb # batch x emb_dim


        
        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                    cell_emb,
                    self.cur_gene_token_embs,
                )['pred']
            mvc_output = mvc_output[:, 1:]

            output['mvc_output'] = mvc_output
       
        
        return output
    
    
    


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


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
    




class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        do_detach: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        if do_detach:
            weight = self.embedding.weight
            self.embedding = nn.Embedding.from_pretrained(weight)
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

class LinearBlock(nn.Module):
    def __init__(self, d_in, d_model):
        super(LinearBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.Linear = nn.Linear(d_in, d_model)
    
    def forward(self, x):
        outputs = self.Linear(x)
        outputs = self.leaky_relu(outputs)
        return outputs
    
    
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        ExprDecoder_nlayers: int = 2,
        ExprDecoder_d_hid: int = 64,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        
        if ExprDecoder_nlayers != 2:
            in_features = [d_in] + [ExprDecoder_d_hid] * (ExprDecoder_nlayers - 1)
            out_features = [ExprDecoder_d_hid] * (ExprDecoder_nlayers - 1) + [d_model]
            self.fc = torch.nn.Sequential(*([
                LinearBlock(in_feature, out_feature) for in_feature, out_feature in zip(in_features, out_features)
            ]  + [
                nn.Linear(d_model, 1),
            ]))
            
            self.explicit_zero_prob = explicit_zero_prob
            if explicit_zero_prob:
                self.zero_logit = torch.nn.Sequential(*([
                    LinearBlock(in_feature, out_feature) for in_feature, out_feature in zip(in_features, out_features)
                ]  + [
                    nn.Linear(d_model, 1),
                ]))
                
        else:
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


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class SimDecoder(nn.Module):
    """
    Decoder for classification task with similarity matrix.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
        projection_dim: int = 2048,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, projection_dim)

        self.cls_token_matrix = nn.Parameter(torch.randn(n_cls, projection_dim))
        self.embed_norm = nn.LayerNorm(projection_dim)
        self.token_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        x = self.out_layer(x)
        x = self.embed_norm(x)

        sim = self.sim_matrix(x, self.cls_token_matrix)

        return sim

    def get_sim_matrix(self):
        return self.cls_token_matrix

    def sim_matrix(self, a, b, eps=1e-8):
        # b = self.token_norm(b)
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        # sim_mt = torch.mm(a, b.transpose(0, 1))
        return sim_mt


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


class FlashTransformerDecoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = FlashMHA(embed_dim=d_model, 
                                  num_heads=nhead, 
                                  attention_dropout=dropout,
                                  **factory_kwargs)
        self.self_attn.batch_first = batch_first
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
            
        self.final_self_attn = FlashMHA(embed_dim=d_model, 
                                  num_heads=nhead, 
                                  attention_dropout=dropout,
                                  **factory_kwargs)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        #==================modify for FlashMHA======================#
        if tgt_mask is not None:
            raise ValueError("FlashTransformerDecoderLayer does not support tgt_mask")

        if not tgt_key_padding_mask.any().item():
            # no padding tokens in tgt
            tgt_key_padding_mask_ = None
        else:
            if tgt_key_padding_mask.dtype != torch.bool:
                tgt_key_padding_mask = tgt_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            tgt_key_padding_mask_ = ~tgt_key_padding_mask
            
            
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask_)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask_))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
        x = self.final_self_attn(x, key_padding_mask=tgt_key_padding_mask_)[0]

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, 
                           key_padding_mask=key_padding_mask,
                          )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    
    
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    
    
def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


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
    
    