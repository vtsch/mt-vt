import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional, Any
import math
from torch import long, nn, Tensor
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from pos_enc import positional_encoding

#https://github.com/sunnyqiny/Unsupervised-Temporal-Embedding-and-Clustering/tree/cc5a41df905efbac11788a43b6151c08c68b8c6c 

def generate_square_subsequent_mask(config):
    ts_length = 6
    t0 = np.floor(ts_length *0.8)
    t0 = t0.astype(int)
    mask = torch.zeros(config.batch_size, ts_length)
    for i in range(0,t0):
        mask[i,t0:] = 1 
    for i in range(t0,ts_length):
        mask[i,i+1:] = 1
    mask = mask.int().masked_fill(mask == 1, float(0.0))#.masked_fill(mask == 1, float('-inf'))
    if config.context:
        mask = torch.cat((mask, torch.ones(config.batch_size, config.context_count)), 1)
        mask = mask.int()
    return mask

# from https://github.com/gzerveas/mvts_transformer 

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, config, activation='gelu'):
        super(TSTransformerEncoder, self).__init__()

        self.config = config

        self.project_inp = nn.Linear(1, self.config.emb_size)

        encoder_layer = TransformerBatchNormEncoderLayer(self.config.emb_size, self.config.n_heads, self.config.dim_feedforward, dropout=self.config.dropout, activation=activation) #use batch normalization

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.config.num_layers)

        self.output_layer = nn.Linear(self.config.emb_size, 1)

        self.act = _get_activation_fn(activation)

        self.dropout = nn.Dropout(self.config.dropout)
        self.max_pool = nn.MaxPool1d(kernel_size=self.config.ts_length)

    def forward(self, indices, data, context, attention_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        data = data.unsqueeze(2)
        indices = indices.unsqueeze(2)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        inp = data.permute(1, 0, 2)
        indices = indices.permute(1, 0, 2)

        inp = self.project_inp(inp) * math.sqrt(self.config.emb_size)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        #add positional encoding
        pos_enc_inp = positional_encoding(self.config, inp, indices)  # # (seq_length, batch_size, d_model)

        if self.config.context:
            # add context to forward_seq
            context = context.unsqueeze(1)
            context = context.permute(2, 0, 1)
            #repeat context to match the shape of pos_enc_inp in dim 2
            context = context.repeat(1, 1, pos_enc_inp.shape[2])
            pos_enc_inp = torch.cat((pos_enc_inp, context), dim=0) # (emb_size + context_size, batch_size, emb_size) )
        
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(pos_enc_inp, src_key_padding_mask=~attention_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout(output)
        # linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, emb_size)
        #output = self.max_pool(output)

        return output


