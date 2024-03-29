import math
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from bunch import Bunch
from pos_enc import positional_encoding
from torch import Tensor, long, nn
from torch.nn.modules import (BatchNorm1d, Dropout, Linear, MultiheadAttention,
                              TransformerEncoderLayer)

# from https://github.com/gzerveas/mvts_transformer


def generate_square_subsequent_mask(config: Bunch) -> torch.Tensor:
    '''
    Generate mask for transformer. Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths, where 1 means keep element at this position (time step) and 0 means mask it.
    Args:
        config: config file
    Returns:
        mask: mask
    '''
    mask = torch.arange(0, config.ts_length).repeat(config.batch_size, 1)
    if config.context:
        mask = torch.cat(
            (mask, torch.ones(config.batch_size, config.context_count)), 1)
        mask = mask.int()
    return mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(
        "activation should be relu/gelu, not {}".format(activation))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu") -> None:
        '''
        This transformer encoder layer block is made up of self-attn and feedforward network.
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of intermediate layer, relu or gelu (default=relu).
        '''
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # normalizes each feature across batch samples and time steps
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
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
        '''
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
            src: the output of the encoder layer
        '''
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

    def __init__(self, config, activation='relu') -> None:
        '''
        Initialize the TST transformer encoder.
        Args:
            config: the config file
            activation: the activation function of intermediate layer, relu or gelu (default=relu).
        '''
        super(TSTransformerEncoder, self).__init__()

        self.config = config

        self.project_inp = nn.Linear(1, self.config.emb_size)

        encoder_layer = TransformerBatchNormEncoderLayer(
            self.config.emb_size, self.config.n_heads, self.config.dim_feedforward, dropout=self.config.dropout, activation=activation)  # use batch normalization

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.config.num_layers)

        self.output_layer = nn.Linear(self.config.emb_size, 1)

        self.act = _get_activation_fn(activation)

        self.dropout = nn.Dropout(self.config.dropout)
        self.max_pool = nn.MaxPool1d(kernel_size=self.config.ts_length)

    def forward(self, indices: pd.DataFrame, data: pd.DataFrame, context: pd.DataFrame, attention_masks: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X: (batch_size, seq_length, 1) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, ts_length + context_dim)
        '''
        # add positional encoding
        # # (ts_length, batch_size, d_model)
        inp = positional_encoding(self.config, data, indices)

        # adjust shapes for transformer
        inp = inp.unsqueeze(1)
        indices = indices.unsqueeze(1)

        if self.config.context:
            # repeat context to match the shape of pos_enc_inp in dim 2
            context = context.unsqueeze(1)
            # (bs, 1, ts_length + context_dim)
            inp = torch.cat((inp, context), dim=2)

        # permute because pytorch convention for transformers is [ts_length, batch_size, feat_dim=1]
        inp = inp.permute(2, 0, 1)
        # [ts_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.project_inp(inp) * math.sqrt(self.config.emb_size)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~attention_masks)
        # the output transformer encoder/decoder embeddings don't include non-linearity  # (seq_length, batch_size, emb_size)
        output = self.act(output)
        output = output.permute(1, 0, 2)  # (batch_size, ts_length, d_model)
        output = self.dropout(output)
        # linear(d_model,feat_dim) vectorizes the operation over (ts_length, batch_size).
        output = self.output_layer(output)  # (batch_size, ts_length, 1)
        # (batch_size, ts_length + context_dim)
        output = output.reshape([output.shape[0], -1])
        return output
