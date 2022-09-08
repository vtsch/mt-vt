import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional, Any
import math
from torch import long, nn, Tensor
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


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
        mask = torch.cat((mask, torch.ones(config.batch_size, 4)), 1)
        mask = mask.int()
    return mask

# from https://github.com/gzerveas/mvts_transformer 


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/ts_length))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/ts_length))
        \text{where pos is the word position and i is the embed idx)
    Args:
        ts_length: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """
    def __init__(self, ts_length, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.ts_length = ts_length

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        pe = torch.zeros(self.max_len, self.ts_length)  # positional encoding
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.ts_length, 2).float() * (-math.log(10000.0) / self.ts_length))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        x = x + pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        ts_length: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, ts_length, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(ts_length, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(ts_length, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, ts_length)

        self.norm1 = BatchNorm1d(ts_length, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(ts_length, eps=1e-5)
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
        src = src + self.dropout1(src2)  # (seq_len, batch_size, ts_length)
        src = src.permute(1, 2, 0)  # (batch_size, ts_length, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * ts_length)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, ts_length)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, ts_length)
        src = src.permute(1, 2, 0)  # (batch_size, ts_length, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, ts_length)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, config, activation='gelu', norm='BatchNorm'):
        super(TSTransformerEncoder, self).__init__()

        self.config = config

        self.project_inp = nn.Linear(self.config.feat_dim, self.config.ts_length)
        self.pos_enc = FixedPositionalEncoding(self.config.ts_length, self.config.dropout, max_len=self.config.ts_length)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.config.ts_length, self.config.n_heads, self.config.dim_feedforward, dropout=self.config.dropout, activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.config.ts_length, self.config.n_heads, self.config.dim_feedforward, dropout=self.config.dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.config.num_layers)

        self.output_layer = nn.Linear(self.config.ts_length, self.config.feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout = nn.Dropout(self.config.dropout)

        self.feat_dim = self.config.feat_dim

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
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = data.permute(1, 0, 2)
        indices = indices.permute(1, 0, 2)

        inp = self.project_inp(inp) * math.sqrt(self.config.ts_length)  # [seq_length, batch_size, ts_length] project input vectors to d_model dimensional space
        if self.config.use_pos_enc:
            inp = self.pos_enc(inp)
        else:
            inp = inp + indices
            inp = self.dropout(inp)
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~attention_masks)  # (seq_length, batch_size, ts_length)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, ts_length)
        output = self.dropout(output)
        # Most probably defining a Linear(ts_length,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


