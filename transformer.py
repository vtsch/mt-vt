from comet_ml import Experiment
from typing import Optional, Any
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoder, TransformerEncoderLayer
import numpy as np
import torch.nn as nn

from transformer_modules import TransformerBlock, DenseInterpolation

class Transformer(nn.Module):

    # https://github.com/Sarunas-Girdenas/transformer_for_time_series/blob/master/transformers_time_series.ipynb 

    def __init__(self, emb: int, 
                heads: int,
                depth: int,
                num_features: int,
                num_out_channels_emb: int,
                dropout: float,
                mask: bool=True):
        """
        Transformer for time series.
        Inputs:
        =======
        emb (int): Embedding dimension
        heads (int): Number of attention heads
        depth (int): Number of transformer blocks
        seq_length (int): length of the sequence
        num_features (int): number of time series features
        mask (bool): if mask diagonal
        """

        super().__init__()

        self.num_features = num_features

        # 1D Conv for actual values of time series
        self.time_series_features_encoding = nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=1,
                bias=False
            )

        # positional embedding for time series
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=emb)

        # transformer blocks
        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=emb,
                    mask=mask,
                    dropout=dropout
                )
            )
        
        # transformer blocks put together
        self.transformer_blocks = nn.Sequential(*tblocks)

        # conv1d for embeddings
        self.conv1d_for_embeddings = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_out_channels_emb,
            kernel_size=1,
            bias=False
        )
        
        # feed forward through maxpooled embeddings to reduce them to probability
        self.feed_forward = torch.nn.Linear(
            emb,
            1
            )

        self.dropout = nn.Dropout(dropout)

        return None
    
    @staticmethod
    def init_weights(layer):
        """Purpose: initialize weights in each
        LINEAR layer.
        Input: pytorch layer
        """

        if isinstance(layer, torch.nn.Linear):
            np.random.seed(42)
            size = layer.weight.size()
            fan_out = size[0] # number of rows
            fan_in = size[1] # number of columns
            variance = np.sqrt(2.0/(fan_in + fan_out))
            # initialize weights
            layer.weight.data.normal_(0.0, variance)
        
    def forward(self, x):
        """
        Forward pass.
        x (torch.tensor): 3D tensor of size (batch, num_features, 1)
        """

        # 1D Convolution to convert time series data to features (kind of embeddings)
        time_series_features = self.time_series_features_encoding(x)
        b, t, e = time_series_features.size()  # b: batch size, t: 1, e: embedding = ts length

        positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        
        # sum encoded time serie features and positional encodings and pass on to transfmer block
        x = time_series_features + positions
        x = self.dropout(x)

        x = self.transformer_blocks(x)

        x = self.conv1d_for_embeddings(x) # (batch, num out channels, emb)
        # maxpool
        x = x.max(dim=1)[0] # (batch, emb)

        x = self.feed_forward(x) # (batch, 1)
        #x = x.reshape(b, -1) # (batch)

        x = torch.sigmoid(x)

        return x

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        x = super(CausalConv1d, self)
        y = F.pad(input, (self.__padding, 0))
        x = x.forward(y)
        return x


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels,embedding_size,k):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x)
        return torch.tanh(x)

class TransformerTimeSeries(torch.nn.Module):
    # https://github.com/sunnyqiny/Unsupervised-Temporal-Embedding-and-Clustering/tree/cc5a41df905efbac11788a43b6151c08c68b8c6c 
    """
    Time Series application of transformers based on paper
    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()

        self.feature_size = 12
        self.dropout = 0.1
        self.num_layers = 2
        self.ts_length = 6

        self.input_embedding = context_embedding(2, self.feature_size, 3)
        self.positional_embedding = torch.nn.Embedding(60, self.feature_size) 

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=2, dropout=self.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=2)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=self.num_layers)

        self.fc1 = torch.nn.Linear(self.feature_size, 1)
        self.f_class = torch.nn.Linear(self.feature_size, 1)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, indices, data, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((data.unsqueeze(1), indices.unsqueeze(1)), 1)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)
        #print("z_emb", z_embedding)
        #print("z_embedding shape: ", z_embedding.shape)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(indices.type(torch.long))
        #print("positional_embeddings", positional_embeddings)
        #print("positional_embeddings shape: ", positional_embeddings.shape)
        positional_embeddings = positional_embeddings.reshape(positional_embeddings.shape[0], self.ts_length, -1).permute(1, 0, 2)
        #print("positional_embeddings shape: ", positional_embeddings.shape)
        input_embedding = z_embedding + positional_embeddings
        #print("input_embedding shape: ", input_embedding.shape)
        transformer_embedding = self.transformer_encoder(input_embedding, attention_masks)
        #print("transformer_embedding shape: ", transformer_embedding.shape)

        transformer_reconstruction = self.transformer_decoder(transformer_embedding, attention_masks)
        #print("transformer_reconstruction shape: ", transformer_reconstruction.shape)
        output = self.fc1(transformer_reconstruction.permute(1, 0, 2))
        output = self.softmax(output) #to restrict to [0,1]
        #print("output shape: ", output.shape)
        psu_class = self.f_class(transformer_reconstruction.permute(1, 0, 2))
        output = output.permute(0, 2, 1)
        return output, psu_class, transformer_embedding, transformer_reconstruction
