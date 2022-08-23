import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#https://github.com/sunnyqiny/Unsupervised-Temporal-Embedding-and-Clustering/tree/cc5a41df905efbac11788a43b6151c08c68b8c6c 

def generate_square_subsequent_mask(ts_length):
        t0 = np.floor(ts_length *0.9)

        t0 = t0.astype(int)
        mask = torch.zeros(ts_length, ts_length)
        for i in range(0,t0):
            mask[i,t0:] = 1 
        for i in range(t0,ts_length):
            mask[i,i+1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))#.masked_fill(mask == 1, float(0.0))
        return mask


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
    """
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

    def __init__(self, config):
        super(TransformerTimeSeries, self).__init__()

        self.input_embedding = context_embedding(2, config.emb_size, 1)
        self.positional_embedding = torch.nn.Embedding(config.max_value, config.emb_size) 

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.emb_size, nhead=config.n_heads, dropout=config.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=config.emb_size, nhead=config.n_heads)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=config.num_layers)

        self.fc1 = torch.nn.Linear(config.emb_size, 1)
        self.f_class = torch.nn.Linear(config.emb_size, 1)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, indices, data, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((data.unsqueeze(1), indices.unsqueeze(1)), 1)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)
   
        # get my positional embeddings (Batch s ize, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(indices.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings
        transformer_embedding = self.transformer_encoder(input_embedding, attention_masks)
        #print("transformer_embedding shape: ", transformer_embedding.shape)

        transformer_reconstruction = self.transformer_decoder(transformer_embedding, attention_masks)
        #print("transformer_reconstruction shape: ", transformer_reconstruction.shape)
        output = self.fc1(transformer_reconstruction.permute(1, 0, 2))
        #print("output shape: ", output.shape)
        psu_class = self.f_class(transformer_reconstruction.permute(1, 0, 2))
        output = output.permute(0, 2, 1)
        return output, psu_class, transformer_embedding, transformer_reconstruction
