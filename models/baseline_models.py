import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, config):
        '''
        Simple Autoencoder, input is (batch_size, seq_length+nr_features)
        '''
        super(SimpleAutoencoder, self).__init__()
        self.fc = nn.Linear(in_features=config.ts_length+config.context_count_size, out_features=config.ts_length+config.context_count_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class DeepAutoencoder(nn.Module):
    def __init__(self, config):
        super(DeepAutoencoder, self).__init__()
        self.fc1 = nn.Linear(in_features=config.ts_length+config.context_count_size, out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=24)
        self.fc4 = nn.Linear(in_features=24, out_features=config.ts_length+config.context_count_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism 

class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.normalization = nn.BatchNorm1d(num_features=hidden_size)
            
        self.pool = nn.MaxPool1d(kernel_size=1)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization(conv1)
        
        x = self.conv_2(x)
        x = self.normalization(x)
        
        conv3 = self.conv_3(x)
        x = self.normalization(conv1+conv3)
        
        x = self.pool(x)
        return x


class CNN(nn.Module):
    #def __init__(self, emb_size, input_size = 1, hid_size = 256, kernel_size = 5):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = ConvNormPool(
            input_size=1,
            hidden_size=config.bl_hidden_size,
            kernel_size=config.kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=config.bl_hidden_size,
            hidden_size=config.bl_hidden_size//2,
            kernel_size=config.kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=config.bl_hidden_size//2, out_features=config.ts_length+config.context_count_size)
        
    def forward(self, input):
        input = input.reshape(input.shape[0], 1, input.shape[1])
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.avgpool(x)        
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

class LSTMencoder(nn.Module):
    ''' Encodes time-series sequence '''
    def __init__(self, config):
        '''
        : param input_size:     the number of features in the input X -> is 1 as 1d time series
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMencoder, self).__init__()
        # define LSTM layer
        self.config = config
        self.lstm = nn.LSTM(input_size = 1, hidden_size = config.bl_hidden_size,
                            num_layers = config.num_layers, batch_first = True)
                            # dropout=dropout_p if num_rnn_layers>1 else 0, bidirectional=bidirectional,
        self.avgpool = nn.AdaptiveAvgPool1d((config.bl_hidden_size//2))
        self.fc = nn.Linear(in_features=config.bl_hidden_size//2, out_features=1)
        

    def forward(self, x):
        '''
        : param x_input:               input of the TS is (batch_size, seq_length), for LSTM must be (batch_size, seq_len, input_size) if batch_first = True, 
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence; (batch_size, seq_len, hidden_size)
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        x = x.reshape(x.shape[0], x.shape[1], -1)
        lstm_out, self.hidden = self.lstm(x)
        emb = self.avgpool(lstm_out)
        emb = self.fc(emb)
        emb = emb.reshape(emb.shape[0], -1)
        return emb
    
