import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, config):
        super(SimpleAutoencoder, self).__init__()
        self.fc = nn.Linear(in_features=config.ts_length, out_features=config.emb_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class DeepAutoencoder(nn.Module):
    def __init__(self, config):
        super(DeepAutoencoder, self).__init__()
        self.fc1 = nn.Linear(in_features=config.ts_length, out_features=12)
        self.fc2 = nn.Linear(in_features=12, out_features=8)
        self.fc4 = nn.Linear(in_features=8, out_features=config.emb_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc4(x)
        #x = self.fc5(x)
        return x


# https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism 

class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size
    ):
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
        self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
        self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
        self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
            
        self.pool = nn.MaxPool1d(kernel_size=1)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))   
        
        x = self.pool(x)
        return x


class CNN(nn.Module):
    #def __init__(self, emb_size, input_size = 1, hid_size = 256, kernel_size = 5):
    def __init__(self, emb_size, input_size, hid_size = 32, kernel_size = 1):
        super().__init__()
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        #self.fc = nn.Linear(in_features=hid_size//4, out_features=emb_size)
        self.fc = nn.Linear(in_features=hid_size//2, out_features=emb_size)
        
    def forward(self, input):
        input = input.unsqueeze(2)
        #print(input.shape)
        x = self.conv1(input)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = self.avgpool(x)        
        #print("shape avgpool", x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        #print("shape view", x.shape)
        x = F.softmax(self.fc(x), dim=1)
        #print("shape fc", x.shape)
        return x


class RNN(nn.Module):
    """RNN module(cell type lstm or gru)"""
    def __init__(
        self,
        input_size,
        hid_size,
        num_rnn_layers=1,
        dropout_p = 0.2,
        bidirectional = False,
        rnn_type = 'lstm',
    ):
        super().__init__()
        
        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            
        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states



class RNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        emb_size,
    ):
        super().__init__()
            
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=1,
        )

        self.rnn_layer = RNN(
            input_size=input_size, 
            hid_size=hid_size, #hid_size * 2 if bidirectional else hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.avgpool = nn.AdaptiveAvgPool1d((32)) #batch size
        self.fc = nn.Linear(in_features=hid_size, out_features=emb_size)

    def forward(self, input):
        input = input.unsqueeze(1)
        #x = self.conv1(input) # input shape: batch_size * num_features (1) * num_channels (186)
        x, _ = self.rnn_layer(input)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)#.squeeze(1)
        return x


class LSTMencoder(nn.Module):
    ''' Encodes time-series sequence '''
    def __init__(self, hidden_size, num_layers):
        '''
        : param input_size:     the number of features in the input X -> is 1 as 1d time series
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMencoder, self).__init__()
        # define LSTM layer
        self.lstm = nn.LSTM(input_size = 1, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
                            # dropout=dropout_p if num_rnn_layers>1 else 0, bidirectional=bidirectional,
        self.avgpool = nn.AdaptiveAvgPool1d((hidden_size//2))
        self.fc = nn.Linear(in_features=hidden_size//2, out_features=1)
        

    def forward(self, x_input):
        '''
        : param x_input:               input of the TS is (batch_size, seq_length), for LSTM must be (batch_size, seq_len, input_size) if batch_first = True, 
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence; (batch_size, seq_len, hidden_size)
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        x = x_input.reshape(x_input.shape[0], x_input.shape[1], 1)
        lstm_out, self.hidden = self.lstm(x)
        emb = self.avgpool(lstm_out)
        emb = self.fc(emb)
        emb = emb.squeeze(2)
        
        return emb     
    
