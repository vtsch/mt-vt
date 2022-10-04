import torch
from bunch import Bunch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    def __init__(self, config: Bunch) -> None:
        '''
        Initialize the model
        Args:
            config: config file
        '''
        super(SimpleAutoencoder, self).__init__()
        self.fc = nn.Linear(in_features=config.ts_length+config.context_count_size, out_features=config.ts_length+config.context_count_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size)
        Returns:
            x: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size)'''
        x = self.fc(x)
        return x

class DeepAutoencoder(nn.Module):
    def __init__(self, config: Bunch) -> None:
        '''
        Initialize the model, 4 layers of the autoencoder
        Args:
            config: config file
        '''
        super(DeepAutoencoder, self).__init__()
        self.fc1 = nn.Linear(in_features=config.ts_length+config.context_count_size, out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=24)
        self.fc4 = nn.Linear(in_features=24, out_features=config.ts_length+config.context_count_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size)
        Returns:
            x: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size)
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class ConvNormPool(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int) -> None:
        '''
        Conv Skip-connection module for CNN
        Args:
            input_size: input size of the layer
            hidden_size: hidden size of the layer
            kernel_size: kernel size of the layer
        '''
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of Conv Norm Pool Layer
        Args:
            input: the input to the model, (batch_size, 1, ts_length+context_count_size)
        Returns:   
            x: output of conv norm pool layer, (batch_size, 1, ts_length+context_count_size)
        '''
        conv1 = self.conv_1(input)
        x = self.normalization(conv1)
        
        x = self.conv_2(x)
        x = self.normalization(x)
        
        conv3 = self.conv_3(x)
        x = self.normalization(conv1+conv3)
        
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self, config: Bunch) -> None:
        '''
        Initialize the model
        Args:
            config: config file
        '''
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model - 2 stacked Conv Norm Pool layers
        Args:
            input: the input to the model, (batch_size, ts_length+context_count_size)
        Returns:
            x: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size)
        '''
        input = input.reshape(input.shape[0], 1, input.shape[1])
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.avgpool(x)        
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

class LSTMencoder(nn.Module):
    ''' Encodes time-series sequence '''
    def __init__(self, config: Bunch) -> None:
        '''
        Encoder of the model
        Args:
            config: config file

        LSTM Layer:
        : input_size:     the number of features in the input X -> is 1 as 1d time series
        : hidden_size:    the number of features in the hidden state h
        : num_layers:     number of recurrent layers
        '''
        super(LSTMencoder, self).__init__()
        # define LSTM layer
        self.config = config
        self.lstm = nn.LSTM(input_size = 1, hidden_size = config.bl_hidden_size,
                            num_layers = 2, batch_first = True)
                            # dropout=dropout_p if num_rnn_layers>1 else 0, bidirectional=bidirectional,
        self.avgpool = nn.AdaptiveAvgPool1d((config.bl_hidden_size//2))
        self.fc = nn.Linear(in_features=config.bl_hidden_size//2, out_features=1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size) for LSTM must be (batch_size, seq_len, input_size) if batch_first = True
        Returns:
            emb: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size), all the hidden states in the sequence; (batch_size, seq_len, hidden_size)
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        x = x.reshape(x.shape[0], x.shape[1], -1)
        lstm_out, self.hidden = self.lstm(x)
        emb = self.avgpool(lstm_out)
        emb = self.fc(emb)
        emb = emb.reshape(emb.shape[0], -1)
        return emb
    
