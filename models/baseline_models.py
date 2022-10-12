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
        self.fc = nn.Linear(in_features=config.ts_length+config.context_count_size,
                            out_features=config.ts_length+config.context_count_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size)
        Returns:
            x: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size)
        '''
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
        self.fc1 = nn.Linear(in_features=config.ts_length +
                             config.context_count_size, out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=24)
        self.fc4 = nn.Linear(
            in_features=24, out_features=config.ts_length+config.context_count_size)

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


class CNN(nn.Module):
    # https://github.com/Wickstrom/MixupContrastiveLearning/blob/main/MixupContrastiveLearningExample.ipynb

    def __init__(self, config: Bunch) -> None:
        '''
        Initialize the model
        Args:
            config: config file
        '''
        super(CNN, self).__init__()
        self.config = config
        out_size = config.ts_length+config.context_count_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(12, 24, kernel_size=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, out_size, kernel_size=1),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size)
        Returns:
            x: the learned representation of the model, (batch_size, emb_size, ts_length+context_count_size)'''
        x = x.unsqueeze(1)
        h = self.encoder(x)
        return h


class LSTMencoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, config: Bunch) -> None:
        '''
        Encoder of the model
        Args:
            config: config file
        '''
        super(LSTMencoder, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=1, hidden_size=config.bl_hidden_size,
                            num_layers=2, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(in_features=config.bl_hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        Args:
            x: the input to the model, (batch_size, ts_length+context_count_size) for LSTM must be (batch_size, seq_len, input_size) if batch_first = True
        Returns:
            emb: the learned representation of the model, 
            LSTM layer returns:
                lstm_out: (batch_size, emb_size, ts_length+context_count_size), all the hidden states in the sequence
                hidden: (2, batch_size, hidden_size), hidden state and cell state for the last element in the sequence 
        '''
        x = x.reshape(x.shape[0], x.shape[1], -1)
        lstm_out, self.hidden = self.lstm(x)
        emb = self.fc(lstm_out)
        emb = emb.reshape(emb.shape[0], -1)
        return emb
