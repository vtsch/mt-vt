import torch
from torch import nn

class TSTCCbase_Model(nn.Module):
    def __init__(self, config):
        '''
        Initialize the model
        Parameters:
            config: the configuration of the model
        '''
        super(TSTCCbase_Model, self).__init__()
        self.config = config
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(24, config.emb_size, kernel_size=2, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(config.emb_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.linear = nn.Linear(8, config.emb_size)

        model_output_dim = config.ts_length
        self.logits_supervised = nn.Linear(model_output_dim * config.emb_size, config.n_clusters_real)
        self.logits = nn.Linear(model_output_dim * config.emb_size, config.ts_length) # as we learn representations, output in ts_length not n_classes

    def forward(self, x_in):
        '''
        Forward pass of the model.
        Parameters:
            x_in: the input to the model, (batch_size, ts_length) should be (batch_size, n_features, ts_length) for conv, if augment before is already correct
        Returns:
            logits: the classifier output of the model, (batch_size, n_clusters)
            x: the embedding output of the model, (batch_size, emb_size, ts_length)
        '''
        x_in = x_in.float()   
        if self.config.tstcc_training_mode != "self_supervised":
            x_in = x_in.reshape(self.config.batch_size, 1, self.config.ts_length+self.config.context_count_size)

        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.config.dataset == "furst":
            x = self.linear(x)

        # classifier of encoded signals
        x_flat = x.reshape(x.shape[0], -1) # (batch_size, emb_size * ts_length)

        if self.config.tstcc_training_mode == "supervised":
            logits = self.logits_supervised(x_flat)
        else:
            logits = self.logits(x_flat)

        return logits, x