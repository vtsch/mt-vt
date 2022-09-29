import torch
from torch import nn

class base_Model(nn.Module):
    def __init__(self, config):
        super(base_Model, self).__init__()
        self.config = config
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(8, config.emb_size, kernel_size=2, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(config.emb_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        model_output_dim = config.ts_length

        self.logits = nn.Linear(model_output_dim * config.emb_size, config.n_clusters)

    def forward(self, x_in):
        # x_in shape: (batch_size, ts_length) should be (batch_size, n_features, ts_length) for conv, if augment before is already correct
        x_in = x_in.float()   
        if self.config.tstcc_training_mode != "self_supervised":
            x_in = x_in.reshape(self.config.batch_size, 1, self.config.ts_length+self.config.context_count_size)

        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x) # (batch_size, emb_size, ts_length)
        
        x_flat = x.reshape(x.shape[0], -1) # (batch_size, emb_size * ts_length)

        # classifier of encoded signals
        logits = self.logits(x_flat)
        return logits, x
        #logits shape: (batch_size, n_clusters)
        #x shape: (batch_size, emb_size, ts_length)