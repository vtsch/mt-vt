import torch
from torch import nn

class base_Model(nn.Module):
    def __init__(self, config):
        super(base_Model, self).__init__()
        self.config = config
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(config.feat_dim, 4, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config.dropout)
        )
        self.conv_block_aug = nn.Sequential(
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
        # x_in shape: (batch_size, time_steps, n_features) should be (batch_size, n_features, time_steps) for conv
        x_in = x_in.float()
        if self.config.tstcc_aug:
            x_in = x_in.reshape(self.config.batch_size, 1, self.config.ts_length)
            x = self.conv_block_aug(x_in)
        else:
            x_in = x_in.permute(0, 2, 1)
            x = self.conv_block1(x_in)

        x = self.conv_block2(x)
        x = self.conv_block3(x) # (batch_size, emb_size, ts_length)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

        #logits shape: (batch_size, n_clusters)
        #x shape: (batch_size, emb_size, ts_length)

