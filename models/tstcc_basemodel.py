from torch import nn

class base_Model(nn.Module):
    def __init__(self, config):
        super(base_Model, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(config.input_channels, 4, kernel_size=8, stride=1, bias=False, padding=4),
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
            nn.Conv1d(8, config.final_out_channels, kernel_size=2, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(config.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        model_output_dim = config.emb_size
        self.logits = nn.Linear(model_output_dim * config.final_out_channels, config.n_clusters)

    def forward(self, x_in):
        # x_in shape: (batch_size, time_steps)
        x_in = x_in.reshape(-1, 1, x_in.shape[1])
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

        #logits shape: (batch_size, n_clusters)
        #x shape: (batch_size, final_out_channels, time_steps)

