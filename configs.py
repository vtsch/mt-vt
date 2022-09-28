import torch
import os

class Config(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_save_dir = "saved_models"

        # kmeans
        self.metric = "dtw"  # metric : {“euclidean”, “dtw”, “softdtw”}
        self.n_clusters = 3
        self.n_clusters_real = 2

        # data
        self.sample_size = 500
        self.PSA_DATA = True
        self.upsample = False
        self.ts_length = 6 

        # experiment
        self.experiment_name = "simple_transformer" # "raw_data", "simple_ac", "deep_ac", "lstm", "cnn", "simple_transformer", "ts_tcc"
        self.tstcc_training_mode = "supervised" # random_init, supervised, self_supervised, fine_tune, train_linear

        # contexts
        self.context = True
        self.context_bmi = True
        self.context_age = True
        self.context_center = True
        self.context_count = 3 if self.context_bmi and self.context_age and self.context_center else 1
        self.context_count_size = self.context_count if self.context else 0 

        # additional info
        self.pos_enc = "learnable_pos_enc" # "absolute_days", "delta_days", "learnable_pos_enc", "age_pos_enc", "none", #"rotary_pos_enc",
        self.emb_size = 6 

        # for training models
        self.loss_fn = torch.nn.CrossEntropyLoss()  #torch.nn.CrossEntropyLoss() #MSELoss for LSTM
        self.lr = 0.001
        self.batch_size = 8
        self.n_epochs = 10
        self.dropout = 0.1
        self.bl_hidden_size = 12
        self.num_layers = 1
        self.kernel_size = 1

        #transformer
        self.max_value = 3000
        self.n_heads = 2
        self.dim_feedforward = 128

        # ts-tcc
        self.tstcc_model_saved_dir = "saved_models/ts_tcc/self_supervised/22-09-22_17-36-27"
        self.hidden_dim = 100
        self.tstcc_aug = False

        self.max_seg = 5
