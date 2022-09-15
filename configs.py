import torch
import os

class Config(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_save_dir = "saved_models"

        # kmeans
        self.metric = "dtw"  # metric : {“euclidean”, “dtw”, “softdtw”}
        self.n_clusters = 2
        self.n_clusters_real = 2

        # data
        self.sample_size = 500
        self.PSA_DATA = True
        self.upsample = False
        self.ts_length = 6 #6, 10 if context - OLD

        # experiment
        self.experiment_name = "ts_tcc" # "raw_data", "simple_ac", "deep_ac", "lstm", "cnn", "simple_transformer", "ts_tcc"
        self.tstcc_training_mode = "train_linear" # random_init, supervised, self_supervised, fine_tune, train_linear

        # additional info
        self.context = True
        self.deltatimes = False
        self.learnable_pos_enc = False
        self.feat_dim = 1 if self.context == False else 5 #TODO change according to nr. of context vectors used
        self.emb_size = self.feat_dim*self.ts_length if self.experiment_name != "simple_transformer" else 6

        # contexts
        self.context_bmi = True
        self.context_age = True
        self.context_center = True
        self.context_race = True

        # for training models
        self.loss_fn = torch.nn.CrossEntropyLoss()  #torch.nn.CrossEntropyLoss() #MSELoss for LSTM
        self.lr = 0.001
        self.batch_size = 8
        self.n_epochs = 30
        self.dropout = 0.1
        self.bl_hidden_size = 12
        self.num_layers = 1
        self.kernel_size = 1

        #transformer
        self.max_value = 3000
        self.n_heads = 2
        self.dim_feedforward = 128
        self.d_model = 6 # dimensionality of the model, must be divisible by n_heads

        # ts-tcc
        self.tstcc_model_saved_dir = "saved_models/ts_tcc/self_supervised/22-09-15_15-14-27"
        self.hidden_dim = 100
        self.tstcc_aug = False

        self.max_seg = 5
