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
        self.sample_size = 10000
        self.PSA_DATA = True
        self.upsample = True
        self.ts_length = 6

        # additional info
        self.DELTATIMES = False
        self.use_pos_enc = False
        self.context = False

        # experiment
        self.experiment_name = "simple_transformer" # "raw_data", "simple_ac", "deep_ac", "lstm", "cnn", "simple_transformer", "ts_tcc"
        self.tstcc_training_mode = "simple_ac" # random_init, supervised, self_supervised, fine_tune, train_linear
       
        # for training models
        self.loss_fn = torch.nn.CrossEntropyLoss()  #torch.nn.CrossEntropyLoss() #MSELoss for LSTM
        self.lr = 0.001
        self.batch_size = 8
        self.n_epochs = 30
        self.emb_size = 6  # only change for simple transformer, else = 6
        self.dropout = 0.1

        # transformer
        self.num_layers = 1
        self.max_value = 3000
        self.n_heads = 2
        self.feat_dim = 1 # dimensionality of data features
        self.d_model = 6 # dimensionality of the model, = nr of features in data
        self.dim_feedforward = 128

        # ts-tcc
        self.tstcc_model_saved_dir = "saved_models/ts_tcc/self_supervised/22-09-01_14-31-45u"
        self.input_channels = 1
        self.final_out_channels = 16  # 16 with k=2 #32             with k=8
        self.hidden_dim = 100

        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5

        self.temperature = 0.2
        self.use_cosine_similarity = True
