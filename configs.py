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
        self.DELTATIMES = False
        self.NOPOSENC = False
        self.ts_length = 6

        # experiment
        self.experiment_name = "ts-tcc" # "simple_ac", "deep_ac", "lstm", "cnn", "simple_transformer", "ts-tcc"
        self.tstcc_training_mode = "supervised" # (random_init), supervised, self_supervised, fine_tune, train_linear
       
        # for training models
        self.loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.MSELoss() 
        self.lr = 0.001
        self.batch_size = 24
        self.n_epochs = 10
        self.emb_size = 6  # needs to be = tslength if baselines and tstcc
        self.dropout = 0.3

        # transformer
        self.num_layers = 1
        self.max_value = 3000
        self.n_heads = 2

        # ts-tcc
        self.tstcc_model_saved_dir = os.path.join(os.path.join(self.model_save_dir, self.experiment_name, f"self_supervised", "saved_models"))
        self.input_channels = 1
        self.kernel_size = 2
        self.stride = 1
        self.final_out_channels = 16  # 16 with k=2 #32 with k=8
        self.hidden_dim = 100

        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5

        self.temperature = 0.2
        self.use_cosine_similarity = True
