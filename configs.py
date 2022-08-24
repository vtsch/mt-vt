import torch

class Config(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_save_dir = "./saved_models"

        # kmeans
        self.metric = "dtw"  # metric : {“euclidean”, “dtw”, “softdtw”}
        self.n_clusters = 3
        self.n_clusters_real = 2

        # data
        self.sample_size = 10000
        self.PSA_DATA = True
        self.upsample = False
        self.DELTATIMES = False
        self.NOPOSENC = False
        self.ts_length = 6

        # models
        self.loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.MSELoss() 
        self.lr = 0.001
        self.batch_size = 24
        self.n_epochs = 5
        self.emb_size = 6  # needs to be = tslength if baselines and tstcc
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.dropout = 0.3

        # experiment
        self.MOD_RAW = True
        self.MOD_SIMPLE_AC = False 
        self.MOD_DEEP_AC = False
        self.MOD_LSTM = False
        self.MOD_CNN = False
        self.MOD_TRANSFORMER = False
        self.MOD_TSTCC = False
        self.experiment_name = "raw_model" if self.MOD_RAW else "simple_ac" if self.MOD_SIMPLE_AC else "deep_ac" if self.MOD_DEEP_AC else "lstm_model" if self.MOD_LSTM else "cnn_model" if self.MOD_CNN else "transformer_model" if self.MOD_TRANSFORMER else "ts-tcc" if self.MOD_TSTCC else "notimplemented"

        # transformer
        self.num_layers = 1
        self.max_value = 3000
        self.n_heads = 2

        # ts-tcc
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
