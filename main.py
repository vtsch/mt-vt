import configparser
import comet_ml
import torch
import os
from torchsummary import summary
from preprocess import upsample_data, load_psa_data_to_pd
from dataloader import data_generator
from clustering_algorithms import run_kmeans, run_kmeans_only
from metrics import calculate_clustering_scores
from umapplot import run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
from utils import get_bunch_config_from_json, build_save_path, build_comet_logger, set_requires_grad
from transformer import TransformerTimeSeries
from models.model import base_Model
from models.TC import TC
from tstcctrainer import TSTCCTrainer
import numpy as np

class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 
    n_clusters = 2
    n_clusters_real = 2
    lr=0.001
    batch_size = 24
    n_epochs = 2
    emb_size = 12 #needs to be = tslength if baselines
    model_save_dir = "./saved_models"
    sample_size = 2000

    PSA_DATA = True
    upsample = True
    DELTATIMES = False
    NOPOSENC = False

    CHECK_FEATURES = False
    MOD_RAW = False
    MOD_SIMPLE_AC = False
    MOD_DEEP_AC = False
    MOD_LSTM = False
    MOD_CNN = False
    MOD_RNN_ATT = False
    MOD_TRANSFORMER = False
    MOD_TSTCC = True

    #transformer config
    dropout = 0.1
    num_layers = 4
    ts_length = 6
    max_value = 3000
    n_heads = 4
    
    #TSTCC
    input_channels = 1
    kernel_size = 2
    stride = 1
    final_out_channels = 16 #16 with k=2 #32 with k=8
    hidden_dim = 100

    num_classes = 2
    dropout = 0.35
    features_len = 6 #6 with k=2 #3 with k=8

    # optimizer parameters
    beta1 = 0.9
    beta2 = 0.99

    drop_last = True

    experiment_name = "raw_model" if MOD_RAW else "loaded features" if CHECK_FEATURES else "simple_ac" if MOD_SIMPLE_AC else "deep_ac" if MOD_DEEP_AC else "lstm_model" if MOD_LSTM else "cnn_model" if MOD_CNN else "rnn_attmodel" if MOD_RNN_ATT else "transformer_model" if MOD_TRANSFORMER else "ts-tcc" if MOD_TSTCC else "notimplemented"


if __name__ == '__main__':

    config = Config()

    save_path = build_save_path(config)
    os.makedirs(save_path)
    config.model_save_path = save_path

    experiment = build_comet_logger(config)
    cwd = os.getcwd()
    experiment.log_asset_folder(folder=cwd, step=None, log_file_name=True, recursive=False)

    # load and preprocess PSA or ECG data 
    if config.PSA_DATA == True:
        file_name = "data/pros_data_mar22_d032222.csv"
        df_psa = load_psa_data_to_pd(file_name, config)
        
    # run kmeans on raw data

    if config.MOD_RAW == True:
        y_real = df_psa['pros_cancer']
        df_psa = df_psa.iloc[:,:-2]
        df_train_values = df_psa.values
        kmeans_labels = run_kmeans(df_train_values, config, experiment)
        run_umap(df_psa, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real.astype(int), kmeans_labels, experiment)

    if config.CHECK_FEATURES == True:
        features = np.load('data/features_last.npy')
        y_real = np.load('data/labels_last.npy')
        kmeans_labels = run_kmeans_only(features, config.n_clusters, config.metric)
        features = features.reshape(features.shape[0], -1)
        run_umap(features, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real.astype(int), kmeans_labels, experiment)


    # run embedding models and kmeans
    if config.MOD_SIMPLE_AC == True:
        model = SimpleAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()
    
        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_DEEP_AC == True:
        model = DeepAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()
        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_LSTM == True: 
        model = RNNModel(input_size=config.ts_length, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_CNN == True:
        model = CNN(emb_size=config.emb_size, hid_size=128)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
    
    if config.MOD_RNN_ATT == True: 
        model = RNNAttentionModel(input_size=1, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=False)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
    

    if config.MOD_TRANSFORMER == True: 
        model = TransformerTimeSeries(config) 
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        total_loss, total_acc, outs, targets, predictions = trainer.eval()
    
        kmeans_labels = run_kmeans(predictions, config, experiment)
        run_umap(predictions, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        #if n_clusters > 2: summarize all >1 to 1 for clustering metrics
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
    
    if config.MOD_TSTCC: 
        # Load datasets
        #train_dl, valid_dl, test_dl = data_generator(df_psa, config)

        # Load Model
        model = base_Model(config).to(config.device)
        model_dict = model.state_dict()

        # delete all the parameters except for logits
        del_list = ['logits']
        pretrained_dict_copy = model_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del model_dict[i]
        set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

        # Trainer
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        total_loss, total_acc, outs, targets, embeddings = trainer.eval()

        kmeans_labels = run_kmeans_only(embeddings, config.n_clusters, config.metric)
        run_umap(embeddings, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)

