import torch
import os
from torchsummary import summary
from dataloader import load_ecg_data_to_pd, upsample_data, load_psa_data_to_pd
from clustering_algorithms import run_kmeans, run_kmeans_only
from metrics import calculate_clustering_scores
from umapplot import run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
from utils import get_bunch_config_from_json, build_save_path, build_comet_logger
from transformer import TransformerTimeSeries
import numpy as np

class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 
    n_clusters = 2
    lr=0.001
    batch_size = 12
    n_epochs = 10
    emb_size = 6
    model_save_directory = "./models"
    sample_size = 6000

    PSA_DATA = True
    DELTATIMES = True

    CHECK_FEATURES = False
    MOD_RAW = False
    MOD_SIMPLE_AC = False
    MOD_DEEP_AC = False
    MOD_LSTM = False
    MOD_CNN = False
    MOD_RNN_ATT = False
    MOD_TRANSFORMER = True

    experiment_name = "raw_model" if MOD_RAW else "loaded features" if CHECK_FEATURES else "simple_ac" if MOD_SIMPLE_AC else "deep_ac" if MOD_DEEP_AC else "lstm_model" if MOD_LSTM else "cnn_model" if MOD_CNN else "rnn_attmodel" if MOD_RNN_ATT else "transformer_model TS" if MOD_TRANSFORMER else "notimplemented"


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
        ts_length = 6
        df_psa = load_psa_data_to_pd(file_name, config)
        
    else:
        file_name_train = 'data/mitbih_train.csv'
        file_name_test = 'data/mitbih_test.csv'
        ts_length = 186

        #load data
        df_mitbih_train, df_mitbih_test = load_ecg_data_to_pd(file_name_train, file_name_test)
        df_psa = upsample_data(df_mitbih_train, n_clusters=config.n_clusters, sample_size=400)
        df_test = upsample_data(df_mitbih_test, n_clusters=config.n_clusters, sample_size=150)
        y_real = df_psa['class']

    # run kmeans on raw data

    if config.MOD_RAW == True:
        y_real = df_psa['pros_cancer']
        df_psa = df_psa.iloc[:,:-2]
        df_train_values = df_psa.values
        kmeans_labels = run_kmeans(df_train_values, config.n_clusters, config.metric, config.experiment_name, experiment)
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
        model = SimpleAutoencoder(ts_length, config.emb_size)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()
    
        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_DEEP_AC == True:
        model = DeepAutoencoder(ts_length, config.emb_size)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()
        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_LSTM == True: 
        model = RNNModel(input_size=ts_length, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)

    if config.MOD_CNN == True:
        model = CNN(emb_size=config.emb_size, hid_size=128)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
    
    if config.MOD_RNN_ATT == True: 
        model = RNNAttentionModel(input_size=1, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=False)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
    

    if config.MOD_TRANSFORMER == True: 
        model = TransformerTimeSeries() 
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        predictions, target = trainer.eval()
    
        kmeans_labels = run_kmeans(predictions, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(predictions, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target.astype(int), kmeans_labels, experiment)
    