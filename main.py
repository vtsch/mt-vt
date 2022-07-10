from pickletools import TAKEN_FROM_ARGUMENT1, TAKEN_FROM_ARGUMENT4U
import torch
from torchsummary import summary
from dataloader import load_ecg_data_to_pd, upsample_data, load_psa_data_to_pd, create_psa_df
from clustering_algorithms import run_kmeans
from utils import plot_centroids, plot_loss, calculate_clustering_scores, run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
import numpy as np
#from transformer import TSTransformerEncoder


class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '../input/mitbih-with-synthetic/attn.pth'
    lstm_state_path = '../input/mitbih-with-synthetic/lstm.pth'
    cnn_state_path = '../input/mitbih-with-synthetic/cnn.pth'
    
    attn_logs = '../input/mitbih-with-synthetic/attn.csv'
    lstm_logs = '../input/mitbih-with-synthetic/lstm.csv'
    cnn_logs = '../input/mitbih-with-synthetic/cnn.csv'

    PSA_DATA = True
    RAW_MOD = False
    SIMPLE_AC = False
    DEEP_AC = False
    LSTM_MOD = False
    CNN_MOD = True
    RNN_ATTMOD = False
    TRANSFORMER_MOD = False



if __name__ == '__main__':

    config = Config()
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_clusters = 2
    lr=1e-3
    batch_size = 32
    n_epochs = 1
    emb_size = 2

    # load and preprocess PSA or ECG data 
    if config.PSA_DATA == True:
        file_name = "data/pros_data_mar22_d032222.csv"
        ts_length = 6

        #load data
        df_raw = load_psa_data_to_pd(file_name)
        df_psa = create_psa_df(df_raw)
        #create train test split
        df_train, df_test = df_psa.iloc[:int(len(df_psa)*0.8)], df_psa.iloc[int(len(df_psa)*0.8):]
        y_real = df_train['pros_cancer']
        
    else:
        file_name_train = 'data/mitbih_train.csv'
        file_name_test = 'data/mitbih_test.csv'
        ts_length = 186

        #load data
        df_mitbih_train, df_mitbih_test = load_ecg_data_to_pd(file_name_train, file_name_test)
        df_train = upsample_data(df_mitbih_train, n_clusters=n_clusters, sample_size=400)
        df_test = upsample_data(df_mitbih_test, n_clusters=n_clusters, sample_size=150)
        y_real = df_train['class']

    # run kmeans on raw data

    if config.RAW_MOD == True:
        name = "Raw Data"
        df_train = df_train.iloc[:,:-2]
        df_train_values = df_train.values
        kmeans_labels = run_kmeans(df_train_values, n_clusters, metric, name)
        run_umap(df_train, y_real, kmeans_labels, name)
        calculate_clustering_scores(y_real, kmeans_labels)


    # run embedding models and kmeans

    if config.SIMPLE_AC == True:
        name = "Simple AC"
        model = SimpleAutoencoder()
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)
    
        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)

    if config.DEEP_AC == True:
        name = "Deep AC"
        model = DeepAutoencoder()
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)
        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)

    if config.LSTM_MOD == True: 
        name = "LSTM"
        model = RNNModel(input_size=ts_length, hid_size=32, emb_size=10, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)

    if config.CNN_MOD == True:
        name = "CNN"
        model = CNN(emb_size=emb_size, hid_size=128)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, train_data=df_train, test_data = df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    
    if config.RNN_ATTMOD == True: 
        name = "RNN Attention Module"
        model = RNNAttentionModel(input_size=1, hid_size=32, emb_size=emb_size, rnn_type='lstm', bidirectional=False)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    
    """
    if config.TRANSFORMER_MOD == True: 
        name = "Transformer"
        model = TSTransformerEncoder(feat_dim=1, max_len=186, d_model=64, n_heads=8, num_layers=3, dim_feedforward=256)
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)
        np.save('ouput_%s.npy' %name, output)
        np.save('target_%s.npy' %name, target)
        print(target.shape)
        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        np.save('kmeans_labels_%s.npy' %name, kmeans_labels)
        print(kmeans_labels.shape)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    """