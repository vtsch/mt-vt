import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchsummary import summary
#from pytorch_model_summary import summary
import pandas as pd
from z_dataloader import load_raw_data_to_pd, upsample_data
from z_clustering_algorithms import sklearnkmeans, k_means_dtw, run_kmeans
from z_utils import plot_centroids, plot_umap, plot_loss
from z_modules import CNN, RNNModel, RNNAttentionModel, RecurrentAutoencoder, SimpleAutoencoder, DeepAutoencoder
from z_train import Trainer
from z_embeddings import umap_embedding
from sklearn.metrics.cluster import adjusted_rand_score



# https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism

class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '../input/mitbih-with-synthetic/attn.pth'
    lstm_state_path = '../input/mitbih-with-synthetic/lstm.pth'
    cnn_state_path = '../input/mitbih-with-synthetic/cnn.pth'
    
    attn_logs = '../input/mitbih-with-synthetic/attn.csv'
    lstm_logs = '../input/mitbih-with-synthetic/lstm.csv'
    cnn_logs = '../input/mitbih-with-synthetic/cnn.csv'

    RAW_MOD = False
    RNN_ATTMOD = False
    LSTM_MOD = False
    CNN_MOD = False
    SIMPLE_AC = False
    DEEP_AC = True


if __name__ == '__main__':

    file_name_train = 'data/mitbih_train.csv'
    file_name_test = 'data/mitbih_test.csv'
    n_clusters = 5
    emb_size = 10
    lr=1e-3
    batch_size = 32
    n_epochs = 3

    config = Config()

    #load data
    df_mitbih_train, df_mitbih_test = load_raw_data_to_pd(file_name_train, file_name_test, n_clusters=n_clusters)
    df_train = upsample_data(df_mitbih_train, n_clusters=n_clusters, sample_size=400)
    df_test = upsample_data(df_mitbih_test, n_clusters=n_clusters, sample_size=150)

    #print(df_train.info())


    if config.RAW_MOD == True:
        umap_emb = umap_embedding(df_train)

        centroids, kmeans_labels = sklearnkmeans(df_train, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids original data")

        #centroids, dtwkmeans_labels = k_means_dtw(df, n_clusters, num_iter=5,w=5)
        #plot_centroids(centroids, n_clusters, "dtw kmeans centroids original data")

        plot_umap(umap_emb, df_train['class'], "umap embedding original data")


    #model = RNNAttentionModel(1, 64, 'lstm', False)
    if config.LSTM_MOD == True: 
        name = "LSTM"
        model = RNNModel(input_size=1, hid_size=32, n_classes=n_clusters, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(n_clusters)

        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))

    if config.CNN_MOD == True:
        name = "CNN"
        model = CNN(num_classes=n_clusters, hid_size=128)
        #print(model)
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data=df_train, test_data = df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(n_clusters)

        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))
    
    if config.SIMPLE_AC == True:
        name = "Simple AC"
        model = SimpleAutoencoder()
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(n_clusters)
    
        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))

    if config.DEEP_AC == True:
        name = "Deep AC"
        model = DeepAutoencoder()
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(n_clusters)
    
        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))
        



