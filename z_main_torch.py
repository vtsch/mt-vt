import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchsummary import summary
import pandas as pd
from z_dataloader import load_raw_data_to_pd, upsample_data
from z_clustering_algorithms import sklearnkmeans, k_means_dtw, run_kmeans
from z_utils import plot_centroids, plot_umap, plot_loss
from z_modules import CNN, RNNModel, RNNAttentionModel, RecurrentAutoencoder
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
    RNN_MOD = True
    CNN_MOD = False
    LSTM_AC = False


if __name__ == '__main__':

    file_name_train = 'data/mitbih_train.csv'
    file_name_test = 'data/mitbih_test.csv'
    n_clusters = 5

    config = Config()

    #load data
    df_mitbih_train, df_mitbih_test = load_raw_data_to_pd(file_name_train, file_name_test, n_clusters=n_clusters)
    df_train = upsample_data(df_mitbih_train, n_clusters=n_clusters, sample_size=500)
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
    if config.RNN_MOD == True: 
        name = "RNN"
        model = RNNModel(1, 64, 'lstm', True)
        print(model)
        summary(model, input_size=(1, 64))
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=1e-3, batch_size=32, num_epochs=2)#100)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))

    if config.CNN_MOD == True:
        name = "CNN"
        model = CNN(num_classes=n_clusters, hid_size=128)
        trainer = Trainer(config=config, train_data=df_train, test_data = df_test, net=model, lr=1e-3, batch_size=32, num_epochs=2)#100)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval()

        kmeans_labels = run_kmeans(output, n_clusters, name)
        print("ARI kmeans: %f" % adjusted_rand_score(target, kmeans_labels))


    if config.LSTM_AC == True:
        model = RecurrentAutoencoder(seq_len=186, n_features=1, embedding_dim=64)
        trainer = Trainer(config=config, train_data = df_train, net=model, lr=1e-3, batch_size=32, num_epochs=3)#100)
        history = trainer.run()
        plot_loss(history, 'LSTM AC Loss')
        output, target = trainer.eval()
        print("output after eval:", output.shape)

        centroids, kmeans_labels = sklearnkmeans(output, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids lstm AC")
        umap_emb = umap_embedding(output)
        plot_umap(umap_emb, kmeans_labels, "umap embedding lstm AC")
        



