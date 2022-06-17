import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from z_dataloader import loadfiveclusters, loadtwoclusters, create_pandas_df, generate_train_test
from z_clustering_algorithms import sklearnkmeans, k_means_dtw
from z_utils import plot_centroids, plot_umap, plot_loss
from z_embeddings import umap_embedding
from z_modules import CNN, RNN, Encoder, RNNAttentionModel, RNN, RNNAttentionModel, RecurrentAutoencoder, Encoder
from z_train import Trainer
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
    RNN_MOD = False
    CNN_MOD = False
    LSTM_AC = True
    LSTM_ENC = False


if __name__ == '__main__':

    file_name_train = 'data/mitbih_train.csv'
    file_name_test = 'data/mitbih_test.csv'
    n_clusters = 5
    sample_size = 600

    config = Config()

    df_mitbih_train = pd.read_csv(file_name_train, header=None)
    df_mitbih_test = pd.read_csv(file_name_test, header=None)
    df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)
    df_mitbih.rename(columns={187: 'class'}, inplace=True)
    #print(df_mitbih.head(5))

    #select sample_size samples from each class
    df = pd.DataFrame()
    for i in range(n_clusters):
        df = pd.concat([df, df_mitbih.loc[df_mitbih['class'] == i].sample(n=sample_size)])
    #shuffle rows of df and remove index column
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)  
    #print(df.head(5))
    
    #print("original data")
    #print(df_mitbih.info())
    #print("subsampled data")
    print(df.info())
 

    if config.RAW_MOD == True:
        umap_emb = umap_embedding(df)

        centroids, kmeans_labels = sklearnkmeans(df, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids original data")

        #centroids, dtwkmeans_labels = k_means_dtw(df, n_clusters, num_iter=5,w=5)
        #plot_centroids(centroids, n_clusters, "dtw kmeans centroids original data")

        plot_umap(umap_emb, df['class'], "umap embedding original data")


    #model = RNNAttentionModel(1, 64, 'lstm', False)
    #model = RNNModel(1, 64, 'lstm', True)

    if config.CNN_MOD == True:
        model = CNN(num_classes=n_clusters, hid_size=128)
        trainer = Trainer(config=config, train_data = df, net=model, lr=1e-3, batch_size=32, num_epochs=3)#100)
        history, embeddings = trainer.run()
        plot_loss(history, 'CNN Loss')
        centroids, kmeans_labels = sklearnkmeans(embeddings, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids CNN")
        umap_emb = umap_embedding(embeddings)
        plot_umap(umap_emb, kmeans_labels, "umap embedding CNN encoder")
        print("ARI kmeans: %f" % adjusted_rand_score(df_mitbih['class'], kmeans_labels))


    if config.LSTM_AC == True:
        model = RecurrentAutoencoder(seq_len=186, n_features=1, embedding_dim=64)
        trainer = Trainer(config=config, train_data = df, net=model, lr=1e-3, batch_size=32, num_epochs=3)#100)
        history, embeddings = trainer.run()
        plot_loss(history, 'LSTM AC Loss')
        centroids, kmeans_labels = sklearnkmeans(embeddings, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids lstm AC")
        umap_emb = umap_embedding(embeddings)
        plot_umap(umap_emb, kmeans_labels, "umap embedding lstm AC")
        

    if config.LSTM_ENC == True:
        model = Encoder(seq_len=186, n_features=1, embedding_dim=32)
        trainer = Trainer(config = config, train_data = df, net=model, lr=1e-3, batch_size=32, num_epochs=100)#100)
        history, embeddings = trainer.run()
        plot_loss(history, 'LSTM AC Loss')
        centroids, kmeans_labels = sklearnkmeans(embeddings, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids lstm encoder")
        umap_emb = umap_embedding(embeddings)
        print(kmeans_labels.shape)
        print(embeddings.shape)
        plot_umap(umap_emb, kmeans_labels, "umap embedding lstm encoder")



