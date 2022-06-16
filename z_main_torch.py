import numpy as np
import torch
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from z_dataloader import create_irregular_ts, reshape_lstm
from z_clustering_algorithms import sklearnkmeans, k_means_dtw
from z_utils import plot_centroids, plot_umap, plot_loss
from z_embeddings import umap_embedding
from z_modules import CNN, RNN, RNNAttentionModel, RNN, RNNAttentionModel
from z_train import Trainer


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
        df = df.append(df_mitbih.loc[df_mitbih['class'] == i].sample(n=sample_size))
    #shuffle rows of df qand remove index column
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    #print(df.head(5))
    
    print("original data")
    print(df_mitbih.info())
    print("subsampled data")
    print(df.info())

    #model = RNNAttentionModel(1, 64, 'lstm', False)
    #model = RNNModel(1, 64, 'lstm', True)
    model = CNN(num_classes=n_clusters, hid_size=128)
    trainer = Trainer(train_data = df, net=model, lr=1e-3, batch_size=96, num_epochs=3)#100)
    trainer.run()
