from pickletools import TAKEN_FROM_ARGUMENT1, TAKEN_FROM_ARGUMENT4U
import torch
from torchsummary import summary
from dataloader import load_psa_data_to_pd, create_psa_df
from clustering_algorithms import run_kmeans
from utils import plot_centroids, plot_loss, calculate_clustering_scores, run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
import numpy as np
from transformer import TSTransformerEncoder

class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    RAW_MOD = False
    SIMPLE_AC = False
    DEEP_AC = False
    LSTM_MOD = True
    CNN_MOD = False
    RNN_ATTMOD = False
    TRANSFORMER_MOD = False



if __name__ == '__main__':

    n_clusters = 2
    lr=1e-3
    batch_size = 32
    n_epochs = 2
    emb_size = 2
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config = Config()

    #load data
    file_name = "data/pros_data_mar22_d032222.csv"
    df_raw = load_psa_data_to_pd(file_name)
    df_psa = create_psa_df(df_raw)
    print(df_psa.head(5))

    #create train test split
    df_train, df_test = df_psa.iloc[:int(len(df_psa)*0.8)], df_psa.iloc[int(len(df_psa)*0.8):]

    if config.RAW_MOD == True:
        name = "Raw Data"
        y_real = df_psa['pros_cancer']
        df_train = df_psa.iloc[:, :-2]
        df_train_values = df_train.values
        kmeans_labels = run_kmeans(df_train_values, n_clusters, metric, name)

        run_umap(df_train, y_real, kmeans_labels, name)
        calculate_clustering_scores(y_real, kmeans_labels)

    if config.SIMPLE_AC == True:
        name = "Simple AC"
        model = SimpleAutoencoder()
        summary(model, input_size=(1, 6))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        print("start evaluation")
        output, target = trainer.eval(emb_size)
    
        print("start clustering")
        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)

    if config.DEEP_AC == True:
        name = "Deep AC"
        model = DeepAutoencoder()
        summary(model, input_size=(1, 6))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)
        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)

    if config.LSTM_MOD == True: 
        name = "LSTM"
        model = RNNModel(input_size=6, hid_size=32, emb_size=10, rnn_type='lstm', bidirectional=True)
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
        summary(model, input_size=(1, 6))
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
        summary(model, input_size=(1, 6))
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    
    """ -- in progress --  
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
        calculate_clustering_scores(target, kmeans_labels)""
    """
    

