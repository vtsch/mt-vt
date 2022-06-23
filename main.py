from pickletools import TAKEN_FROM_ARGUMENT1, TAKEN_FROM_ARGUMENT4U
import torch
from torchsummary import summary
from dataloader import load_raw_data_to_pd, upsample_data
from clustering_algorithms import run_kmeans
from utils import plot_centroids, plot_loss, calculate_clustering_scores, run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer



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
    SIMPLE_AC = False
    DEEP_AC = False
    LSTM_MOD = False
    CNN_MOD = True
    RNN_ATTMOD = False
    


if __name__ == '__main__':

    file_name_train = 'data/mitbih_train.csv'
    file_name_test = 'data/mitbih_test.csv'
    n_clusters = 2
    emb_size = 10
    lr=1e-3
    batch_size = 32
    n_epochs = 2
    emb_size = 5
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 

    config = Config()

    #load data
    df_mitbih_train, df_mitbih_test = load_raw_data_to_pd(file_name_train, file_name_test)
    df_train = upsample_data(df_mitbih_train, n_clusters=n_clusters, sample_size=400)
    df_test = upsample_data(df_mitbih_test, n_clusters=n_clusters, sample_size=150)

    #print(df_train.info())


    if config.RAW_MOD == True:
        name = "Raw Data"
        y_real = df_train['class']
        df_train = df_train.iloc[:,:186]
        kmeans_labels = run_kmeans(df_train, n_clusters, metric, name)

        run_umap(df_train, y_real, kmeans_labels, name)
        calculate_clustering_scores(y_real, kmeans_labels)


    #model = RNNAttentionModel(1, 64, 'lstm', False)
    if config.LSTM_MOD == True: 
        name = "LSTM"
        model = RNNModel(input_size=1, hid_size=32, n_classes=n_clusters, rnn_type='lstm', bidirectional=True)
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
        model = CNN(num_classes=n_clusters, hid_size=128)
        #print(model)
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data=df_train, test_data = df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    
    if config.SIMPLE_AC == True:
        name = "Simple AC"
        model = SimpleAutoencoder()
        summary(model, input_size=(1, 186))
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
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data = df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        history = trainer.run()
        plot_loss(history, '%s Loss' %name)
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, name)
        run_umap(output, target, kmeans_labels, name)
        calculate_clustering_scores(target, kmeans_labels)
    

        



