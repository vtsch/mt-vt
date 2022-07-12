import torch
import os
from comet_ml import Experiment
from torchsummary import summary
from dataloader import load_ecg_data_to_pd, upsample_data, load_psa_data_to_pd, create_psa_df
from clustering_algorithms import run_kmeans
from metrics import calculate_clustering_scores
from umapplot import run_umap
from modules import CNN, RNNModel, RNNAttentionModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
from utils import get_bunch_config_from_json, build_save_path, build_comet_logger
#from transformer import TSTransformerEncoder


class Config:
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    metric = "dtw" #metric : {“euclidean”, “dtw”, “softdtw”} 
    n_clusters = 2
    lr=1e-3
    batch_size = 32
    n_epochs = 2
    emb_size = 2
    model_save_directory = "./models"
    use_comet_experiments = True

    PSA_DATA = True
    MOD_RAW = True
    MOD_SIMPLE_AC = False
    MOD_DEEP_AC = False
    MOD_LSTM = False
    MOD_CNN = False
    MOD_RNN_ATT = False
    MOD_TRANSF = False

    experiment_name = "raw_model" if MOD_RAW else "simple_ac" if MOD_SIMPLE_AC else "deep_ac" if MOD_DEEP_AC else "lstm_model" if MOD_LSTM else "cnn_model" if MOD_CNN else "rnn_attmodel" if MOD_RNN_ATT else "transformer_model" if MOD_TRANSF else "notimplemented"


if __name__ == '__main__':

    config = Config()

    save_path = build_save_path(config)
    os.makedirs(save_path)
    config.model_save_path = save_path

    experiment = build_comet_logger(config)

    # load and preprocess PSA or ECG data 
    if config.PSA_DATA == True:
        file_name = "data/pros_data_mar22_d032222.csv"
        ts_length = 6

        #load data
        df_raw = load_psa_data_to_pd(file_name)
        df_psa = create_psa_df(df_raw)
        df_psa = upsample_data(df_psa, n_clusters=config.n_clusters, sample_size=3000)
        #create train test split
        df_train, df_test = df_psa.iloc[:int(len(df_psa)*0.8)], df_psa.iloc[int(len(df_psa)*0.8):]
        y_real = df_train['pros_cancer']
        
    else:
        file_name_train = 'data/mitbih_train.csv'
        file_name_test = 'data/mitbih_test.csv'
        ts_length = 186

        #load data
        df_mitbih_train, df_mitbih_test = load_ecg_data_to_pd(file_name_train, file_name_test)
        df_train = upsample_data(df_mitbih_train, n_clusters=config.n_clusters, sample_size=400)
        df_test = upsample_data(df_mitbih_test, n_clusters=config.n_clusters, sample_size=150)
        y_real = df_train['class']

    # run kmeans on raw data

    if config.MOD_RAW == True:
        df_train = df_train.iloc[:,:-2]
        df_train_values = df_train.values
        kmeans_labels = run_kmeans(df_train_values, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(df_train, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real, kmeans_labels, experiment)


    # run embedding models and kmeans

    if config.MOD_SIMPLE_AC == True:
        model = SimpleAutoencoder()
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, train_data = df_train, test_data=df_test, net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.n_epochs)
        trainer.run()
        output, target = trainer.eval(config.emb_size)
    
        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target, kmeans_labels, experiment)

    if config.MOD_DEEP_AC == True:
        model = DeepAutoencoder()
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, train_data = df_train, test_data=df_test, net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.n_epochs)
        trainer.run()
        output, target = trainer.eval(config.emb_size)
        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target, kmeans_labels, experiment)

    if config.MOD_LSTM == True: 
        model = RNNModel(input_size=ts_length, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, train_data=df_train, test_data=df_test, net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.n_epochs)
        trainer.run()
        output, target = trainer.eval(config.emb_size)

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target, kmeans_labels, experiment)

    if config.MOD_CNN == True:
        model = CNN(emb_size=config.emb_size, hid_size=128)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, train_data=df_train, test_data = df_test, net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.n_epochs)
        trainer.run()
        output, target = trainer.eval(config.emb_size)

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target, kmeans_labels, experiment)
    
    if config.MOD_RNN_ATT == True: 
        model = RNNAttentionModel(input_size=1, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=False)
        summary(model, input_size=(1, ts_length))
        trainer = Trainer(config=config, experiment=experiment, train_data=df_train, test_data=df_test, net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.n_epochs)
        trainer.run()
        output, target = trainer.eval(config.emb_size)

        kmeans_labels = run_kmeans(output, config.n_clusters, config.metric, config.experiment_name, experiment)
        run_umap(output, target, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(target, kmeans_labels, experiment)
    
    """
    if config.TRANSFORMER_MOD == True: 
        name = "Transformer"
        model = TSTransformerEncoder(feat_dim=1, max_len=186, d_model=64, n_heads=8, num_layers=3, dim_feedforward=256)
        summary(model, input_size=(1, 186))
        trainer = Trainer(config=config, train_data=df_train, test_data=df_test, net=model, lr=lr, batch_size=batch_size, num_epochs=n_epochs)
        trainer.run()
        output, target = trainer.eval(emb_size)

        kmeans_labels = run_kmeans(output, n_clusters, metric, config.experiment_name)
        run_umap(output, target, kmeans_labels, config.experiment_name)
        calculate_clustering_scores(target, kmeans_labels)
    """