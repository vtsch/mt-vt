import comet_ml
import torch
import os
import numpy as np
from torchsummary import summary
from configs import Config
from preprocess import load_psa_data_to_pd, load_psa_df
from kmeans import run_kmeans_and_plots, run_kmeans_only, plot_datapoints
from metrics import calculate_clustering_scores, log_cluster_combinations
from utils import get_bunch_config_from_json, build_save_path, build_comet_logger, set_requires_grad
from umapplot import run_umap
from train import Trainer
from traintstcc import TSTCCTrainer
from models.baseline_models import CNN, LSTMencoder, SimpleAutoencoder, DeepAutoencoder
from models.transformer import TSTransformerEncoder
from models.tstcc_basemodel import base_Model


if __name__ == '__main__':

    config = Config()

    file_name = "data/pros_data_mar22_d032222.csv"

    save_path = build_save_path(config)
    os.makedirs(save_path)
    config.model_save_path = save_path

    experiment = build_comet_logger(config)
    cwd = os.getcwd()
    experiment.log_asset_folder(folder=cwd, step=None, log_file_name=True, recursive=False)
        
    # run kmeans on raw data
    if config.experiment_name == "raw_data":
        df_psa = load_psa_df(file_name)
        y_real = df_psa['pros_cancer']
        df_psa = df_psa.iloc[:,:-1]
        df_train_values = df_psa.values
        kmeans_labels = run_kmeans_and_plots(df_train_values, config, experiment)
        df = df_psa.to_numpy()
        run_umap(df_psa, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real.astype(int), kmeans_labels, experiment)

    # load and preprocess data 
    df_psa = load_psa_data_to_pd(file_name, config)

    # run embedding models and kmeans
    if config.experiment_name == "simple_ac":
        model = SimpleAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()
    
        kmeans_labels = run_kmeans_and_plots(output, config, experiment)
        run_umap(output, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(true_labels.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "deep_ac":
        model = DeepAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()
        kmeans_labels = run_kmeans_and_plots(output, config, experiment)
        run_umap(output, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(true_labels.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "lstm": 
        model = LSTMencoder(config)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()

        kmeans_labels = run_kmeans_and_plots(output, config, experiment)
        run_umap(output, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(true_labels.astype(int), kmeans_labels, experiment)


    if config.experiment_name == "cnn":
        model = CNN(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()

        kmeans_labels = run_kmeans_and_plots(output, config, experiment)
        run_umap(output, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(true_labels.astype(int), kmeans_labels, experiment)
    

    if config.experiment_name == "simple_transformer": 
        model = TSTransformerEncoder(config) 
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, predictions, _ = trainer.eval()
    
        kmeans_labels = run_kmeans_and_plots(predictions, config, experiment)
        run_umap(predictions, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(true_labels.astype(int), kmeans_labels, experiment)
    
    if config.experiment_name == "ts_tcc":
        experiment.set_name(config.experiment_name+config.tstcc_training_mode)
        model = base_Model(config).to(config.device)

        if config.tstcc_training_mode == "random_init":
            model_dict = model.state_dict()
            # delete all the parameters except for logits
            del_list = ['logits']
            pretrained_dict_copy = model_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del model_dict[i]
            set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

        if config.tstcc_training_mode == "fine_tune":
            # load saved model of this experiment
            chkpoint = torch.load(os.path.join(config.tstcc_model_saved_dir, "ckp_last.pt"), map_location=config.device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if config.tstcc_training_mode == "train_linear":
            chkpoint = torch.load(os.path.join(config.tstcc_model_saved_dir, "ckp_last.pt"), map_location=config.device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # delete these parameters (Ex: the linear layer at the end)
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.
        
        # Trainer
        trainer = TSTCCTrainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()

        if config.tstcc_training_mode != "self_supervised":
            # Testing
            outs = trainer.model_evaluate()
            total_loss, total_acc, pred_labels, true_labels, embeddings = outs

            #plot_datapoints(embeddings, pred_labels.astype(int), config.experiment_name+"pred", experiment)
            #run_umap(embeddings, true_labels, pred_labels, config.experiment_name+config.tstcc_training_mode+"pred", experiment)
            #calculate_clustering_scores(true_labels.astype(int), pred_labels.astype(int), experiment)

            kmeans_labels = run_kmeans_only(embeddings, config)
            plot_datapoints(embeddings, kmeans_labels, config.experiment_name+"kmeans", experiment)
            run_umap(embeddings, true_labels, kmeans_labels, config.experiment_name+config.tstcc_training_mode+"kmeans", experiment)
            calculate_clustering_scores(true_labels.astype(int), kmeans_labels.astype(int), experiment)

    # calculate F1 score for all combination of labels
    if config.n_clusters == 3 and config.experiment_name != "raw_data":
        log_cluster_combinations(true_labels.astype(int), kmeans_labels, experiment)




