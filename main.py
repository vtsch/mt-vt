import comet_ml
import torch
import os
from torchsummary import summary
from preprocess import load_psa_data_to_pd, load_psa_df
from clustering_algorithms import run_kmeans, run_kmeans_only
from metrics import calculate_clustering_scores
from umapplot import run_umap
from models.baseline_models import CNN, RNNModel, SimpleAutoencoder, DeepAutoencoder
from train import Trainer
from utils import get_bunch_config_from_json, build_save_path, build_comet_logger, set_requires_grad, _calc_metrics, copy_Files
from models.transformer import TransformerTimeSeries
from models.model import base_Model
import numpy as np
from configs import Config
from traintstcc import TSTCCTrainer


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
        df_psa = load_psa_data_to_pd(file_name, config)
        
    # run kmeans on raw data

    if config.experiment_name == "raw_data":
        df_psa = load_psa_df(file_name)
        y_real = df_psa['pros_cancer']
        df_psa = df_psa.iloc[:,:-2]
        df_train_values = df_psa.values
        kmeans_labels = run_kmeans(df_train_values, config, experiment)
        run_umap(df_psa, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real.astype(int), kmeans_labels, experiment)


    # run embedding models and kmeans
    if config.experiment_name == "simple_ac":
        model = SimpleAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        targets, output, _ = trainer.eval()
    
        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "deep_ac":
        model = DeepAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        targets, output, _ = trainer.eval()
        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "lstm": 
        model = RNNModel(input_size=config.ts_length, hid_size=32, emb_size=config.emb_size, rnn_type='lstm', bidirectional=True)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        targets, output, _ = trainer.eval()

        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "cnn":
        model = CNN(input_size=config.ts_length, emb_size=config.emb_size, hid_size=128)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        targets, output, _ = trainer.eval()

        kmeans_labels = run_kmeans(output, config, experiment)
        run_umap(output, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
    

    if config.experiment_name == "simple_transformer": 
        model = TransformerTimeSeries(config) 
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        targets, predictions, _ = trainer.eval()
    
        kmeans_labels = run_kmeans(predictions, config, experiment)
        run_umap(predictions, targets, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
        #if n_clusters > 2: summarize all >1 to 1 for clustering metrics
        kmeans_labels[kmeans_labels >= 1] = 1
        calculate_clustering_scores(targets.astype(int), kmeans_labels, experiment)
    
    if config.experiment_name == "ts-tcc":
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

        # Trainer
        trainer = TSTCCTrainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()

        if config.tstcc_training_mode == "train_linear":
            load_from = os.path.join(os.path.join(config.model_save_dir, f"tstcc_self_supervised/22-08-25_16-05-45"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=config.device)
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
   
    
        #if config.tstcc_training_mode == "self_supervised":  # to do it only once
        #    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description))

        if config.tstcc_training_mode != "self_supervised":
            # Testing
            outs = trainer.model_evaluate()
            total_loss, total_acc, pred_labels, true_labels, embeddings = outs
            _calc_metrics(pred_labels, true_labels, config.model_save_path)
        
            kmeans_labels = run_kmeans_only(embeddings, config.n_clusters, config.metric)
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            run_umap(embeddings, true_labels, kmeans_labels, config.experiment_name+config.tstcc_training_mode+"kmeans", experiment)
            run_umap(embeddings, true_labels, pred_labels, config.experiment_name+config.tstcc_training_mode+"pred", experiment)
            print("kmeans_labels")
            calculate_clustering_scores(true_labels.astype(int), kmeans_labels.astype(int), experiment)
            print("pred_labels")
            calculate_clustering_scores(true_labels.astype(int), pred_labels.astype(int), experiment)


