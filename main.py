import comet_ml
import torch
import os
from torchsummary import summary
import argparse
from configs import Config
from preprocess import load_psa_data_to_pd, load_plco_df, load_furst_df
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

    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    parser.add_argument('--d', type=str, default='plco', help='plco or furst dataset')
    parser.add_argument('--n', default=2, type=int,
                        help='nr of clusters')
    parser.add_argument('--exp', default="raw_data", type=str,
                        help='raw_data, simple_ac, deep_ac, lstm, cnn, simple_transformer, ts_tcc')
    parser.add_argument('--tstcc_tm', default="supervised", type=str, help='random_init, supervised, self_supervised, fine_tune, train_linear')
    parser.add_argument('--pos_enc', default="none", type=str, help='none, absolute_days, delta_days, learnable_pos_enc, age_pos_enc')
    parser.add_argument('--c', default=False, type=bool, help='use context')
    parser.add_argument('--c_a', default=True, type=bool, help='use age as context')
    parser.add_argument('--c_b', default=True, type=bool, help='use bmi as context')
    parser.add_argument('--c_c', default=True, type=bool, help='use center as context')
    parser.add_argument('--tstcc_mod_sdir', default="saved_models/ts_tcc/self_supervised/absolute_days/22-09-29_11-58-03all", type=str, help='ts_tcc model save dir')
    

    args = parser.parse_args()

    config = Config()
    config.dataset = args.d
    config.n_clusters = args.n
    config.experiment_name = args.exp
    config.tstcc_training_mode = args.tstcc_tm
    config.pos_enc = args.pos_enc
    config.context = args.c
    config.context_age = args.c_a
    config.context_bmi = args.c_b
    config.context_center = args.c_c
    config.tstcc_model_saved_dir = args.tstcc_mod_sdir

    save_path = build_save_path(config)
    os.makedirs(save_path)
    config.model_save_path = save_path

    experiment = build_comet_logger(config)
    cwd = os.getcwd()
    experiment.log_asset_folder(folder=cwd, step=None, log_file_name=True, recursive=False)

    # load and preprocess data 
    if config.dataset == "plco":
        file_name = "data/pros_data_mar22_d032222.csv"
        df_psa = load_psa_data_to_pd(file_name, config)
    elif config.dataset == "furst":
        file_name = "data/furst_data.csv"
        # TODO: load and preprocess furst data
    else:
        raise ValueError("Dataset not supported")
        
    # run kmeans on raw data
    if config.experiment_name == "raw_data":
        if config.dataset == "plco":
            df_psa = load_plco_df(file_name)
            y_real = df_psa['pros_cancer']
            df_psa = df_psa.iloc[:,:-1]
        elif config.dataset == "furst":
            df_psa = load_furst_df(file_name)
            #TODO
        else:
            raise ValueError("Dataset not supported")
        df_train_values = df_psa.values
        kmeans_labels = run_kmeans_and_plots(df_train_values, config, experiment)
        df = df_psa.to_numpy()
        run_umap(df_psa, y_real, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(y_real.astype(int), kmeans_labels, experiment)


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




