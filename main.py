import comet_ml
import torch
import os
from torchsummary import summary
from preprocess import load_psa_data_to_pd
from kmeans import kmeans, run_kmeans_and_plots, plot_datapoints
from metrics import calculate_clustering_scores, log_cluster_combinations
from utils import get_args, build_save_path, build_comet_logger, set_required_grad, get_bunch_config_from_json, complete_config
from plots import run_umap
from train import Trainer
from traintstcc import TSTCCTrainer
from models.baseline_models import CNN, LSTMencoder, SimpleAutoencoder, DeepAutoencoder
from models.transformer import TSTransformerEncoder
from models.tstcc_basemodel import TSTCCbase_Model


if __name__ == '__main__':

    args = get_args()
    config = get_bunch_config_from_json(args.config)
    config = complete_config(config, args)
    
    print("config.experiment_name: ", config.experiment_name)
    print("config.n_clusters: ", config.n_clusters)
    print("config.pos_enc: ", config.pos_enc)
    
    # build model save path
    config.model_save_path = save_path = build_save_path(config)
    os.makedirs(config.model_save_path)
    
    # build comet logger
    experiment = build_comet_logger(config)
    cwd = os.getcwd()
    experiment.log_asset_folder(folder=cwd, step=None, log_file_name=True, recursive=False)

    # load and preprocess data 
    if config.dataset == "plco":
        file_name = "data/pros_data_mar22_d032222.csv"
    elif config.dataset == "furst":
        file_name = "data/psadata_furst.csv"
    else:
        raise ValueError("Dataset not found")
    df_psa = load_psa_data_to_pd(file_name, config)

    # run kmeans on raw data
    if config.experiment_name == "raw_data":


        true_labels = df_psa[config.classlabel]
        if config.dataset == "plco":
            if config.pos_enc == "absolute_days":
                for i in range(0, 6):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_level' + str(i)] + df_psa['psa_days' + str(i)]
                    df_psa.drop('psa_level' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_days' + str(i), axis=1, inplace=True)
            elif config.pos_enc == "delta_days":
                for i in range(0, 6):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_level' + str(i)] + df_psa['psa_delta' + str(i)]
                    df_psa.drop('psa_level' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_delta' + str(i), axis=1, inplace=True)
            elif config.pos_enc == "age_pos_enc":
                for i in range(0, 6):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_level' + str(i)] + df_psa['psa_age' + str(i)]
                    df_psa.drop('psa_level' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_age' + str(i), axis=1, inplace=True)
            else:
                pass
        elif config.dataset == "furst":
            if config.pos_enc == "absolute_days":
                for i in range(0, 20):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_' + str(i)] + df_psa['psa_absolute' + str(i)]
                    df_psa.drop('psa_' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_absolute' + str(i), axis=1, inplace=True)
            elif config.pos_enc == "delta_days":
                for i in range(0, 20):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_' + str(i)] + df_psa['psa_delta' + str(i)]
                    df_psa.drop('psa_' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_delta' + str(i), axis=1, inplace=True)
            elif config.pos_enc == "age_pos_enc":
                for i in range(0, 20):
                    df_psa["psa_with_enc" + str(i)] = df_psa['psa_' + str(i)] + df_psa['psa_age' + str(i)]
                    df_psa.drop('psa_' + str(i), axis=1, inplace=True)
                    df_psa.drop('psa_age' + str(i), axis=1, inplace=True)
            else:
                pass

        else:
            raise ValueError("Dataset not supported")

        df_psa.drop([config.classlabel], axis=1, inplace=True)
        df_train_values = df_psa.values

        kmeans_labels = run_kmeans_and_plots(config, df_train_values, true_labels, experiment)
        df = df_psa.to_numpy()
        run_umap(df_psa, true_labels, kmeans_labels, config.experiment_name, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)


    # run embedding models and kmeans
    if config.experiment_name == "simple_ac":
        model = SimpleAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()
    
        kmeans_labels = run_kmeans_and_plots(config, output, true_labels, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "deep_ac":
        model = DeepAutoencoder(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()

        kmeans_labels = run_kmeans_and_plots(config, output, true_labels, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)

    if config.experiment_name == "lstm": 
        model = LSTMencoder(config)
        print(model)
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()

        kmeans_labels = run_kmeans_and_plots(config, output, true_labels, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)


    if config.experiment_name == "cnn":
        model = CNN(config)
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()

        kmeans_labels = run_kmeans_and_plots(config, output, true_labels, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)
    

    if config.experiment_name == "simple_transformer": 
        model = TSTransformerEncoder(config) 
        summary(model, input_size=(1, config.ts_length))
        trainer = Trainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()
        true_labels, output, _ = trainer.eval()
    
        kmeans_labels = run_kmeans_and_plots(config, output, true_labels, experiment)
        calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels, experiment)
    
    if config.experiment_name == "ts_tcc":
        experiment.set_name(config.experiment_name+config.tstcc_training_mode)
        model = TSTCCbase_Model(config).to(config.device)

        if config.tstcc_training_mode == "random_init":
            model_dict = model.state_dict()
            # delete all the parameters except for logits
            del_list = ['logits']
            pretrained_dict_copy = model_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del model_dict[i]
            set_required_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

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
            set_required_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.
        
        # Trainer
        trainer = TSTCCTrainer(config=config, experiment=experiment, data=df_psa, net=model)
        trainer.run()

        if config.tstcc_training_mode != "self_supervised":
            # Testing
            outs = trainer.model_evaluate()
            _, _, pred_labels, true_labels, embeddings = outs

            #plot_datapoints(embeddings, pred_labels.astype(int), config.experiment_name+"pred", experiment)
            #run_umap(embeddings, true_labels, pred_labels, config.experiment_name+config.tstcc_training_mode+"pred", experiment)
            #calculate_clustering_scores(true_labels.astype(int), pred_labels.astype(int), experiment)

            kmeans_labels = run_kmeans_and_plots(config, embeddings, true_labels, experiment)
            calculate_clustering_scores(config, true_labels.astype(int), kmeans_labels.astype(int), experiment)

    # calculate F1 score for all combination of labels
    if config.n_clusters > 2 and config.tstcc_training_mode != "self_supervised":
        log_cluster_combinations(config, true_labels.astype(int), kmeans_labels, experiment)




