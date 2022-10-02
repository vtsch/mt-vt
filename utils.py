
from comet_ml import Experiment
import argparse
import json
import os
import torch
from datetime import datetime
from bunch import Bunch

def get_args() -> argparse.Namespace:
    """
    Get the arguments from the command line.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", help="Path to the config file", default="config.json")    
    argparser.add_argument("-exp", "--experiment_name", help="model to run, options: raw_data, simple_ac, deep_ac, lstm, cnn, simple_transformer, ts_tc", default="simple_transformer", type=str)
    argparser.add_argument("-n_clusters", "--n_clusters", help="number of clusters to predict", default=2, type=int)
    argparser.add_argument("-pos_enc", "--positional_encoding", help="positional encoding to use, options: none, absolute_days, delta_days, age_pos_enc, learnable_pos_enc", default="none", type=str)
    argparser.add_argument("-tstcc_tm", "--tstcc_training_mode", help="training mode for TS-TCC, options: supervised, self_supervised, fine_tune, train_linear", default="supervised", type=str)
    argparser.add_argument("-tstcc_dir", "--tstcc_last_dir", help="path to the saved TS-TCC model", default="", type=str)
    return argparser.parse_args()

def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    '''
    Parameters:
        json_file_path: path to the json file
    Returns:
        config: Bunch object with the configuration parameters
    '''
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)

def complete_config(config: Bunch, args: argparse.Namespace) -> Bunch:
    '''
    Parameters:
        config: config file
        args: arguments from the command line
    Returns:
        config: config file with the arguments from the command line and context computations
    '''
    config.experiment_name = args.experiment_name
    config.n_clusters = args.n_clusters
    config.pos_enc = args.positional_encoding
    config.tstcc_training_mode = args.tstcc_training_mode
    config.tstcc_model_saved_dir = os.path.join(config.tstcc_model_dir, config.pos_enc, args.tstcc_last_dir)

    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.context_count = 3 if config.context_bmi and config.context_age and config.context_center else 1
    config.context_count_size = config.context_count if config.context else 0 

    if config.dataset == 'furst':
        config.classlabel = 'cancer'
        config.ts_length = 20
    if config.dataset == 'plco':
        config.classlabel = 'pros_cancer'
        config.ts_length = 6

    return config

def build_save_path(config: Bunch) -> str:
    '''
    Parameters:
        config: config file
    Returns:
        save_path: path to where the model will be saved
    '''
    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    return os.path.join(
        config.model_save_dir, config.experiment_name, config.tstcc_training_mode, config.pos_enc, current_timestamp
    )

def build_comet_logger(config: Bunch) -> Experiment:
    '''
    Parameters:
        config: config file
    Returns:
        experiment: Comet logger
    '''
    experiment = Experiment(
        api_key="HzUytPeFFfa9aiGmafVP6CMkP",
        project_name="mt-vt-results",
        workspace="vtsch",
    )
    experiment.set_name(config.experiment_name)
    experiment.log_parameters(config)
    return experiment

# for TS-TCC
def set_required_grad(model, dict_, requires_grad=True) -> None:
    '''
    Parameters:
        model: model to set the required grad
        dict_: dictionary with the parameters to set the required grad
        requires_grad: boolean to set the required grad
    '''
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad
