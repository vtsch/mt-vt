
from re import A
from comet_ml import Experiment
import argparse
import json
import os
import torch
from datetime import datetime
from bunch import Bunch

def get_args() -> argparse.Namespace:
    '''
    Get the configs from the command line
    Returns:
        args: arguments 
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", help="Path to the config file", default="config.json")    
    argparser.add_argument("-exp", "--experiment_name", help="model to run, options: raw_data, simple_ae, deep_ac, lstm, cnn, simple_transformer, ts_tc", default="cnn", type=str)
    argparser.add_argument("-n_clusters", "--n_clusters", help="number of clusters to predict", default=2, type=int)
    argparser.add_argument("-pos_enc", "--positional_encoding", help="positional encoding to use, options: none, absolute_days, delta_days, age_pos_enc, learnable_pos_enc", default="none", type=str)
    argparser.add_argument("-tstcc_tm", "--tstcc_training_mode", help="training mode for TS-TCC, options: supervised, self_supervised, fine_tune, train_linear", default="supervised", type=str)
    argparser.add_argument("-tstcc_dir", "--tstcc_last_dir", help="path to the saved TS-TCC model", default="", type=str)
    return argparser.parse_args()

def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    '''
    Load the configs from the json file
    Args:
        json_file_path: path to the json file
    Returns:
        config: Bunch object with the configuration parameters
    '''
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)

def complete_config(config: Bunch, args: argparse.Namespace) -> Bunch:
    '''
    Complete the config file with the arguments from the command line, additional calculations for context configs and dataset-specific configs
    Args:
        config: config file
        args: arguments from the command line
    Returns:
        config: config file with the arguments from the command line and context computations
    '''
    config.experiment_name = args.experiment_name
    config.n_clusters = args.n_clusters
    config.pos_enc = args.positional_encoding
    config.tstcc_training_mode = args.tstcc_training_mode
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # dataset-specific configs
    if config.dataset == 'furst':
        config.classlabel = 'cancer'
        config.ts_length = 20
    if config.dataset == 'plco':
        config.classlabel = 'pros_cancer'
        config.ts_length = 6

    # context configs
    config.context_count = 3 if config.context_bmi and config.context_age and config.context_center else 1
    config.context_count_size = config.context_count if config.context else 0 
    if config.context:
        if config.context_age and config.context_bmi and config.context_center:
            config.con_str = "all"
        elif config.context_age:
            config.con_str = "a"
        elif config.context_bmi:
            config.con_str = "b"
        elif config.context_center:
            config.con_str = "c"
        else:
            ValueError("invalid context configuration")
            pass
    else:
        config.con_str = "f"
    
    # model save dir
    if config.upsample:
        config.tstcc_model_saved_dir = os.path.join(config.tstcc_model_dir, config.pos_enc, config.con_str, "bal", args.tstcc_last_dir)
    else:
        config.tstcc_model_saved_dir = os.path.join(config.tstcc_model_dir, config.pos_enc, config.con_str, args.tstcc_last_dir)

    return config

def build_save_path(config: Bunch) -> str:
    '''
    Build the path for where to save the model
    Args:
        config: config file
    Returns:
        save_path: path to where the model will be saved
    '''
    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    path1 = os.path.join(config.model_save_dir, config.experiment_name, config.tstcc_training_mode, config.pos_enc, config.con_str)
    if config.upsample:
        path2 = os.path.join(path1, "bal")
    else:
        path2 = path1
    path = os.path.join(path2, current_timestamp)    
    return path

def build_comet_logger(config: Bunch) -> Experiment:
    '''
    Build the comet logger
    Args:
        config: config file
    Returns:
        experiment: Comet logger
    '''
    experiment = Experiment(
        api_key="HzUytPeFFfa9aiGmafVP6CMkP",
        project_name=config.comet_project_name,
        workspace="vtsch",
    )
    experiment.set_name(config.experiment_name)
    experiment.log_parameters(config)
    return experiment

# for TS-TCC
def set_required_grad(model, dict_, requires_grad=True) -> None:
    '''
    Set the requires_grad attribute of the parameters in the model
    Args:
        model: model to set the required grad
        dict_: dictionary with the parameters to set the required grad
        requires_grad: boolean to set the required grad
    '''
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad
