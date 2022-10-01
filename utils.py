
from comet_ml import Experiment
import argparse
import json
import os
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
    return argparser.parse_args()

def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch object.
    :param json_file:
    :return: config as Bunch object:
    """
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)

def build_save_path(config: Bunch) -> str:
    """
    Build the path to save the model.
    """
    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    return os.path.join(
        config.model_save_dir, config.experiment_name, config.tstcc_training_mode, config.pos_enc, current_timestamp
    )

def build_comet_logger(config: Bunch) -> Experiment:
    """
    Build the comet logger.
    """
    experiment = Experiment(
        api_key="HzUytPeFFfa9aiGmafVP6CMkP",
        project_name="mt-vt",
        workspace="vtsch",
    )
    experiment.set_name(config.experiment_name)
    experiment.log_parameters(config)
    return experiment

# for TS-TCC
def set_required_grad(model, dict_, requires_grad=True) -> None:
    """
    Set the requires_grad attribute of the parameters in the model.
    """
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad
