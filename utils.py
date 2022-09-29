
from comet_ml import Experiment
import argparse
import json
import os
from datetime import datetime
from bunch import Bunch

def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Add the Configuration file that has all the relevant parameters",
    )
    argparser.add_argument(
        "-t",
        "--test_model_path",
        help="Path to saved model to be used for test set prediction",
    )
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

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

def build_save_path(config: Bunch) -> str:
    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    return os.path.join(
        config.model_save_dir, config.experiment_name, config.tstcc_training_mode, config.pos_enc, current_timestamp
    )

def build_comet_logger(config: Bunch) -> Experiment:
    experiment = Experiment(
        api_key="HzUytPeFFfa9aiGmafVP6CMkP",
        project_name="mt-vt",
        workspace="vtsch",
    )
    experiment.set_name(config.experiment_name)
    experiment.log_parameters(config)
    return experiment
