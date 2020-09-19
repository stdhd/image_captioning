import yaml
import os
from datetime import datetime


def parse_yaml(model_name: str, file_path: str, create_folder: bool = True) -> dict:
    """
    Read parameter dictionary from yaml file
    :param model_name: name of yaml file
    :param file_path: path of yaml file
    :param create_folder: create folder for saved_models, in case has not been created yet
    :return: dictionary of model parameters
    """
    file_path = os.path.join(file_path, model_name + ".yaml")
    print("Reading paramfile {}".format(file_path))
    try:
        with open(file_path) as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise (RuntimeError("Could not load model parameters from " + file_path + "."))

    if create_folder:
        # Find save directory
        if not os.path.exists("runs"):
            os.mkdir("runs")
        save_dir = os.path.join("runs", model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Copy yaml to save dir
        with open(os.path.join(save_dir, f'copy_{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.yaml'), 'w') as outfile:
            yaml.dump(param, outfile, default_flow_style=True)
    else:
        raise RuntimeError("No folders can be created in read-only mode.")

    return param
