import os
import os.path as osp
import random
import torch

basedir = os.path.abspath(os.path.dirname(__file__))
CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class ModelConfig(object):
    if os.getenv("FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = osp.realpath(osp.join(CURRENT_PATH, ".."))
    model_dir = osp.join(root_dir, "models")
    data_dir = osp.join(root_dir, "data")


model_config = ModelConfig()