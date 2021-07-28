import os.path
import glob
from config import *
from pathlib import Path
import re
import torch

def create_config(log_dir, configurations=configurations):
    from ruamel.yaml import YAML
    YAML = YAML()
    with open(os.path.join(log_dir, 'configs.yaml'), mode='w', encoding='utf-8') as file:
        YAML.dump(configurations, file)

# from ruamel import yaml
# with open("configs.yaml", "r", encoding="utf-8") as f:
#     print(yaml.load(f.read(), Loader=yaml.Loader))

def load_weight(net, load_from):
    if os.path.exists(load_from):
        pretrain_weights = torch.load(load_from)
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        # print(public_keys)
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if net.state_dict()[k].numel()==pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        raise FileNotFoundError("[INFO] not found weights file: {}...".format(load_from))
    return net

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    # if path.exists() and not exist_ok:
    suffix = path.suffix
    path = path.with_suffix('')
    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 0  # increment number
    path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return str(path)
