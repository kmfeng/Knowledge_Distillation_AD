from argparse import ArgumentParser

from torch import nn
from random import randrange
from models.network import get_networks
from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_train.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    normal_class = config["normal_class"]

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['dataset_name'])
    train_output_path = "./outputs/{}/{}/train_outputs/".format(config['experiment_name'], config['dataset_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(train_output_path).mkdir(parents=True, exist_ok=True)

    epsilon = float(config['eps'])
    alpha = float(config['alpha'])

    train_dataloader, _, _ = load_data(config)

    vgg,model = get_networks(config)


if __name__ == '__main__':
    main()