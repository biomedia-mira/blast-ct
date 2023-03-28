import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn

from blast_ct.read_config import get_model, get_optimizer, get_loss, get_train_loader, get_valid_loader, \
    get_test_loader, get_training_hooks
from blast_ct.trainer.model_trainer import ModelTrainer


def run_job(job_dir, train_csv_path, valid_csv_path, config_file, num_epochs, device, random_seed):
    with open(config_file, 'r') as f:
        config = json.load(f)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    set_random_seed(random_seed)
    device = set_device(device)
    print('Setting up configuration...')
    task = config['data']['task']
    model = get_model(config)
    use_cuda = device.type != 'cpu'
    train_loader = get_train_loader(config, model, train_csv_path, use_cuda)
    valid_loader = get_valid_loader(config, model, valid_csv_path, use_cuda)
    test_loader = get_test_loader(config, model, valid_csv_path, use_cuda)
    optimizer = get_optimizer(config, model)
    criterion = get_loss(config)
    hooks = get_training_hooks(job_dir, config, device, valid_loader, test_loader)

    # train the model
    print('Starting Training...')
    train_model = ModelTrainer(job_dir, device, model, criterion, optimizer, hooks, task)
    sucess = train_model(train_loader, num_epochs)
    return sucess


def set_random_seed(random_seed):
    # Set random seeds
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)


def set_device(device):
    device = int(device) if device != 'cpu' else device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Warning: running on CPU!')
    return device


def run_ensemble(job_dir, train_csv_path, valid_csv_path, config_file, num_epochs, device, random_seeds, overwrite):
    random_seeds = [int(seed) for seed in random_seeds.split()]

    if len(np.unique(random_seeds)) != len(random_seeds):
        raise ValueError("Duplicate random seeds were provided.")

    for random_seed in random_seeds:
        run_dir = os.path.join(job_dir, 'random_seed_' + str(random_seed))
        if os.path.exists(run_dir):
            if overwrite:
                print('Run already exists, overwriting...')
            else:
                print('Run already exists, not overwriting...')
                continue
        run_job(run_dir, train_csv_path, valid_csv_path, config_file, num_epochs, device, random_seed)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='Directory for checkpoints, exports, and logs.')
    parser.add_argument('--config-file',
                        required=True,
                        type=str,
                        help='A json configuration file for the job (see example files)')
    parser.add_argument('--train-csv-path',
                        required=True,
                        type=str,
                        help='Path to train csv file with paths of images, targets and masks.')
    parser.add_argument('--valid-csv-path',
                        required=True,
                        type=str,
                        help='Path to validation csv file with paths of images, targets and masks.')
    parser.add_argument('--num-epochs',
                        required=True,
                        type=int,
                        help='Number of epoch to train the model.')
    parser.add_argument('--device',
                        required=True,
                        type=str,
                        help='Device to use for computation')
    parser.add_argument('--random-seeds',
                        default="1 2 3 4 5",
                        type=str,
                        help='List of random seeds for training.')
    parser.add_argument('--overwrite',
                        default=False,
                        type=bool,
                        help='Whether to overwrite run if already exists')

    parse_args, unknown = parser.parse_known_args()

    run_ensemble(**parse_args.__dict__)


if __name__ == "__main__":
    train()
