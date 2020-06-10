import os
import argparse
import json
from blast_ct.trainer.inference import ModelInference, ModelInferenceEnsemble
from blast_ct.train import set_device
from blast_ct.read_config import get_model, get_test_loader
from blast_ct.nifti.savers import NiftiPatchSaver


def run_inference(job_dir, test_csv_path, config_file, device, saved_model_paths, overwrite):
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    else:
        if overwrite:
            print('Run already exists, overwriting...')
        else:
            print('Run already exists, not overwriting...')
            return

    with open(config_file, 'r') as f:
        config = json.load(f)

    model = get_model(config)
    device = set_device(device)
    use_cuda = device.type != 'cpu'
    test_loader = get_test_loader(config, model, test_csv_path, use_cuda)
    extra_output_names = config['test']['extra_output_names'] if 'extra_output_names' in config['test'] else None
    saver = NiftiPatchSaver(job_dir, test_loader, extra_output_names=extra_output_names)
    saved_model_paths = saved_model_paths.split()
    n_models = len(saved_model_paths)
    task = config['data']['task']
    if n_models == 1:
        ModelInference(job_dir, device, model, saver, saved_model_paths[0], task)(test_loader)
    elif n_models > 1:
        ModelInferenceEnsemble(job_dir, device, model, saver, saved_model_paths, task)(test_loader)


def inference():
    install_dir = os.path.dirname(os.path.realpath(__file__))
    default_config = os.path.join(install_dir, 'data/config.json')
    saved_model_paths = [os.path.join(install_dir, f'data/saved_models/model_{i:d}.pt') for i in range(1, 13)]
    default_model_paths = ' '.join(saved_model_paths)

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='Directory for checkpoints, exports, and '
                             'logs. Use an existing directory to load a '
                             'trained model, or a new directory to retrain')
    parser.add_argument('--test-csv-path',
                        default=None,
                        type=str,
                        help='Path to test csv file with paths of images and masks.')
    parser.add_argument('--config-file',
                        default=default_config,
                        type=str,
                        help='A json configuration file for the job (see example files)')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Device to use for computation')
    parser.add_argument('--saved-model-paths',
                        default=default_model_paths,
                        type=str,
                        help='Path to saved model or list of paths separated by spaces.')
    parser.add_argument('--overwrite',
                        default=False,
                        type=bool,
                        help='Whether to overwrite run if already exists')

    parse_args, unknown = parser.parse_known_args()

    run_inference(**parse_args.__dict__)
