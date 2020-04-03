import sys
import json
import os
import argparse
import pandas as pd
import shutil
from .trainer.inference import ModelInference, ModelInferenceEnsemble
from .train import set_device
from .read_config import get_model, get_test_loader
from .nifti.savers import NiftiPatchSaver


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def console_tool():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='input', type=path, help='Path to input image.')
    parser.add_argument('--output', metavar='output', type=str, help='Path to output image.')
    parser.add_argument('--ensemble', help='Whether to use the ensemble (slower but more precise)', type=bool,
                        default=False)
    parser.add_argument('--device', help='GPU device index (int) or \'cpu\' (str)', default='cpu')

    parse_args, unknown = parser.parse_known_args()
    if not (parse_args.input[-7:] == '.nii.gz' or parse_args.input[-4:] == '.nii'):
        raise IOError('Input file must be of type .nii or .nii.gz')

    install_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(install_dir, 'examples/config.json'), 'r') as f:
        config = json.load(f)

    device = set_device(parse_args.device)
    if device.type == 'cpu':
        config['test']['batch_size'] = 32
    job_dir = '/tmp/blast_ct'
    os.makedirs(job_dir, exist_ok=True)
    test_csv_path = os.path.join(job_dir, 'test.csv')
    pd.DataFrame(data=[[0, parse_args.input]], columns=['id', 'image']).to_csv(test_csv_path)

    model = get_model(config)
    test_loader = get_test_loader(config, model, test_csv_path, use_cuda=not device.type == 'cpu')
    saver = NiftiPatchSaver(job_dir, test_loader, write_prob_maps=False)

    if not parse_args.ensemble:
        model_path = os.path.join(install_dir, 'examples/saved_models/model_1.pt')
        ModelInference(job_dir, device, model, saver, model_path, 'segmentation')(test_loader)
    else:
        model_paths = [os.path.join(install_dir, f'examples/saved_models/model_{i:d}.pt') for i in range(1, 13)]
        ModelInferenceEnsemble(job_dir, device, model, saver, model_paths, task='segmentation')(test_loader)
    output_dataframe = pd.read_csv(os.path.join(job_dir, 'predictions/prediction.csv'))
    shutil.copyfile(output_dataframe.loc[0, 'prediction'], parse_args.output)

