import os
import argparse
import json
from blast_ct.trainer.inference import ModelInference, ModelInferenceEnsemble
from blast_ct.train import set_device
from blast_ct.read_config import get_model, get_test_loader
from blast_ct.nifti.savers import NiftiPatchSaver
from blast_ct.localisation.CT_to_template_reg_CP3_rigandaff_usedintheend_tointegrate import RegistrationToCTTemplate
from blast_ct.localisation.localise_lesion_volumes_CP_to_integrate import LesionVolumeLocalisationMNI

def run_inference(job_dir, test_csv_path, config_file, device, saved_model_paths, write_prob_maps, localisation,
                  write_reg_param, no_runs,overwrite, native_space, target_names):
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
    saver = NiftiPatchSaver(job_dir, test_loader, write_prob_maps=write_prob_maps,
                            extra_output_names=extra_output_names)
    saved_model_paths = saved_model_paths.split()
    n_models = len(saved_model_paths)
    task = config['data']['task']
    # Both classes called here are in trainer/inference.py
    if n_models == 1:
        print('entered model inference nmodel=1')
        ModelInference(job_dir, device, model, saver, saved_model_paths[0], task)(test_loader)
    elif n_models > 1:
        print('entered model inference nmodel>1')
        ModelInferenceEnsemble(job_dir, device, model, saver, saved_model_paths, task)(test_loader)

    # Lesion localisation

    data_index_csv = os.path.join(job_dir, 'predictions', 'prediction.csv')
    # Does it make sense to include this? If data_index_csv doesn't exist, the problem is in ModelInference class
    if not os.path.exists(data_index_csv):
        raise FileNotFoundError(f'File {data_index_csv:s} does not exist.')
    if localisation:
        print('entered localisation')
        csv_to_localise = RegistrationToCTTemplate()(data_index_csv, write_reg_param, no_runs)
        LesionVolumeLocalisationMNI(native_space)(csv_to_localise, target_names)


def inference():
    install_dir = os.path.dirname(os.path.realpath(__file__))
    # What is this config.json file?
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
    parser.add_argument('--write-prob-maps',
                        default=False,
                        type=bool,
                        help='Whether to write probability maps images to disk')
    parser.add_argument('--localisation',
                        default=False,
                        type=bool,
                        help='Whether to run localisation or not')
    parser.add_argument('--write-registration-info',
                        default=False,
                        type=bool,
                        help='Whether to write registration iterations and SM values to the csv and the resampled image'
                             'to the disk.')
    parser.add_argument('--number-of-runs',
                        default=1,
                        type=int,
                        help='How many times to run registration between native scan and CT template.')
    parser.add_argument('--overwrite',
                        default=False,
                        type=bool,
                        help='Whether to overwrite run if already exists')
    parser.add_argument('--native-space',
                        default=True,
                        type=bool,
                        help='Whether to calculate the volumes in native space or atlas space.')
    parser.add_argument('--target-names',
                        nargs='+',
                        required=True,
                        help='List of target names to be localised.')
    # Only makes sense if native space = True
    #parser.add_argument('--write-atlas-and-mask-in-native-space',
    #                    default=False,
    #                    type=bool,
    #                    help='Whether to write parcellated atlas and correspondent mask images in native space to disk.')

    #write_atlas_mask_ns = parse_args.write_atlas_and_mask_in_native_space
    #native_space_arg = parse_args.native_space
    #if write_atlas_mask_ns != native_space:
    #    raise ValueError('If localisation is done in atlas space, can\'t save atlas and mask in native space')

    parse_args, unknown = parser.parse_known_args()

    run_inference(**parse_args.__dict__)


if __name__ == "__main__":
    inference()
