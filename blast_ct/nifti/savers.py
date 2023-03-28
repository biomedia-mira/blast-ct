import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import pandas as pd
from blast_ct.localisation.localise_lesions import LesionVolumeLocalisationMNI
from blast_ct.localisation.register_to_template import RegistrationToCTTemplate
from blast_ct.nifti.datasets import FullImageToOverlappingPatchesNiftiDataset
from blast_ct.nifti.patch_samplers import get_patch_and_padding
from blast_ct.nifti.rescale import create_reference_reoriented_image

CLASS_NAMES = ['background', 'iph', 'eah', 'oedema', 'ivh']


def add_predicted_volumes_to_dataframe(dataframe, id_, array, resolution):
    voxel_volume_ml = np.prod(resolution) / 1000.
    for i, class_name in enumerate(CLASS_NAMES):
        if i == 0:
            continue
        volume = np.sum(array == i) * voxel_volume_ml
        dataframe.loc[id_, f'{class_name:s}_predicted_volume_ml'] = round(volume, 2)
    return dataframe


def save_image(output_array, input_image, path, resolution):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image = sitk.GetImageFromArray(output_array)
    reference = create_reference_reoriented_image(input_image)
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    image.SetSpacing(resolution)
    image = sitk.Resample(image, input_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    sitk.WriteImage(image, path)
    return image


def get_num_maps(patches):
    shape = patches[0].shape
    if len(shape) == 3:
        return 1
    elif len(shape) == 4:
        return shape[0]
    else:
        raise ValueError('Trying to save a tensor with dimensionality which is not 3 or 4.')


def reconstruct_image(patches, image_shape, center_points, patch_shape):
    num_maps = get_num_maps(patches)
    assert len(patches) == len(center_points)
    padded_shape = tuple(s - s % ps + ps for s, ps in zip(image_shape, patch_shape))
    reconstruction = np.zeros(shape=(num_maps,) + padded_shape)
    for center, patch in zip(center_points, patches):
        slices, _ = get_patch_and_padding(padded_shape, patch_shape, center)
        reconstruction[(slice(0, num_maps, 1),) + tuple(slices)] = patch
    reconstruction = reconstruction[(slice(0, num_maps, 1),) + tuple(slice(0, s, 1) for s in image_shape)]
    reconstruction = reconstruction.transpose(tuple(range(1, reconstruction.ndim)) + (0,))
    return reconstruction


class Localisation(object):
    def __init__(self, localisation_dir, num_runs, native_space, write_registration_info):
        if not os.path.exists(localisation_dir):
            os.makedirs(localisation_dir)
        self.localisation_dir = localisation_dir

        asset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                 'data/localisation_files')
        target_template_path = os.path.join(asset_dir, 'ct_template.nii.gz')
        atlas_label_map_path = os.path.join(asset_dir, 'atlas_template_space.nii.gz')
        brain_mask_path = os.path.join(asset_dir, 'ct_template_mask.nii.gz')
        roi_dictionary_csv = os.path.join(asset_dir, 'atlas_labels.csv')

        self.register = RegistrationToCTTemplate(localisation_dir, target_template_path, num_runs=num_runs)
        self.localise = LesionVolumeLocalisationMNI(localisation_dir, native_space, atlas_label_map_path,
                                                    brain_mask_path, roi_dictionary_csv, 'prediction',
                                                    write_registration_info)

    def __call__(self, data_index, image_id, input_image, prediction):
        transform, data_index = self.register(data_index, input_image, image_id)
        if transform is not None:
            return self.localise(transform, data_index, image_id, prediction)
        return data_index


class NiftiPatchSaver(object):
    def __init__(self, job_dir, dataloader, write_prob_maps=True, extra_output_names=None, do_localisation=False,
                 num_reg_runs=1, native_space=True, write_registration_info=False):
        assert isinstance(dataloader.dataset, FullImageToOverlappingPatchesNiftiDataset)

        self.prediction_dir = os.path.join(job_dir, 'predictions')
        self.prediction_csv_path = os.path.join(self.prediction_dir, 'prediction.csv')
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.write_prob_maps = write_prob_maps
        self.patches = []
        self.extra_output_patches = {key: [] for key in extra_output_names} if extra_output_names is not None else {}
        self.image_index = 0
        if os.path.exists(self.prediction_csv_path):
            self.prediction_index = pd.read_csv(self.prediction_csv_path, index_col='id')
        else:
            self.prediction_index = pd.DataFrame(columns=self.dataset.data_index.columns)

        localisation_dir = os.path.join(job_dir, 'localisation')
        self.localisation = Localisation(localisation_dir, num_reg_runs, native_space, write_registration_info) if do_localisation else None

    def reset(self):
        self.image_index = 0
        self.patches = []
        if self.extra_output_patches is not None:
            self.extra_output_patches = {key: [] for key in self.extra_output_patches}

    def append(self, state):
        if self.write_prob_maps:
            self.patches += list(state['prob'].cpu().detach())
        else:
            self.patches += list(state['pred'].cpu().detach())
        for name in self.extra_output_patches:
            self.extra_output_patches[name] += list(state[name].cpu().detach())

    def __call__(self, state):
        self.append(state)
        target_shape, center_points = self.dataset.image_mapping[self.image_index]
        target_patch_shape = self.dataset.patch_sampler.target_patch_size
        patches_in_image = len(center_points)

        if len(self.patches) >= patches_in_image:
            to_write = {}
            image_id = self.dataset.data_index.iloc[self.image_index].name
            input_image = sitk.ReadImage(self.dataset.data_index.loc[image_id, self.dataset.channels[0]])
            patches = list(torch.stack(self.patches[0:patches_in_image]).numpy())
            self.patches = self.patches[patches_in_image:]
            reconstruction = reconstruct_image(patches, target_shape, center_points, target_patch_shape)

            if self.write_prob_maps:
                to_write['prob_maps'] = reconstruction
                to_write['prediction'] = np.argmax(reconstruction, axis=-1).astype(np.float64)
            else:
                to_write['prediction'] = reconstruction

            for name in self.extra_output_patches:
                patches = list(torch.stack(self.extra_output_patches[name][0:patches_in_image]).numpy())
                self.extra_output_patches[name] = self.extra_output_patches[name][patches_in_image:]
                images = reconstruct_image(patches, target_shape, center_points, target_patch_shape)
                to_write[name] = images

            resolution = self.dataset.resolution if self.dataset.resolution is not None else input_image.GetSpacing()
            columns = self.dataset.data_index.columns
            self.prediction_index.loc[image_id, columns] = self.dataset.data_index.loc[image_id, columns]
            for name, array in to_write.items():
                path = os.path.join(self.prediction_dir, f'{str(image_id):s}_{name:s}.nii.gz')
                self.prediction_index.loc[image_id, name] = path
                try:
                    output_image = save_image(array, input_image, path, resolution)
                    if name == 'prediction':
                        self.prediction_index = add_predicted_volumes_to_dataframe(self.prediction_index, image_id,
                                                                                   sitk.GetArrayFromImage(output_image),
                                                                                   output_image.GetSpacing())
                        if self.localisation is not None:
                            self.prediction_index = self.localisation(self.prediction_index, image_id, input_image,
                                                                      output_image)
                    message = f"{self.image_index:d}/{len(self.dataset.data_index):d}: Saved prediction for {str(image_id)}."
                except:
                    message = f"{self.image_index:d}/{len(self.dataset.data_index):d}: Error saving prediction for {str(image_id)}."
                    continue

            self.prediction_index.to_csv(self.prediction_csv_path, index_label='id')

            self.image_index += 1
            if self.image_index >= len(self.dataset.image_mapping):
                self.reset()

            return message

        return None
