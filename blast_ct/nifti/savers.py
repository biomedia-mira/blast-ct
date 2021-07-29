import os
import torch
import numpy as np
import SimpleITK as sitk
import time
from blast_ct.nifti.datasets import FullImageToOverlappingPatchesNiftiDataset
from blast_ct.nifti.patch_samplers import get_patch_and_padding
from blast_ct.nifti.rescale import create_reference_reoriented_image
from blast_ct.localisation.ct_to_template_reg_CP3_rigandaff_usedintheend_tointegrate import RegistrationToCTTemplate
from blast_ct.localisation.localise_lesion_volumes_CP_to_integrate import LesionVolumeLocalisationMNI

CLASS_NAMES = ['background', 'iph', 'eah', 'oedema', 'ivh']


def add_predicted_volumes_to_dataframe(dataframe, id_, array, resolution):
    voxel_volume_ml = np.prod(resolution) / 1000.
    for i, class_name in enumerate(CLASS_NAMES):
        if i == 0:
            continue
        volume = np.sum(array == i) * voxel_volume_ml
        dataframe.loc[dataframe['id'] == id_, f'{class_name:s}_predicted_volume_ml'] = volume
    return dataframe


def save_image(output_array, input_image, path, resolution=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image = sitk.GetImageFromArray(output_array)
    reference = create_reference_reoriented_image(input_image)
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    image.SetSpacing(resolution) if resolution is not None else image.SetSpacing(reference.GetSpacing())
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

def localise(data_index, input_image, prediction_, localisation_dir, image_id, write_registration_info,
               number_of_runs, native_space):
    if not os.path.exists(os.path.dirname(localisation_dir)):
        os.makedirs(localisation_dir)
    start_reg = time.time()
    transform, data_index_post_reg = RegistrationToCTTemplate(localisation_dir)(data_index, write_registration_info,
                                                                       number_of_runs, input_image, image_id)
    time_elapsed = time.time() - start_reg
    passed = time_elapsed
    print(f'Finished registration took {passed}s')
    data_index_post_localise = LesionVolumeLocalisationMNI(localisation_dir, native_space)(transform, data_index_post_reg, image_id,
                                                                         prediction_, write_registration_info)

    return data_index_post_localise

class NiftiPatchSaver(object):
    def __init__(self, job_dir, dataloader, write_prob_maps=True, extra_output_names=None,
                 localisation = False, number_of_runs = 1, native_space = True,
                 write_registration_info=False):
        assert isinstance(dataloader.dataset, FullImageToOverlappingPatchesNiftiDataset)

        self.prediction_dir = os.path.join(job_dir, 'predictions')
        self.localisation_dir = os.path.join(job_dir, 'localisation')
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.write_prob_maps = write_prob_maps

        self.patches = []
        self.extra_output_patches = {key: [] for key in extra_output_names} if extra_output_names is not None else {}

        self.image_index = 0
        self.data_index = self.dataset.data_index.copy()

        self.localisation = localisation
        self.number_of_runs = number_of_runs
        self.native_space = native_space
        self.write_registration_info = write_registration_info

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
        start_savers = time.time()
        self.append(state)
        target_shape, center_points = self.dataset.image_mapping[self.image_index]
        target_patch_shape = self.dataset.patch_sampler.target_patch_size
        patches_in_image = len(center_points)

        if len(self.patches) >= patches_in_image:
            to_write = {}
            id_ = self.dataset.data_index.loc[self.image_index]['id']
            input_image = sitk.ReadImage(self.dataset.data_index.loc[self.image_index][self.dataset.channels[0]])
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
            resolution = self.dataset.resolution
            for name, array in to_write.items():
                path = os.path.join(self.prediction_dir, f'{str(id_):s}_{name:s}.nii.gz')
                self.data_index.loc[self.data_index['id'] == id_, name] = path
                time_elapsed = time.time() - start_savers
                passed_savers = time_elapsed
                print(f'Since it entered savers until entering localisation took {passed_savers}s')
                print('entered localisation')
                saved_image = save_image(array, input_image, path, resolution)

                if name == 'prediction':
                    resolution_ = resolution if resolution is not None else input_image.GetSpacing()
                    self.data_index = add_predicted_volumes_to_dataframe(self.data_index, id_, array, resolution_)
                    if self.localisation:
                        self.data_index = localise(self.data_index, input_image, saved_image, self.localisation_dir,
                                                   id_, self.write_registration_info,
                                                   self.number_of_runs, self.native_space)

            self.image_index += 1
            message = f"{self.image_index:d}/{len(self.dataset.data_index):d}: Saved prediction for {str(id_)}."
            if self.image_index >= len(self.dataset.image_mapping):
                self.data_index.to_csv(os.path.join(self.prediction_dir, 'prediction.csv'), index=False)
                self.reset()

            return message

        return None
