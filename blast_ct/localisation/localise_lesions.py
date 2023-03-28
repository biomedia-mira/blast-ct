import os

import SimpleITK as sitk
import numpy as np
import pandas as pd


class LesionVolumeLocalisationMNI(object):
    def __init__(self, localisation_dir, native_space, atlas_label_map_path, brain_mask_path, roi_dictionary_csv,
                 label_map_name, write_registration_info):

        self.atlas_label_map = sitk.ReadImage(atlas_label_map_path)
        self.brain_mask = sitk.ReadImage(brain_mask_path, sitk.sitkInt8)
        roi_dictionary = pd.read_csv(roi_dictionary_csv)
        self.roi_dictionary = \
            {name: label for name, label in zip(roi_dictionary['ROIName'], roi_dictionary['ROIIndex'])}
        self.class_names = ['background', 'iph', 'eah', 'oedema', 'ivh']
        self.native_space = native_space
        self.localisation_dir = localisation_dir
        self.label_map_name = label_map_name
        self.write_registration_info = write_registration_info

    # Calculate total volume of a region or of the brain
    @staticmethod
    def calc_volume_ml(image):
        array = sitk.GetArrayFromImage(image)
        return np.sum(array) * np.prod(image.GetSpacing()) / 1000.

    def localise_lesion_volumes(self, label_map, atlas_label_map, brain_mask):
        # Creating a dictionary with 4 nested dictionaries (one per class). Inside those, the keys are the regions
        # labels and so far it is empty.
        localised_volumes = {class_name: {atlas_label_name: None for atlas_label_name in self.roi_dictionary}
                             for class_name in self.class_names}
        # Creating a dictionary to store the region volumes
        region_volumes = {atlas_label_name: None for atlas_label_name in self.roi_dictionary}
        for roi_name, roi_label in self.roi_dictionary.items():
            # if region is background (image_id 0)
            if roi_label == 0:
                continue
            # Create a mask for each region (each atlas label)
            region_mask = atlas_label_map == roi_label

            region_volumes[roi_name] = self.calc_volume_ml(sitk.Mask(region_mask, brain_mask))

            # Get a mask of the portion of each region occupied by the lesion
            masked_label_map = sitk.Mask(label_map, region_mask)
            for class_label, class_name in enumerate(self.class_names):
                # class_label = 0 -> background
                if class_label == 0:
                    continue

                masked_label_map_array = sitk.GetArrayFromImage(masked_label_map)
                masked_label_map_class = np.where(masked_label_map_array == class_label, 1, 0)
                masked_label_map_class = sitk.GetImageFromArray(masked_label_map_class)

                # Calculate the volume of each class type in the overlap between the region and lesion
                localised_volumes[class_name][roi_name] = self.calc_volume_ml(masked_label_map_class)
                # localised_volumes[class_name][roi_name] = self.calc_volume_ml(masked_label_map == class_label)

        return localised_volumes, region_volumes

    def __call__(self, aff_transform, data_index, image_id, label_map):
        # get the atlas_label_map, brain mask and label_map in the native or atlas space
        atlas_label_map = self.atlas_label_map
        brain_mask = self.brain_mask

        if self.native_space:
            atlas_label_map \
                = sitk.Resample(atlas_label_map, label_map, aff_transform.GetInverse(), sitk.sitkNearestNeighbor, 0)
            brain_mask = sitk.Resample(brain_mask, label_map, aff_transform.GetInverse(), sitk.sitkNearestNeighbor, 0)
        else:
            label_map = sitk.Resample(label_map, atlas_label_map, aff_transform, sitk.sitkNearestNeighbor, 0)

        localised_volumes, region_volumes = self.localise_lesion_volumes(label_map, atlas_label_map, brain_mask)
        for class_name in localised_volumes.keys():
            for roi_name in localised_volumes[class_name].keys():
                volume = localised_volumes[class_name][roi_name]
                if volume is not None:
                    data_index.loc[image_id, f'{self.label_map_name}_{class_name:s}_{roi_name:s}_ml'] = volume

        # add region volumes
        data_index.loc[image_id, f'Brain_volume_ml'] = self.calc_volume_ml(brain_mask)
        for roi_name in region_volumes.keys():
            volume = region_volumes[roi_name]
            if volume is not None:
                data_index.loc[image_id, f'{roi_name:s}_volume_ml'] = region_volumes[roi_name]

        if self.write_registration_info and self.native_space:
            atlas_native_space_path = os.path.join(self.localisation_dir, f'{str(image_id):s}_parc_atlas_native.nii.gz')
            sitk.WriteImage(atlas_label_map, atlas_native_space_path)
            brain_mask_native_space_path = os.path.join(self.localisation_dir,
                                                        f'{str(image_id):s}_brain_mask_native.nii.gz')
            sitk.WriteImage(brain_mask, brain_mask_native_space_path)
            data_index.loc[image_id, 'atlas_in_native_space'] = atlas_native_space_path
            data_index.loc[image_id, 'brain_mask_native_space'] = brain_mask_native_space_path
        return data_index
