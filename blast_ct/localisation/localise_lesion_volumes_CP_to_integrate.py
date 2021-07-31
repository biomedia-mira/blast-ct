import os
import SimpleITK as sitk
import pandas as pd
import numpy as np

class LesionVolumeLocalisationMNI(object):
    def __init__(self, localisation_dir, native_space, localisation_files_list):
        # The parcellated atlas in study specific space (CT template space)
        atlas_label_map_path = localisation_files_list[0]
        brain_mask_path = localisation_files_list[1]
        roi_dictionary_csv = localisation_files_list[2]


        # Reading all the images
        self.atlas_label_map = sitk.ReadImage(atlas_label_map_path)
        self.brain_mask = sitk.ReadImage(brain_mask_path, sitk.sitkInt8)
        roi_dictionary = pd.read_csv(roi_dictionary_csv)
        self.roi_dictionary = {name: label for name, label in
                               zip(roi_dictionary['ROIName'], roi_dictionary['ROIIndex'])}
        self.class_names = ['background', 'iph', 'eah', 'oedema', 'ivh']
        self.native_space = native_space
        self.localisation_dir = localisation_dir

    # Calculate total volume of a region or of the brain
    @staticmethod
    def calc_volume_ml(image):
        # Turning the image into an array of intensity values
        # As the image will be a mask, the array will just be 1s and 0s
        array = sitk.GetArrayFromImage(image)
        # np.sum gives the sum of every element in the array
        # .GetSpacing gets the spacing (x, y, z) between voxels -> so
        # np.prod(image.GetSpacing()) = volume of one voxel
        return np.sum(array) * np.prod(image.GetSpacing()) / 1000.

    def localise_lesion_volumes(self, label_map, atlas_label_map, brain_mask):
        # A dictionary with n nested dictionaries (one per class). Inside those, the keys are the regions
        # labels and so far it is empty.
        localised_volumes = {class_name: {atlas_label_name: None for atlas_label_name in self.roi_dictionary}
                             for class_name in self.class_names}
        # Creating a dictionary to store the region volumes
        region_volumes = {atlas_label_name: None for atlas_label_name in self.roi_dictionary}
        for roi_name, roi_label in self.roi_dictionary.items():
            # if region is background (index 0)
            if roi_label == 0:
                continue
            # Create a mask for each region (each atlas label)
            region_mask = atlas_label_map == roi_label
            # brain mask -> from the parcellated atlas
            region_volumes[roi_name] = self.calc_volume_ml(sitk.Mask(region_mask, brain_mask))
            # If the label map is the lesion map (it is), then here we get a mask of the portion of each region
            # occupied by the lesion
            masked_label_map = sitk.Mask(label_map, region_mask)
            for class_label, class_name in enumerate(self.class_names):
                # Class with index 0 is also background
                # Here, maybe it would be helpful to flag and not continue, as if a lesion is identified
                # to be in the background, it should be flagged
                if class_label == 0:
                    continue

                masked_label_map_array = sitk.GetArrayFromImage(masked_label_map)
                masked_label_map_class = np.where(masked_label_map_array == class_label, 1, 0)
                masked_label_map_class = sitk.GetImageFromArray(masked_label_map_class)

                # Calculate the volume of each class type in the overlap between the region and lesion
                localised_volumes[class_name][roi_name] = self.calc_volume_ml(masked_label_map_class)
                # localised_volumes[class_name][roi_name] = self.calc_volume_ml(masked_label_map == class_label)
                # From this function we take the volume per lesion class and per anatomical roi; and each roi's volume
                # What if we have several lesions in one scan? the label_map is per CT scan or per lesion?
        return localised_volumes, region_volumes

    # data_index_csv is the csv file the user submits with the lesion maps paths (I think)
    def __call__(self, aff_transform, data_index, image_id, image, write_registration_info):
        target_name = 'prediction'
        # get the atlas_label_map, brain mask and label_map in the native or atlas space
        label_map = image                       # Predicted segmentation
        atlas_label_map = self.atlas_label_map  # Parcellated atlas
        brain_mask = self.brain_mask
        pixeltype = label_map.GetPixelIDTypeAsString()
        print('pixel type: ', pixeltype)

        if self.native_space:
            # If we want to work on native space, we need to put everything that is in MNI space (parcellated
            # atlas and correspondent mask) in native space, using the inverse.
            # If we project the parcellated atlas to the study-specific CT template, we just need to use the
            # inverse of the rig+affine
            atlas_label_map = sitk.Resample(atlas_label_map, label_map, aff_transform.GetInverse(),
                                            sitk.sitkNearestNeighbor, 0)
            brain_mask = sitk.Resample(brain_mask, label_map, aff_transform.GetInverse(), sitk.sitkNearestNeighbor, 0)
        else:
            label_map = sitk.Resample(label_map, atlas_label_map, aff_transform, sitk.sitkNearestNeighbor, 0)

        # add summary statistics
        # Calculate full volume of each lesion class
        #for class_label, class_name in enumerate(self.class_names):
        #    if class_label == 0:
        #        continue
            # The max of label_map == class_label is always 1
        #    label_map_array = sitk.GetArrayFromImage(label_map)
        #    label_map_class = np.where(label_map_array == class_label, 1, 0)
        #    label_map_class = sitk.GetImageFromArray(label_map_class)
        #    data_index.loc[data_index['id'] == image_id, f'{target_name}_{class_name:s}_ml'] = self.calc_volume_ml(label_map_class)

            # data_index.loc[data_index['id']==image_id, f'{target_name}_{class_name:s}_ml'] = self.calc_volume_ml(label_map == class_label)

        # localise lesion volumes
        # localised_volumes is a dictionary: volume of lesion per lesion class and per anatomical ROI
        localised_volumes, region_volumes = self.localise_lesion_volumes(label_map, atlas_label_map, brain_mask)
        for class_name in localised_volumes.keys():
            for roi_name in localised_volumes[class_name].keys():
                volume = localised_volumes[class_name][roi_name]
                if volume is not None:
                    data_index.loc[data_index['id']==image_id, f'{target_name}_{class_name:s}_{roi_name:s}_ml'] = volume

        # add region volumes
        data_index.loc[data_index['id']==image_id, f'Brain_volume_ml'] = self.calc_volume_ml(brain_mask)
        for roi_name in region_volumes.keys():
            volume = region_volumes[roi_name]
            if volume is not None:
                data_index.loc[data_index['id']==image_id, f'{roi_name:s}_volume_ml'] = region_volumes[roi_name]
        if write_registration_info and self.native_space:
            atlas_native_space_path = os.path.join(self.localisation_dir, f'{str(image_id):s}_parc_atlas_native.nii.gz')
            sitk.WriteImage(atlas_label_map, atlas_native_space_path)
            brain_mask_native_space_path = os.path.join(self.localisation_dir, f'{str(image_id):s}_brain_mask_native.nii.gz')
            sitk.WriteImage(brain_mask, brain_mask_native_space_path)
            data_index.loc[data_index['id'] == image_id, 'atlas_in_native_space'] = atlas_native_space_path
            data_index.loc[data_index['id'] == image_id, 'brain_mask_native_space'] = brain_mask_native_space_path
        return data_index