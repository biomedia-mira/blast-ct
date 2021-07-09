import argparse
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import tqdm


class LesionVolumeLocalisationMNI(object):
    def __init__(self, class_names, native_space=False):
        # The parcellated atlas in study specific space (CT template space)
        atlas_label_map_path = '/vol/biomedic3/cpicarr1/template_integration/LesionLocalisationAnalysis/mean_template_7_u_lobe_atlas.nii.gz'
        roi_dictionary_csv = '/vol/biomedic3/cpicarr1/template_integration/LesionLocalisationAnalysis/lobe_labels.csv'
        brain_mask_path = '/vol/biomedic3/cpicarr1/template_integration/LesionLocalisationAnalysis/mean_template_7_u_mask.nii.gz'

        # Reading all the images
        self.atlas_label_map = sitk.ReadImage(atlas_label_map_path)
        self.brain_mask = sitk.ReadImage(brain_mask_path,sitk.sitkInt8)
        roi_dictionary = pd.read_csv(roi_dictionary_csv)
        self.roi_dictionary = {name: label for name, label in
                               zip(roi_dictionary['ROIName'], roi_dictionary['ROIIndex'])}
        self.class_names = class_names
        self.native_space = native_space

    # Calculate volume
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
        # class_names is dependent on the number of classes indicated by the user
        # Creating n empty nested dictionaries, in which the keys are brain regions. n = number of classes
        localised_volumes = {class_name: {atlas_label_name: None for atlas_label_name in self.roi_dictionary}
                             for class_name in self.class_names}

        # Creating an empty dictionary to store the region volumes
        region_volumes = {atlas_label_name: None for atlas_label_name in self.roi_dictionary}

        for roi_name, roi_label in self.roi_dictionary.items():
            # If region is background (index 0)
            if roi_label == 0:
                continue
            # Create a mask for each region (each atlas label)
            region_mask = atlas_label_map == roi_label

            # Why do we use sitk.Mask(region_mask, brain_mask) and not just region_mask?? Bc the region is expanded
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
                # Calculate the volume of each class type in the overlap between the region and lesion
                localised_volumes[class_name][roi_name] = self.calc_volume_ml(masked_label_map == class_label)
                # From this function we take the volume per lesion class and per anatomical roi; and each roi's volume
                # What if we have several lesions in one scan? the label_map is per CT scan or per lesion?
        return localised_volumes, region_volumes

    # data_index_csv is the csv file the user submits with the lesion maps paths (I think)
    def __call__(self, data_index_csv, target_names):
        data_index = pd.read_csv(data_index_csv, index_col='id')
        #  .iterrows() is used to iterate over a dataset's rows in the form of (index, series) pair
        for id_, item in tqdm(data_index.iterrows()):
            # This csv file includes one column with the transform from the blast-ct output to mni space?
            # Transform from the native scan to the study specific
            aff_transform = sitk.ReadTransform(item['aff_transform'])
            for target_name in target_names:
                if not isinstance(item[target_name], str):
                    continue
                # get the atlas_label_map, brain mask and label_map in the native or atlas space
                label_map = sitk.ReadImage(item[target_name])  # Predicted segmentation
                atlas_label_map = self.atlas_label_map         # Parcellated atlas
                brain_mask = self.brain_mask

                # native_space = label_map space, so the lesion map space/subject specific space? yes
                if self.native_space:
                    # If we want to work on native space, we need to put everything that is in MNI space (parcellated
                    # atlas and correspondent mask) in native space, using the inverse.
                    # If we project the parcellated atlas to the study-specific CT template, we just need to use the
                    # inverse of the rig+affine
                    atlas_label_map = sitk.Resample(atlas_label_map, label_map, aff_transform.GetInverse(), sitk.sitkNearestNeighbor, 0)
                    brain_mask = sitk.Resample(brain_mask, label_map, aff_transform.GetInverse(), sitk.sitkNearestNeighbor, 0)
                else:
                    label_map = sitk.Resample(label_map, atlas_label_map, aff_transform, sitk.sitkNearestNeighbor, 0)

                # add summary statistics
                # Calculate full volume of each lesion class
                for class_label, class_name in enumerate(self.class_names):
                    if class_label == 0:
                        continue
                    # The max of label_map == class_label is always 1
                    data_index.loc[id_, f'{target_name}_{class_name:s}_ml'] = self.calc_volume_ml(label_map == class_label)

                # localise lesion volumes
                # localised_volumes is a dictionary: volume of lesion per lesion class and per anatomical ROI
                localised_volumes, region_volumes = self.localise_lesion_volumes(label_map, atlas_label_map, brain_mask)
                for class_name in localised_volumes.keys():
                    for roi_name in localised_volumes[class_name].keys():
                        volume = localised_volumes[class_name][roi_name]
                        if volume is not None:
                            data_index.loc[id_, f'{target_name}_{class_name:s}_{roi_name:s}_ml'] = volume

                # add region volumes
                data_index.loc[id_, f'Brain_volume_ml'] = self.calc_volume_ml(brain_mask)
                for roi_name in region_volumes.keys():
                    volume = region_volumes[roi_name]
                    if volume is not None:
                        data_index.loc[id_, f'{roi_name:s}_volume_ml'] = region_volumes[roi_name]
            id_scan = id_.split("/")[1]
            atlas_native_space_path= '/vol/biomedic3/cpicarr1/template_integration/CT_4_class/parcellated_atlas_native_space/parc_atlas_native_{0}.nii.gz'.format(id_scan)
            brain_mask_path= '/vol/biomedic3/cpicarr1/template_integration/CT_4_class/parcellated_atlas_native_space/brain_mask_native_{0}.nii.gz'.format(id_scan)
            sitk.WriteImage(brain_mask, brain_mask_path)
            sitk.WriteImage(atlas_label_map, atlas_native_space_path)
            data_index.loc[id_, 'atlas_in_native_space']= atlas_native_space_path
            data_index.loc[id_, 'brain_mask_native_space']= brain_mask_path

        return data_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-index-csv',
                        required=True,
                        type=str,
                        help='Path to data index csv')
    parser.add_argument('--num-classes',
                        required=True,
                        type=int,
                        help='The number of classes, must be 4 or 7')
    parser.add_argument('--target-names',
                        nargs='+',
                        required=True,
                        help='List of target names to be localised.')
    parser.add_argument('--native-space',
                        default=True,
                        type=bool,
                        help='Whether to calculate the volumes in native space or atlas space.')

    parse_args, unknown = parser.parse_known_args()

    num_classes = parse_args.num_classes
    if num_classes not in [4, 7]:
        raise ValueError('Number of classes must be 4 or 7.')

    # data_index_csv: csv file with the path to the images to be processed (or to the lesion map already?)
    data_index_csv = parse_args.data_index_csv
    if not os.path.exists(data_index_csv):
        raise FileNotFoundError(f'File {data_index_csv:s} does not exist.')
    if num_classes == 4:
        class_names = ['background', 'iph', 'eah', 'oedema', 'ivh']
    else:
        class_names = ['background', 'contusion_core', 'ivh', 'sah', 'edh', 'sdh', 'oedema', 'petechial_haemorrhage']
    #The target is the segmentation? Then we can just indicate the column like we did with the image column, no?
    target_names = parse_args.target_names
    native_space = parse_args.native_space

    vl = LesionVolumeLocalisationMNI(class_names, native_space)
    data_index_localised = vl(data_index_csv, target_names)

    suffix = 'native_space.csv' if native_space else 'atlas_space.csv'
    data_index_csv.replace('.csv', '_localised_volumes_in_' + suffix)
    # Replace here the path to my csv
    # Arguments that should work: --data-index-csv
    # /vol/vipdata/data/brain/center-tbi/data/CT/CT_CENTER_EXTRA_ICOMETRIX_Aug_2020/predictions.csv
    # --num-classes 4
    # --target-names prediction or matching_target
    data_index_localised.to_csv(data_index_csv.replace('.csv', '_localised_volumes_in_' + suffix))
