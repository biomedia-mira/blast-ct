import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import argparse
import os
import re
import operator
import time

class RegistrationToCTTemplate(object):
    def __init__(self, localisation_dir):
        self.target_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'mean_template_7_u.nii.gz')
        self.target_template = sitk.ReadImage(self.target_template_path)
        self.localisation_dir = localisation_dir

    def register_image_to_atlas(self, image):
        start_rigid = time.time()
        # image was already read before
        image = sitk.Threshold(image, lower=-1024.0, upper=1e6, outsideValue=-1024.0)
        dimension = self.target_template.GetDimension()
        # Rigid registration of image to target_template

        # Set the initial moving transforms.
        initial_transform_rig = sitk.CenteredTransformInitializer(self.target_template, image,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

        registration_method_rig = sitk.ImageRegistrationMethod()

        # Similarity metric settings:
        registration_method_rig.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)

        # Sampling settings:
        registration_method_rig.SetMetricSamplingStrategy(registration_method_rig.REGULAR)
        registration_method_rig.SetMetricSamplingPercentage(0.2)

        # Interpolator settings:
        registration_method_rig.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings:

        registration_method_rig.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                                                    numberOfIterations=200,
                                                                    convergenceMinimumValue=1e-6,
                                                                    convergenceWindowSize=5)


        # As we are working with a transformation which parameter space includes both translation and rotation, and
        # mm and radians are not commensurate, and we would like the change of on mm and one radian to have a similar
        # effect, we scale these parameters during the optimization:
        registration_method_rig.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework:
        registration_method_rig.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method_rig.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method_rig.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()  # mm

        # Set the initial moving and optimized transforms.
        # SetMovingInitialTransform: maps points from the virtual img domain to the moving img domain,never modified.
        # SetInitialTransform: composed with the moving initial transform, maps points from the virtual image domain
        # to the moving image domain, modified during optimization.

        # optimized_transform_rig = sitk.Euler3DTransform()   # Identity
        # registration_method_rig.SetMovingInitialTransform(initial_transform_rig)
        # registration_method_rig.SetInitialTransform(optimized_transform_rig)
        registration_method_rig.SetInitialTransform(initial_transform_rig)
        time_elapsed = time.time() - start_rigid
        passed = time_elapsed % 3600 % 60
        print(f'Finished defining rigid parameter took {passed}s')
        # Using the composite transformation we just add them in (stack based, first in - last applied).
        # mni_transform = sitk.ReadTransform(self.transform_mni)
        # final_rig_transform = sitk.CompositeTransform([registration_method_rig.Execute(self.target_template, image), initial_transform_rig])
        start_execute_rigid = time.time()
        final_rig_transform = registration_method_rig.Execute(self.target_template, image)
        time_elapsed = time.time() - start_execute_rigid
        passed = time_elapsed % 3600 % 60
        print(f'Finished executing rigid parameter took {passed}s')
        iterations_rig = registration_method_rig.GetOptimizerIteration()
        final_metric_value_rig = registration_method_rig.GetMetricValue()

        ######### Affine registration ################
        start_parameters_affine = time.time()
        registration_method_rig.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration_method_rig.SetMetricSamplingStrategy(registration_method_rig.REGULAR)
        registration_method_rig.SetMetricSamplingPercentage(0.2)
        registration_method_rig.SetInterpolator(sitk.sitkLinear)
        registration_method_rig.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                                                    numberOfIterations=200,
                                                                    convergenceMinimumValue=1e-6,
                                                                    convergenceWindowSize=5)
        registration_method_rig.SetOptimizerScalesFromPhysicalShift()
        registration_method_rig.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method_rig.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method_rig.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()  # mm
        registration_method_rig.SetMovingInitialTransform(final_rig_transform)
        optimized_transform_aff = sitk.AffineTransform(dimension)
        registration_method_rig.SetInitialTransform(optimized_transform_aff, inPlace=True)

        time_elapsed = time.time() - start_parameters_affine
        passed = time_elapsed % 3600 % 60
        print(f'Finished affine parameter took {passed}s')

        start_execute_affine = time.time()
        registration_method_rig.Execute(self.target_template, image)
        time_elapsed = time.time() - start_execute_affine
        passed = time_elapsed % 3600 % 60
        print(f'Finished executing affine took {passed}s')
        #final_aff_transform = sitk.CompositeTransform([final_rig_transform, optimized_transform_aff])
        start_last_affine = time.time()
        final_aff_transform = sitk.Transform()
        final_aff_transform.AddTransform(final_rig_transform)
        final_aff_transform.AddTransform(optimized_transform_aff)
        iterations_aff = registration_method_rig.GetOptimizerIteration()
        final_metric_value_aff = registration_method_rig.GetMetricValue()

        filter = sitk.MinimumMaximumImageFilter()
        filter.Execute(image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.target_template)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(filter.GetMinimum())
        resampler.SetTransform(final_aff_transform)
        image_resampled_aff = resampler.Execute(image)
        time_elapsed = time.time() - start_last_affine
        passed = time_elapsed % 3600 % 60
        print(f'Finished executing last affine part took {passed}s')


        return final_aff_transform, iterations_rig, final_metric_value_rig, iterations_aff, final_metric_value_aff, image_resampled_aff

    def best_run(self, final_metric_aff_dict, no_runs):
        sm_values = {}
        for iteration in range(0, no_runs):
            sm_values[iteration] = final_metric_aff_dict.get(iteration,{}).get('final_metric_aff')
        best_iter = min(sm_values.items(), key=operator.itemgetter(1))[0]
        transform, iterations_rig, final_metric_rig, iterations_aff, final_metric_aff, image_resampled_aff = final_metric_aff_dict[best_iter].values()
        return transform, iterations_rig, final_metric_rig, iterations_aff, final_metric_aff, image_resampled_aff

    def __call__(self, data_index, write_reg_param, no_runs, image_id):
        image_column = 'image'
        final_metric_aff_dict={}
        #image_path = data_index.data_index.loc[data_index.data_index['id'] == image_id, image_column].item()
        #image = sitk.ReadImage(image_path)
        try:
            image = sitk.ReadImage(data_index.data_index.loc[data_index.data_index['id'] == image_id, image_column].item())
            print('Sucessfully read image ' + data_index.data_index.loc[data_index.data_index['id'] == image_id, image_column].item())
        except RuntimeError:
            print(f'Could not read image: {image_id:s}')
            return data_index


        for iteration in range(0, no_runs):
            try:
                transform, iterations_rig, final_metric_rig, iterations_aff, final_metric_aff, image_resampled_aff = \
                    self.register_image_to_atlas(image)
                print(f'{image_id:s} image registered.')
                final_metric_aff_dict[iteration] = {'transform': transform, 'iterations_rig': iterations_rig,
                                                    'final_metric_rig': final_metric_rig,
                                                    'iterations_aff': iterations_aff,
                                                    'final_metric_aff': final_metric_aff,
                                                    'image_resampled_aff': image_resampled_aff}
            except RuntimeError:
                print(f'Could not register image: {image_id:s}.')
                continue

        transform, iterations_rig, final_metric_rig, iterations_aff, final_metric_aff, image_resampled_aff = self.best_run(
            final_metric_aff_dict, no_runs)
        if write_reg_param:
            resampled_image_path = os.path.join(self.localisation_dir, f'{str(image_id):s}_resampled.nii.gz')
            sitk.WriteImage(image_resampled_aff, resampled_image_path)
            print('wrote resampled image')
            print(self.localisation_dir)
            transform_path = os.path.join(self.localisation_dir, f'{str(image_id):s}_transform.tfm')
            sitk.WriteTransform(transform, transform_path)

            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'iterations_rig'] = iterations_rig
            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'final_metric_rig'] = final_metric_rig
            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'iterations_aff'] = iterations_aff
            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'final_metric_aff'] = final_metric_aff
            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'image_resampled'] = resampled_image_path
            data_index.data_index.loc[data_index.data_index['id'] == image_id, 'aff_transform'] = transform_path
            data_index = data_index.data_index

        return transform, data_index