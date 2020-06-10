import numpy as np
import SimpleITK as sitk


def get_physical_size(image):
    return [(sz - 1) * spc for sz, spc in zip(image.GetSize(), image.GetSpacing())]


def get_size_from_spacing(spacing, physical_size):
    return [int(round(phys_sz / spc + 1)) for spc, phys_sz in zip(spacing, physical_size)]


# Create reference image with origin at 0
def get_reference_image(image, new_spacing):
    physical_size = get_physical_size(image)
    size = get_size_from_spacing(new_spacing, physical_size)
    reference_image = sitk.Image(size, image.GetPixelID())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    reference_image.SetSpacing(new_spacing)
    return reference_image


def rescale(spacing, image: sitk.Image, is_discrete=False):
    reference_image = get_reference_image(image, spacing)
    mode = sitk.sitkNearestNeighbor if is_discrete else sitk.sitkLinear
    default_value = 0 if is_discrete else float(np.min(sitk.GetArrayViewFromImage(image)))
    return sitk.Resample(image, reference_image, sitk.Transform(), mode, default_value)


def sitk_is_vector(sitk_image):
    return sitk_image.GetPixelID() in range(12, 22)


def sitk_to_numpy(sitk_image, dtype=np.float32):
    if sitk_image is None:
        return None
    array = sitk.GetArrayFromImage(sitk_image).astype(dtype)
    if sitk_is_vector(sitk_image):
        if sitk_image.GetNumberOfComponentsPerPixel() == 1:
            array = np.expand_dims(array, axis=0)
        else:
            array = np.transpose(array, (-1,) + tuple(range(len(array.shape) - 1)))
    return array
