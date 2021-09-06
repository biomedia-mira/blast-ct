import SimpleITK as sitk
import numpy as np


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


def create_reference_reoriented_image(image):
    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    reference = sitk.Image(new_size.tolist(), image.GetPixelIDValue())
    reference.SetSpacing(new_spacing.tolist())
    reference.SetOrigin(new_origin.tolist())
    try:
        reference.SetDirection(new_dir.flatten().tolist())
    except RuntimeError:
        print('Could not reorient image due to singular direction matrix, proceeding with image not reoriented!')
        return image
    return reference


def reorient_image(image, is_discrete):
    """Reorients an image to standard radiology view."""
    default_value = 0 if is_discrete else float(np.min(sitk.GetArrayViewFromImage(image)))
    reference = create_reference_reoriented_image(image)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetDefaultPixelValue(default_value)
    resample.SetTransform(sitk.Transform())
    if is_discrete:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)
