import numpy as np
from .augmention import RandomAugmentation


# All patch samplers can't alter state after __init__ because of dataset class when num_workers > 0

def get_patch_and_padding(shape, patch_size, center):
    padding = [[0, 0], [0, 0], [0, 0]]
    patch = []
    for dim, (size, ps, c) in enumerate(zip(shape, patch_size, center)):
        start = c - ps // 2
        end = start + ps
        if start < 0:
            padding[dim][0] = -start
            start = 0
        if end > size:
            padding[dim][1] = end - size
            end = size
        patch.append(slice(start, end, 1))

    return patch, padding


class PatchSampler(object):
    def __init__(self, image_patch_shape, target_patch_shape, augmentation=None):
        if any((i < l) for i, l in zip(image_patch_shape, target_patch_shape)):
            raise ValueError('Label map patch size must be smaller or equal to image patch size.')
        self.image_patch_size = image_patch_shape
        self.target_patch_size = target_patch_shape
        self.augmentation = [] if augmentation is None else augmentation
        assert isinstance(self.augmentation, list)
        for el in self.augmentation:
            assert isinstance(el, RandomAugmentation)

    def get_target_patch(self, target, center):
        if target is None:
            return target
        patch, padding = get_patch_and_padding(target.shape, self.target_patch_size, center)
        target_patch = target[tuple(patch)]
        if any([any(p) for p in padding]):
            target_patch = np.pad(target_patch, padding, mode='constant', constant_values=0)
        return target_patch

    def get_image_patch(self, image, center):
        image_shape = image.shape
        patch, padding = get_patch_and_padding(image_shape[1:], self.image_patch_size, center)
        padding = [[0, 0]] + padding
        patch = [slice(0, image_shape[0], 1)] + patch

        image_patch = image[tuple(patch)]

        if any([any(p) for p in padding]):
            image_patch = np.pad(image_patch, padding, mode='edge')
        return image_patch

    def get_patches(self, center, image, target, mask):
        image_patch = self.get_image_patch(image, center)
        target_patch = self.get_target_patch(target, center)
        mask_patch = self.get_target_patch(mask, center)
        return image_patch, target_patch, mask_patch

    def sample_patch_center(self, target, mask):
        raise NotImplementedError

    def __call__(self, image, target, mask=None):
        # ensure that the mask has at least one point
        mask = np.ones_like(target) if mask is None else mask
        if np.sum(mask) == 0:
            raise ValueError('Empty sampling mask')

        center = self.sample_patch_center(target, mask)
        image_patch, target_patch, mask_patch = self.get_patches(center, image, target, mask)
        for augmentation in self.augmentation:
            image_patch, target_patch, mask_patch = augmentation(image_patch, target_patch, mask_patch)
        # these copies are only here because of flipping and should be removed in the future
        return image_patch.copy(), target_patch.copy(), mask_patch.copy()


class StochasticPatchSampler(PatchSampler):
    def get_sampling_mask(self, target, mask):
        raise NotImplementedError

    def sample_patch_center(self, target, mask):
        sampling_mask = self.get_sampling_mask(target, mask)
        points = np.argwhere(sampling_mask)
        center = points[np.random.choice(len(points))]
        return center


class RandomPatchSampler(StochasticPatchSampler):
    def __init__(self, image_patch_shape, target_patch_shape, augmentation=None):
        super().__init__(image_patch_shape, target_patch_shape, augmentation)

    def get_sampling_mask(self, target, mask):
        return mask


class ConditionalPatchSampler(StochasticPatchSampler):
    def __init__(self, image_patch_shape, target_patch_shape,
                 class_probabilities, augmentation=None, n_tries=3):
        super().__init__(image_patch_shape, target_patch_shape, augmentation)
        self.class_probabilities = class_probabilities
        self.n_tries = n_tries

    def get_sampling_mask(self, target, mask):
        # try sampling points from a random label a maximum of n_tries
        for i in range(self.n_tries):
            label = np.random.choice(len(self.class_probabilities), p=self.class_probabilities)
            sampling_mask = np.logical_and(target == label, mask)
            if sampling_mask.any():
                return sampling_mask
        # if sampling mask is empty default to mask
        return mask


class ForegroundBackgroundPatchSampler(StochasticPatchSampler):
    def __init__(self, image_patch_shape, target_patch_shape,
                 foreground_probability=.5, augmentation=None, n_tries=3):
        super().__init__(image_patch_shape, target_patch_shape, augmentation)
        self.foreground_probability = foreground_probability
        self.n_tries = n_tries

    def get_sampling_mask(self, target, mask):
        # try sampling points from a random label a maximum of n_tries
        for i in range(self.n_tries):
            is_foreground = np.random.choice((False, True),
                                             p=(1 - self.foreground_probability, self.foreground_probability))
            if is_foreground:
                sampling_mask = np.logical_and(target != 0, mask)
            else:
                sampling_mask = np.logical_and(target == 0, mask)
            if sampling_mask.any():
                return sampling_mask
        # if sampling mask is empty default to mask
        return mask


class BoundingBoxCenteredPatchSampler(PatchSampler):
    def __init__(self, image_patch_shape, target_patch_shape, augmentation):
        super().__init__(image_patch_shape, target_patch_shape, augmentation)

    def sample_patch_center(self, target, mask):
        return np.array(tuple((np.max(arr) - np.min(arr)) // 2 for arr in np.where(mask)))
