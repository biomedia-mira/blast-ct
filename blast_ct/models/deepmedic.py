import torch.nn as nn
import torch
from .base import BiomedicalBlock, DownSample, UpSample, PreActBlock, crop_center

SCALE_FACTORS = ((5, 5, 5), (3, 3, 3), (1, 1, 1))
FEATURE_MAPS = (30, 30, 40, 40, 40, 40, 50, 50)
FULLY_CONNECTED = (250, 250)
DROPOUT = (.0, .5, .5)


class Path(BiomedicalBlock):
    def __init__(self, scale_factor, input_channels, feature_maps):
        super().__init__()
        self.layers = list()
        self.scale_factor = tuple(scale_factor)

        self.layers.append(DownSample(self.scale_factor))
        for i, feature_map in enumerate(feature_maps):
            in_channels = feature_maps[i - 1] if i > 0 else input_channels
            self.layers.append(PreActBlock(in_channels, feature_map))
        self.layers.append(UpSample(self.scale_factor))

        self.path = nn.Sequential(*self.layers)

    def forward(self, x, output_size):
        input_size = self.calculate_input_size(output_size)
        out = crop_center(x, input_size)
        out = self.path(out)
        out = crop_center(out, output_size)
        return out


class DeepMedic(BiomedicalBlock):
    def __init__(self,
                 input_channels,
                 num_classes,
                 scale_factors=SCALE_FACTORS,
                 feature_maps=FEATURE_MAPS,
                 fully_connected=FULLY_CONNECTED,
                 dropout=DROPOUT):

        super().__init__()
        # assert all scale factors are equal or less than the next one
        assert all([all(l[i] >= l[i + 1] for i in range(len(l) - 1)) for l in [i for i in list(zip(*scale_factors))]])
        self.scale_factors = tuple(scale_factors)
        self.feature_maps = tuple(feature_maps)
        self.output_size = None

        self.paths = []
        for i, scale_factor in enumerate(scale_factors):
            path = Path(scale_factor, input_channels, feature_maps)
            self.paths.append(path)
            self.add_module('path' + str(scale_factor) + str(i), path)

        assert len(fully_connected) + 1 == len(dropout)
        fms = []
        channels = (feature_maps[-1] * len(self.paths),) + tuple(fully_connected) + (num_classes, )
        for i in range(len(channels[:-1])):
            fms.append(PreActBlock(channels[i], channels[i + 1], kernel_size=(1, 1, 1), dropout_prob=dropout[i]))

        self.fully_connected = nn.Sequential(*fms)

        # to calculate sizes
        self.layers = self.paths[0].layers

    def forward(self, image, **kwargs):
        input_size = tuple(image.shape[2:])
        output_size = self.get_output_size(input_size)

        activations = []
        for i, path in enumerate(self.paths):
            out = path(image, output_size)
            activations.append(out)

        out = torch.cat(tuple(activations), dim=1)
        out = self.fully_connected(out)
        return out, {}
