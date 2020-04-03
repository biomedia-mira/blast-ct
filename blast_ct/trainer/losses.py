import torch
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)
