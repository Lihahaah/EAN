import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ean_module(nn.Module):
    def __init__(self, channels, groups=16, mode='l2'):
        """
        EAN: An Efficient Attention Module Guided by Normalization for Deep Neural Networks

        Args:
            channels (int): Number of input channels.
            groups (int): Number of groups for GroupNorm.
            mode (str): Normalization mode ('l2' or 'l1').
        """
        super(ean_module, self).__init__()
        assert mode in ['l1', 'l2'], "mode must be 'l1' or 'l2'"

        self.groups = groups
        self.groupnorm = nn.GroupNorm(num_groups=channels // groups, num_channels=channels // groups, affine=True)
        self.activation = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1, channels // groups, 1, 1))
        self.delta = nn.Parameter(torch.zeros(1, channels // groups, 1, 1))
        self.epsilon = 1e-5
        self.mode = mode

    @staticmethod
    def get_module_name():
        return "ean"

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xs = self.groupnorm(x)

        weight = self.groupnorm.weight.view(1, -1, 1, 1)

        if self.mode == 'l2':
            weights = (weight.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.alpha * (weight / weights)
        elif self.mode == 'l1':
            weights = torch.abs(weight).mean(dim=1, keepdim=True) + self.epsilon
            norm = self.alpha * (weight / weights)

        out = x * self.activation(xs * norm + self.delta)
        out = out.view(b, -1, h, w)

        return out

