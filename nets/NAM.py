import torch.nn as nn
from torch.nn import functional as F
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.BatchNormalization = nn.BatchNorm2d(self.in_channels, affine=True)

    def forward(self, x):
        backup = x
        x = self.BatchNormalization(x)
        channel_weight = self.BatchNormalization.weight.abs() / torch.sum(self.BatchNormalization.weight.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(channel_weight, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * backup

        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels=3):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.InstanceNormalization = nn.InstanceNorm2d(self.in_channels, affine=True)

    def forward(self, x):
        backup = x
        x = self.InstanceNormalization(x)
        spatial_weight = self.InstanceNormalization.weight.abs() / torch.sum(self.InstanceNormalization.weight.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(spatial_weight, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * backup

        return x


class cbam_block(nn.Module):
    def __init__(self, channel):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel)
        self.spatialattention = SpatialAttention(channel)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

