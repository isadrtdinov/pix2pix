import torch
from torch import nn
import torch.nn.functional as F


def get_norm(norm, channels=None):
    if norm == 'instance':
        return nn.InstanceNorm2d(channels)
    elif norm == 'none':
        return nn.Identity()
    else:
        raise ValueError('Unknown norm type')


class InputLayer(nn.Module):
    def __init__(self, image_channels, conv_channels, conv_kernel=3, norm='instance'):
        super(InputLayer, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.conv1 = nn.Conv2d(image_channels, conv_channels,
                               conv_kernel, padding=padding)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels,
                               conv_kernel, padding=padding)
        self.norm = get_norm(norm, conv_channels)

    def forward(self, inputs):
        outputs = F.relu(self.conv1(inputs))
        outputs = F.relu(self.norm(self.conv2(outputs)))
        return outputs


class DownsampleBlock(nn.Module):
    def __init__(self, num_channels, conv_kernel=3, norm='instance', dropout=0.5):
        super(DownsampleBlock, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(num_channels, num_channels * 2,
                               conv_kernel, padding=padding)
        self.conv2 = nn.Conv2d(num_channels * 2, num_channels * 2,
                               conv_kernel, padding=padding)
        self.norm = get_norm(norm, num_channels * 2)

    def forward(self, inputs):
        outputs = self.dropout(self.max_pool(inputs))
        outputs = F.relu(self.conv1(outputs))
        outputs = F.relu(self.norm(self.conv2(outputs)))
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, conv_kernel=3, norm='instance', dropout=0.5):
        super(UpsampleBlock, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.conv_transpose = nn.ConvTranspose2d(num_channels * 2, num_channels,
                                                 kernel_size=4, stride=2, padding=1)
        self.norm = get_norm(norm, num_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(num_channels * 2, num_channels,
                               conv_kernel, padding=padding)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               conv_kernel, padding=padding)

    def forward(self, inputs, features):
        outputs = self.conv_transpose(inputs)
        outputs = torch.tanh(self.norm(outputs))
        outputs = torch.cat([outputs, features], dim=1)
        outputs = self.dropout(outputs)
        outputs = F.relu(self.conv1(outputs))
        outputs = F.relu(self.conv2(outputs))
        return outputs


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, norm='instance', neg_slope=0.2):
        super(ConvNormRelu, self).__init__()
        self.neg_slope = neg_slope
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=4, stride=2, padding=1)
        self.norm = get_norm(norm, out_channels)

    def forward(self, inputs):
        outputs = self.norm(self.conv(inputs))
        outputs = F.leaky_relu(outputs, self.neg_slope)
        return outputs
