import torch
from torch import nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, image_channels, conv_channels, conv_kernel=3):
        super(InputLayer, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.conv1 = nn.Conv2d(image_channels, conv_channels,
                               conv_kernel, padding=padding)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels,
                               conv_kernel, padding=padding)
        self.norm = nn.InstanceNorm2d(conv_channels)

    def forward(self, inputs):
        outputs = F.relu(self.conv1(inputs))
        outputs = F.relu(self.norm(self.conv2(outputs)))
        return outputs


class DownsampleBlock(nn.Module):
    def __init__(self, num_channels, conv_kernel=3, dropout=0.5):
        super(DownsampleBlock, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(num_channels, num_channels * 2,
                               conv_kernel, padding=padding)
        self.conv2 = nn.Conv2d(num_channels * 2, num_channels * 2,
                               conv_kernel, padding=padding)
        self.norm = nn.InstanceNorm2d(num_channels * 2)

    def forward(self, inputs):
        outputs = self.dropout(self.max_pool(inputs))
        outputs = F.relu(self.conv1(outputs))
        outputs = F.relu(self.norm(self.conv2(outputs)))
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, conv_kernel=3, dropout=0.5):
        super(UpsampleBlock, self).__init__()
        padding = (conv_kernel - 1) // 2

        self.conv_transpose = nn.ConvTranspose2d(num_channels * 2, num_channels,
                                                 kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(num_channels * 2)
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
