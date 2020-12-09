from torch import nn
from .layers import InputLayer, DownsampleBlock, UpsampleBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, conv_channels=64,
                 num_layers=4, conv_kernel=3, norm='instance', dropout=0.5):
        super(Generator, self).__init__()
        block_channels = [conv_channels * 2 ** power for power in range(num_layers)]
        self.input_layer = InputLayer(in_channels, conv_channels, conv_kernel, norm)

        self.downsample_layers = [DownsampleBlock(num_channels, conv_kernel, norm, dropout)
                                  for num_channels in block_channels]
        self.downsample_layers = nn.ModuleList(self.downsample_layers)

        self.upsample_layers = [UpsampleBlock(num_channels, conv_kernel, norm, dropout)
                                for num_channels in block_channels[::-1]]
        self.upsample_layers = nn.ModuleList(self.upsample_layers)

        self.head = nn.Conv2d(conv_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        features_list = [self.input_layer(inputs)]
        outputs = features_list[-1]

        for layer in self.downsample_layers:
            features_list += [layer(outputs)]
            outputs = features_list[-1]

        for layer, features in zip(self.upsample_layers, features_list[-2::-1]):
            outputs = layer(outputs, features)

        outputs = self.head(outputs)
        return outputs


def build_generator(params):
    return Generator(params.in_channels, params.out_channels, params.generator_channels,
                     params.generator_layers, params.generator_kernel,
                     params.generator_norm, params.generator_dropout)
