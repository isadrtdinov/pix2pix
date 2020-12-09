from torch import nn
from .layers import ConvNormRelu


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 conv_channels=64, num_layers=3, norm='instance'):
        super(Discriminator, self).__init__()
        channels = [conv_channels * 2 ** power for power in range(num_layers)]

        self.layers = [ConvNormRelu(in_channels + out_channels, conv_channels, norm)]
        for i in range(1, num_layers):
            self.layers += [ConvNormRelu(channels[i - 1], channels[i], norm)]

        self.layers += [nn.Conv2d(channels[-1], 1, kernel_size=4, stride=2, padding=1)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


def build_discriminator(params):
    return Discriminator(params.in_channels, params.out_channels, params.discriminator_channels,
                         params.discriminator_layers, params.discriminator_norm)
