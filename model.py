import torch
import torch.nn as nn


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """ResBlock used by BGAN and DBGAN"""
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()

        self.conv_relu = _make_layer(ConvReLU, num_layers=5, in_channels=channels, out_channels=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        x = self.conv(x) + res
        return x


class BGAN_G(nn.Module):
    """the G of BGAN"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=9):
        super(BGAN_G, self).__init__()

        # because of the noise concat
        self.in_channels = in_channels + 4
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)
        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # TODO(jkhu29): try different res
        x = self.conv_relu1(x)
        res = x
        x = self.res1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = torch.clamp(res + x, min=0, max=1)
        del res
        x = self.conv3(x)
        return x


class DBGAN_G(BGAN_G):
    """the G of DBGAN"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=16):
        super(DBGAN_G, self).__init__()
        self.in_channels = in_channels
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)
        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)


class GAN_D(nn.Module):
    """GAN_D: VGG19"""
    def __init__(self):
        super(GAN_D, self).__init__()
        self.net_d = [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self._conv_block(64, 64, with_stride=True)
        self._conv_block(64, 128)
        self._conv_block(128, 128, with_stride=True)
        self._conv_block(128, 256)
        self._conv_block(256, 256, with_stride=True)
        self._conv_block(256, 512)
        self._conv_block(512, 512, with_stride=True)
        self.net_d.extend([nn.AdaptiveAvgPool2d(1)])
        self.net_d.extend([nn.Conv2d(512, 1024, kernel_size=1)])
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self.net_d.extend([nn.Conv2d(1024, 1, kernel_size=1)])
        self.net_d = nn.Sequential(*self.net_d)

    def _conv_block(self, in_channels, out_channels, with_batch=True, with_stride=False):
        if with_stride:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)])
        else:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)])

        if with_batch:
            self.net_d.extend([nn.BatchNorm2d(out_channels)])
        self.net_d.extend([nn.LeakyReLU(0.2)])

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net_d(x).view(batch_size))


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
