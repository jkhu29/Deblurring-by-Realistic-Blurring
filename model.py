import torch
import torch.nn as nn


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=False):
        super(ConvReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranReLU(nn.Module):
    """ConvTranReLU: conv trans 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=False):
        super(ConvTranReLU, self).__init__()
        self.withbn = withbn
        self.convtran = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.convtran(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvPixelShuffle(nn.Module):
    """ConvPixelShuffle"""
    def __init__(self, in_channels, out_channels, num_scale=2):
        super(ConvPixelShuffle, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(num_scale)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """ResBlock used by BGAN and DBGAN"""
    def __init__(self, num_conv=5, channels=64):
        super(ResBlock, self).__init__()

        self.conv_relu = _make_layer(ConvReLU, num_layers=num_conv, in_channels=channels, out_channels=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        x = self.dropout(x)
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
        x = self.conv_relu1(x)
        res = x
        x = self.res1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = res + x
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


class BlurGAN_G(nn.Module):
    """the G of BlurGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=9):
        super(BlurGAN_G, self).__init__()
        self.in_channels = in_channels + 4
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)

        # down sample
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels * 2, withbn=True)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels * 4, withbn=True)

        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=5, channels=out_channels * 4)

        # up sample
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 4, out_channels=out_channels * 2, withbn=True)
        self.convup_relu2 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels, withbn=True)

        self.conv4 = nn.Conv2d(out_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_relu1(x)
        res = x
        x = self.conv_relu2(x)
        x = self.conv_relu3(x)
        x = self.res1(x)
        x = self.convup_relu1(x)
        x = self.convup_relu2(x)
        x = res + x
        del res
        x = self.conv4(x)
        x = self.tanh(x)
        return x


class DeblurGAN_G(BlurGAN_G):
    """the G of DeblurGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=16):
        super(DeblurGAN_G, self).__init__()
        self.in_channels = in_channels
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)

        # down sample
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels * 2, withbn=True)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels * 4)

        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=5, channels=out_channels * 4)

        # up sample
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 4, out_channels=out_channels * 2, withbn=True)
        self.convup_relu2 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels)

        self.conv4 = nn.Conv2d(out_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)


class SFTLayer(nn.Module):
    """SFTLayer"""
    def __init__(self, channels):
        super(SFTLayer, self).__init__()
        self.scale_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.scale_conv2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.shift_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.shift_conv2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        scale = self.scale_conv2(self.relu1(self.scale_conv1(x[1])))
        shift = self.shift_conv2(self.relu2(self.shift_conv1(x[1])))
        return x[0] * (scale + 1) + shift
        

class ResSFT(nn.Module):
    """Resblock for SFTGAN"""
    def __init__(self, channels=64):
        super(ResSFT, self).__init__()
        self.sft1 = SFTLayer(channels)
        self.sft2 = SFTLayer(channels)

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=channels, out_channels=channels)
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=channels, out_channels=channels)

    def forward(self, x):
        res = x
        x = self.sft1(x)
        x = self.conv_relu1(x)
        x = self.sft2((x, res[1]))
        x = self.conv_relu2(x)
        return(res[0] + x, res[1])


class SFTGAN_G(nn.Module):
    """SFTGAN_G"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=9):
        super(SFTGAN_G, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # sft branch
        self.res1 = _make_layer(ResSFT, num_layers=num_resblocks)
        self.sft1 = SFTLayer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # hr branch
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=2, in_channels=out_channels, out_channels=out_channels)
        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # for seg
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=in_channels, out_channels=out_channels)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=3, in_channels=out_channels, out_channels=out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # x[0]: img x[1]: seg
        cond = self.conv_relu2(x[1])
        cond = self.conv_relu3(cond)
        cond = self.conv4(cond)

        fea = self.conv1(x[0])
        x = self.res1((fea, cond))
        x = self.sft1((fea, cond))
        x = self.conv2(x)
        x = fea + x

        x = self.convup_relu1(x)
        x = self.conv_relu1(x)
        x = self.conv3(x)
        return x


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
