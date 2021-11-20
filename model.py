from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


# ----------------
# Basic Block
# ----------------
class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""

    def __init__(self, in_channels, out_channels, withbn=False, stride: int = 2):
        super(ConvReLU, self).__init__()

        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranReLU(nn.Module):
    """ConvTranReLU: conv trans 64 * 3 * 3 + leakyrelu"""

    def __init__(self, in_channels, out_channels, withbn=False, stride: int = 2):
        super(ConvTranReLU, self).__init__()

        self.withbn = withbn
        self.convtran = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
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
        self.conv = nn.Conv2d(in_channels, out_channels * (num_scale ** 2), kernel_size=3, stride=1, padding=1, bias=False)
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

        self.conv_relu = _make_layer(ConvReLU, num_layers=num_conv, in_channels=channels, out_channels=channels, stride=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        x = self.conv(x) + res
        return x


# TODO
class RepResBlock(nn.Module):
    """ResBlock in RepVGG"""

    def __init__(self, num_conv=1, channels=64):
        super(RepResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        return x_conv1 + x_conv2 + x


# TODO
class UNetConvBlock(nn.Module):
    def __init__(self):
        super(UNetConvBlock, self).__init__()


class HINBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super(HINBlock, self).__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False, track_running_stats=True)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = torch.cat([self.norm(x1), x2], dim=1)
        return x


# ----------------
# Attention Block
# ----------------
class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 64, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class RCAB(nn.Module):
    def __init__(self, num_features: int = 64, reduction: int = 8):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class SupervisedAttention(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 3, kernel_size: int = 3):
        super(SupervisedAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, bias=False)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size, bias=False)
        self.conv3 = nn.Conv2d(out_channels, channels, kernel_size, bias=False)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class NonLocalAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_embed1 = self.relu1(self.conv1(x))
        x_embed2 = self.relu2(self.conv2(x))
        x_assembly = self.relu3(self.conv3(x))

        n, c, h, w = x_embed1.shape
        x_embed1 = x_embed1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed2 = x_embed2.view(n, c, h * w)
        score = torch.matmul(x_embed1, x_embed2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(n, -1, h * w).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly).permute(0, 2, 1).view(n, -1, h, w)

        return x_final


# ----------------
# Normalize Block
# ----------------
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def calc_mean_std(self, features, eps=1e-8):
        n, c, h, w = features.shape
        var = features.view(n, c, -1).var(dim=2) + eps
        std = var.sqrt().view(n, c, 1, 1)
        mean = features.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
        return mean, std

    def forward(self, features_content, features_style):
        assert features_content.shape[:2] == features_style.shape[:2], "make sure the features have the same shape, " \
                                                                       "get {} and {}".format(features_content.shape,
                                                                                              features_style.shape)
        size = features_content.shape
        mean_style, std_style = self.calc_mean_std(features_style)
        mean_content, std_content = self.calc_mean_std(features_content)
        features_normalized = (features_content - mean_content.expand(size)) / std_content.expand(size)
        return features_normalized * std_style.expand(size) + mean_style.expand(size)


# ----------------
# Models we use
# ----------------
class BlurGAN_G(nn.Module):
    """the G of BlurGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=3):
        super(BlurGAN_G, self).__init__()

        self.in_channels = in_channels * 2 + 1
        self.num_resblocks = num_resblocks

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_channels, out_channels, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels, withbn=True)

        # down sample
        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=1, channels=out_channels)

        # up sample
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels, withbn=True, stride=2)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_channels, in_channels, kernel_size=7, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_relu1(x)
        x = self.res1(x)
        x = self.convup_relu1(x)
        x = self.conv2(x)
        return x


class DeblurGAN_G(nn.Module):
    """the G of DeblurGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=5):
        super(DeblurGAN_G, self).__init__()
        self.in_channels = in_channels
        self.num_resblocks = num_resblocks

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.in_channels, out_channels, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_relu1 = _make_layer(
            ConvReLU, num_layers=1, 
            in_channels=out_channels, out_channels=out_channels,
            withbn=True, stride=1
        )

        # down sample
        self.nla1 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels, stride=1)
        self.conv_relu3 = _make_layer(
            ConvReLU, num_layers=1, 
            in_channels=out_channels, out_channels=out_channels * 2,
            withbn=True
        )

        # res block
        self.res1 = _make_layer(RCAB, num_layers=self.num_resblocks, num_features=out_channels * 2)

        # up sample
        self.convup_relu1 = _make_layer(
            ConvTranReLU, num_layers=1, 
            in_channels=out_channels * 2, out_channels=out_channels,
            withbn=True
        )
        self.nla2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels, stride=1)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_channels, in_channels, kernel_size=7, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_relu1(x)
        res1 = x
        x = self.nla1(x)
        x = self.conv_relu3(x)
        res2 = x
        x = self.res1(x) + res2
        x = self.convup_relu1(x) + res1
        x = self.nla2(x)
        del res1, res2
        x = self.conv2(x)
        return x


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
        return (res[0] + x, res[1])


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
        fea = self.res1((fea, cond))
        x = self.sft1((fea, cond))
        x = self.conv2(x)
        x = fea + x

        x = self.convup_relu1(x)
        x = self.conv_relu1(x)
        x = self.conv3(x)
        return x


# TODO
class MPRNet(nn.Module):
    def __init__(self):
        super(MPRNet, self).__init__()


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


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
        out = self.net_d(x).view(batch_size)
        return out


if __name__ == "__main__":
    from torchsummary import summary
    a = DeblurGAN_G(in_channels=3).cuda()
    b = BlurGAN_G(in_channels=3).cuda()
    summary(a, (3, 64, 64))
    summary(b, (7, 64, 64))
