import torch
import torch.nn as nn
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def weights_init(model):
    """init from article"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0., 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0)


def clac_gram(x):
    (n, c, h, w) = x.size()
    f = x.view(n, c, w * h)
    f_trans = f.transpose(1, 2)
    gram = f.bmm(f_trans) / (c * h * w)
    return gram


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(GradientPenaltyLoss, self).__init__()
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, inputs):
        if self.grad_outputs.size() != inputs.size():
            self.grad_outputs.resize_(inputs.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


def calc_ssim(img1, img2):
    """calculate SSIM"""
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return ssim(img1, img2, multichannel=True)
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image channel.')
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_pnsr(img1, img2):
    """calculate PNSR"""
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return cv2.PSNR(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return cv2.PSNR(img1 * 255., img2 * 255.)
        elif img1.shape[2] == 1:
            return cv2.PSNR(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image channel.')
    else:
        raise ValueError('Wrong input image dimensions.')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def upsampling(img, x, y):
    func = nn.Upsample(size=[x, y], mode='bilinear', align_corners=True)
    return func(img)


def generate_noise(size, channels=1, type='gaussian', scale=2):
    if type == 'gaussian':
        noise = torch.randn(channels, size[0], round(size[1]/scale), round(size[2]/scale))
        noise = upsampling(noise, size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(channels, size[0], size[1], size[2]) + 5
        noise2 = torch.randn(channels, size[0], size[1], size[2])
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(channels, size[0], size[1], size[2])
    return noise


def concat_noise(img, **kwargs):
    noise = generate_noise(**kwargs)
    noise = noise.to(img.device)
    mixed_img = torch.cat((img, noise), 1)
    return mixed_img
