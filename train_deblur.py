import random
import warnings

import torch

import config
from train_basic import BasicDBGAN


def train_deblur_init():
    opt = config.get_dbgan_options()

    # device init
    CUDA_ENABLE = torch.cuda.is_available()
    if CUDA_ENABLE and opt.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    elif CUDA_ENABLE and not opt.cuda:
        warnings.warn("WARNING: You have CUDA device, so you should probably run with --cuda")
    elif not CUDA_ENABLE and opt.cuda:
        assert CUDA_ENABLE, "ERROR: You don't have a CUDA device"

    device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

    # seed init
    manual_seed = opt.seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    return opt, device


if __name__ == '__main__':
    opt, device = train_deblur_init()
    dbgan = BasicDBGAN(opt.blur_model_path, device)
    dbgan._train_batch()