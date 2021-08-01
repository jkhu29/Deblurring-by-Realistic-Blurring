import copy
import random
import warnings

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
import torchvision

from tqdm import tqdm

from model import BGAN_G, DBGAN_G, GAN_D, FeatureExtractor
import config
import dataset
import utils


opt = config.get_dbgan_options()

# deveice init
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

# ----------------------
# bgan
# ----------------------
# models init
model_blur = BGAN_G().to(device)
pth_path = opt.blur_model_path
model_blur.load_state_dict(torch.load(pth_path))
model_blur.to(device)
model_blur.eval()

# ----------------------
# dbgan
# ----------------------
# models init
deblurmodel_g = DBGAN_G().to(device)
deblurmodel_d = GAN_D().to(device)

# dataset init, train file need .h5
train_dataset = dataset.TrainDatasetDeblur(opt.train_file)
train_dataloader = dataloader.DataLoader(
                                    dataset=train_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.workers,
                                    pin_memory=True,
                                    drop_last=True
                                    )

valid_dataset = dataset.ValidDatasetDeblur(opt.valid_file)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

deblurmodel_g.apply(utils.weights_init)
deblurmodel_d.apply(utils.weights_init)

# criterion init
criterion_g = nn.L1Loss()
criterion_d = nn.BCEWithLogitsLoss()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)

# optim init
if opt.adam:
    deblurmodel_g_optimizer = optim.Adam(deblurmodel_g.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
else:
    deblurmodel_g_optimizer = optim.RMSprop(deblurmodel_g.parameters(), lr=opt.lr)

deblurmodel_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(deblurmodel_g_optimizer, T_max=opt.niter)

# pre-train dbgan_g first
for epoch in range(opt.niter):

    deblurmodel_g.train()

    epoch_losses = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for sharp in train_dataloader:

            sharp = sharp.to(device)
            sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
            # the blur image is made by bgan_g
            blur = model_blur(sharp_noise)

            sharp_fake = deblurmodel_g(blur)

            deblurmodel_g_optimizer.zero_grad()

            loss = criterion_g(sharp_fake, sharp)
            loss.backward()
            epoch_losses.update(loss.item(), len(blur))

            deblurmodel_g_optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(sharp))

    deblurmodel_g_scheduler.step()

torch.save(deblurmodel_g.state_dict(), "%s/models/dbgan_generator_pretrain.pth" % opt.output_dir)

del deblurmodel_g_optimizer
del deblurmodel_g_scheduler

# train dbgan_d
if opt.adam:
    deblurmodel_g_optimizer = optim.Adam(deblurmodel_g.parameters(), lr=opt.lr*0.01, eps=1e-8, weight_decay=1)
    deblurmodel_d_optimizer = optim.Adam(deblurmodel_d.parameters(), lr=opt.lr*0.01, eps=1e-8, weight_decay=1)
else:
    deblurmodel_g_optimizer = optim.RMSprop(deblurmodel_g.parameters(), lr=opt.lr*0.01)
    deblurmodel_d_optimizer = optim.RMSprop(deblurmodel_d.parameters(), lr=opt.lr*0.01)

deblurmodel_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(deblurmodel_g_optimizer, T_max=opt.niter)
deblurmodel_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(deblurmodel_d_optimizer, T_max=opt.niter)

for epoch in range(opt.niter):

    deblurmodel_g.train()
    deblurmodel_d.train()

    epoch_losses_d = utils.AverageMeter()
    epoch_losses_total = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for sharp in train_dataloader:

            sharp = sharp.to(device)
            sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
            blur = model_blur(sharp_noise)

            # get the sharp real and fake
            sharp_real = Variable(sharp).to(device)
            sharp_fake = deblurmodel_g(blur).to(device)

            # update model d
            target_real = Variable(torch.rand(opt.batch_size) * 0.5 + 0.7).to(device)
            target_fake = Variable(torch.rand(opt.batch_size) * 0.3).to(device)

            deblurmodel_d.zero_grad()
            loss_real_d = criterion_d(deblurmodel_d(sharp_real), target_real)
            loss_fake_d = criterion_d(deblurmodel_d(Variable(sharp_fake)), target_fake)
            
            loss_d = (loss_real_d + loss_fake_d) * 0.5
            loss_d.backward()
            epoch_losses_d.update(loss_d.item(), len(sharp))
            deblurmodel_d_optimizer.step()

            # update model g
            deblurmodel_g.zero_grad()

            # get the features of real blur images and fake blur images
            features_real = Variable(feature_extractor(sharp_real.data)).to(device)
            features_fake = feature_extractor(sharp_fake.data).to(device)

            # get loss_perceptual
            loss_perceptual = 0.
            grams_real = utils.calc_gram(features_real)
            grams_fake = utils.calc_gram(features_fake)
            for gram_fake, gram_real in zip(grams_fake, grams_real):
                loss_perceptual += criterion_g(gram_fake, gram_real)

            # get loss content
            loss_content = criterion_g(sharp_real, sharp_fake)
            
            # get loss_rbl
            loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                         torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

            total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

            total_loss.backward()
            epoch_losses_total.update(total_loss.item(), len(blur))

            deblurmodel_g_optimizer.step()

            t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg), loss_d='{:.6f}'.format(epoch_losses_d.avg))
            t.update(len(blur))

    deblurmodel_g_scheduler.step()
    deblurmodel_d_scheduler.step()

    # test
    deblurmodel_g.eval()
    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()
    #
    for data in valid_dataloader:
        sharp = data
    #
        sharp = sharp.to(device)
        sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
        blur = model_blur(sharp_noise)
    #
        with torch.no_grad():
            preds = deblurmodel_g(blur)[0]
            epoch_pnsr.update(utils.calc_pnsr(preds, sharp[0]), len(sharp))
            epoch_ssim.update(utils.calc_ssim(preds, sharp[0]), len(sharp))
    #
    print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))
    torch.cuda.empty_cache()

torch.save(deblurmodel_g.state_dict(), '%s/models/dbgan_generator.pth' % opt.output_dir)
