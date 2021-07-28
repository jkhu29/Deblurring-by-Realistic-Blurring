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

from model import BGAN_G, GAN_D, FeatureExtractor
import config
import dataset
import utils


# TODO(jkhu29): add DBGAN train
opt = config.get_options()

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

# models init
model_g = BGAN_G().to(device)
model_d = GAN_D().to(device)

# criterion init
criterion_g = nn.MSELoss()
criterion_d = nn.BCEWithLogitsLoss()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)

# dataset init, train file need .h5
train_dataset = dataset.TrainDataset(opt.train_file)
train_dataloader = dataloader.DataLoader(
                                    dataset=train_dataset,
                                    batch_size=opt.batch_size, 
                                    shuffle=True, 
                                    num_workers=opt.workers, 
                                    pin_memory=True, 
                                    drop_last=True
                                    )

valid_dataset = dataset.ValidDataset(opt.valid_file)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

model_g.apply(utils.weights_init)
model_d.apply(utils.weights_init)

# optim init
if opt.adam:
    model_g_optimizer = optim.Adam(model_g.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
else:
    model_g_optimizer = optim.RMSprop(model_g.parameters(), lr=opt.lr)

model_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_g_optimizer, T_max=opt.niter)

if opt.save_model_pdf:
    from torchviz import make_dot
    sampleData = torch.rand(1, 3, 128, 128).to(device)
    out = model_g(sampleData)
    out_d = model_d(out)
    d = make_dot(out_d)
    d.render('modelviz.pdf', view=False)

# pre-train bgan_g first
for epoch in range(opt.niter):

    model_g.train()

    epoch_losses = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for data in train_dataloader:
            blur, sharp = data

            blur = blur.to(device)
            sharp = sharp.to(device)

            blur_fake = model_g(sharp)

            model_g_optimizer.zero_grad()

            loss = criterion_g(blur_fake, blur)
            loss.backward()
            epoch_losses.update(loss.item(), len(sharp))

            model_g_optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(sharp))

    model_g_scheduler.step()

    # test
    model_g.eval()

    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    for data in valid_dataloader:
        blur, sharp = data

        blur = blur.to(device)
        sharp = sharp[0].to(device)

        with torch.no_grad():
            preds = model_g(sharp)
            epoch_pnsr.update(utils.calc_pnsr(preds, blur[0]), len(blur))
            epoch_ssim.update(utils.calc_ssim(preds, blur[0]), len(blur))

    print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))

torch.save(model_g.state_dict(), "%s/models/bgan_generator_pretrain.pth" % opt.output_dir)

# train bgan_d
if opt.adam:
    model_g_optimizer = optim.Adam(model_g.parameters(), lr=opt.lr*0.01, eps=1e-8, weight_decay=1)
    model_d_optimizer = optim.Adam(model_d.parameters(), lr=opt.lr*0.01, eps=1e-8, weight_decay=1)
else:
    model_g_optimizer = optim.RMSprop(model_g.parameters(), lr=opt.lr*0.01)
    model_d_optimizer = optim.RMSprop(model_d.parameters(), lr=opt.lr*0.01)

model_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_g_optimizer, T_max=opt.niter)
model_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_d_optimizer, T_max=opt.niter)

for epoch in range(opt.niter):

    model_g.train()
    model_d.train()

    epoch_losses_d = utils.AverageMeter()
    epoch_losses_total = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for data in train_dataloader:
            blur, sharp = data
            sharp = sharp.to(device)

            # get the blur real and fake
            blur_real = Variable(blur).to(device)
            blur_fake = model_g(sharp).to(device)

            # update model_d
            target_real = Variable(torch.rand(opt.batch_size) * 0.5 + 0.7).to(device)
            target_fake = Variable(torch.rand(opt.batch_size) * 0.3).to(device)

            model_d.zero_grad()
            loss_real_d = criterion_d(model_d(blur_real), target_real)
            loss_fake_d = criterion_d(model_d(Variable(blur_fake)), target_fake)
            loss_d = loss_real_d + loss_fake_d
            loss_d.backward()
            epoch_losses_d.update(loss_d.item(), len(inputs))

            # update model g
            model_g.zero_grad()

            # get the features of real blur images and fake blur images
            features_real = Variable(feature_extractor(blur_real.data)).to(device)
            features_fake = feature_extractor(blur_fake.data).to(device)

            # get loss_perceptual
            loss_perceptual = 0.
            grams_real = [utils.gram(fmap) for fmap in features_real]
            grams_fake = [utils.gram(fmap) for fmap in features_fake]
            for i in range(len(grams_fake)):
                loss_perceptual += criterion_g(grams_fake[i], grams_real[i][:len(blur)])

            # get loss_rbl
            loss_rbl = - torch.log(loss_real_d - loss_fake_d) \
                       - torch.log(1 - loss_fake_d - loss_real_d)
            total_loss = 0.01 * loss_rbl + loss_perceptual

            total_loss.backward()
            epoch_losses_total.update(total_loss.item(), len(sharp))

            model_g_optimizer.step()

            t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg))
            t.update(len(inputs))

    model_g_scheduler.step()
    model_d_scheduler.step()

    # test
    model_g.eval()
    model_d.eval()

    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    for data in valid_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels[0].to(device)

        with torch.no_grad():
            preds = model_g(inputs)[0]
            epoch_pnsr.update(utils.calc_pnsr(preds, labels), len(inputs))
            epoch_ssim.update(utils.calc_ssim(preds, labels), len(inputs))

    print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))

torch.save(model_g.state_dict(), '%s/models/bgan_generator_final.pth' % opt.output_dir)
torch.save(model_d.state_dict(), '%s/models/bgan_discriminator_final.pth' % opt.output_dir)
