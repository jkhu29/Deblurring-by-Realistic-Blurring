import copy
import random
import warnings
import itertools

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
import torchvision

from torch.cuda.amp import autocast as autocast

from tqdm import tqdm

from model import BlurGAN_G, DeblurGAN_G, GAN_D
import config
import dataset
import utils


opt = config.get_cyclegan_options()

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

# cyclegan for bgan, init
model_g_x2y = BlurGAN_G().to(device)
model_g_y2x = DeblurGAN_G().to(device)
model_d_x = GAN_D().to(device)
model_d_y = GAN_D().to(device)

model_g_x2y.apply(utils.weights_init)
model_g_y2x.apply(utils.weights_init)
model_d_x.apply(utils.weights_init)
model_d_y.apply(utils.weights_init)

# criterion init
criterion_generate = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# dataset init, need .h5
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

# optim init
optimizer_g = optim.Adam(
    itertools.chain(model_g_x2y.parameters(), model_g_y2x.parameters()), 
    lr=opt.lr, betas=(0.5, 0.999)
    )
optimizer_d_x = optim.Adam(model_d_x.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_d_y = optim.Adam(model_d_y.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# lr init
model_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=opt.niter)
model_scheduler_d_x = optim.lr_scheduler.CosineAnnealingLR(optimizer_d_x, T_max=opt.niter)
model_scheduler_d_y = optim.lr_scheduler.CosineAnnealingLR(optimizer_d_y, T_max=opt.niter)

# amp init
scaler_g = torch.cuda.amp.GradScaler()
# scaler_d_x = torch.cuda.amp.GradScaler()
# scaler_d_y = torch.cuda.amp.GradScaler()

# train cyclegan
cnt = 0
for epoch in range(opt.niter):
    model_g_x2y.train()
    model_g_y2x.train()
    model_d_x.train()
    model_d_y.train()

    epoch_losses_g = utils.AverageMeter()
    epoch_losses_d_x = utils.AverageMeter()
    epoch_losses_d_y = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for data in train_dataloader:
            cnt += 1
            blur, sharp = data

            blur_real = blur.to(device)
            sharp_real = sharp.to(device)

            blur_noise = utils.concat_noise(blur_real, (4, 128, 128), blur_real.size()[0])
            sharp_noise = utils.concat_noise(sharp_real, (4, 128, 128), sharp_real.size()[0])

            # --------------------
            # generator train(2 * model_g)
            # --------------------
            target_real = Variable(torch.rand(opt.batch_size) * 0.5 + 0.7).to(device)

            optimizer_g.zero_grad()

            with autocast():
                # loss identity(ATTN!: `a_same = model_a2b(a_real)`)
                blur_same = model_g_x2y(blur_noise)            # model_g_x2y: sharp --> blur
                loss_identity_blur = criterion_identity(blur_same, blur_real) * 5.

                sharp_fake = model_g_y2x(sharp_real)           # model_g_y2x: blur --> sharp
                loss_identity_sharp = criterion_identity(sharp_fake, sharp_real) * 5.

                # loss gan
                blur_fake = model_g_x2y(sharp_noise)
                blur_pred_fake = model_d_y(blur_fake)          # get blur features
                loss_gan_x2y = criterion_generate(blur_pred_fake, target_real)

                sharp_fake = model_g_y2x(blur_real)
                sharp_pred_fake = model_d_x(sharp_fake)        # get sharp features
                loss_gan_y2x = criterion_generate(sharp_pred_fake, target_real)

                sharp_fake_noise = utils.concat_noise(sharp_fake, (4, 128, 128), blur_real.size()[0])

                # loss cycle
                blur_recover = model_g_x2y(sharp_fake_noise)   # recover the blur: blur->sharp->blur
                loss_cycle_x2y = criterion_cycle(blur_recover, blur_real) * 10.

                sharp_recover = model_g_y2x(blur_fake)         # recover the sharp: sharp->blur->sharp
                loss_cycle_y2x = criterion_cycle(sharp_recover, sharp_real) * 10.

                # loss total
                loss_total = loss_identity_blur + loss_identity_sharp + \
                             loss_gan_x2y + loss_gan_y2x + \
                             loss_cycle_x2y + loss_cycle_y2x

            if cnt % 4 == 0:
                scaler_g.scale(loss_total).backward()
                scaler_g.step(optimizer_g)
                epoch_losses_g.update(scaler_g.scale(loss_total).item(), len(sharp))
                scaler_g.update()

            # --------------------
            # discriminator sharp train(model_d_x)
            # -------------------- 
            optimizer_d_x.zero_grad()

            with autocast():
                target_fake = Variable(torch.rand(opt.batch_size) * 0.3).to(device)

                # loss real
                pred_sharp_real = model_d_x(sharp_real)
                loss_sharp_real = criterion_generate(pred_sharp_real, target_real)

                # loss fake
                sharp_fake_ = copy.deepcopy(sharp_fake.data)
                pred_sharp_fake = model_d_x(sharp_fake_.detach())
                loss_sharp_fake = criterion_generate(pred_sharp_fake, target_fake)

                # loss rbl TODO(jkhu29): something strange
                loss_sharp_rbl = - torch.log(abs(loss_sharp_real - loss_sharp_fake)) - \
                                   torch.log(abs(1 - loss_sharp_fake - loss_sharp_real))

                # loss total
                loss_total_d_x = (loss_sharp_real + loss_sharp_fake) * 0.5 + loss_sharp_rbl * 0.01
                
            if cnt % 4 == 0:
                loss_total_d_x.backward()
                epoch_losses_d_x.update(loss_total_d_x.item(), len(sharp))
                optimizer_d_x.step()

                # scaler_d_x.scale(loss_total_d_x).backward()
                # scaler_d_x.step(optimizer_d_x)
                # epoch_losses_d_x.update(scaler_d_x.scale(loss_total_d_x).item(), len(sharp))
                # scaler_d_x.step()

            # --------------------
            # discriminator blur train(model_d_y)
            # -------------------- 
            optimizer_d_y.zero_grad()

            with autocast():
                # loss real
                pred_blur_real = model_d_x(blur_real)
                loss_blur_real = criterion_generate(pred_blur_real, target_real)

                # loss fake
                blur_fake_ = copy.deepcopy(blur_fake.data)
                pred_blur_fake = model_d_y(blur_fake_.detach())
                loss_blur_fake = criterion_generate(pred_blur_fake, target_fake)

                # loss rbl
                loss_blur_rbl = - torch.log(abs(loss_blur_real - loss_blur_fake)) - \
                                  torch.log(abs(1 - loss_blur_fake - loss_blur_real))

                # loss total
                loss_total_d_y = (loss_blur_real + loss_blur_fake) * 0.5 + loss_blur_rbl * 0.01
                 
            if cnt % 4 == 0:
                loss_total_d_y.backward()
                epoch_losses_d_y.update(loss_total_d_y.item(), len(sharp))
                optimizer_d_y.step()

                # scaler_d_y.scale(loss_total_d_y).backward()
                # scaler_d_y.step(optimizer_d_y)
                # epoch_losses_d_y.update(scaler_d_y.scale(loss_total_d_y).item(), len(sharp))
                # scaler_d_y.update()

            t.set_postfix(
                loss_g='{:.6f}'.format(epoch_losses_g.avg), 
                loss_d_sharp='{:.6f}'.format(epoch_losses_d_x.avg), 
                loss_d_blur='{:.6f}'.format(epoch_losses_d_y.avg)
                )
            t.update(len(sharp))

    model_scheduler_g.step()
    model_scheduler_d_x.step()
    model_scheduler_d_y.step()

torch.save(model_g_x2y.state_dict(), "%s/models/bgan_generator.pth" % opt.output_dir)
torch.save(model_g_y2x.state_dict(), "%s/models/dbgan_generator_cycle.pth" % opt.output_dir)
