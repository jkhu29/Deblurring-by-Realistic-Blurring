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

from tqdm import tqdm

from model import BlurGAN_G, DBGAN_G, GAN_D
from criterion import GradientPenaltyLoss
import config
import dataset
import utils


class BasicCycleGAN(object):
    """"""
    def __init__(self, train_file, valid_file, device):
        super(BasicCycleGAN, self).__init__()
        self.device = device
        self.niter = 10
        self.batch_size = 32
        self.workers = 4
        self.batch_scale = 4
        self.lr = 1e-4

        # cyclegan for bgan, init
        self.model_g_x2y = BlurGAN_G().to(device)
        self.model_g_y2x = DBGAN_G().to(device)
        self.model_d_x = GAN_D().to(device)
        self.model_d_y = GAN_D().to(device)

        self.model_g_x2y.apply(utils.weights_init)
        self.model_g_y2x.apply(utils.weights_init)
        self.model_d_x.apply(utils.weights_init)
        self.model_d_y.apply(utils.weights_init)

        # criterion init
        self.criterion_generate = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # dataset init
        train_dataset = dataset.BasicDataset(train_file)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.workers, 
            pin_memory=True, 
            drop_last=True
        )
        self.data_length = len(train_dataset)

        valid_dataset = dataset.BasicDataset(valid_file)
        self.valid_dataloader = dataloader.DataLoader(
            dataset=valid_dataset, 
            batch_size=1
        )

        # optim init
        self.optimizer_g = optim.Adam(
            itertools.chain(self.model_g_x2y.parameters(), self.model_g_y2x.parameters()), 
            lr=self.lr, betas=(0.75, 0.999)
        )
        self.optimizer_d_x = optim.Adam(
            self.model_d_x.parameters(), 
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_d_y = optim.Adam(
            self.model_d_y.parameters(), 
            lr=self.lr, betas=(0.5, 0.999)
        )

        # lr init
        self.model_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.niter
        )
        self.model_scheduler_d_x = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_x, T_max=self.niter
        )
        self.model_scheduler_d_y = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_y, T_max=self.niter
        )

    def _train_batch(self):
        cnt = 0
        for epoch in range(self.niter):
            self.model_g_x2y.train()
            self.model_g_y2x.train()

            epoch_losses_g = utils.AverageMeter()
            epoch_losses_d_x = utils.AverageMeter()
            epoch_losses_d_y = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for data in self.train_dataloader:
                    self.model_d_x.eval()
                    self.model_d_y.eval()

                    cnt += 1
                    blur, sharp = data

                    blur_real = blur.to(self.device)
                    sharp_real = sharp.to(self.device)

                    blur_noise = utils.concat_noise(blur_real, (4, 128, 128), blur_real.size()[0])
                    sharp_noise = utils.concat_noise(sharp_real, (4, 128, 128), sharp_real.size()[0])

                    # --------------------
                    # generator train(2 * model_g)
                    # --------------------
                    target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)

                    loss_total = self._calc_loss_g(blur_noise, blur_real, sharp_real, sharp_noise)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_g.zero_grad()
                        loss_total.backward()
                        epoch_losses_g.update(loss_total.item(), len(sharp))
                        self.optimizer_g.step()

                    # --------------------
                    # discriminator sharp train(model_d_x)
                    # -------------------- 
                    self.model_d_x.train()

                    target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(device)

                    loss_total_d_x = self._calc_loss_d(sharp_real, target_real, sharp_fake, target_fake)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_x.zero_grad()
                        loss_total_d_x.backward()
                        epoch_losses_d_x.update(loss_total_d_x.item(), len(sharp))
                        self.optimizer_d_x.step()

                    # --------------------
                    # discriminator blur train(model_d_y)
                    # -------------------- 
                    self.model_d_y.train()

                    loss_total_d_y = self._calc_loss_d(blur_real, target_real, blur_fake, target_fake)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_y.zero_grad()
                        loss_total_d_y.backward()
                        epoch_losses_d_y.update(loss_total_d_y.item(), len(sharp))
                        self.optimizer_d_y.step()

                    t.set_postfix(
                        loss_g='{:.6f}'.format(epoch_losses_g.avg), 
                        loss_d_sharp='{:.6f}'.format(epoch_losses_d_x.avg), 
                        loss_d_blur='{:.6f}'.format(epoch_losses_d_y.avg)
                        )
                    t.update(len(sharp))

                torch.save(model_g_x2y.state_dict(), "{}/models/bgan_generator_snapshot_{}.pth".format(self.output_dir, epoch))

            self.model_scheduler_g.step()
            self.model_scheduler_d_x.step()
            self.model_scheduler_d_y.step()

    def _calc_loss_g(self, blur_noise, blur_real, sharp_real, sharp_noise):
        # loss identity(ATTN!: `a_same = model_a2b(a_real)`)
        blur_same = self.model_g_x2y(blur_noise)            # model_g_x2y: sharp --> blur
        loss_identity_blur = self.criterion_identity(blur_same, blur_real)

        sharp_fake = self.model_g_y2x(sharp_real)           # model_g_y2x: blur --> sharp
        loss_identity_sharp = self.criterion_identity(sharp_fake, sharp_real)

        # loss gan
        blur_fake = self.model_g_x2y(sharp_noise)
        blur_pred_fake = self.model_d_y(blur_fake)          # get blur features
        loss_gan_x2y = self.criterion_generate(blur_pred_fake, target_real) * 5.

        sharp_fake = self.model_g_y2x(blur_real)
        sharp_pred_fake = self.model_d_x(sharp_fake)        # get sharp features
        loss_gan_y2x = self.criterion_generate(sharp_pred_fake, target_real) * 5.

        sharp_fake_noise = utils.concat_noise(sharp_fake, (4, 128, 128), blur_real.size()[0])

        # loss cycle
        blur_recover = self.model_g_x2y(sharp_fake_noise)   # recover the blur: blur->sharp->blur
        loss_cycle_x2y = self.criterion_cycle(blur_recover, blur_real) * 10.

        sharp_recover = self.model_g_y2x(blur_fake)         # recover the sharp: sharp->blur->sharp
        loss_cycle_y2x = self.criterion_cycle(sharp_recover, sharp_real) * 10.

        # loss total
        loss_total = loss_identity_blur + loss_identity_sharp + \
                     loss_gan_x2y + loss_gan_y2x + \
                     loss_cycle_x2y + loss_cycle_y2x

        return loss_total

    def _calc_loss_d(self, sharp_real, target_real, sharp_fake, target_fake):
        # loss real
        pred_sharp_real = self.model_d_x(sharp_real)
        loss_sharp_real = self.criterion_generate(pred_sharp_real, target_real)

        # loss fake
        sharp_fake_ = copy.deepcopy(sharp_fake.data)
        pred_sharp_fake = self.model_d_x(sharp_fake_.detach())
        loss_sharp_fake = self.criterion_generate(pred_sharp_fake, target_fake)

        # loss rbl
        loss_sharp_rbl = - torch.log(abs(loss_sharp_real - loss_sharp_fake)) - \
                           torch.log(abs(1 - loss_sharp_fake - loss_sharp_real))

        # loss total
        loss_total = (loss_sharp_real + loss_sharp_fake) * 0.5 + loss_sharp_rbl * 0.01

        return loss_total


class BasicGAN(object):
    """BasicGAN"""
    def __init__(self, blur_model_path, device):
        super(BasicGAN, self).__init__()
        self.device = device
        self.niter = 10
        self.batch_size = 32
        self.workers = 4
        self.batch_scale = 4

        # ----------------------
        # bgan
        # ----------------------
        # models init
        model_blur = BlurGAN_G().to(device)
        model_blur.load_state_dict(torch.load(blur_model_path))
        model_blur.to(device)
        model_blur.eval()

        # ----------------------
        # dbgan
        # ----------------------
        # models init
        deblurmodel_g = DeblurGAN_G().to(device)
        deblurmodel_d = GAN_D().to(device)
        deblurmodel_g.apply(utils.weights_init)
        deblurmodel_d.apply(utils.weights_init)

        # dataset init
        train_dataset = dataset.BasicDatasetDeblur(train_file)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )

        valid_dataset = dataset.BasicDatasetDeblur(valid_file)
        self.valid_dataloader = dataloader.DataLoader(
            dataset=valid_dataset, 
            batch_size=1
        )

        # criterion init
        self.criterion_g = nn.L1Loss()
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.feature_extractor = FeatureExtractor(
            torchvision.models.vgg19(pretrained=True)
        ).to(device)

        # optim init
        self.deblurmodel_g_optimizer = optim.Adam(
            deblurmodel_g.parameters(), 
            lr=self.lr, eps=1e-8, weight_decay=1
        )
        self.deblurmodel_d_optimizer = optim.Adam(
            deblurmodel_d.parameters(), 
            lr=self.lr, eps=1e-8, weight_decay=1
        )

        # lr init
        self.deblurmodel_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.deblurmodel_g_optimizer, T_max=self.niter
        )
        self.deblurmodel_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.deblurmodel_d_optimizer, T_max=self.niter
        )

    def _train_batch(self):
        cnt = 0
        for epoch in range(self.niter):
            deblurmodel_g.train()
            deblurmodel_d.train()

            epoch_losses_d = utils.AverageMeter()
            epoch_losses_total = utils.AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for sharp in self.train_dataloader:
                    cnt += 1

                    sharp = sharp.to(self.device)
                    sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_real = Variable(sharp).to(self.device)
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)
                    target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)

                    loss_real_d = self.criterion_d(self.deblurmodel_d(sharp_real), target_real)
                    loss_fake_d = self.criterion_d(self.deblurmodel_d(Variable(sharp_fake)), target_fake)
                    loss_d = (loss_real_d + loss_fake_d) * 0.5

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_d.zero_grad()
                        loss_d.backward()
                        epoch_losses_d.update(loss_d.item(), len(sharp))
                        self.deblurmodel_d_optimizer.step()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real blur images and fake blur images
                    features_real = Variable(self.feature_extractor(sharp_real.data)).to(self.device)
                    features_fake = self.feature_extractor(sharp_fake.data).to(self.device)
                    
                    # get loss_perceptual
                    grams_real = utils.calc_gram(features_real) * 10.
                    grams_fake = utils.calc_gram(features_fake) * 10.
                    loss_perceptual = self.criterion_g(grams_fake, grams_real) * 4.

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)
                    
                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_total.update(total_loss.item(), len(blur))
                        self.deblurmodel_g_optimizer.step()

                    t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg), loss_d='{:.6f}'.format(epoch_losses_d.avg))
                    t.update(len(blur))

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()

        torch.save(self.deblurmodel_g.state_dict(), '%s/models/dbgan_generator.pth' % self.output_dir)

    def _valid(self):
        # valid
        self.deblurmodel_g.eval()
        epoch_pnsr = utils.AverageMeter()
        epoch_ssim = utils.AverageMeter()

        with torch.no_grad():
            for data in self.valid_dataloader:
                sharp = data

                sharp = sharp.to(device)
                sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
                blur = self.model_blur(sharp_noise)

                preds = self.deblurmodel_g(blur)[0]
                epoch_pnsr.update(utils.calc_psnr(preds, sharp[0]), len(sharp))
                epoch_ssim.update(utils.calc_ssim(preds, sharp[0]), len(sharp))

        print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))


class BasicWGAN(BasicGAN):
    """BasicWGAN clipping"""
    def __init__(self, blur_model_path, device):
        super(BasicWGAN, self).__init__()
        # optim init
        self.deblurmodel_g_optimizer = optim.RMSprop(
            deblurmodel_g.parameters(), 
            lr=self.lr
        )
        self.deblurmodel_d_optimizer = optim.RMSprop(
            deblurmodel_d.parameters(), 
            lr=self.lr
        )

        self.weight_cliping_limit = 0.01

    def _train_batch(self):
        cnt = 0
        one = torch.FloatTensor([1])
        mone = one * -1
        for epoch in range(self.niter):
            deblurmodel_g.train()
            deblurmodel_d.train()

            epoch_losses_d = utils.AverageMeter()
            epoch_losses_total = utils.AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for sharp in self.train_dataloader:
                    cnt += 1

                    sharp = sharp.to(self.device)
                    sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_real = Variable(sharp).to(self.device)
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    for p in self.deblurmodel_d.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)
                    target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)

                    loss_real_d = self.criterion_d(self.deblurmodel_d(sharp_real), target_real)
                    loss_fake_d = self.criterion_d(self.deblurmodel_d(Variable(sharp_fake)), target_fake)
                    loss_d = - loss_real_d + loss_fake_d

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_d.zero_grad()
                        loss_real_d.backward(one)
                        loss_fake_d.backward(mone)
                        epoch_losses_d.update(loss_d.item(), len(sharp))
                        self.deblurmodel_d_optimizer.step()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real blur images and fake blur images
                    features_real = Variable(self.feature_extractor(sharp_real.data)).to(self.device)
                    features_fake = self.feature_extractor(sharp_fake.data).to(self.device)
                    
                    # get loss_perceptual
                    grams_real = utils.calc_gram(features_real) * 10.
                    grams_fake = utils.calc_gram(features_fake) * 10.
                    loss_perceptual = self.criterion_g(grams_fake, grams_real) * 4.

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)
                    
                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_total.update(total_loss.item(), len(blur))
                        self.deblurmodel_g_optimizer.step()

                    t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg), loss_d='{:.6f}'.format(epoch_losses_d.avg))
                    t.update(len(blur))

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()

        torch.save(self.deblurmodel_g.state_dict(), '%s/models/dbgan_generator.pth' % self.output_dir)


class BasicWGANGP(object):
    """BasicWGANGP"""
    def __init__(self, blur_model_path, device):
        super(BasicWGANGP, self).__init__()
        self.gradient_penalty = GradientPenaltyLoss()
    
    def _train_batch(self):
        cnt = 0
        one = torch.FloatTensor([1])
        mone = one * -1
        for epoch in range(self.niter):
            deblurmodel_g.train()
            deblurmodel_d.train()

            epoch_losses_d = utils.AverageMeter()
            epoch_losses_total = utils.AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for sharp in self.train_dataloader:
                    cnt += 1

                    sharp = sharp.to(self.device)
                    sharp_noise = utils.concat_noise(sharp, (4, 128, 128), sharp.size()[0])
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_real = Variable(sharp).to(self.device)
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    for p in self.deblurmodel_d.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)
                    target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)

                    loss_real_d = self.criterion_d(self.deblurmodel_d(sharp_real), target_real)
                    loss_fake_d = self.criterion_d(self.deblurmodel_d(Variable(sharp_fake)), target_fake)
                    loss_d = - loss_real_d + loss_fake_d

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_d.zero_grad()
                        loss_real_d.backward(mone)
                        loss_fake_d.backward(one)
                        epoch_losses_d.update(loss_d.item(), len(sharp))
                        self.deblurmodel_d_optimizer.step()
                        # train with gradient penalty
                        loss_gradient_penalty = self.gradient_penalty(sharp_real.data, sharp_fake.data)
                        loss_gradient_penalty.backward()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real blur images and fake blur images
                    features_real = Variable(self.feature_extractor(sharp_real.data)).to(self.device)
                    features_fake = self.feature_extractor(sharp_fake.data).to(self.device)
                    
                    # get loss_perceptual
                    grams_real = utils.calc_gram(features_real) * 10.
                    grams_fake = utils.calc_gram(features_fake) * 10.
                    loss_perceptual = self.criterion_g(grams_fake, grams_real) * 4.

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)
                    
                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_total.update(total_loss.item(), len(blur))
                        self.deblurmodel_g_optimizer.step()

                    t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg), loss_d='{:.6f}'.format(epoch_losses_d.avg))
                    t.update(len(blur))

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()

        torch.save(self.deblurmodel_g.state_dict(), '%s/models/dbgan_generator.pth' % self.output_dir)
