import os
import copy
import itertools

from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from torchvision import transforms
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm
# from tqdm.notebook import tqdm

from model import *
import config
import utils


class BasicCycleGAN(object):
    """"""
    def __init__(self):
        super(BasicCycleGAN, self).__init__()
        opt = config.get_cyclegan_options()
        self.train_file = opt.train_file
        self.valid_file = opt.valid_file
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.niter = opt.niter
        self.batch_size = opt.batch_size
        self.workers = opt.workers
        self.batch_scale = opt.batch_scale
        self.lr = opt.lr
        self.output_dir = opt.output_dir

        torch.backends.cudnn.benchmark = True

        self.target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)
        self.target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)

        # cyclegan for bgan, init
        self.model_g_x2y = BlurGAN_G().to(self.device)
        self.model_g_y2x = DeblurGAN_G().to(self.device)
        self.model_d_x = GAN_D().to(self.device)
        self.model_d_y = GAN_D().to(self.device)
        self.vgg = Vgg16().to(self.device)

        if os.path.exists("bgan_pretrain.pth"):
            bgan_params = torch.load("bgan_pretrain.pth")
            dbgan_params = torch.load("dbgan_pretrain.pth")
            self.model_g_x2y.load_state_dict(bgan_params)
            self.model_g_y2x.load_state_dict(dbgan_params)
        else:
            self.model_g_x2y.apply(utils.weights_init)
            self.model_g_y2x.apply(utils.weights_init)

        self.model_d_x.apply(utils.weights_init)
        self.model_d_y.apply(utils.weights_init)

        # criterion init
        self.criterion_generate = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # dataset init
        description = {
            "blur": "byte",
            "sharp": "byte",
            "size": "int",
        }
        train_dataset = TFRecordDataset(self.train_file, None, description, shuffle_queue_size=1024)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True, 
            drop_last=True
        )
        self.data_length = int(44184 / 4 * self.workers)
        valid_dataset = TFRecordDataset(self.valid_file, None, description)
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

        # prep init
        self.prep = transforms.Compose(
            [
                # transforms.Lambda(lambda x: x.mul_(1 / 255)),
                transforms.Normalize(
                    mean=[0.40760392, 0.4595686, 0.48501961],
                    std=[0.225, 0.224, 0.229]
                ),
                # transforms.Lambda(lambda x: x.mul_(255)),
                # WARNING(hujiakui): Lambda --> inplace ops, can't backward
            ]
        )

    def train_batch(self):
        print("-----------------train-----------------")
        cnt = 0
        for epoch in range(self.niter):
            epoch_losses_g_content = utils.AverageMeter()
            epoch_losses_g_style = utils.AverageMeter()
            epoch_losses_d_x = utils.AverageMeter()
            epoch_losses_d_y = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for record in self.train_dataloader:
                    cnt += 1
                    blur = record["blur"].reshape(
                        self.batch_size, 
                        3, 
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    sharp = record["sharp"].reshape(
                        self.batch_size, 
                        3, 
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    n, c, h, w = sharp.shape
                    blur_noise = utils.concat_noise(blur, (c + 1, h, w), n)
                    sharp_noise = utils.concat_noise(sharp, (c + 1, h, w), n)

                    # --------------------
                    # generator train(2 * model_g)
                    # --------------------
                    loss_content, blur_fake, sharp_fake = self._calc_loss_g(blur_noise, blur, sharp_noise, sharp)
                    loss_style = self._calc_loss_style(self.vgg(self.prep(blur / 255) * 255), self.vgg(self.prep(blur_fake / 255) * 255))
                    loss_total = 0.01 * loss_content + loss_style

                    self.model_g_x2y.train()
                    self.model_g_y2x.train()
                    if cnt % self.batch_scale == 0:
                        self.optimizer_g.zero_grad()
                        loss_total.backward()
                        epoch_losses_g_content.update(loss_content.item(), self.batch_size)
                        epoch_losses_g_style.update(loss_style.item(), self.batch_size)
                        self.optimizer_g.step()
                    self.model_g_x2y.eval()
                    self.model_g_y2x.eval()

                    # --------------------
                    # discriminator sharp train(model_d_x)
                    # -------------------- 
                    self.model_d_x.train()
                    loss_total_d_x = self._calc_loss_d(self.model_d_x, sharp_fake, sharp)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_x.zero_grad()
                        loss_total_d_x.backward()
                        epoch_losses_d_x.update(loss_total_d_x.item(), self.batch_size)
                        self.optimizer_d_x.step()
                    self.model_d_x.eval()

                    # --------------------
                    # discriminator blur train(model_d_y)
                    # -------------------- 
                    self.model_d_y.train()
                    loss_total_d_y = self._calc_loss_d(self.model_d_y, blur_fake, blur)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_y.zero_grad()
                        loss_total_d_y.backward()
                        epoch_losses_d_y.update(loss_total_d_y.item(), self.batch_size)
                        self.optimizer_d_y.step()
                    self.model_d_y.eval()

                    t.set_postfix(
                        loss_content='{:.6f}'.format(epoch_losses_g_content.avg), 
                        loss_style='{:.6f}'.format(epoch_losses_g_style.avg), 
                        loss_d_sharp='{:.6f}'.format(epoch_losses_d_x.avg), 
                        loss_d_blur='{:.6f}'.format(epoch_losses_d_y.avg)
                    )
                    t.update(self.batch_size)

                torch.save(self.model_g_x2y.state_dict(), "{}/bgan_generator_snapshot_{}.pth".format(self.output_dir, epoch))
                torch.save(self.model_g_y2x.state_dict(), "{}/dbgan_generator_snapshot_{}.pth".format(self.output_dir, epoch))

            self.model_scheduler_g.step()
            self.model_scheduler_d_x.step()
            self.model_scheduler_d_y.step()

    def _calc_loss_g(self, blur_noise, blur_real, sharp_noise, sharp_real):
        # loss identity(ATTN!: `a_same = model_a2b(a_real)`)
        _, c, h, w = blur_real.shape
        blur_same = self.model_g_x2y(blur_noise)            # model_g_x2y: sharp --> blur
        loss_identity_blur = self.criterion_identity(blur_same, blur_real)

        sharp_fake = self.model_g_y2x(sharp_real)           # model_g_y2x: blur --> sharp
        loss_identity_sharp = self.criterion_identity(sharp_fake, sharp_real)

        # loss gan
        blur_fake = self.model_g_x2y(sharp_noise)
        blur_fake_pred = self.model_d_y(blur_fake)          # get blur features
        loss_gan_x2y = self.criterion_generate(blur_fake_pred, self.target_real)

        sharp_fake = self.model_g_y2x(blur_real)
        sharp_fake_pred = self.model_d_x(sharp_fake)        # get sharp features
        loss_gan_y2x = self.criterion_generate(sharp_fake_pred, self.target_real)

        sharp_fake_noise = utils.concat_noise(sharp_fake, (c + 1, h, w), blur_real.size()[0])

        # loss cycle
        blur_recover = self.model_g_x2y(sharp_fake_noise)   # recover the blur: blur->sharp->blur
        loss_cycle_x2y = self.criterion_cycle(blur_recover, blur_real) * 2

        sharp_recover = self.model_g_y2x(blur_fake)         # recover the sharp: sharp->blur->sharp
        loss_cycle_y2x = self.criterion_cycle(sharp_recover, sharp_real) * 2

        # loss total
        loss_total = loss_identity_blur + loss_identity_sharp + \
                     loss_gan_x2y + loss_gan_y2x + \
                     loss_cycle_x2y + loss_cycle_y2x

        return loss_total, blur_fake, sharp_fake
    
    def _calc_loss_style(self, features_fake, features_real, loss_style=0):
        for f_fake, f_real in zip(features_fake, features_real):
            gram_fake = utils.calc_gram(f_fake)
            gram_real = utils.calc_gram(f_real)
            loss_style += self.criterion_generate(gram_fake, gram_real)
        return loss_style

    def _calc_loss_d(self, model_d, fake, real):
        # loss real
        pred_real = torch.sigmoid(model_d(real))
        loss_real = self.criterion_generate(pred_real, self.target_real)

        # loss fake
        fake_ = copy.deepcopy(fake.data)
        pred_fake = torch.sigmoid(model_d(fake_.detach()))
        loss_fake = self.criterion_generate(pred_fake, self.target_fake)

        # loss rbl
        loss_rbl = - torch.log(abs(loss_real - loss_fake)) - \
                     torch.log(abs(1 - loss_fake - loss_real))

        # loss total
        loss_total = (loss_real + loss_fake) * 0.5 + loss_rbl * 0.01

        return loss_total


class BasicGAN(object):
    """BasicGAN"""
    def __init__(self):
        super(BasicGAN, self).__init__()
        opt = config.get_dbgan_options()
        self.train_file = opt.train_file
        self.valid_file = opt.valid_file
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.niter = opt.niter
        self.batch_size = opt.batch_size
        self.workers = opt.workers
        self.batch_scale = opt.batch_scale
        self.lr = opt.lr
        self.output_dir = opt.output_dir
        self.blur_model_path = opt.blur_model_path

        torch.backends.cudnn.benchmark = True

        self.target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)
        self.target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)

        # ----------------------
        # bgan
        # ----------------------
        # models init
        self.model_blur = BlurGAN_G().to(self.device)
        self.model_blur.load_state_dict(torch.load(self.blur_model_path))
        self.model_blur.to(self.device)
        self.model_blur.eval()
        for params in self.model_blur.parameters():
            params.required_grad = False

        # ----------------------
        # dbgan
        # ----------------------
        # models init
        self.deblurmodel_g = DeblurGAN_G().to(self.device)
        self.deblurmodel_d = GAN_D().to(self.device)
        params = torch.load("dbgan_pretrain.pth")
        self.deblurmodel_g.load_state_dict(params)
        self.deblurmodel_d.apply(utils.weights_init)
        self.vgg = Vgg16().to(self.device)

        # dataset init
        description = {
            "image": "byte",
            "size": "int",
        }
        train_dataset = TFRecordDataset(self.train_file, None, description, shuffle_queue_size=1024)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.data_length = int(67032 / 4 * self.workers)

        valid_dataset = TFRecordDataset(self.valid_file, None, description)
        self.valid_dataloader = dataloader.DataLoader(
            dataset=valid_dataset, 
            batch_size=1
        )

        # criterion init
        self.criterion_g = nn.MSELoss()
        self.criterion_d = nn.BCELoss()

        # optim init
        self.deblurmodel_g_optimizer = optim.RMSprop(
            self.deblurmodel_g.parameters(), 
            lr=self.lr, eps=1e-8
        )
        self.deblurmodel_d_optimizer = optim.RMSprop(
            self.deblurmodel_d.parameters(), 
            lr=self.lr, eps=1e-8
        )

        # lr init
        self.deblurmodel_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.deblurmodel_g_optimizer, T_max=self.niter
        )
        self.deblurmodel_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.deblurmodel_d_optimizer, T_max=self.niter
        )

        # prep init
        self.prep = transforms.Compose(
            [
                # transforms.Lambda(lambda x: x.mul_(1 / 255)),
                transforms.Normalize(
                    mean=[0.40760392, 0.4595686, 0.48501961],
                    std=[0.225, 0.224, 0.229]
                ),
                # transforms.Lambda(lambda x: x.mul_(255)),
                # WARNING(hujiakui): Lambda --> inplace ops, can't backward
            ]
        )

    def train_batch(self):
        cnt = 0
        for epoch in range(self.niter):
            epoch_losses_d = utils.AverageMeter()
            epoch_losses_total = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for record in self.train_dataloader:
                    cnt += 1
                    sharp_real = record["image"].reshape(
                        self.batch_size, 
                        3, 
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    n, c, h, w = sharp_real.shape
                    sharp_noise = utils.concat_noise(sharp_real, (c + 1, h, w), n)
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    self.deblurmodel_d.train()
                    loss_real_d = self.criterion_d(torch.sigmoid(self.deblurmodel_d(sharp_real)), self.target_real)
                    loss_fake_d = self.criterion_d(torch.sigmoid(self.deblurmodel_d(Variable(sharp_fake))), self.target_fake)
                    loss_d = (loss_real_d + loss_fake_d) * 0.5

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_d.zero_grad()
                        loss_d.backward()
                        epoch_losses_d.update(loss_d.item(), self.batch_size)
                        self.deblurmodel_d_optimizer.step()
                    self.deblurmodel_d.eval()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real sharp images and fake sharp images
                    features_real = self.vgg(self.prep(sharp_real / 255) * 255)
                    features_fake = self.vgg(self.prep(sharp_fake / 255) * 255)

                    # get loss_perceptual
                    loss_perceptual = 0
                    for f_fake, f_real in zip(features_fake, features_real):
                        gram_fake = utils.calc_gram(f_fake)
                        gram_real = utils.calc_gram(f_real)
                        loss_perceptual += self.criterion_g(gram_fake, gram_real)

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)

                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

                    self.deblurmodel_g.train()
                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_total.update(total_loss.item(), self.batch_size)
                        self.deblurmodel_g_optimizer.step()
                    self.deblurmodel_g.eval()

                    t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg), loss_d='{:.6f}'.format(epoch_losses_d.avg))
                    t.update(self.batch_size)

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()

            torch.save(self.deblurmodel_g.state_dict(), '{}/dbgan_generator_snapshot_{}.pth'.format(self.output_dir, epoch))

    def _valid(self):
        # valid
        torch.cuda.empty_cache()
        epoch_pnsr = utils.AverageMeter()
        epoch_ssim = utils.AverageMeter()
        cnt = 0
        for record in self.valid_dataloader:
            sharp = record["image"].reshape(
                1,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to(self.device)
            n, c, h, w = sharp.shape
            sharp_noise = utils.concat_noise(sharp, (c + 1, h, w), n)
            blur = self.model_blur(sharp_noise)

            preds = self.deblurmodel_g(blur)
            del blur
            epoch_pnsr.update(utils.calc_psnr(preds, sharp), 1)
            epoch_ssim.update(utils.calc_ssim(preds, sharp), 1)
            cnt += 1
            if cnt >= 5:
                break

        print('eval psnr: {:.6f} eval ssim: {:.6f}'.format(epoch_pnsr.avg, epoch_ssim.avg))


class BasicWGAN(BasicGAN):
    """BasicWGAN clipping"""
    def __init__(self):
        super(BasicWGAN, self).__init__()
        # optim init
        self.deblurmodel_g_optimizer = optim.RMSprop(
            self.deblurmodel_g.parameters(),
            lr=self.lr
        )
        self.deblurmodel_d_optimizer = optim.RMSprop(
            self.deblurmodel_d.parameters(),
            lr=self.lr
        )

        self.weight_cliping_limit = 0.01

    def train_batch(self):
        cnt = 0
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        for epoch in range(self.niter):
            epoch_losses_d = utils.AverageMeter()
            epoch_losses_total = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for record in self.train_dataloader:
                    cnt += 1
                    sharp_real = record["image"].reshape(
                        self.batch_size, 
                        3, 
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    n, c, h, w = sharp_real.shape
                    sharp_noise = utils.concat_noise(sharp_real, (c + 1, h, w), n)
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    for p in self.deblurmodel_d.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                    loss_real_d = self.deblurmodel_d(sharp_real).mean()
                    loss_fake_d = self.deblurmodel_d(Variable(sharp_fake)).mean()
                    loss_d = loss_real_d - loss_fake_d

                    self.deblurmodel_d.train()
                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_d.zero_grad()
                        loss_real_d.backward(mone)
                        loss_fake_d.backward(one)
                        self.deblurmodel_d_optimizer.step()
                        epoch_losses_d.update(loss_d.item(), self.batch_size)
                    self.deblurmodel_d.eval()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real blur images and fake blur images
                    features_real = self.vgg(self.prep(sharp_real.data / 255) * 255)
                    features_fake = self.vgg(self.prep(sharp_fake.data / 255) * 255)

                    # get loss_perceptual
                    loss_perceptual = 0
                    for f_fake, f_real in zip(features_fake, features_real):
                        gram_fake = utils.calc_gram(f_fake)
                        gram_real = utils.calc_gram(f_real)
                        loss_perceptual += self.criterion_g(gram_fake, gram_real)

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)

                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.01 * loss_rbl

                    self.deblurmodel_g.train()
                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_total.update(total_loss.item(), self.batch_size)
                        self.deblurmodel_g_optimizer.step()
                    self.deblurmodel_g.eval()

                    t.set_postfix(
                        total_loss='{:.6f}'.format(epoch_losses_total.avg), 
                        loss_d='{:.6f}'.format(epoch_losses_d.avg)
                    )
                    t.update(self.batch_size)

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()
            torch.save(self.deblurmodel_g.state_dict(), '{}/dbgan_generator_snapshot_{}.pth'.format(self.output_dir, epoch))


class BasicWGANGP(BasicWGAN):
    """BasicWGANGP"""
    def __init__(self):
        super(BasicWGANGP, self).__init__()
    
    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = epsilon * real + (1 - epsilon) * fake
        interpolates = interpolates.clone().detach().requires_grad_(True)
        gradients = torch.autograd.grad(
            self.deblurmodel_d(interpolates),
            interpolates,
            grad_outputs=self.target_real,
            create_graph=True
        )[0]

        return ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()

    def train_batch(self):
        cnt = 0
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        for epoch in range(self.niter):
            epoch_losses_d = utils.AverageMeter()
            epoch_losses_perceptual = utils.AverageMeter()
            epoch_losses_content = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for record in self.train_dataloader:
                    cnt += 1
                    sharp_real = record["image"].reshape(
                        self.batch_size, 
                        3, 
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    n, c, h, w = sharp_real.shape
                    sharp_noise = utils.concat_noise(sharp_real, (c + 1, h, w), n)
                    blur = self.model_blur(sharp_noise)

                    # get the sharp real and fake
                    sharp_fake = self.deblurmodel_g(blur).to(self.device)

                    # --------------
                    # update model d
                    # --------------
                    self.deblurmodel_d.train()
                    for p in self.deblurmodel_d.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                    for _ in range(self.batch_scale):
                        loss_real_d = self.deblurmodel_d(sharp_real).mean()
                        loss_fake_d = self.deblurmodel_d(Variable(sharp_fake)).mean()

                        self.deblurmodel_d.zero_grad()
                        loss_real_d.backward(mone)
                        loss_fake_d.backward(one)

                        # train with gradient penalty
                        loss_gradient_penalty = self.gradient_penalty(sharp_real, sharp_fake)
                        loss_gradient_penalty.backward()

                        self.deblurmodel_d_optimizer.step()

                        loss_d = loss_real_d - loss_fake_d
                        epoch_losses_d.update(loss_d.item(), self.batch_size)
                    self.deblurmodel_d.eval()

                    # --------------
                    # update model g
                    # --------------
                    # get the features of real blur images and fake blur images
                    self.deblurmodel_g.train()
                    features_real = self.vgg(self.prep(sharp_real.data / 255) * 255)
                    features_fake = self.vgg(self.prep(sharp_fake.data / 255) * 255)

                    # get loss_perceptual
                    loss_perceptual = 0
                    for f_fake, f_real in zip(features_fake, features_real):
                        gram_fake = utils.calc_gram(f_fake)
                        gram_real = utils.calc_gram(f_real)
                        loss_perceptual += self.criterion_g(gram_fake, gram_real)

                    # get loss content
                    loss_content = self.criterion_g(sharp_real, sharp_fake)

                    # get loss_rbl
                    loss_rbl = - torch.log(abs(loss_real_d.detach() - loss_fake_d.detach())) - \
                                 torch.log(abs(1 - loss_fake_d.detach() - loss_real_d.detach()))

                    total_loss = 0.005 * loss_content + loss_perceptual + 0.1 * loss_rbl

                    if cnt % self.batch_scale == 0:
                        self.deblurmodel_g.zero_grad()
                        total_loss.backward()
                        epoch_losses_perceptual.update(loss_perceptual.item(), self.batch_size)
                        epoch_losses_content.update(loss_content.item(), self.batch_size)
                        self.deblurmodel_g_optimizer.step()
                    self.deblurmodel_g.eval()

                    t.set_postfix(
                        loss_d='{:.6f}'.format(epoch_losses_d.avg), 
                        loss_content='{:.6f}'.format(epoch_losses_content.avg), 
                        loss_perceptual='{:.6f}'.format(epoch_losses_perceptual.avg)
                    )
                    t.update(self.batch_size)

                self._valid()

            self.deblurmodel_g_scheduler.step()
            self.deblurmodel_d_scheduler.step()
            torch.save(self.deblurmodel_g.state_dict(), '{}/dbgan_generator_snapshot_{}.pth'.format(self.output_dir, epoch))
