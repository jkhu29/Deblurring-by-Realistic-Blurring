import cv2
import torch
from skimage.color import rgb2yiq, yiq2rgb

import utils
from model import *


bgan = BlurGAN_G().to("cuda") 
bgan_state_dict = torch.load("bgan_pretrain.pth")
bgan.load_state_dict(bgan_state_dict)
bgan.eval()

# dbgan = DeblurGAN_G().to("cuda")
# dbgan_state_dict = torch.load("dbgan_pretrain.pth")
# dbgan.load_state_dict(dbgan_state_dict)
# dbgan.eval()

img = cv2.imread("img/model_blur.png")
img_yiq = rgb2yiq(img)
img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to("cuda")
n, c, h, w = img.shape
img_noise = utils.concat_noise(img, (4, h, w), img.size()[0])
img_bgan = bgan(img_noise)[0].detach().cpu().numpy()
g = utils.lum(img_bgan)
img_yiq[..., 0] = g
img_blur = yiq2rgb(img_yiq)
cv2.imshow("blur", img_blur)

# img_dbgan = dbgan(torch.from_numpy(img_bgan.transpose(2, 0, 1)).float().unsqueeze(0).to("cuda"))[0].detach().cpu().numpy()
# img_sharp = img_dbgan.transpose(1, 2, 0)
# cv2.imshow("deblur", img_sharp/255)
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.imwrite("img/model_blur.png", img_blur)
# cv2.imwrite("img/model_deblur.png", img_sharp)
# print(utils.calc_psnr(torch.from_numpy(img_dbgan).cuda(), img))
