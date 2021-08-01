import cv2
import numpy as np
import torch

from model import BGAN_G
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_blur = BGAN_G().to(device)
pth_path = "/home/junru/code/Deblurring-by-Realistic-Blurring-main/models/bgan_generator.pth"
model_blur.load_state_dict(torch.load(pth_path))
model_blur.to(device)
model_blur.eval()

img = cv2.imread("bgan_and_dbgan.png").transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0).to(device)
print(img.shape, img.size()[2], img.size()[3])
noise = utils.concat_noise(img, (4, img.size()[2], img.size()[3]))
img_blur = model_blur(noise).detach().squeeze().cpu().numpy().transpose(1, 2, 0)

cv2.imshow("blur", img_blur)
cv2.imwrite("blur.png", img_blur*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
