import cv2
import numpy as np
import torch

from model import DBGAN_G
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_deblur = DBGAN_G().to(device)
pth_path = "/home/junru/code/Deblurring-by-Realistic-Blurring-main/models/dbgan_generator_pretrain.pth"
model_deblur.load_state_dict(torch.load(pth_path))
model_deblur.to(device)
model_deblur.eval()

img = cv2.imread("deblur.jpg") / 255.
img = img.transpose(2, 0, 1)
img = img.astype(np.float32)
img = torch.from_numpy(img).unsqueeze(0).to(device)
img_sharp = model_deblur(img).detach()
img_sharp = img_sharp.squeeze().cpu().numpy().transpose(1, 2, 0)

cv2.imshow("sharp", img_sharp)
cv2.imwrite("deblur_pretrain_result.jpg", img_sharp*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
