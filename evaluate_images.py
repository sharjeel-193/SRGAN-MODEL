import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import cv2

from model import Generator

import os

def check_similarity(hrImg, outImg):
  img1 = np.asarray(hrImg)
  img2 = np.asarray(outImg)
  errorL2 = cv2.norm( img1, img2, cv2.NORM_L2 )
  percentage = 1 - errorL2 / ( img1.shape[0] * img1.shape[1] )
  return percentage


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

input_path = 'test_imgs/input'
HR_path = 'test_imgs/hr'
input_imgs = os.listdir(input_path)
HR_imgs = os.listdir(HR_path)
for file in input_imgs:
  if(file=='gitkeep'):
    continue
  img_path = (os.path.join(input_path, file))
  image = Image.open(img_path)
  hr_img = Image.open(os.path.join(HR_path, file[0:4]+'.png'))
  image = Variable(ToTensor()(image)).unsqueeze(0)
  if TEST_MODE:
      image = image.cuda()

  start = time.process_time()
  out = model(image)
  elapsed = (time.process_time() - start)
  print('For Image '+str(file))
  print('\tcost ' + str(elapsed) + 's')
  out_img = ToPILImage()(out[0].data.cpu())
  print('\tshape ' + str(np.asarray(out_img).shape))
  # score = ssim(np.asarray(out_img), np.asarray(hr_img))
  print('Similarity: '+ str(check_similarity(hr_img, out_img)))
  out_img.save('test_imgs/output/SR_' + file)
  # print(filename)

