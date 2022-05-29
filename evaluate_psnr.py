import os
from PIL import Image
import numpy as np
from math import log10, sqrt

def PSNR(original, srgOUT):
	mse = np.mean((original - srgOUT) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

output_path = 'test_imgs/output'
HR_path = 'test_imgs/hr'
output_imgs = os.listdir(output_path)
HR_imgs = os.listdir(HR_path)
sumPSNR = 0
for file in HR_imgs:
  if file=='.gitkeep':
    continue
  hrImg = Image.open(os.path.join(HR_path, file))
  outImg = Image.open(os.path.join(output_path, 'SR_'+file[0:4]+'x4.png'))
  val = PSNR(np.asarray(hrImg), np.asarray(outImg))
  sumPSNR += val
  print('For File '+file+' : '+str(val))

print('Avg PSNR: '+str(sumPSNR/16))