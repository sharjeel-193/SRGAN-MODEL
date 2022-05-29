# Read images (of size 255 x 255) from file.
import tensorflow as tf

def ssim_img(original,srganOut):
    tf.shape(original)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
    tf.shape(srganOut)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
    im1 = tf.expand_dims(original, axis=0)
    im2 = tf.expand_dims(srganOut, axis=0)
    # Compute SSIM over tf.uint8 Tensors.
    ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim2
output_path = 'test_imgs/output'
HR_path = 'test_imgs/hr'
output_imgs = os.listdir(output_path)
HR_imgs = os.listdir(HR_path)
sumSSIM = 0
for file in HR_imgs:
  if file=='.gitkeep':
    continue
  hrImg = Image.open(os.path.join(HR_path, file))
  outImg = Image.open(os.path.join(output_path, 'SR_'+file[0:4]+'x4.png'))
  val = ssim_img(np.asarray(hrImg), np.asarray(outImg))
  sumSSIM += val[0]
  print('For File '+file+' : '+str(val))

print('Avg SSIM: '+str(sumSSIM/16))