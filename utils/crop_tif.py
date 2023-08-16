'''
Utility script to crop any .tif image such that it meets
the dimensions specified in Darktable_constants.py

How to use: python ./utils/crop_tif.py [PATH TO .TIF FILE] [PATH TO DIRECTORY AT WHICH TO SAVE CROPPED IMAGE]
'''
# TODO: update to support Demosaic

import sys
import torch
import numpy as np
import tifffile
import imageio
from PIL import Image, ImageOps
from torch.nn.functional import interpolate
from torchvision import transforms

IMG_SIZE = 736

# Command-line args
path = sys.argv[1]
crop_path = sys.argv[2]

try:
    image_uncropped = Image.open(path)
    image_uncropped = ImageOps.exif_transpose(image_uncropped)
except:
    image_uncropped = imageio.imread(path)

# Transforming TIFF to torch tensor
to_tensor_transform = transforms.Compose([transforms.ToTensor()])
image_uncropped = to_tensor_transform(image_uncropped)
image_uncropped = torch.squeeze(image_uncropped, dim=0)
print(f"Uncropped img size: {str(image_uncropped.size())}")
if len(image_uncropped.size()) == 3:
    num_channels, width, height = image_uncropped.size()
else:
    num_channels = 1
    width, height = image_uncropped.size()

# Cropping image tensor
if IMG_SIZE % 4 != 0:
    raise ValueError("IMG_SIZE in Constants.py must be a multiple of 4 (default is 736).")
mid_width = width // 2
width_low = int(mid_width - (IMG_SIZE / 2))
while width_low % 4 != 0: # Ensuring that image crops are along mosaic boundaries
    width_low -= 1
width_high = width_low + IMG_SIZE
mid_height = height // 2
height_low = int(mid_height - (IMG_SIZE / 2))
while height_low % 4 != 0:
    height_low -= 1
height_high = height_low + IMG_SIZE

if len(image_uncropped.size()) == 3:
    image_cropped = image_uncropped[:, width_low:width_high, height_low:height_high]
else:	
    image_cropped = image_uncropped[width_low:width_high, height_low:height_high]

# Saving tensor
image_ndarray = image_cropped.numpy()
image_ndarray = np.moveaxis(image_ndarray, 0, -1)
print(f"Cropped img size: {str(image_ndarray.shape)}")
tifffile.imwrite(crop_path, image_ndarray)
#plt.imsave(os.path.join(crop_path, 'crop.tif'), image_ndarray, format='tif')
