'''
Utility script to crop any .tif image such that it meets
the dimensions laid out in Darktable_constants.py\

How to use: python ./utils/crop_tif.py [PATH TO .TIF FILE]  [PATH TO DIRECTORY AT WHICH TO SAVE CROPPED IMAGE]
'''

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import transforms

# Local files
import Darktable_constants as c

# Command-line args
path = sys.argv[1]
crop_path = sys.argv[2]

# Opening file
image_uncropped = Image.open(path)

# Transforming TIFF to torch tensor
to_tensor_transform = transforms.Compose([transforms.ToTensor()])
image_uncropped = to_tensor_transform(image_uncropped)
image_uncropped = interpolate(image_uncropped[None, :, :, :], scale_factor=0.25, mode='bilinear')
image_uncropped = torch.squeeze(image_uncropped, dim=0)
num_channels, width, height = image_uncropped.size()

# Cropping image tensor
if width % 2 == 0:
    mid_width = width / 2
else:
    mid_width = (width - 1) / 2
if height % 2 == 0:
    mid_height = height / 2
else:
    mid_height = (height - 1) / 2
image_cropped = image_uncropped[:, int(mid_width - (c.IMG_SIZE / 2)):int(mid_width +
                (c.IMG_SIZE / 2)), int(mid_height - (c.IMG_SIZE / 2)):int(mid_height +
                (c.IMG_SIZE / 2))]

# Saving tensor
image_ndarray = image_cropped.numpy()
image_ndarray = np.moveaxis(image_ndarray, 0, -1)
plt.imsave(os.path.join(crop_path, 'crop.' + c.CROP_FORMAT), image_ndarray, format=c.CROP_FORMAT)
