# File to define all of the loss functions used throughout this codebase

import numpy as np
import torch
import torch.nn as nn

# local files
from Models import PerceptualLoss

'''
Spatial gradient loss

Adapted from Neural Nano Optics (Tseng et al. 2021):
https://github.com/Ethan-Tseng/Neural_Nano-Optics/blob/master/loss.py 
'''
# TODO: test me!
class SpatialLoss():
    def __init__(self):

        def spatial_gradient(x):
            diag_down = x[:, 1:, 1:, :] - x[:, :-1, :-1, :]
            dv = x[:, 1:, :, :] - x[:, :-1, :, :]
            dh = x[:, :, 1:, :] - x[:, :, :-1, :]
            diag_up = x[:, :-1, 1:, :] - x[:, 1:, :-1, :]

            return [dh, dv, diag_down, diag_up]
        self.spatial_gradient = spatial_gradient
        
    def __call__(self, output_img, GT_img):
        total_loss = 0.0

        for i in range(np.shape(output_img)[0]):
            gx = self.spatial_gradient(output_img[i:i+1,:,:,:])
            gy = self.spatial_gradient(GT_img)

            loss = 0
            for xx, yy in zip(gx, gy):
                loss = loss + torch.mean(torch.abs(xx - yy))
            total_loss += loss
        
        return total_loss


# Dictionary to store references to all loss functions
losses = {
    'L1': nn.L1Loss,
    'MSE': nn.MSELoss,
    'Spatial': SpatialLoss,
    'Perceptual': PerceptualLoss #TODO: move me over from Models.py + debug
}
