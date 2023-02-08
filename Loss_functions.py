# File to define all of the loss functions used throughout this codebase

import numpy as np
import torch.nn as nn

'''
Spatial gradient loss

Adapted from Neural Nano Optics (Tseng et al. 2021):
https://github.com/Ethan-Tseng/Neural_Nano-Optics/blob/master/loss.py 
'''
# TODO: test me!
def Spatial_loss(output_img, GT_img):

    def spatial_gradient(x):
        diag_down = x[:, 1:, 1:, :] - x[:, :-1, :-1, :]
        dv = x[:, 1:, :, :] - x[:, :-1, :, :]
        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_up = x[:, :-1, 1:, :] - x[:, 1:, :-1, :]

        return [dh, dv, diag_down, diag_up]

    total_loss = 0.0
    for i in range(np.shape(output_img)[0]):

        gx = spatial_gradient(output_img[i:i+1,:,:,:])
        gy = spatial_gradient(GT_img)

        loss = 0
        for xx, yy in zip(gx, gy):
            loss = loss + np.mean(np.abs(xx - yy))
        total_loss += loss
    
    return total_loss


# Dictionary to store references to all loss functions
losses = {
    'L1': nn.L1Loss,
    'MSE': nn.MSELoss,
    'Spatial': Spatial_loss,
    'Perceptual': None #TODO: move me over from Models.py + debug
}