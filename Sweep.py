# Utility script to evaluate the learned parameter domain of a proxy.
# 
# Uniformly sweeps across the parameter range for a given proxy, saving numbered outputs,
# formatted for use with ffmpeg to create animations. 
# 
# Sample ffmpeg command: ffmpeg -r 25 -f image2 -s 1920x1080 -i [image name]_[proxy type]_[parameter]_sweep_%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p [VIDEO NAME].mp4
#
# How to use: python Sweep.py [proxy type]_[parameter] [path to .DNG image to use as input] [integer number of values to sweep over]
#TODO: Support multiple params per proxy

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local files
from Models import UNet, load_checkpoint, eval
from Dataset import Darktable_Dataset
from Loss_functions import losses
import Constants as c
import PyDarktable as dt
from utils.npy_convert import convert

# Constants
add_params = True
skip_connect = True
clip_output = True
learning_rate = 0.0001
gamma = 0.1
step_size = 7

'''
Sweeping across the parameter ranges
'''
def sweep(proxy_type, param, possible_values, num):
    min = possible_values[0][0]
    max = possible_values[0][1]
    
    # Writing num equally-spaced floats to a .npy file
    vals = [value for value in np.linspace(min, max, int(num))]
    vals_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_sweep_params.npy')
    convert(vals, vals_path)
        
    return vals, vals_path

'''
Procedure to set up data loading
'''
def dataloader(proxy_type, param, input_path, vals_path):

    since = time.time()

    # Create dataset
    image_dataset = Darktable_Dataset(
        root_dir=c.IMAGE_ROOT_DIR, 
        stage=1, 
        proxy_type=proxy_type, 
        param=param, 
        input_dir=input_path, 
        params_file=vals_path, 
        sweep=True
        )

    # Wrap dataset with a DataLoader
    #TODO: remove dataloader?
    data = torch.utils.data.DataLoader(
        image_dataset, 
        batch_size=c.PROXY_MODEL_BATCH_SIZE, 
        sampler=image_dataset.sweep_sampler,
        num_workers=1
        )

    time_elapsed = time.time() - since
    print('Data Loade prepared in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()

    return data

'''
Procedure to initialize the proxy and load in model weights.
'''
def model_init(model_out_dir, possible_params):
    
    # Set up model.
    unet = UNet(
        num_input_channels=c.NUM_IMAGE_CHANNEL + len(possible_params),
        num_output_channels=c.NUM_IMAGE_CHANNEL, 
        skip_connect=skip_connect, 
        add_params=add_params, 
        clip_output=clip_output
        )

    start_epoch = load_checkpoint(unet, model_out_dir) #weight_out_dir
    
    if use_gpu:
        unet.cuda()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        
    # criterion is the loss function, which can be nn.L1Loss() or nn.MSELoss()
    #criterion = nn.MSELoss()
    criterion = losses[c.WHICH_LOSS[proxy_type][0]]
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    
    print('Model initialized')
    return unet, criterion, optimizer

if __name__ == '__main__':
    proxy_type, param = sys.argv[1].split('_') # Type of proxy to evaluate [module, parameter]
    img_path = sys.argv[2]                     # Path of the image to use as input
    num = int(sys.argv[3])                     # Number of parameter values to sweep over

    # Constants
    possible_params = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)
    model_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)
    use_gpu = torch.cuda.is_available()

    # Getting the path at which to save model outputs
    output_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.SWEEP_OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Created directory for sweep outputs at: \n' + output_path)
    
    # Sweeping over the param range and writing to a .npy file
    vals, vals_path = sweep(proxy_type, param, possible_params, num)
    
    # Setting up dataloader
    data = dataloader(proxy_type, param, img_path, vals_path)

    # Initializing the model
    model, criterion, optimizer = model_init(model_out_dir, possible_params)
    
    # Sweeping the proxy
    eval(
        model, 
        data, 
        criterion, 
        optimizer, 
        use_gpu, 
        proxy_type,
        param,
        sweep=True,
        outputs_path=output_path
        )
    
    print('Parameter sweep completed.')