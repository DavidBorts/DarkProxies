# Utility script to evaluate the learned parameter domain of a proxy.
# 
# Uniformly sweeps across the parameter range for a given proxy, saving numbered outputs,
# formatted for use with ffmpeg to create animations. 
# 
# Sample ffmpeg command: ffmpeg -r 25 -f image2 -s 1920x1080 -i [image name]_[proxy type]_[parameter]_sweep_%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p [VIDEO NAME].mp4
#
# How to use: python Darktable_sweep.py [proxy type]_[parameter] [path to .DNG image to use as input] [integer number of values to sweep over]

import sys
import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local files
from Models import UNet, load_checkpoint, eval
from Darktable_dataset import Darktable_Dataset
import Darktable_constants as c
import PyDarktable as dt
from npy_convert import convert

# Constants
add_params = True
skip_connect = True
clip_output = True
use_checkpoint = True
learning_rate = 0.0001
gamma = 0.1
step_size = 7
input_dir = c.SWEEP_INPUT_DIR
output_dir = c.SWEEP_OUTPUT_DIR

def sweep(proxy_type, param, possible_values, num):
    min = possible_values[0][0]
    max = possible_values[0][1]
    
    # Writing num equally-spaced floats to a .npy file
    vals = [value for value in np.linspace(min, max, int(num))]
    vals_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_sweep_params.npy')
    convert(vals, vals_path)
        
    return vals, vals_path
        
def copy_img(img_path, input_dir, vals, render_first=False):
    
    # Getting the name of the image
    image = img_path.split('\\')[-1]
    print(f'image: {image}')
    
    # Tracking the first input to avoid rendering the same image twice
    first_input = None
    
    # Extracting necessary params from the source image
    raw_prepare_params, temperature_params = dt.read_dng_params(img_path)
    
    for val in vals[0]:
    
        # Getting path of the corresponding input image
        input_path = os.path.join(input_dir, f'{image}_{proxy_type}_{param}')
        input_path = (repr(input_path).replace('\\\\', '/')).strip("'") + f'_{val}.tif' # Dealing with Darktable CLI pickiness

        # Making sure that the same input image is not generated multiple times
        if first_input is None:
            first_input = input_path # Path of the original input image to be copied later

            # Assembling a dictionary of all of the original params for the source image
            # (used to render proxy input)
            if render_first:
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering an unchanged copy of the source image for model input
                dt.render(img_path, input_path, original_params)
        else:
            _ = shutil.copyfile(first_input, input_path) # Copying the first input image to get the others
                
    print('Input data copied.')

def dataloader(proxy_type, param, input_path, vals_path):
    # Set up data loading.
    since = time.time()
    image_dataset = Darktable_Dataset(root_dir = c.IMAGE_ROOT_DIR, stage=1, proxy_type=proxy_type, param=param, input_dir=input_path, params_file=vals_path,sweep=True)
    data = torch.utils.data.DataLoader(image_dataset, 
                                               batch_size=c.PROXY_MODEL_BATCH_SIZE, 
                                               sampler=image_dataset.sweep_sampler,
                                               num_workers=1)#num_workers should be 4
    time_elapsed = time.time() - since
    print('Data Loade prepared in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()
    return data

def model_init(model_out_dir, possible_params):
    
    # Set up model.
    unet = UNet(num_input_channels=c.NUM_IMAGE_CHANNEL + len(possible_params),
                num_output_channels=c.NUM_IMAGE_CHANNEL, skip_connect=skip_connect, add_params=add_params, clip_output=clip_output)
    if use_checkpoint:
        start_epoch = load_checkpoint(unet, model_out_dir) #weight_out_dir
        lr = learning_rate * (gamma ** (start_epoch//step_size))
    else:
        start_epoch = 0 
    if use_gpu:
        unet.cuda()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        
    # criterion is the loss function, which can be nn.L1Loss() or nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    
    print('Model initialized')
    return unet, criterion, optimizer

if __name__ == '__main__':
    proxy_type, param = sys.argv[1].split('_') # Type of proxy to evaluate [module, parameter]
    img_path = sys.argv[2]                     #  Path of the image to use as input
    num = int(sys.argv[3])                     # Number of parameter values to sweep over

    # Constants
    possible_params = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)
    model_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)
    use_gpu = torch.cuda.is_available()

    # Getting the paths for model inputs and outputs
    input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + input_dir)
    output_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + output_dir)
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # Sweeping over the param range and writing to a .npy file
    vals, vals_path = sweep(proxy_type, param, possible_params, num)
    
    # Copying the image at img_path num times to create the input dataset
    copy_img(img_path, input_path, vals)
    
    # Setting up dataloader
    data = dataloader(proxy_type, param, input_path, vals_path)

    # Initializing the model
    model, criterion, optimizer = model_init(model_out_dir, possible_params)
    
    # Sweeping
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