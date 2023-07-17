'''
Script to configure and process a RAW image with a chain 
of NN proxies of Darktable image processing modules.

The functions and classes in this script are used elsewhere
in this codebase to perform experiments that require
multiple proxies instead of one.

This script can also be run directly to instantiate a 
differentiable ISP and evaluate it on a set of 
input RAW images. See below.

Arguments:
[param_file]: path to .txt file with the proxy params for each generated image
[dng_path]: path to directory with desired .DNG source images
[input_path]: path at which to store rendered pipeline inputs
[output_path]: path at which to store pipeline output predictions
[label_path]: path at which to store rendered ground truth labels

Example param file**:
                    0.5,3.3
                    0.9,4.5
                    -0.4,0.7
**(to generate 3 images using a pipeline with 2 proxies)

IMPORTANT NOTE:
Before running this script, edit the pipeline_config.txt file to
toggle which proxies to include in the pipeline. Each proxy has
a boolean on its right which enables or disables it. DO NOT modify
this file in any other way.

Example pipeline_config.txt file:**
                            sharpen_amount 1
                            exposure_exposure 0
                            colorbalancergb_contrast 1
                            hazeremoval_strength 0
                            lowpass_radius 0
                            censorize_pixelate 0
**(to include only sharpen and contrast in the pipeline)

Example script usage: python ./Proxy_pipeline.py [param_file] [dng_path] [input_path] [output_path] [label_path]
'''

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Local files
import Constants as c
from Dataset import Darktable_Dataset
from Generate_data import generate_pipeline
from Models import UNet, ChenNet, DemosaicNet, generic_load
from utils.misc import get_possible_values

class ProxyPipeline:
    def __init__(self, proxies_list, use_gpu):
        
        self.num_proxies = len(proxies_list)
        self.use_gpu = use_gpu

        # Loading in all of the specified proxies
        # in the correct order
        proxies = []
        possible_values_list = []
        img_channels_list = []
        for proxy_name, params in proxies_list:

            proxy_type = proxy_name.split('_')[0]

            # Getting necessary params to load in the proxy
            params = params.split('_')
            possible_values = get_possible_values(proxy_type, params)
            possible_values_list.append(possible_values)
            if proxy_type in c.SINGLE_IMAGE_CHANNEL:
                img_channels_list.append(1)
            elif proxy_type == "demosaic":
                img_channels_list.append(4)
            else:
                img_channels_list.append(c.NUM_IMAGE_CHANNEL)

            proxy_type = proxy_name

            # Location of model weights
            #TODO: refactor to get rid of 'full' checks
            if params[0].lower() != 'full':
                weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + ''.join([f'{param}_' for param in params]) + c.MODEL_WEIGHTS_PATH)
            else:
                weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + c.MODEL_WEIGHTS_PATH)

            # Loding proxy
            proxies.append(load_model(proxy_type, params, possible_values, weight_out_dir, self.use_gpu))
        self.models = proxies
        self.possible_values = possible_values_list #NOTE: this is a list of lists
        self.img_channels = img_channels_list
    
    def process(self, orig_tensor, input_tensors):

        img_channels = self.img_channels

        # Filling in the image for the input to the first proxy
        input_tensors[0].data[:, 0:img_channels[0], :, :] = orig_tensor[:, 0:img_channels[0], :, :]

        # Storing the pipeline outputs
        outputs = []

        # evaluating
        for num in range(self.num_proxies):
            model = self.models[num]
            input = input_tensors[num]

            output = model(input)

            # Filling in the output of the previous proxy into the input tensor of the
            # following proxy
            if (num + 1) < self.num_proxies:
                input_tensors[num + 1].data[:, 0:img_channels[num], :, :] = output

            outputs.append(output)

        return outputs

def load_model(proxy_type, params, possible_values, weight_out_dir, use_gpu):
    
    # TODO: rework to consider -c edge case
    if params[0] != 'full':
        print(f"Loading in model: {proxy_type}: " + "".join(f" {param}" for param in params))
    else:
        print(f"Loading in model: {proxy_type}")

    # Getting model weights
    weights_list = os.listdir(weight_out_dir)
    if params[0].lower() != 'full':
        weights_list.sort(key=lambda x: float(x.strip(proxy_type + '_' + "".join(f"{param}_" for param in params)).strip('.pkl')))
    else:
        weights_list.sort(key=lambda x: float(x.strip(proxy_type + '_').strip('.pkl')))
    final_weights = weights_list[-1]
    print(f'Using model weights from {final_weights}.')

    # Loading the weights into a Unet
    num_input_channels = c.NUM_IMAGE_CHANNEL + len(possible_values)
    state_dict = generic_load(weight_out_dir, final_weights)
    if proxy_type == "demosaic":
        model = ChenNet(0, clip_output=True, add_params=False)
    else:
        model = UNet(
                    num_input_channels=num_input_channels,
                    num_output_channels=c.NUM_IMAGE_CHANNEL
                    )
    model.load_state_dict(state_dict)
    
    # Locking the weights in the U-Net
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad=False

     # Adjusting for GPU usage
    if use_gpu:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model

def evaluate(param_file, input_path, label_path, output_path, use_gpu):

    # Reading in proxy order
    proxy_order = []
    with open(os.path.join(c.IMAGE_ROOT_DIR, c.CONFIG_FILE), 'r') as file:
        lines = file.readlines()
        for line in lines:
            proxy, params, enable = line.split(' ')
            if int(enable) == 1:
                proxy_order.append((proxy, params))


    # Differentiable ISP
    isp = ProxyPipeline(proxy_order, use_gpu)

    # Setting up dataset
    print("Preparing dataset" )
    sys.stdout.flush()
    image_dataset = Darktable_Dataset(
                                      root_dir = c.IMAGE_ROOT_DIR, 
                                      stage=3, 
                                      input_dir=input_path,
                                      output_dir=label_path
                                      )

    # Adjusting for GPU usage
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor

    # Reading the list of input param values
    params_list = None
    with open(param_file, 'r') as file:
        params_list = file.readlines()

    # Creating input tensors for the proxies
    input_tensors = []
    for num in range(isp.num_proxies):

        # Getting possible values
        proxy_type, params = proxy_order[num]
        #FIXME: bring everything below this line up to speed
        possible_values = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)

        # Getting correct tensor size
        num_input_channels = c.NUM_IMAGE_CHANNEL + len(possible_values)

        input_tensor = torch.tensor((), dtype=torch.float).new_ones((1 , \
                       num_input_channels, width, height)).type(dtype)
        
        params = params_list[num].split(',')

        # Filling the input tensors with the
        # correct param value for each proxy
        for param_idx in range(len(params)):
            param = np.array(params)[param_idx]
            input_tensor.data[:, c.NUM_IMAGE_CHANNEL+param_idx, :, :] = param

        input_tensors.append(input_tensor)

    # Evaluating
    for index in range(len(image_dataset)):
        name, orig_tensor, label_tensor = image_dataset[index]
        _, width, height = orig_tensor.size()

        outputs = isp.process(orig_tensor, input_tensors)
        final_output = outputs[isp.num_proxies-1]

        # Getting loss
        loss = nn.L1Loss(final_output, label_tensor)

        # Saving outputs
        #TODO: implement me!

if __name__ == '__main__':

    param_file = sys.argv[1]
    dng_path = sys.argv[2]
    input_path = sys.argv[3]
    label_path = sys.argv[4]
    output_path = sys.argv[5]

    # Adjusting for GPU usage
    use_gpu = torch.cuda.is_available()

    # Rendering ground truth labels using Darktable CLI
    generate_pipeline(
                     param_file, 
                     input_path, 
                     label_path, 
                     dng_path=dng_path
                     )

    # Getting pipeline outputs
    evaluate(
            param_file, 
            input_path, 
            label_path, 
            output_path, 
            use_gpu
            )