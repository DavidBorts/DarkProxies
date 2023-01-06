import sys
sys.path.append('../')
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn 
import torch.optim as optim
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

# Local files
import Darktable_constants as c
from Models import UNet
from Darktable_dataset import Darktable_Dataset
from Darktable_pipeline import proxyPipeline
from Darktable_generate_data import generate_pipeline
from Darktable_finetune_parameters import initial_guess, decide, project_param_values

# Constants
image_root_dir = c.IMAGE_ROOT_DIR
num_iters = c.PIPELINE_REGRESSION_NUM_ITERS

def regress(
            isp,
            proxy_order,
            orig_tensor, 
            label_tensor,
            param_tensor,
            optimizer,
            scheduler,
            num_iters,
            name,
            dtype,
            possible_values
            ):

    # MAE loss
    criterion = nn.L1Loss()
    
    # Starting off with an initial guess
    best_params = initial_guess(possible_values)
    print('Initial Parameters: ')
    print(best_params)

    # Getting parameter ranges
    param_ranges = []
    for vals in possible_values:
        if type(vals) is tuple:
            param_ranges.append(np.absolute(vals[1] - vals[0]))
        else:
            param_ranges.append(np.absolute(vals[-1] - vals[0]))
    
    # Tracking loss
    loss = None

    # Tracking the number of frames in the saved animation
    frame = 0
    
    # Path to save animation
    animations_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, 'pipeline' + c.ANIMATIONS_DIR)
    if not os.path.exists(animations_path):
        os.mkdir(animations_path)
        print('Animation directory created at: ' + animations_path)

    # Converting the tensors to the correct datatype
    orig_tensor = orig_tensor.type(dtype)
    tm_tensor = tm_tensor.type(dtype)

    # Refining guess num_iters times
    for i in range(num_iters):

        scheduler.step()

        _, width, height = orig_tensor.size()
        num_input_channels = c.NUM_IMAGE_CHANNEL + 1

        # Fill in the best guess into the hyper-param tensor
        for param_idx in range(len(best_params)):
            best_param = np.array(best_params)[param_idx]
            param_tensor.data[param_idx, :, :, :, :] = best_param

        # Create list of pipeline input tensors and fill them in
        # with the best guess for all hyper-parameter channels
        #TODO: does this support multiple params per proxy?
        input_tensors = []
        for proxy_num in range(isp.num_proxies):
            input_tensor = torch.tensor((), \
                           dtype=torch.float).new_ones((1 , \
                            num_input_channels, width, height)).type(dtype)
            
            # Fill in hyper-parameter guesses
            best_param = np.array(best_params)[proxy_num]
            input_tensor.data[:, c.NUM_IMAGE_CHANNEL, :, :] = param_tensor[proxy_num, :, :, :, :].squeeze(0)#FIXME ?
            
            input_tensors.append(input_tensor)
                
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            # Processing through ISP
            # Outputs is a list of every proxy output tensor
            outputs = isp.process(orig_tensor, input_tensors)

            loss = criterion(outputs[-1], tm_tensor)
            loss.backward()
            optimizer.step()

            # Saving outputs
            save_output = decide(i, num_iters)
            if save_output and c.PIPELINE_CREATE_ANIMATION:
                frame += 1
                outputs_ndarray = outputs[-1].detach().cpu().clone().numpy()
                outputs_ndarray = np.moveaxis(outputs_ndarray, 0, -1)
                outputs_path = os.path.join(animations_path, f'pipeline_frame_{frame:04}.png')
                plt.imsave(outputs_path, outputs_ndarray, format=c.PIPE_REGRESSION_ANIMATION_FORMAT)
            
        # Getting out current best param values
        for param in isp.num_proxies:
            if dtype == torch.cuda.FloatTensor:
                param_temp = param_tensor.cpu().clone().detach()[param, :, :, :, :]
            else:
                param_temp = param_tensor.clone().detach()[param, :, :, :, :]

            best_params[param] = param_temp.mean(0).mean(0).mean(0).mean(0).numpy()
        print('current params: ')
        print(best_params)
                    
        # statistics
        print("\tIter {:d}/{:d}, Current Loss: {:3f}".format(i + 1, num_iters, loss.item()),end='\r')
        sys.stdout.flush()
                
    # At the end of finetuning, project onto acceptable parameters.
    best_params = project_param_values(best_params, possible_values, finalize=False, dtype=dtype)
    print('Final Loss: {} \nFinal Parameters: {}\n'.format(loss.item(), best_params))

    if c.PIPELINE_CREATE_ANIMATION:       
        print('Animation saved.')
    
    return best_params

def regression_procedure(input_path, label_path, use_gpu):

    # Constants
    param_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_PARAM_DIR)
    interactive = c.INTERACTIVE

    # Checking if user wants to regress params
    regress = None
    while regress is not 'n' and regress is not 'y' and interactive:
        regress = input('Proceed to pipeline regression? (y/n)\n')
    if regress is 'n':
        print('quitting')
        quit()
    
    # Adjusting for GPU usage
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor

    # Reading in proxy order
    proxy_order = []
    with open(os.path.join(c.IMAGE_ROOT_DIR, c.CONFIG_FILE), 'r') as file:
        lines = file.readlines()
        for line in lines:
            proxy, enable = line.split(' ')
            if int(enable) == 1:
                proxy_order.append(proxy)

    
    # Differentiable ISP
    isp = proxyPipeline(proxy_order, use_gpu)

    # Creating any directories that do nto already exist
    if not os.path.exists(param_out_dir):
        os.mkdir(param_out_dir)
        print('Directory for regressed params created at: ' + param_out_dir)
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Input directory created at: ' + input_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)
        print('Label directory created at: ' + label_path)
    
    # Setting up dataset
    print("Preparing dataset" )
    sys.stdout.flush()
    image_dataset = Darktable_Dataset(
                                      root_dir = c.IMAGE_ROOT_DIR, 
                                      stage=3, 
                                      input_dir=input_path,
                                      output_dir=label_path
                                      )

    # Getting possible param values
    #TODO: does this support multiple params per proxy?
    possible_values = isp.possible_values

    # Initializing an empty matrix to store param guesses
    #TODO: does this support multiple params per proxy?
    best_params_mat = np.empty((isp.num_proxies, len(image_dataset)))

    # Iterating over every ground truth image in the dataset
    for index in range(len(image_dataset)):
        
        name, orig_tensor, label_tensor = image_dataset[index]
        _, width, height = orig_tensor.size() 

        # Creating a PyTorch Variable that contains every parameter guess
        # It is N x B x C x W x H:
        # N = Number of proxies in the ISP
        # B = Batch size
        # C = Number of hyper-parameter channels (currently this can only be 1)
        # W = Width
        # H = Height
        # This is the tensor that will be regressed
        #TODO: does this support multiple params per proxy?
        param_tensor = Variable(torch.tensor((), dtype=torch.float).new_ones((isp.num_proxies, \
                       1, 1, width, height)).type(dtype), requires_grad=True)

        print("Optimizing Hyperparameters for {} index {}".format(name, str(index)))
        print("-"*40)
        sys.stdout.flush()

        orig_tensor.unsqueeze_(0)
        label_tensor.unsqueeze_(0)

        optimizer = optim.Adam([param_tensor], lr=0.25)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        best_params_mat[:, index] = regress(
                                        isp,
                                        proxy_order,
                                        orig_tensor, 
                                        label_tensor,
                                        param_tensor,
                                        optimizer,
                                        scheduler,
                                        num_iters,
                                        name,
                                        dtype,
                                        possible_values
                                    )
    np.save(os.path.join(param_out_dir, f'pipeline_optimized_params.npy'), best_params_mat)
    print('Finished pipeline regression.')

if __name__ == '__main__':

    # Creating stage 3 directory if it does not already exist
    stage_3_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH)
    if not os.path.exists(stage_3_path):
        os.mkdir(stage_3_path)
        print('Directory created at: ' + stage_3_path)

    # Adjuting for gpu usage
    use_gpu = torch.cuda.is_available()

    # Arguments
    param_file = sys.argv[1]

    # Getting paths to input and label data
    input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_INPUT_DIR)
    label_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_OUTPUT_DIR)

    # Generating data
    generate_pipeline(param_file, input_path, label_path)

    # Running regression procedure
    regression_procedure(input_path, label_path, use_gpu)