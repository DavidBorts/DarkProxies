import sys
sys.path.append('../')
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

# Local files
import Constants as c
from Loss_functions import losses
from Dataset import Darktable_Dataset
from Proxy_pipeline import ProxyPipeline
from Generate_data import generate_pipeline
from utils.regression import initial_guess, decide, project_param_values

# Constants
image_root_dir = c.IMAGE_ROOT_DIR

def regress(
            isp,
            orig_tensor, 
            param_tensors,
            optimizer,
            scheduler,
            num_iters,
            dtype,
            possible_values
            ):

    # MAE loss
    criterion = losses[c.STAGE_3_LOSS_FN]
    
    # Starting off with an initial guess
    # NOTE: this is a list of lists
    best_params = [initial_guess(vals) for vals in possible_values]
    print('Initial Parameters: ')
    print(best_params)
    
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

    # Regressing guess num_iters times
    for i in range(num_iters):

        scheduler.step()

        _, width, height = orig_tensor.size()
        num_input_channels = [c.NUM_IMAGE_CHANNEL + len(params) for params in best_params]

        # Fill in the best guess into the hyper-param tensor
        for param_tensor, params in zip(param_tensors, best_params):
            params_ndarray = np.array(params) #TODO: is this redundant?
            for param_idx, param in enumerate(params_ndarray):
                param_tensor.data[:, param_idx, :, :] = param

        # Assemble list of pipeline input tensors and fill them in
        # with the best guess for all hyper-parameter channels
        input_tensors = []
        for proxy_num in range(isp.num_proxies):
            input_tensor = torch.tensor((), \
                           dtype=torch.float).new_ones((1 , \
                            num_input_channels[proxy_num], \
                                width, height)).type(dtype)
            
            # Fill in hyper-parameter guesses
            if param_tensors[proxy_num] is not None:
                input_tensor.data[:, c.NUM_IMAGE_CHANNEL:, :, :] = param_tensors[proxy_num]
            input_tensors.append(input_tensor)
                
        # forward
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            # Processing through ISP
            #NOTE: Outputs is a list of every proxy output tensor
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
        for idx, param_tensor in enumerate(param_tensors):
            if param_tensor is None:
                continue
            _, num_params, _, _ = param_tensor.shape
            for param_idx in range(num_params):
                if dtype == torch.cuda.FloatTensor:
                    param_temp = param_tensor.cpu().clone().detach()[:, param_idx, :, :]
                else:
                    param_temp = param_tensor.clone().detach()[:, param_idx, :, :]
                
                best_params[idx][param_idx] = param_temp.mean(0).mean(0).mean(0).numpy()
        print('current params: ')
        print(best_params)
        sys.stdout.flush()
                    
        # Statistics
        print("\tIter {:d}/{:d}, Current Loss: {:3f}".format(i + 1, num_iters, loss.item()),end='\r')
        sys.stdout.flush()
                
    # At the end of finetuning, project onto acceptable parameters.
    for idx in range(isp.num_proxies):
        best_params[idx] = project_param_values(best_params[idx], possible_values[idx], finalize=False, dtype=dtype)
    print('Final Loss: {} \nFinal Parameters: {}\n'.format(loss.item(), best_params))

    if c.PIPELINE_CREATE_ANIMATION:       
        print('Animation completed and saved.')
    
    best_params = [param for params in best_params for param in params]
    return best_params

def regression_procedure(proxy_order, input_path, label_path, use_gpu):

    # Constants
    param_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_PARAM_DIR)
    num_iters = c.PIPELINE_REGRESSION_NUM_ITERS
    
    # Adjusting for GPU usage
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor

    # Differentiable ISP
    isp = ProxyPipeline(proxy_order, use_gpu)

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
    # NOTE: this is a list of lists
    possible_values = isp.possible_values

    # Initializing an empty matrix to store param guesses
    param_dims = [len(params) for params in possible_values]
    total_params_size = sum(param_dims) # total size of all params concatenated together
    best_params_mat = np.empty((total_params_size, len(image_dataset)))

    # Iterating over every ground truth image in the dataset
    for index in range(len(image_dataset)):
        
        name, orig_tensor, label_tensor = image_dataset[index]
        _, width, height = orig_tensor.size() 

        '''
        A list of N PyTorch Variables that contains every parameter guess
        Each Variable is B x Cn x W x H:
        N = Number of proxies in the ISP
        B = Batch size
        Cn = Number of hyper-parameter channels for proxy n
        W = Width
        H = Height
        These are the tensors that will be regressed
        NOTE: B is always 1, as this is an evaluation method
        '''
        param_tensors = [Variable(torch.tensor((), dtype=torch.float).new_ones((\
                       1, len(params), width, height)).type(dtype), requires_grad=True) \
                        if len(params) > 0 else None for params in isp.possible_values]

        print("Optimizing Hyperparameters for {} index {}".format(name, str(index)))
        print("-"*40)
        sys.stdout.flush()

        orig_tensor.unsqueeze_(0)
        label_tensor.unsqueeze_(0)

        optimizer = optim.Adam(param_tensors, lr=0.25)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        best_params_mat[:, index] = regress(
                                        isp,
                                        orig_tensor, 
                                        param_tensors,
                                        optimizer,
                                        scheduler,
                                        num_iters,
                                        dtype,
                                        possible_values
                                    )
    np.save(os.path.join(param_out_dir, f'pipeline_optimized_params.npy'), best_params_mat)
    print('Finished pipeline regression.')

if __name__ != '__main__':
    raise RuntimeError("This script is only configured to be called directly by the user!")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip", help="[OPTIONAL] Skip data generation", default=False, 
                    action=argparse.BooleanOptionalAction)
args = parser.parse_args()
skip_data = args.skip

config_path = os.path.join(c.IMAGE_ROOT_DIR, c.CONFIG_FILE)

# Creating stage 3 directory if it does not already exist
stage_3_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH)
if not os.path.exists(stage_3_path):
    os.mkdir(stage_3_path)
    print('Directory created at: ' + stage_3_path)

# Adjuting for gpu usage
use_gpu = torch.cuda.is_available()

# Getting paths to input and label data
input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_INPUT_DIR)
label_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, c.STAGE_3_OUTPUT_DIR)

# Reading in proxy order
proxy_order = []
with open(config_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        proxy, params, enable = line.split(' ')
        if int(enable) == 1:
            proxy_order.append((proxy, params))

# Generating data
if not skip_data:
    print("Generating data.")
    generate_pipeline(proxy_order, input_path, label_path)
else:
    print("Skipping data generation.")

# Running regression procedure
regression_procedure(proxy_order, input_path, label_path, use_gpu)