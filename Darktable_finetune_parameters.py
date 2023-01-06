# Stage 2 code

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

'''
Returns an initial guess vector. The values of the initial guess are determined randomly uniformly.
'''
def initial_guess(possible_values):
    x_0 = []
    for vals in possible_values:
        if type(vals) is list:
            x_0.append(vals[random.randint(0,len(vals)-1)])
        elif type(vals) is tuple:
            x_0.append(random.uniform(vals[0], vals[1]))
        else:
            raise TypeError('Possible values must be given as list (discrete) or tuple (continuous)')
    return np.array(x_0)
    
'''
Projects parameter values so that they lie in the valid range.
Inputs:
    unprojected_param_values: Numpy array of parameter values before projection.
    possible_values: List of possible values that parameters can take.
    finalize: Applies to discrete parameters. Set to True if you want to round discrete parameters to 
       to nearest valid integer. Set to False during training so that discrete parameters lie within
       smallest and largest integer.
    dtype: Datatype of the tensor. Either torch.FloatTensor, or torch.cuda.FloatTensor.
Outputs:
    projected_param_values: Numpy array of parameter values after projection.
'''
def project_param_values(unprojected_param_values, possible_values, finalize, dtype):
    def _project_onto_discrete(value, vals_list):
        array = np.array(vals_list)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def _project_onto_discrete_range(value, vals_list):
        if value < vals_list[0]:
            value = vals_list[0]
        if value > vals_list[-1]:
            value = vals_list[-1]
        return value
    
    def _project_onto_continuous(value, range_tuple):
        if value < range_tuple[0]:
            value = range_tuple[0]
        if value > range_tuple[1]:
            value = range_tuple[1]
        return value
    
    projected_param_values = []
    for i in range(len(possible_values)):
        vals = possible_values[i]
        if type(vals) == tuple:
            projected_param_values.append(_project_onto_continuous(unprojected_param_values[i], vals))
        elif finalize:
            projected_param_values.append(_project_onto_discrete(unprojected_param_values[i], vals))
        else:
            projected_param_values.append(_project_onto_discrete_range(unprojected_param_values[i], vals))
    return np.array(projected_param_values, dtype=np.float64)

'''
This method is needed to load a model weight file that was trained using DataParallel. For some reason the keys have
'module.' prefixed to them when trained using DataParallel, but not otherwise.
'''
def generic_load(model_weight_file):
    state_dict = torch.load(os.path.join(model_weight_file))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:6] == 'module':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

'''
Regime that finetunes parameters. 

orig_tensor is the original tensor
tm_tensor is the output of the tone-mapping operator tensor
input_tensor is the built up tensor consisting of both orig_tensor and parameter channels
'''
def finetune_parameters(model, orig_tensor, tm_tensor, input_tensor, proxy_type, param, ground_truth, possible_values, criterion, optimizer, scheduler, num_iters, dtype):
    
    # Starting off with an initial guess
    best_params = initial_guess(possible_values)
    print('Initial Parameters: {}'.format(best_params))
    
    # Tracking loss
    loss = None

    # Tracking the number of frames in the saved animation
    frame = 0
    
    # Path to save animation
    animations_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_PATH, f'{proxy_type}_{param}_' + c.ANIMATIONS_DIR)
    if not os.path.exists(animations_path):
        os.mkdir(animations_path)

    # Converting the tensors to the right datatype
    orig_tensor = orig_tensor.type(dtype)
    tm_tensor= tm_tensor.type(dtype)

    # Refining guess num_iters times
    for i in range(num_iters):

        scheduler.step()

        # Fill in the image
        input_tensor.data[:, 0:c.NUM_IMAGE_CHANNEL, :, :] = orig_tensor[:, 0:c.NUM_IMAGE_CHANNEL, :, :]
        
        # Fill in the guess to all hyper-parameter channels
        for param_idx in range(len(best_params)):
            best_param = np.array(best_params)[param_idx]
            input_tensor.data[:, c.NUM_IMAGE_CHANNEL+param_idx, :, :] = best_param
                
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion(outputs, tm_tensor)
            loss.backward()
            optimizer.step()

            # Saving outputs
            save_output = decide(i, num_iters)
            if save_output and c.CREATE_ANIMATION:
                frame += 1
                outputs_ndarray = outputs[0].detach().cpu().clone().numpy()
                outputs_ndarray = np.moveaxis(outputs_ndarray, 0, -1)
                outputs_path = os.path.join(animations_path, f'{proxy_type}_{param}_{ground_truth[:5]}_frame_{frame:04}.png')
                plt.imsave(outputs_path, outputs_ndarray, format='png')
            
        # Getting out current param values
        if dtype == torch.cuda.FloatTensor:
            input_tensor_temp = input_tensor.cpu().clone().detach()[:,3:,:,:]
        else:
            input_tensor_temp = input_tensor.clone().detach()[:,3:,:,:]
        best_params = input_tensor_temp.mean(0).mean(1).mean(1).tolist()
        print('current params: {}\n'.format(str(best_params)))
        best_params = project_param_values(best_params, possible_values, finalize=False, dtype=dtype)
                    
        # statistics
        print("\tIter {:d}/{:d}, Current Loss: {:3f}".format(i + 1, num_iters, loss.item()),end='\r')
        sys.stdout.flush()
                
    # At the end of finetuning, project onto acceptable parameters.
    best_params = project_param_values(best_params, possible_values, finalize=False, dtype=dtype)
    print('Final Loss: {} \nFinal Parameters: {}\n'.format(loss.item(), best_params))

    if c.CREATE_ANIMATION:       
        print('Animation saved.')
    
    return best_params

def decide(i, num_iters):
    if i == 0 or i == (num_iters - 1):
        return True
    
    save_output = i % (math.pow(i, 1.75) * (1 / num_iters))
    
    if int(save_output) == 0:
        return True
    else:
        return False

'''
Method that is called by main. An example of the inputs can be found in the main method.
'''
def run_finetune_procedure(
    image_root_dir, 
    param_out_dir,
    weight_out_dir, #model_weight_file,
    possible_values,
    num_iters,
    use_gpu,
    proxy_type,
    param,
    interactive
):

    # Checking if user wants to finetune params
    finetune = None
    while finetune is not 'n' and finetune is not 'y' and interactive:
        finetune = input('Proceed to finetuning? (y/n)\n')
    if finetune is 'n':
        print('quitting')
        quit()

    # Getting model weights
    weights_list = os.listdir(weight_out_dir)
    weights_list.sort(key=lambda x: float(x.strip(proxy_type + '_' + param + '_').strip('.pkl')))
    final_weights = weights_list[-1]
    model_weight_file = os.path.join(weight_out_dir, final_weights)
    print(f'Using model weights from {final_weights}.')

    # Creating directory to store optimized param guesses, if it does not already exist
    if not os.path.exists(param_out_dir):
        os.mkdir(param_out_dir)
        print('directory created at: ' + param_out_dir)

    # Adjusting for GPU usage
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor
    
    # Getting dataset
    print("Preparing dataset" )
    sys.stdout.flush()
    image_dataset = Darktable_Dataset(root_dir = image_root_dir, stage=2, proxy_type=proxy_type, param=param)
    #if c.SAVE_CROPS:
    #    image_dataset.save_crops()
    
    # Setting up U-net model
    print("Loading in model") 
    num_input_channels = c.NUM_IMAGE_CHANNEL + len(possible_values)
    state_dict = generic_load(model_weight_file)
    unet = UNet(num_input_channels=num_input_channels,
                num_output_channels=c.NUM_IMAGE_CHANNEL)
    unet.load_state_dict(state_dict)
    unet.eval()
    
    # Lock the weights in the U-Net
    for parameter in unet.parameters():
        parameter.requires_grad=False
        
    # Adjusting for GPU usage
    if use_gpu:
        unet.cuda()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)

    criterion = nn.L1Loss()

    # Getting the range of each param
    param_ranges = []
    for vals in possible_values:
        if type(vals) is tuple:
            param_ranges.append(np.absolute(vals[1] - vals[0]))
        else:
            param_ranges.append(np.absolute(vals[-1] - vals[0]))

    # Initializing an empty matrix to store param guesses
    best_params_mat = np.empty((len(possible_values), len(image_dataset)))

    for index in range(len(image_dataset)):
        # Run the optimization over 'input_tensor'. We will fill this tensor with the image and 
        # current hyper-parameter values at each training and validation iteration.
        name, orig_tensor, tm_tensor = image_dataset[index]
        _, width, height = orig_tensor.size() 

        input_tensor = Variable(torch.tensor((), dtype=torch.float).new_ones((1, \
                       num_input_channels, width, height)).type(dtype), requires_grad=True)

        print("Optimizing Hyperparameters for {} index {}".format(name, str(index)))
        print("-"*40)
        sys.stdout.flush()

        ground_truth = name.split('_')[3].strip('.tif')
        orig_tensor.unsqueeze_(0)
        tm_tensor.unsqueeze_(0)

        optimizer = optim.Adam([input_tensor], lr=0.25)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        best_params_mat[:, index] = finetune_parameters(
                                        unet, 
                                        orig_tensor, 
                                        tm_tensor,
                                        input_tensor,
                                        proxy_type,
                                        param,
                                        ground_truth,
                                        possible_values,
                                        criterion,
                                        optimizer,
                                        scheduler,
                                        num_iters,
                                        dtype
                                    )
    np.save(os.path.join(param_out_dir, f'{proxy_type}_{param}_optimized_params.npy'), best_params_mat)
