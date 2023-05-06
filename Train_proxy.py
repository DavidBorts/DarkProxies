# Various functions to train proxies (stage 1)

import sys
sys.path.append('../') # TODO: is this necessary?
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Local files
from Models import UNet, ChenNet, DemosaicNet, train_model, load_checkpoint, eval
from Dataset import Darktable_Dataset
from Loss_functions import losses
import Constants as c

# Constants
# TODO: move these into Constants.py
skip_connect = True
clip_output = c.CLIP_OUTPUT
num_epoch = c.PROXY_MODEL_NUM_EPOCH
use_checkpoint = True
learning_rate = 0.0001 # was 0.0001
gamma = 0.1 # was 0.1
step_size = 1000 # was 7

def get_num_input_channels(proxy_type, possible_params, append_params):
    '''
    TODO: add comment
    '''
    # Bayer mosaics are always unpacked into 4 channels
    if proxy_type == "demosaic":
        return 4
    
    num_channels = c.NUM_IMAGE_CHANNEL # input tensor
    params_size = (len(possible_params), c.IMG_SIZE, c.IMG_SIZE) # tuple size of the appended params
    
    # Depending on c.EMBEDDING_TYPE, some number of parameter
    # channels might be appended to the input tensor
    #TODO: move this into Models.py
    if append_params:
        num_params = len(possible_params)
        embedding_type = c.EMBEDDING_TYPES[c.EMBEDDING_TYPE]

        if embedding_type == "none":
            num_channels += len(possible_params)

        elif embedding_type == "linear_to_channel":
            channels = int(np.ceil(float(num_params) / c.EMBEDDING_RATIO))
            if c.EMBED_TO_SINGLE:
                channels = 1
            params_size = (channels, c.IMG_SIZE/2, c.IMG_SIZE/2)
            num_channels += channels

        else: # embedding type is "linear_to_value"
            final = int(np.ceil(num_params*1.0 / c.EMBEDDING_RATIO))
            if c.EMBED_TO_SINGLE:
                final = 1
            params_size = (final, c.IMG_SIZE, c.IMG_SIZE)
            num_channels += final
    return num_channels, params_size


def run_training_procedure(model_out_dir, batch_size, num_epochs, use_gpu, possible_params, proxy_type, params, append_params, name, dataset_name, gt_list=None):
    '''
    Sets up the dataset, Dataloader, model, and training regime, then begins training.

    inputs:
        [model_out_dir]: Path to directory at which to save model weights
        [batch_size]: Training batch size
        [num_epochs]: Number of epochs for which to train
        [use_gpu]: (Boolean) whether or not GPU is available
        [possible_params]: (list of tuples of floats) ranges of each input parameter
        [proxy_type]: Name of the block to learn
        [params]: if None, train proxy on full parameter space, else train only on params list
        [append_params]: (Boolean) whether or not the given proxy has input parameters
        [name]: Prefix used for all directories/files corresponding to this specific proxy
        [dataset_name]: Name of custom pre-existing dataset to train on
        [gt_list]: (OPTIONAL) list of ground truth image names (in correct order)
    '''

    # Constants
    image_root_dir = c.IMAGE_ROOT_DIR
    
    # Checking if user wants to train a proxy
    interactive = c.INTERACTIVE
    train = None
    while train not in ['y', 'n', 'Y', 'N'] and interactive:
        train = input('Continue to training? (y/n)\n')
    if train in ['n', 'N']:
        skip = None
        while skip not in ['y', 'n', 'Y', 'N']:
            skip = input('Skip to model finetuning? (y/n)\n')
        if skip in ['n', 'N']:
            print('quitting.')
            quit()
        else:
            return

    print('Begin proxy training (stage 1)')

    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
        print('directory created at: ' + model_out_dir)

    # Set up data loading.
    since = time.time()
    if dataset_name is not None:
        print(f'Loading data from: {dataset_name}')
        image_dataset = Darktable_Dataset(image_root_dir, 
                                        1, 
                                        proxy_type, 
                                        params, 
                                        not append_params,
                                        dataset_name,
                                        param_ranges=possible_params,
                                        gt_list=gt_list)
    else:
        image_dataset = Darktable_Dataset(image_root_dir, 
                                        1, 
                                        proxy_type, 
                                        params, 
                                        not append_params,
                                        name,
                                        param_ranges=possible_params,
                                        gt_list=gt_list)
    train_loader = torch.utils.data.DataLoader(image_dataset, 
                                               batch_size=batch_size, 
                                               sampler=image_dataset.train_sampler,
                                               num_workers=1)
    val_loader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=batch_size,
                                             sampler=image_dataset.val_sampler,
                                             num_workers=1)
    time_elapsed = time.time() - since
    print('Data Loaders prepared in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {
        'train': len(image_dataset.train_sampler),
        'val': len(image_dataset.val_sampler)
    }
    
    # Setting up model
    num_channels, params_size = get_num_input_channels(proxy_type, possible_params, append_params)
    model = None
    if proxy_type == "demosaic":
        #model = DemosaicNet(num_input_channels=num_channels, num_output_channels=12,
                            #skip_connect=skip_connect, clip_output=clip_output)
        model = ChenNet(0, clip_output=clip_output, add_params=False)
    else:
        channel_list = [32, 64, 128, 256, 512]
        if c.NPF_BASELINE:
            channel_list = [16, 32, 64, 128, 256]
        if append_params:
            model = UNet(num_input_channels=num_channels, num_output_channels=c.NUM_IMAGE_CHANNEL, 
                    skip_connect=skip_connect, add_params=append_params, clip_output=clip_output,
                    num_params=len(possible_params), params_size=params_size, channel_list=channel_list)
        else:
             model = UNet(num_input_channels=num_channels, num_output_channels=c.NUM_IMAGE_CHANNEL, 
                    skip_connect=skip_connect, add_params=append_params, clip_output=clip_output,
                    channel_list=channel_list)
    if use_checkpoint:
        start_epoch = load_checkpoint(model, model_out_dir) #weight_out_dir
    else:
        start_epoch = 0 

    # GPU & DataParallel configuration
    if use_gpu:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    # Setting up training regime.
    # (criterion is the loss function, as set in Constants.py)
    criterion = None	
    if c.WHICH_LOSS[proxy_type][0] == "Perceptual":	
        criterion = losses[c.WHICH_LOSS[proxy_type][0]](nn.MSELoss(), use_gpu)	
    else:	
        criterion = losses[c.WHICH_LOSS[proxy_type][0]]()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Train the model.
    train_model(
        model, 
        dataloaders, 
        dataset_sizes, 
        criterion, 
        optimizer, 
        scheduler,
        model_out_dir,
        num_epochs,
        start_epoch,
        use_gpu,
        proxy_type,
        params,
        name
    )

    print(f'{name}: proxy training completed.')

'''
Evaluate the model on a any input(s)
'''
# TODO: add suppport for colorin & colorout (+ demosaic)
# TODO: move this to eval file?
# TODO: bring up to speed with run_training_procedure
def run_eval_procedure(image_root_dir, model_out_dir, use_gpu, params_file, possible_params, proxy_type, param, append_params):

    # Constants
    input_path = os.path.join(c.IMAGE_ROOT_DIR, c.EVAL_PATH, f'{proxy_type}_{param}_input')
    output_path = os.path.join(c.IMAGE_ROOT_DIR, c.EVAL_PATH, f'{proxy_type}_{param}_output')

    # Creating input and output directories, if they do not already exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Directory created at: ' + output_path)
    
    # Set up data loading.
    since = time.time()
    image_dataset = Darktable_Dataset(root_dir = image_root_dir, stage=1, proxy_type=proxy_type, param=param, input_dir=input_path, output_dir=output_path, params_file=params_file)
    eval_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, num_workers=1)
    time_elapsed = time.time() - since
    print('Data Loader prepared in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()
    
    # Set up model.
    unet = UNet(num_input_channels=c.NUM_IMAGE_CHANNEL + len(possible_params),
                num_output_channels=c.NUM_IMAGE_CHANNEL, skip_connect=skip_connect, add_params=append_params, clip_output=clip_output)
    if use_checkpoint:
        start_epoch = load_checkpoint(unet, model_out_dir) #weight_out_dir
    if use_gpu:
        unet.cuda()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        
    # Set up training regime.
    # criterion is the loss function, which can be nn.L1Loss() or nn.MSELoss()
    #criterion = nn.MSELoss()
    criterion = losses[c.WHICH_LOSS[proxy_type]]()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    eval(
        unet, 
        eval_loader, 
        criterion, 
        optimizer, 
        use_gpu, 
        proxy_type,
        param
    )
