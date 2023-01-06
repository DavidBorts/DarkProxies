# Stage 1 code

import sys
sys.path.append('../')
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence

# Local files
from Models import UNet, train_model, load_checkpoint, eval
from Darktable_dataset import Darktable_Dataset
import Darktable_constants as c

# Constants
add_params = True
skip_connect = True
clip_output = True
num_epoch = c.PROXY_MODEL_NUM_EPOCH
use_checkpoint = True
learning_rate = 0.0001
gamma = 0.1
step_size = 7

'''
Run the stage 1 training and save model outputs
'''
def run_training_procedure(image_root_dir, model_out_dir, batch_size, num_epochs, use_gpu, possible_params, proxy_type, param, interactive):
    
    # Checking if user wants to train a proxy
    train = None
    while train != 'y' and train != 'n' and interactive:
        train = input('Continue to training? (y/n)\n')
    if train == 'n':
        skip = None
        while skip != 'y' and skip != 'n':
            skip = input('Skip to model finetuning? (y/n)\n')
        if skip == 'n':
            quit()
        else:
            return

    print('Begin proxy training (stage 1)')

    # Set up data loading.
    since = time.time()
    image_dataset = Darktable_Dataset(root_dir = image_root_dir, stage=1, proxy_type=proxy_type, param=param)
    train_loader = torch.utils.data.DataLoader(image_dataset, 
                                               batch_size=batch_size, 
                                               sampler=image_dataset.train_sampler,
                                               num_workers=1)#num_workers should be 4
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
        
    # Set up training regime.
    # criterion is the loss function, which can be nn.L1Loss() or nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Train the model.
    train_model(
        unet, 
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
        param
    )
    
    print(f'{proxy_type}: {param} proxy training completed.')

'''
Evaluate the model on a any input(s)
'''
def run_eval_procedure(image_root_dir, model_out_dir, use_gpu, params_file, possible_params, proxy_type, param):

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
    eval_loader = torch.utils.data.DataLoader(image_dataset, 
                                               batch_size=batch_size, 
                                               num_workers=1)#num_workers should be 4
    time_elapsed = time.time() - since
    print('Data Loader prepared in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()
    
    # Set up model.
    unet = UNet(num_input_channels=c.NUM_IMAGE_CHANNEL + len(possible_params),
                num_output_channels=c.NUM_IMAGE_CHANNEL, skip_connect=skip_connect, add_params=add_params, clip_output=clip_output)
    if use_checkpoint:
        start_epoch = load_checkpoint(unet, model_out_dir) #weight_out_dir
    if use_gpu:
        unet.cuda()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        
    # Set up training regime.
    # criterion is the loss function, which can be nn.L1Loss() or nn.MSELoss()
    criterion = nn.MSELoss()
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
