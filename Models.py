# Code related to models and their definitions and training is located here.
# The FCNet is a fully connected network.
# The UNet definition is in here.
# The ChenNet allows for processing RAW images (RGGB).

import copy
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import Darktable_constants as c

DEFAULT_UNET_CHANNEL_LIST = [32, 64, 128, 256, 512] # Same as in 'Learning to see in the Dark'
DEFAULT_DEMOSAICNET_CHANNEL_LIST = [32, 64, 128, 256, 512] # Same as in 'Learning to see in the Dark'
DEFAULT_FCNET_CHANNEL_LIST = [32, 64, 128, 256, 256, 128, 64, 32, 1]

def fc_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )

class FCNet(nn.Module):
    def __init__(self, num_input_channels, channel_list = DEFAULT_FCNET_CHANNEL_LIST):
        super().__init__()
        if not len(channel_list) == 9:
            raise ValueError('channel_list argument should be a list of 9.')
        self.fc_1 = fc_relu(num_input_channels, channel_list[0]) 
        self.fc_2 = fc_relu(channel_list[0], channel_list[1]) 
        self.fc_3 = fc_relu(channel_list[1], channel_list[2]) 
        self.fc_4 = fc_relu(channel_list[2], channel_list[3]) 
        self.fc_5 = fc_relu(channel_list[3], channel_list[4]) 
        self.fc_6 = fc_relu(channel_list[4], channel_list[5]) 
        self.fc_7 = fc_relu(channel_list[5], channel_list[6]) 
        self.fc_8 = fc_relu(channel_list[6], channel_list[7]) 
        self.fc_9 = nn.Linear(channel_list[7], channel_list[8])

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)
        x = self.fc_5(x)
        x = self.fc_6(x)
        x = self.fc_7(x)
        x = self.fc_8(x)
        x = self.fc_9(x)
        return x


'''
Basic building block of the UNet
'''
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

'''
Basic building block of the DemosaicNet
'''
def double_conv_leaky(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True, negative_slope=0.2)
    )

'''
UNet implementation. This UNet allows for a single skip connection from input to output.
'''
class UNet(nn.Module):   
    def __init__(self, num_input_channels, num_output_channels, skip_connect=True, add_params=True, clip_output=True, channel_list=DEFAULT_UNET_CHANNEL_LIST):
        super().__init__()
        if not len(channel_list) == 5:
            raise ValueError('channel_list argument should be a list of integers of size 5.')
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_connect = skip_connect
        self.add_params = add_params
        self.clip_output = clip_output

        if self.add_params:
            print('Appending parameters onto every layer.')
            # Downsamples for param channels.
            in_channel_list = []
            for channel in channel_list:
                self.down1 = nn.AvgPool2d(2)
                self.down2 = nn.AvgPool2d(2)
                self.down3 = nn.AvgPool2d(2) 
                self.down4 = nn.AvgPool2d(2)
                in_channel_list.append(channel + num_input_channels - num_output_channels) 
        else:
            in_channel_list = channel_list 
        
        if self.skip_connect:
            print('Adding skip connection from input to output.')
        if self.clip_output:
            print('Clipping output of model.')
            
        # Down
        self.conv_down_1 = double_conv(self.num_input_channels, channel_list[0])              # num_input_channels -> 32
        self.maxpool_1 = nn.MaxPool2d(2)
        self.conv_down_2 = double_conv(in_channel_list[0], channel_list[1])                      # 32 (52)  -> 64
        self.maxpool_2 = nn.MaxPool2d(2)
        self.conv_down_3 = double_conv(in_channel_list[1], channel_list[2])                      # 64 (84) -> 128
        self.maxpool_3 = nn.MaxPool2d(2)
        self.conv_down_4 = double_conv(in_channel_list[2], channel_list[3])                      # 128 (148) -> 256
        self.maxpool_4 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = double_conv(in_channel_list[3], channel_list[4])                           # 256 (276) -> 512
        
        # Up: Currently, we do not append hyperparameters on the upsampling portion.
        # Can use this for upsample: self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.ConvTranspose2d(channel_list[4], channel_list[3], 2, stride=2)   # 512 -> 256
        self.conv_up_4 = double_conv(channel_list[4], channel_list[3])                     # 256 + 256 (+ 20) -> 256

        self.upsample_3 = nn.ConvTranspose2d(channel_list[3], channel_list[2], 2, stride=2)   # 256 -> 128
        self.conv_up_3 = double_conv(channel_list[3], channel_list[2])                     # 128 + 128 (+ 20) -> 128

        self.upsample_2 = nn.ConvTranspose2d(channel_list[2], channel_list[1], 2, stride=2)   # 128 -> 64
        self.conv_up_2 = double_conv(channel_list[2], channel_list[1])                     # 64 + 64 (+ 20) -> 64

        self.upsample_1 = nn.ConvTranspose2d(channel_list[1], channel_list[0], 2, stride=2)   # 64  -> 32
        self.conv_up_1 = double_conv(channel_list[1], channel_list[0])                     # 32 + 32 (+ 20) -> 32

        # Final
        # Note that in 'Learning to see in the Dark' the authors use the Tensorflow 'SAME' padding to ensure
        # that output dimensions are maintained. However, here we need to determine the padding ourselves, which
        # is why the final convolution does not have any padding.
        self.conv_final = nn.Conv2d(channel_list[0], self.num_output_channels, 1, padding=0)  # 32  -> num_output_channels 

    def forward(self, x):
        if self.add_params:
            # Get param channels first.
            param_half = self.down1(x[:,3:,:,:])
            param_fourth = self.down2(param_half)
            param_eighth = self.down3(param_fourth)
            param_sixteenth = self.down4(param_eighth)

        # Down
        conv1  = self.conv_down_1(x)
        pool1  = self.maxpool_1(conv1)
        if self.add_params: 
            pool1 = torch.cat([pool1, param_half], dim=1)

        conv2  = self.conv_down_2(pool1)
        pool2  = self.maxpool_2(conv2)
        if self.add_params:
            pool2 = torch.cat([pool2, param_fourth], dim=1)        

        conv3  = self.conv_down_3(pool2)
        pool3  = self.maxpool_3(conv3)
        if self.add_params:
            pool3 = torch.cat([pool3, param_eighth], dim=1)
        
        conv4  = self.conv_down_4(pool3)
        pool4  = self.maxpool_4(conv4)
        if self.add_params:
            pool4 = torch.cat([pool4, param_sixteenth], dim=1)        

        # Bridge
        conv5  = self.bridge(pool4)
        
        # Up
        up6    = self.upsample_4(conv5)
        #print('merge6')
        #print(conv4.size())
        #print(up6.size())
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6  = self.conv_up_4(merge6)
        
        up7    = self.upsample_3(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7  = self.conv_up_3(merge7)
        
        up8    = self.upsample_2(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8  = self.conv_up_2(merge8)
        
        up9    = self.upsample_1(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9  = self.conv_up_1(merge9)
        
        # Final
        out    = self.conv_final(conv9)
        if self.skip_connect:
            out =  out + x[:,0:self.num_output_channels,:,:] # Skip connection from input to output
        
        if self.clip_output:
            return torch.min(torch.max(out, torch.zeros_like(out)), torch.ones_like(out))
        
        return out
    
    
'''
ChenNet. Implementation of the net from 'Learning to see in the Dark' using a UNet for the conv step.
'''
class ChenNet(nn.Module):
    def __init__(self, num_params, clip_output=True, add_params=True, channel_list=DEFAULT_UNET_CHANNEL_LIST):        
        super().__init__()
        if not len(channel_list) == 5:
            raise ValueError('channel_list argument should be a list of integers of size 5.')
        self.num_input_channels = 4+num_params
        self.num_output_channels = 12
        self.clip_output = clip_output
        self.add_params = add_params 

        if self.add_params:
            print('Appending parameters onto every down layer.')
            # Downsamples for param channels.
            self.down1 = nn.AvgPool2d(2)
            self.down2 = nn.AvgPool2d(2)
            self.down3 = nn.AvgPool2d(2) 
            self.down4 = nn.AvgPool2d(2)
            in_channel_list = []
            for channel in channel_list:
                in_channel_list.append(channel + num_params)
        else:
            in_channel_list = channel_list
            
        if self.clip_output:
            print('Clipping output of model.')
        
        # Down
        self.conv_down_1 = double_conv(self.num_input_channels, channel_list[0])              # num_input_channels -> 32
        self.maxpool_1 = nn.MaxPool2d(2)
        self.conv_down_2 = double_conv(in_channel_list[0], channel_list[1])                      # 32 (52)  -> 64
        self.maxpool_2 = nn.MaxPool2d(2)
        self.conv_down_3 = double_conv(in_channel_list[1], channel_list[2])                      # 64 (84) -> 128
        self.maxpool_3 = nn.MaxPool2d(2)
        self.conv_down_4 = double_conv(in_channel_list[2], channel_list[3])                      # 128 (148) -> 256
        self.maxpool_4 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = double_conv(in_channel_list[3], channel_list[4])                           # 256 (276) -> 512
        
        # Up
        # Can use this for upsample: self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.ConvTranspose2d(channel_list[4], channel_list[3], 2, stride=2)   # 512 -> 256
        self.conv_up_4 = double_conv(channel_list[4], channel_list[3])                     # 256 + 256 (+ 20) -> 256
        self.upsample_3 = nn.ConvTranspose2d(channel_list[3], channel_list[2], 2, stride=2)   # 256 -> 128
        self.conv_up_3 = double_conv(channel_list[3], channel_list[2])                     # 128 + 128 (+ 20) -> 128
        self.upsample_2 = nn.ConvTranspose2d(channel_list[2], channel_list[1], 2, stride=2)   # 128 -> 64
        self.conv_up_2 = double_conv(channel_list[2], channel_list[1])                     # 64 + 64 (+ 20) -> 64
        self.upsample_1 = nn.ConvTranspose2d(channel_list[1], channel_list[0], 2, stride=2)   # 64  -> 32
        self.conv_up_1 = double_conv(channel_list[1], channel_list[0])                     # 32 + 32 (+ 20) -> 32

        # Final
        # Note that in 'Learning to see in the Dark' the authors use the Tensorflow 'SAME' padding to ensure
        # that output dimensions are maintained. However, here we need to determine the padding ourselves, which
        # is why the final convolution does not have any padding.
        self.conv_final = nn.Conv2d(channel_list[0], self.num_output_channels, 1, padding=0)  # 32  -> num_output_channels
        self.pixel_shuffle = nn.PixelShuffle(2) # Used to un-tile the 12 channels into an sRGB image with 3 channels
        
    def forward(self, x):
        if self.add_params:
            # Get param channels first.
            param_half = self.down1(x[:,4:,:,:])
            param_fourth = self.down2(param_half)
            param_eighth = self.down3(param_fourth)
            param_sixteenth = self.down4(param_eighth)

        # Down
        conv1  = self.conv_down_1(x)
        pool1  = self.maxpool_1(conv1)
        if self.add_params: 
            pool1 = torch.cat([pool1, param_half], dim=1)

        conv2  = self.conv_down_2(pool1)
        pool2  = self.maxpool_2(conv2)
        if self.add_params:
            pool2 = torch.cat([pool2, param_fourth], dim=1)        

        conv3  = self.conv_down_3(pool2)
        pool3  = self.maxpool_3(conv3)
        if self.add_params:
            pool3 = torch.cat([pool3, param_eighth], dim=1)
        
        conv4  = self.conv_down_4(pool3)
        pool4  = self.maxpool_4(conv4)
        if self.add_params:
            pool4 = torch.cat([pool4, param_sixteenth], dim=1)        

        # Bridge
        conv5  = self.bridge(pool4)
        
        # Up
        up6    = self.upsample_4(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6  = self.conv_up_4(merge6)
        
        up7    = self.upsample_3(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7  = self.conv_up_3(merge7)
        
        up8    = self.upsample_2(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8  = self.conv_up_2(merge8)
        
        up9    = self.upsample_1(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9  = self.conv_up_1(merge9)
        
        # Final
        out    = self.conv_final(conv9)
        out    = self.pixel_shuffle(out)
        
        if self.clip_output:
            return torch.min(torch.max(out, torch.zeros_like(out)), torch.ones_like(out))
        
        return out

class DemosaicNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, skip_connect=True, clip_output=True, channel_list=DEFAULT_DEMOSAICNET_CHANNEL_LIST):
        super().__init__()
        if not len(channel_list) == 5:
            raise ValueError('channel_list argument should be a list of integers of size 5.')
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_connect = skip_connect
        self.clip_output = clip_output

        in_channel_list = channel_list
        
        if self.skip_connect:
            print('Adding skip connection from input to output.')
        if self.clip_output:
            print('Clipping output of model.')
            
        # Down
        self.conv_down_1 = double_conv_leaky(self.num_input_channels, channel_list[0])              # num_input_channels -> 32
        self.maxpool_1 = nn.MaxPool2d(2)
        self.conv_down_2 = double_conv_leaky(in_channel_list[0], channel_list[1])                      # 32 (52)  -> 64
        self.maxpool_2 = nn.MaxPool2d(2)
        self.conv_down_3 = double_conv_leaky(in_channel_list[1], channel_list[2])                      # 64 (84) -> 128
        self.maxpool_3 = nn.MaxPool2d(2)
        self.conv_down_4 = double_conv_leaky(in_channel_list[2], channel_list[3])                      # 128 (148) -> 256
        self.maxpool_4 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = double_conv_leaky(in_channel_list[3], channel_list[4])                           # 256 (276) -> 512
        
        # Up: Currently, we do not append hyperparameters on the upsampling portion.
        # Can use this for upsample: self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.ConvTranspose2d(channel_list[4], channel_list[3], 2, stride=2)   # 512 -> 256
        self.conv_up_4 = double_conv_leaky(channel_list[4], channel_list[3])                     # 256 + 256 (+ 20) -> 256

        self.upsample_3 = nn.ConvTranspose2d(channel_list[3], channel_list[2], 2, stride=2)   # 256 -> 128
        self.conv_up_3 = double_conv_leaky(channel_list[3], channel_list[2])                     # 128 + 128 (+ 20) -> 128

        self.upsample_2 = nn.ConvTranspose2d(channel_list[2], channel_list[1], 2, stride=2)   # 128 -> 64
        self.conv_up_2 = double_conv_leaky(channel_list[2], channel_list[1])                     # 64 + 64 (+ 20) -> 64

        self.upsample_1 = nn.ConvTranspose2d(channel_list[1], channel_list[0], 2, stride=2)   # 64  -> 32
        self.conv_up_1 = double_conv_leaky(channel_list[1], channel_list[0])                     # 32 + 32 (+ 20) -> 32

        # Final
        # Note that in 'Learning to see in the Dark' the authors use the Tensorflow 'SAME' padding to ensure
        # that output dimensions are maintained. However, here we need to determine the padding ourselves, which
        # is why the final convolution does not have any padding.
        self.conv_final = nn.Conv2d(channel_list[0], self.num_output_channels, 1, padding=0)  # 32  -> num_output_channels 

    def forward(self, x):

        # Down
        #print("forward(x) size: " + str(x.size()))
        conv1  = self.conv_down_1(x)
        pool1  = self.maxpool_1(conv1)

        conv2  = self.conv_down_2(pool1)
        pool2  = self.maxpool_2(conv2)       

        conv3  = self.conv_down_3(pool2)
        pool3  = self.maxpool_3(conv3)
        
        conv4  = self.conv_down_4(pool3)
        pool4  = self.maxpool_4(conv4)       

        # Bridge
        conv5  = self.bridge(pool4)
        
        # Up
        up6    = self.upsample_4(conv5)

        merge6 = torch.cat([conv4, up6], dim=1)
        conv6  = self.conv_up_4(merge6)
        
        up7    = self.upsample_3(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7  = self.conv_up_3(merge7)
        
        up8    = self.upsample_2(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8  = self.conv_up_2(merge8)
        
        up9    = self.upsample_1(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9  = self.conv_up_1(merge9)
        
        # Final
        out    = self.conv_final(conv9)
        if self.skip_connect:
            out =  out + x[:,0:self.num_output_channels,:,:] # Skip connection from input to output
        
        if self.clip_output:
            return torch.min(torch.max(out, torch.zeros_like(out)), torch.ones_like(out))
        
        # Unpacking channels into 3-channel RGB image
        out = f.pixel_shuffle(out, 2)
        
        return out

class PerceptualLoss():
    def __init__(self, loss, use_gpu):
        self.criterion = loss
        self.vgg_func= self.content_func(use_gpu)


    def content_func(self, use_gpu):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        if use_gpu:
            cnn = cnn.cuda()
        model = nn.Sequential()
        if use_gpu:
            model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
        
    def get_loss(self, output, label):
        f_out = self.vgg_func.forward(output)
        f_label = self.vgg_func.forward(label)
        f_label_no_grad = f_label.detach()
        loss = self.criterion(f_out, f_label_no_grad)
        return loss

    def __call__(self, output, label):
        return self.get_loss(output, label)

'''
This method is needed to load a model weight file that was trained using DataParallel. For some reason the keys have
'module.' prefixed to them when trained using DataParallel, but not otherwise.
'''
def generic_load(model_weight_dir, model_weight_file):
    state_dict = torch.load(os.path.join(model_weight_dir, model_weight_file))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:6] == 'module':
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v 
    return new_state_dict


def load_checkpoint(model, model_weight_dir):
    print('Loading from Checkpoint.')
    # Find the latest checkpoint
    saved_weight_filenames = os.listdir(model_weight_dir)
    latest_weight = ''
    latest_epoch = 0
    for weight_filename in saved_weight_filenames:
        if '.pkl' in weight_filename:
            print('weight filename: ' + weight_filename)
            epoch = int(weight_filename.split('_')[2][0:-4]) + 1
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_weight = weight_filename
    # If no checkpoint is found, return without doing anything.
    if latest_weight == '':
        print('No Checkpoint found.')
        latest_epoch = 0
        return latest_epoch
    print('Loading from {}/{}'.format(model_weight_dir, latest_weight))
    state_dict = generic_load(model_weight_dir, latest_weight)
    model.load_state_dict(state_dict)
    return latest_epoch

'''
Training routine for the model. For example of inputs, go to Darktable_train_proxy.py
'''
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, weight_out_dir, num_epochs, start_epoch, use_gpu, proxy_type, param, save_every_epoch = True, save_outputs=True):
    save_output_frequency = c.SAVE_OUTPUT_FREQ
    num_epochs = num_epochs + start_epoch
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    if not os.path.exists(weight_out_dir):
        os.mkdir(weight_out_dir)
    
    # Creating directory to store model predictions
    predictions_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_'  + c.OUTPUT_PREDICTIONS_PATH)
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        sys.stdout.flush()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            max_iter_per_epoch = len(dataloaders[phase])
            # Iterate over data.
            i = 0
            num_inputs_seen = 0
            
            for names, inputs, labels in dataloaders[phase]:  
                inputs = inputs.type(dtype)
                labels = labels.type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()
                
#                print("inputs {}".format(inputs.size()))
#                print("labels {}".format(labels.size()))
#                sys.stdout.flush()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # Saving model outputs during training
                    if save_outputs and (epoch % save_output_frequency) == 0:
                        outputs_ndarray = outputs[0].detach().cpu().numpy()
                        outputs_ndarray = np.moveaxis(outputs_ndarray, 0, -1)
                        outputs_path = os.path.join(predictions_path, f'{proxy_type}_{param}_pred_epoch-{epoch}_{names[0]}')
                        plt.imsave(outputs_path, outputs_ndarray, format='png')

                    
#                    print("outputs {}".format(outputs.size()))
#                    sys.stdout.flush()
                    
                    loss = criterion(outputs, labels)
                    
#                    print("loss {}".format(loss))
#                    sys.stdout.flush()
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        
#                        print("backward loss")
#                        sys.stdout.flush()
                        
                        optimizer.step()
                        
#                        print("step optimizer")
#                        sys.stdout.flush()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                num_inputs_seen += inputs.size(0)
                print("\tbatch {:d}/{:d}, Average Loss: {:3f}".format(
                    i + 1, max_iter_per_epoch, running_loss/num_inputs_seen), end='\r')
                sys.stdout.flush()
                i += 1
                
            epoch_loss = running_loss / dataset_sizes[phase]

            print('\tDone with {} phase. Average Loss: {:.4f}'.format(
                phase, epoch_loss))
            sys.stdout.flush()
            # deep copy the model
            if phase == 'val' and ((epoch_loss < best_loss or epoch == (num_epochs-1)) or save_every_epoch):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Saved model')
                torch.save(best_model_wts, os.path.join(weight_out_dir, proxy_type + '_' + param + '_' + '{:03d}.pkl'.format(epoch)))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.flush()

'''
Evaluate the model on a any input(s)
'''
def eval(model, dataloader, criterion, optimizer, use_gpu, proxy_type, param, sweep=False, outputs_path=None):
    dtype = torch.FloatTensor
    if use_gpu:
        dtype = torch.cuda.FloatTensor

    if sweep:
        print ('Sweeping...')
    else:
        print('Evaluating...')
    print('-' * 10)
    sys.stdout.flush()

    model.eval()   # Set model to evaluate mode
    
    # Lock the weights in the U-Net	
    for parameter in model.parameters():	
        parameter.requires_grad=False

    # List of all the image names + losses
    eval_info = []

    running_loss = 0.0
    max_iter_per_epoch = len(dataloader)
    # Iterate over data.
    i = 0
    num_inputs_seen = 0
    
    if sweep:
        # sweeping (forward)
        frame = 0

        for index in range(len(dataloader)):
            for param in range(dataloader.num_params):
                
                name, input = dataloader[index, param]
                input = input.type(dtype)
            
                frame += 1

                # zero the parameter gradients
                #TODO: delete this?
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(False):
                    output = model(input)

                    # Saving model output
                    outputs_ndarray = output[0].detach().cpu().numpy()
                    outputs_ndarray = np.moveaxis(outputs_ndarray, 0, -1)
                    output_path = os.path.join(outputs_path, f'{name[0].split(".")[0]}_{proxy_type}_{param}_sweep_{frame:04}.png')
                    #print('output path: ' + output_path)
                    plt.imsave(output_path, outputs_ndarray, format=c.OUTPUT_PREDICTIONS_FORMAT)

        print('Done with sweep.')
        sys.stdout.flush()
        return

    # evaluating (forward)
    for names, inputs, labels in dataloader:  
                inputs = inputs.type(dtype)
                labels = labels.type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(False):
                    outputs = model(inputs)

                    # Saving model output
                    outputs_ndarray = outputs[0].detach().cpu().numpy()
                    outputs_ndarray = np.moveaxis(outputs_ndarray, 0, -1)
                    outputs_path = os.path.join(c.IMAGE_ROOT_DIR, c.EVAL_PATH, f'{proxy_type}_{param}_eval_{names[0]}')
                    plt.imsave(outputs_path, outputs_ndarray, format=c.OUTPUT_PREDICTIONS_FORMAT)

                loss = criterion(outputs, labels)

                # Saving image name and loss
                eval_info.append('image name: ' + names[0] + ', loss: ' + str(loss.item()))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                num_inputs_seen += inputs.size(0)
                print("\tbatch {:d}/{:d}, Average Loss: {:3f}".format(
                    i + 1, max_iter_per_epoch, running_loss/num_inputs_seen), end='\r')
                sys.stdout.flush()
                i += 1

    print('\tDone with evaluation. Average Loss: {:.4f}'.format(
    running_loss))

    # Printing image statistics
    for info in eval_info:
        print(info)

    sys.stdout.flush()
