# Data loader 
#
# Stage 1: Loads in image data and concatenates parameters as channels to the image
# Stage 2: Loads in image data

import Darktable_constants as c
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import interpolate
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

'''
Face_Dataset. This is the dataset for the Face Dataset, and is used for both the 
training of the proxy model, as well as the fine-tuning of the parameters.
'''
class Darktable_Dataset(Dataset):
    '''
    Initialize the object.
    Inputs:
        root_dir: root_directory of the images.
        stage: Specifies whether we are training proxy (stage_1), or finetuning hyperparameters (stage_2)
        val_split: fraction of variables to put into validation set.
        shuffle_seed: Random seed that determines the train/val split.
        transform: Transforms to be applied onto PIL Image. DON'T INCLUDE THE ToTensor transform in this,
            since I already handle that.
    '''
    def __init__(self, root_dir, stage = 1, val_split=0.25, shuffle_seed=0, input_dir=None, output_dir=None, params_file=None, transform=None, proxy_type='colorbalancergb', param='contrast', sweep=False):
        if not stage == 1 and not stage == 2:
            raise ValueError("Please enter either 1 or 2 for the stage parameter.")
            
        self.root_dir = root_dir
        self.stage = stage
        self.stage_path = getattr(c, 'STAGE_' + str(self.stage) + '_PATH')
        self.transform = transform
        self.proxy_type = proxy_type
        self.param = param
        self.sweep = sweep
        
        if input_dir is None:
            self.input_image_dir = os.path.join(self.root_dir, self.stage_path, self.proxy_type + '_' + param + '_' + c.INPUT_DIR)
        else:
            self.input_image_dir = input_dir
        if output_dir is None:
            self.output_image_dir = os.path.join(self.root_dir, self.stage_path, self.proxy_type + '_' + param + '_' + c.OUTPUT_DIR)
        else:
            self.output_image_dir = output_dir
            
        if sweep:
            self.output_image_dir = None
        
        self.image_name_list = os.listdir(self.input_image_dir)
        self.image_name_list.sort()
        print('Images in the dataset: ')
        print(self.image_name_list)

        if self.stage == 1:
            if params_file is None:
                self.param_mat = np.load(os.path.join(root_dir, self.stage_path, f'{proxy_type}_{param}_params.npy'))
            else:
                self.param_mat = np.load(params_file)
                
        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_indices, val_indices = self._get_split(val_split, shuffle_seed)
        # the samplers are created within the class itself.
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.sweep_sampler = SubsetRandomSampler(indices)

    def _get_split(self, val_split, shuffle_seed):
        dataset_size = len(self)
        indices = list(range(dataset_size))
        if not shuffle_seed is None:
            np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
        split = int(np.floor(val_split * dataset_size))
        return indices[split:], indices[:split]
    
    def __getitem__(self, index):
        '''
        Appends parameter channels to the image tensor. Note that image is already a tensor in this case. 
        '''        
        def _format_input_with_parameters(image, params):
            num_channels, width, height = image.size()
            
            param_tensor = torch.tensor((), dtype=torch.float).new_ones((len(params), width, height)).type(torch.FloatTensor)
            channel_index = 0
            for parameter in params:
                param_tensor[channel_index, :, :] = \
                    torch.tensor((), dtype=torch.float).new_full((width, height), parameter)
                channel_index += 1
            return torch.cat([image, param_tensor], dim=0)

        to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        
        image_name = self.image_name_list[index]
        print('image name: ' + image_name)
        input_image = Image.open(os.path.join(self.input_image_dir, image_name))
        
        if self.transform is not None:
            input_image = self.transform(input_image)
            
        proxy_model_input = to_tensor_transform(input_image)
        proxy_model_input = interpolate(proxy_model_input[None, :, :, :], scale_factor=0.25, mode='bilinear')
        proxy_model_input = torch.squeeze(proxy_model_input, dim=0)
        num_channels, width, height = proxy_model_input.size()
        
        if width % 2 == 0:
            mid_width = width / 2
        else:
            mid_width = (width - 1) / 2
        if height % 2 == 0:
            mid_height = height / 2
        else:
            mid_height = (height - 1) / 2
            
        proxy_model_input = proxy_model_input[:, int(mid_width - (c.IMG_SIZE / 2)):int(mid_width + (c.IMG_SIZE / 2)), int(mid_height - (c.IMG_SIZE / 2)):int(mid_height + (c.IMG_SIZE / 2))]
        
        if not self.sweep:
            output_image = Image.open(os.path.join(self.output_image_dir, image_name))
            
            if self.transform is not None:
                output_image = self.transform(output_image)
            
            proxy_model_label = to_tensor_transform(output_image)
            proxy_model_label = interpolate(proxy_model_label[None, :, :, :], scale_factor=0.25, mode='bilinear')
            proxy_model_label = torch.squeeze(proxy_model_label, dim=0)[:, int(mid_width - (c.IMG_SIZE / 2)):int(mid_width + (c.IMG_SIZE / 2)), int(mid_height - (c.IMG_SIZE / 2)):int(mid_height + (c.IMG_SIZE / 2))]
        
        if self.stage == 1:
            if c.SAVE_CROPS and self.sweep == False:
                input_ndarray = proxy_model_input.detach().cpu().numpy()
                input_ndarray = np.moveaxis(input_ndarray, 0, -1)
                input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.proxy_type + '_' + self.param + '_' + c.CROPPED_INPUT_DIR, f'crop_{image_name}')
                plt.imsave(input_path, input_ndarray, format='png')
                label_ndarray = proxy_model_label.detach().cpu().numpy()
                label_ndarray = np.moveaxis(label_ndarray, 0, -1)
                label_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.proxy_type + '_'  + self.param + '_' +  c.CROPPED_OUTPUT_DIR, f'crop_{image_name}')
                plt.imsave(label_path, label_ndarray, format='png')
            params = self.param_mat[:,index]
            proxy_model_input = _format_input_with_parameters(proxy_model_input, params)
        if self.sweep:
            return image_name, proxy_model_input
        return image_name, proxy_model_input, proxy_model_label

    def __len__(self):
        return len(self.image_name_list)
