''' 
Data loader for Darktable

Stage 1: Loads in image data and concatenates parameters as channels to the image
Stages 2 & 3: Loads in image data
'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import interpolate
from torchvision import transforms

# Local files
import Darktable_constants as c

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

'''
Darktable_Dataset. This is the dataset for any Darktable data, and is used for the
training of the proxy model, the fine-tuning of its parameters, and for ISP-tuning
experiments.
'''
class Darktable_Dataset(Dataset):
    '''
    Initialize the object.
    Inputs:
        root_dir: root_directory of the images.
        stage: Specifies whether we are training proxy (stage_1), or finetuning hyperparameters (stage_2)
        val_split: fraction of variables to put into validation set.
        shuffle_seed: Random seed that determines the train/val split.
        input_dir: path to the directory of input images
        otuput_dir: path to the directory of the output images
        params_file: path to the .npy file with parameter values
        transform: Transforms to be applied onto PIL Image.
        proxy_type: proxy type of data
        param: parameter type of data
        sweep: Toggles sweep mode of the dataloader for Darktable_sweep.py
    '''
    def __init__(self, root_dir, stage = 1, val_split=0.25, shuffle_seed=0, input_dir=None, output_dir=None, params_file=None, transform=None, proxy_type=None, param=None, sweep=False, vary_input=False):
        
        if stage != 1 and stage != 2 and stage != 3:
            raise ValueError("Please enter either 1, 2, or 3 for the stage parameter.")
            
        self.root_dir = root_dir
        self.stage = stage
        self.stage_path = getattr(c, 'STAGE_' + str(self.stage) + '_PATH')
        self.transform = transform
        self.proxy_type = proxy_type
        self.param = param
        self.sweep = sweep
        self.vary_input = vary_input
        
        # Configuring input & output directories
        if input_dir is None:
            self.input_image_dir = os.path.join(self.root_dir, self.stage_path, self.proxy_type + '_' + param + '_' + c.INPUT_DIR)
        else:
            self.input_image_dir = input_dir
        if output_dir is None:
            self.output_image_dir = os.path.join(self.root_dir, self.stage_path, self.proxy_type + '_' + param + '_' + c.OUTPUT_DIR)
        else:
            self.output_image_dir = output_dir
        
        # Getting image name list
        if sweep:
            # Sweep mode
            self.image_name_list = os.listdir(self.input_image_dir)
            print("Sweeping for: ")
            print(self.image_name_list)
        else:
            # Sorting image list
            self.image_name_list = os.listdir(self.output_image_dir)
            self.image_name_list.sort(key=lambda x: (x.split(".")[0], float(x.split("_")[3].split(".tif")[0])))
            print('Images in the dataset:')
            print(self.image_name_list)

        # Getting path to params file (stage 1 only)
        if self.stage == 1:
            if params_file is None:
                self.param_mat = np.load(os.path.join(root_dir, self.stage_path, f'{proxy_type}_{param}_params.npy'))
            else:
                self.param_mat = np.load(params_file)
            if self.sweep:
                self.num_params = len(self.param_mat)

        # Creating samplers
        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_indices, val_indices = self._get_split(val_split, shuffle_seed)
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.sweep_sampler = SubsetRandomSampler(indices)

    def _get_split(self, val_split, shuffle_seed):
        '''
        Returns the data indices, randomly split with seed shuffle_seed and ratio val_split.
        '''
        dataset_size = len(self)
        indices = list(range(dataset_size))
        if not shuffle_seed is None:
            np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
        split = int(np.floor(val_split * dataset_size))
        return indices[split:], indices[:split]
    
    def __getitem__(self, indexValue):
        '''
        By default, appends parameter channels to the image tensor. Note that image is already a tensor in this case. 
        '''        
        def _format_input_with_parameters(image, params):
            '''
            Concatentate image and parameter tensors.
            '''
            _, width, height = image.size()
            
            param_tensor = torch.tensor((), dtype=torch.float).new_ones((len(params), width, height)).type(torch.FloatTensor)
            channel_index = 0
            for parameter in params:
                param_tensor[channel_index, :, :] = \
                    torch.tensor((), dtype=torch.float).new_full((width, height), parameter)
                channel_index += 1
            return torch.cat([image, param_tensor], dim=0)

        to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        # Supporting a different indexing format for sweep mode
        if self.sweep:
            index, param_num = indexValue
        else:
            index = indexValue
        
        # Determining whether or not to append parameter channels
        append_params = self.proxy_type not in c.NO_PARAMS
        
        image_name = self.image_name_list[index]
        if not self.sweep:
            print('ground truth image name: ' + image_name)
        input_image_name = image_name.split(".")[0] + '.tif'
        if self.vary_input:
            input_image_name = image_name
        print('input image name: ' + input_image_name)
        input_image = Image.open(os.path.join(self.input_image_dir, input_image_name))
        
        if self.transform is not None:
            input_image = self.transform(input_image)
            
        proxy_model_input = to_tensor_transform(input_image)
        proxy_model_input = interpolate(proxy_model_input[None, :, :, :], scale_factor=0.25, mode='bilinear')
        proxy_model_input = torch.squeeze(proxy_model_input, dim=0)
        _, width, height = proxy_model_input.size()
        
        # Cropping input tensor
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
            # Saving crops
            if c.SAVE_CROPS and self.sweep == False:
                input_ndarray = proxy_model_input.detach().cpu().numpy()
                input_ndarray = np.moveaxis(input_ndarray, 0, -1)
                input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.proxy_type + '_' + self.param + '_' + c.CROPPED_INPUT_DIR, f'crop_{image_name}')
                plt.imsave(input_path, input_ndarray, format=c.CROP_FORMAT)
                label_ndarray = proxy_model_label.detach().cpu().numpy()
                label_ndarray = np.moveaxis(label_ndarray, 0, -1)
                label_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.proxy_type + '_'  + self.param + '_' +  c.CROPPED_OUTPUT_DIR, f'crop_{image_name}')
                plt.imsave(label_path, label_ndarray, format=c.CROP_FORMAT)

            # Appending parameter tensor
            if append_params:
                params = self.param_mat[:,index]
                if self.sweep:
                    params = self.param_mat[:, param_num]
                proxy_model_input = _format_input_with_parameters(proxy_model_input, params)
            
        if self.sweep:
            return image_name, proxy_model_input
        return image_name, proxy_model_input, proxy_model_label

    def __len__(self):
        return len(self.image_name_list)