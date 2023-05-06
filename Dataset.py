''' 
Data loader for Darktable

Stage 1: Loads in image data and concatenates parameters as channels to the image
Stages 2 & 3: Loads in image data
'''

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tifffile	
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import interpolate
from torchvision import transforms

# Local files
import Constants as c
from utils.extract_RAW import get_cfa

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Darktable_Dataset(Dataset):
    '''
    Darktable_Dataset. This is the dataset for any Darktable data, and is used for the
    training of the proxy model, the fine-tuning of its parameters, and for ISP-tuning
    experiments.
    '''
    def __init__(self, root_dir, stage, proxy_type, params, sampler, name, val_split=0.25, shuffle_seed=0, input_dir=None, output_dir=None, params_file=None, transform=None, sweep=False, param_ranges=None, gt_list=None):
        '''
        Initialize the object.
        Inputs:
            [root_dir]: root_directory of the images.
            [stage]: Specifies whether we are training proxy (stage_1), 
                     or finetuning hyperparameters (stage_2)
            [proxy_type]: proxy type of data
            [params]: list of parameters that vary across data
            [sampler]:
            [name]:
            [val_split]: fraction of variables to put into validation set.
            [shuffle_seed]: Random seed that determines the train/val split.
            [input_dir]: path to the directory of input images
            [otuput_dir]: path to the directory of the output images
            [params_file]: path to the .npy file with parameter values
            [transform]: Transforms to be applied onto PIL Image.
            [sweep]: Toggles sweep mode of the dataloader for Darktable_sweep.py
            [param_ranges]: 
        '''
        
        if stage not in [1, 2, 3]:
            raise ValueError("Please enter either 1, 2, or 3 for the stage parameter.")
            
        self.root_dir = root_dir
        self.stage = stage
        self.stage_path = getattr(c, 'STAGE_' + str(self.stage) + '_PATH')
        self.transform = transform
        self.proxy_type = proxy_type
        self.params = params
        self.append_params = self.proxy_type not in c.NO_PARAMS
        self.sweep = sweep
        self.sampler = sampler
        self.name = name
        self.embedding_type = c.EMBEDDING_TYPES[c.EMBEDDING_TYPE]
        self.param_ranges = param_ranges
        self.gt_list = gt_list

        if self.param_ranges is not None:
            self.param_lower_bounds = np.array([range[0] for range in param_ranges])
            self.param_upper_bounds = np.array([range[1] for range in param_ranges])
            self.param_diff = self.param_upper_bounds - self.param_lower_bounds
        
        # Configuring input & output directories
        if input_dir is None:
            self.input_image_dir = os.path.join(self.root_dir, self.stage_path, self.name + c.INPUT_DIR)
        else:
            self.input_image_dir = input_dir
        if output_dir is None:
            self.output_image_dir = os.path.join(self.root_dir, self.stage_path, self.name + c.OUTPUT_DIR)
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
            if self.proxy_type == "demosaic":#TODO: temporary hack - delete me!
                self.image_name_list.sort()
            else:
                self.image_name_list.sort(key=lambda x: (x.split(".")[0], float(x.split("_")[3].split(".tif")[0])))
            if self.gt_list is not None:
                self.image_name_list = self.gt_list
            print('Images in the dataset:')
            print(self.image_name_list)

        # Getting path to params file (stage 1 only)
        if self.stage == 1 and proxy_type not in c.NO_PARAMS:
            if params_file is None:
                self.param_mat = np.load(os.path.join(root_dir, self.stage_path, self.name + 'params.npy'))
            else: # For sweep mode
                self.param_mat = np.load(params_file)
            if self.sweep:
                self.num_params = len(self.param_mat)
        else:
            self.param_mat = None

        # Creating index samplers
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
        if dataset_size == 1:
            return indices, []
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
        
        def pack_input_demosaic(image, cfa):
            '''
            Packs image tensor into a smaller 4-channel image. For neural demosaicing.
            '''

            mosaic = torch.unsqueeze(image, 2)	
            H, W, _ = mosaic.size()

            # Checking for 2x2 Bayer pattern
            if len(cfa) != 4:
                print("Provided CFA: " + cfa)
                raise ValueError("Only 2 X 2 Bayer patterns are currently supported.")

            # Checking for pattern with 2 green tiles
            if cfa.upper().count('G') != 2:
                print("Provided CFA: " + cfa)
                raise ValueError("Only CFA patterns with 2 green tiles are currently supported.")

            # Distinguishing G's in CFA
            g_indices = [index for index, char in enumerate(cfa.upper()) if char == 'G']
            cfa_new = ""
            for idx, c in enumerate(cfa):
                if idx == g_indices[0]:
                    cfa_new += '1'
                elif idx == g_indices[1]:
                    cfa_new += '2'
                else:
                    cfa_new += c
            #cfa[g_indices[0]] = 'G1'
            #cfa[g_indices[1]] = 'G2'

            # Getting color channels
            TL = mosaic[0:H:2, 0:W:2, :]
            TR = mosaic[0:H:2, 1:W:2, :]
            BL = mosaic[1:H:2, 0:W:2, :]
            BR = mosaic[1:H:2, 1:W:2, :]
            channels = [TL, TR, BL, BR]

            # Assigning channels to R, G, & B
            COLORS = ['R', '1', 'B', '2']
            for i in range(len(cfa_new)):
                color = cfa_new[i].upper()

                # Checking for RGB-only CFA
                if color not in COLORS:
                    print("Provided CFA: " + cfa)
                    raise ValueError("Only RGB-based CFAs are currently supported.")
                
                # Re-mapping each channel to the correct location, based on the Bayer pattern
                idx = COLORS.index(color)
                if idx != i:
                    channels[idx], channels[i] = channels[i], channels[idx]

            # Packing the channels together as RGBG
            # (axes rearranged to be C x H x W)
            return torch.moveaxis(torch.cat(channels, dim=2), -1, 0)

        to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        # Supporting a different indexing format for sweep mode
        if self.sweep:
            index, param_num = indexValue
        else:
            index = indexValue
        
        image_name = self.image_name_list[index].strip('\n')
        #if not self.sweep:
            #print('ground truth image name: ' + image_name)
        input_image_name = image_name.split(".")[0] + '.tif'
        if self.sampler: # colorin/colorout/demosaic
            input_image_name = image_name
            #input_image_name = image_name.split(".")[0] + '_' + str(image_name.split('_')[3]) # Temporary hack - rename colorin data and delete me!
            if self.proxy_type == "demosaic": # TODO: Temporary hack - delete me!
                input_image_name = image_name.split('.')[0] + '.tif'
        #print('input image name: ' + input_image_name)
        #input_image = imageio.imread(os.path.join(self.input_image_dir, input_image_name))	
        #input_image = tifffile.imread(os.path.join(self.input_image_dir, input_image_name))	
        input_image = Image.open(os.path.join(self.input_image_dir, input_image_name))	
        input_image = ImageOps.exif_transpose(input_image)	
        #print("input image shape:")	
        #print(input_image.shape)
        #input_image = input_image.astype(np.float32) #TODO: do we need this??
        
        if self.transform is not None:
            print("Performing additional image transform.")
            input_image = self.transform(input_image)

        if self.proxy_type == "demosaic":
            #pre_pack_input = np.squeeze(np.array(input_image).copy(), axis=2)
            pre_pack_input = to_tensor_transform(input_image)
            pre_pack_input = torch.squeeze(pre_pack_input, dim=0)
            #pre_pack_input = (pre_pack_input * 255).astype(np.uint8)
            #print("pre pack shape: ")
            #print(np.shape(pre_pack_input))
            
        proxy_model_input = to_tensor_transform(input_image)
        if c.TAPOUTS[self.proxy_type] is None and c.DOWNSAMPLE_IMAGES:
            #print('Downsampling input tensor.')
            proxy_model_input = interpolate(proxy_model_input[None, :, :, :], scale_factor=0.25, mode='bilinear')
        proxy_model_input = torch.squeeze(proxy_model_input, dim=0)
        
        if len(proxy_model_input.size()) == 3:
            _, width, height = proxy_model_input.size()
        else:
            width, height = proxy_model_input.size()
        
        # Cropping input tensor
        if c.IMG_SIZE % 4 != 0:
            raise ValueError("IMG_SIZE in Constants.py must be a multiple of 4 (default is 736).")
        mid_width = width // 2
        width_low = int(mid_width - (c.IMG_SIZE / 2))
        while width_low % 4 != 0: # Ensuring that image crops are along mosaic boundaries
            width_low -= 1
        width_high = width_low + c.IMG_SIZE
        mid_height = height // 2
        height_low = int(mid_height - (c.IMG_SIZE / 2))
        while height_low % 4 != 0:
            height_low -= 1
        height_high = height_low + c.IMG_SIZE

        if len(proxy_model_input.size()) == 3:
            proxy_model_input = proxy_model_input[:, width_low:width_high, height_low:height_high]
        else:	
            proxy_model_input = proxy_model_input[width_low:width_high, height_low:height_high]
        
        if self.proxy_type == 'demosaic':
            dng_name = input_image_name.split('_')[0].split('.')[0] + '.dng'
            dng_path = os.path.join(c.IMAGE_ROOT_DIR, getattr(c, 'STAGE_' + str(self.stage) + '_DNG_PATH'), dng_name)
            #print('dng path: ' + str(dng_path))
            cfa = get_cfa(dng_path)
            proxy_model_input = np.squeeze(pack_input_demosaic(proxy_model_input, cfa))
            #print('proxy_model_input shape after packing: ' + str(proxy_model_input.shape))
        
        if not self.sweep:
            if c.TAPOUTS[self.proxy_type] is None:
                output_image = Image.open(os.path.join(self.output_image_dir, image_name))
            else:
                output_image = imageio.imread(os.path.join(self.output_image_dir, image_name))	
            #output_image = output_image.astype(np.float32) #TODO: do we need this??	
            # output_image = Image.open(os.path.join(self.output_image_dir, image_name), mode='r', formats=None) FIXME: NOT WORKING!!
            
            if self.transform is not None:
                output_image = self.transform(output_image)
            
            proxy_model_label = to_tensor_transform(output_image)
            if c.TAPOUTS[self.proxy_type] is None and c.DOWNSAMPLE_IMAGES:
                #print('Downsampling ground truth tensor')
                proxy_model_label = interpolate(proxy_model_label[None, :, :, :], scale_factor=0.25, mode='bilinear')
            proxy_model_label = torch.squeeze(proxy_model_label, dim=0)
            proxy_model_label = proxy_model_label[:, width_low:width_high, height_low:height_high]
        
        if self.stage == 1:
            # Saving crops
            if c.SAVE_CROPS and self.sweep == False:

                if self.proxy_type != "demosaic":	
                    input_ndarray = proxy_model_input.detach().cpu().numpy()	
                else:	
                    input_ndarray = pre_pack_input.detach().cpu().numpy()
                input_ndarray = np.moveaxis(input_ndarray, 0, -1).copy(order='C')

                crop_input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.name + c.CROPPED_INPUT_DIR)
                if not os.path.exists(crop_input_path):
                    os.mkdir(crop_input_path)
                    print('directory created at: ' + crop_input_path)
                    
                input_path = os.path.join(crop_input_path, f'crop_{image_name}')
                if self.proxy_type == "demosaic" and not os.path.exists(input_path):
                    #Image.fromarray(input_ndarray).save(input_path, c.CROP_FORMAT)
                    #imageio.imwrite(input_path, input_ndarray, format=c.CROP_FORMAT)
                    tifffile.imwrite(input_path, input_ndarray)
                    #print('crop saved: input')
                elif not os.path.exists(input_path):
                    plt.imsave(input_path, input_ndarray, format=c.CROP_FORMAT)
                    #print('crop saved: input')

                label_ndarray = proxy_model_label.detach().cpu().numpy()
                label_ndarray = np.moveaxis(label_ndarray, 0, -1).copy(order='C')

                crop_label_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, self.name + c.CROPPED_OUTPUT_DIR)
                if not os.path.exists(crop_label_path):
                    os.mkdir(crop_label_path)
                    print('directory created at: ' + crop_label_path)

                label_path = os.path.join(crop_label_path, f'crop_{image_name}')
                if not os.path.exists(label_path):
                    plt.imsave(label_path, label_ndarray, format=c.CROP_FORMAT)
                    #print('crop saved: ground truth')

            # Appending parameter tensor
            if self.append_params:
                params = self.param_mat[:,index]
                #params = float(params) # TODO: is this necessary? does it work??

                if self.sweep:
                    params = self.param_mat[:, param_num]

                #print("image name + params: " + image_name + str(params))
                sys.stdout.flush()

                # Normalizing param values to [0, 1] range
                if len(params) != len(self.param_lower_bounds):
                    raise ValueError("ERROR: param possible ranges should be the same length as params")
                if self.param_ranges is not None:
                    params = (params + self.param_lower_bounds) / self.param_diff
                
                if self.embedding_type != "none":
                    return image_name, proxy_model_input, params, proxy_model_label

                proxy_model_input = _format_input_with_parameters(proxy_model_input, params)
            
        if self.sweep:
            return image_name, proxy_model_input
        return image_name, proxy_model_input, proxy_model_label

    def __len__(self):
        return len(self.image_name_list)
