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

def pack_input_demosaic(image, cfa):
    '''
    Packs image tensor into a smaller 4-channel image. For neural demosaicing.
    '''
    mosaic = torch.squeeze(image, 0)
    mosaic = torch.unsqueeze(mosaic, 2)
    try:
        H, W, _ = mosaic.size()
    except:
        print(mosaic.size())
        _, H, W, _ = mosaic.size()

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

class Darktable_Dataset(Dataset):
    '''
    Darktable_Dataset. This is the dataset for any Darktable data, and is used for the
    training of the proxy model, the fine-tuning of its parameters, and for ISP-tuning
    experiments.
    '''
    def __init__(self,
                 root_dir,
                 stage,
                 proxy_type,
                 params,
                 param_ranges,
                 name=None,
                 sampler=False,
                 val_split=0.25,
                 shuffle_seed=0,
                 input_dir=None,
                 output_dir=None,
                 params_file=None,
                 transform=None,
                 sweep=False,
                 gt_list=None,
                 proxy_order=None
                 ):
        '''
        Initialize the object.
        Inputs:
            [root_dir]: root_directory of the images.
            [stage]: Specifies whether we are training proxy (stage_1), 
                     or finetuning hyperparameters (stage_2)
            [proxy_type]: proxy type of data
            [params]: list of parameters that vary across data
            [param_ranges]:
            [name]:
            [sampler]:
            [val_split]: fraction of variables to put into validation set.
            [shuffle_seed]: Random seed that determines the train/val split.
            [input_dir]: path to the directory of input images
            [otuput_dir]: path to the directory of the output images
            [params_file]: path to the .npy file with parameter values
            [transform]: Transforms to be applied onto PIL Image.
            [sweep]: Toggles sweep mode of the dataloader for Darktable_sweep.py 
            [gt_list]:
            [proxy_order]:
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
        self.proxy_order = proxy_order

        if param_ranges is not None:
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
            #TODO: add gt_list arg for sweep mode
            self.image_name_list = os.listdir(self.input_image_dir)
            print("Sweeping for: ")
            print(self.image_name_list)
        else:
            #TODO: implement gt_list universally to get rid of if checks
            if self.gt_list is not None:
                self.image_name_list = self.gt_list
            else:
                self.image_name_list = os.listdir(self.output_image_dir)
                self.image_name_list.sort()
            print('Images in the dataset:')
            print(self.image_name_list)

        # Getting path to params file (stage 1 only)
        if self.stage == 1 and proxy_type not in c.NO_PARAMS:
            if params_file is None:
                self.param_mat = np.load(os.path.join(root_dir, self.stage_path, self.name + 'params.npy'))
            else:
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
        def get_crop_indices(tensor):
            '''
            Return the correct indices to crop a tensor to a square shape 
            (Constants.IMG_SIZE X Constants.IMG_SIZE)
            '''
            if c.IMG_SIZE % 4 != 0:
                raise ValueError("IMG_SIZE in Constants.py must be a multiple of 4 (default is 736).")
            
            if len(tensor.size()) == 3:
                _, width, height = tensor.size()
            else:
                width, height = tensor.size()

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

            return width_low, width_high, height_low, height_high

        def save_cropped_tensors(input_tensor, gt_tensor, gt_image_name, model_name):
            '''
            Saving cropped input and ground truth tensors as TIFF images
            '''
            crop_gt_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, model_name.name + c.CROPPED_OUTPUT_DIR)
            crop_gt_path = os.path.join(crop_gt_dir, f'crop_{gt_image_name}')
            if os.path.exists(crop_gt_path):
                return
            if not os.path.exists(crop_gt_dir):
                os.mkdir(crop_gt_dir)
                print('directory created at: ' + crop_gt_dir)

            gt_ndarray = gt_tensor.numpy()
            gt_ndarray = np.moveaxis(gt_ndarray, 0, -1).copy(order='C')
            tifffile.imwrite(crop_gt_path, gt_ndarray)
            print('crop saved: ground truth')

            crop_input_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, model_name.name + c.CROPPED_INPUT_DIR)
            crop_input_path = os.path.join(crop_input_dir, f'crop_{gt_image_name}')
            if os.path.exists(crop_input_path):
                return
            if not os.path.exists(crop_input_dir):
                os.mkdir(crop_input_dir)
                print('directory created at: ' + crop_input_dir)

            input_ndarray = input_tensor.numpy()
            input_ndarray = np.moveaxis(input_ndarray, 0, -1).copy(order='C')
            tifffile.imwrite(crop_input_path, input_ndarray)
            print('crop saved: input')

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
        
        gt_image_name = self.image_name_list[index].strip('\n')
        input_image_name = gt_image_name.split(".")[0] + '.tif'
        if self.sampler or self.stage == 3: # colorin/colorout/demosaic
            input_image_name = gt_image_name

        # Loading in input tensor
        input_image_path = os.path.join(self.input_image_dir, input_image_name)
        try:
            input_image = Image.open(input_image_path)
            input_image = ImageOps.exif_transpose(input_image)
        except:
            input_image = imageio.imread(input_image_path)
        
        # Reformatting input tensor
        if self.transform is not None:
            print("Performing additional image transform.")
            input_image = self.transform(input_image)
        proxy_model_input = to_tensor_transform(input_image)
        if c.DOWNSAMPLE_IMAGES:
            print('Downsampling input tensor.')
            proxy_model_input = interpolate(proxy_model_input[None, :, :, :], scale_factor=0.25, mode='bilinear')
        proxy_model_input = torch.squeeze(proxy_model_input, dim=0)
        
        # Cropping input tensor
        width_low, width_high, height_low, height_high = get_crop_indices(proxy_model_input)
        if len(proxy_model_input.size()) == 3:
            proxy_model_input = proxy_model_input[:, width_low:width_high, height_low:height_high]
        else:	
            proxy_model_input = proxy_model_input[width_low:width_high, height_low:height_high]
            proxy_model_input = proxy_model_input.unsqueeze(0)
        
        # Extracting cfa and packing input for demosaic proxy
        if self.proxy_type == 'demosaic' or (self.proxy_order is not None and self.proxy_order[0][0].split('_')[0] == "demosaic"):
            dng_name = input_image_name.split('_')[0].split('.')[0] + '.dng'
            dng_path = os.path.join(c.IMAGE_ROOT_DIR, getattr(c, 'STAGE_' + str(self.stage) + '_DNG_PATH'), dng_name)
            cfa = get_cfa(dng_path)
            pre_pack_input = proxy_model_input.clone().unsqueeze(dim=0)
            proxy_model_input = np.squeeze(pack_input_demosaic(proxy_model_input, cfa))
        
        # Loading in and reformatting/cropping ground truth tensor
        if not self.sweep: # NOTE: this is not necessary when sweeping a proxy model
            try:
                output_image = Image.open(os.path.join(self.output_image_dir, gt_image_name))
            except:
                output_image = imageio.imread(os.path.join(self.output_image_dir, gt_image_name))	
            
            if self.transform is not None:
                output_image = self.transform(output_image)
            proxy_model_label = to_tensor_transform(output_image)
            if c.DOWNSAMPLE_IMAGES:
                print('Downsampling ground truth tensor')
                proxy_model_label = interpolate(proxy_model_label[None, :, :, :], scale_factor=0.25, mode='bilinear')
            proxy_model_label = torch.squeeze(proxy_model_label, dim=0)
            if len(proxy_model_label.size()) == 3:
                proxy_model_label = proxy_model_label[:, width_low:width_high, height_low:height_high]
            else:
                proxy_model_label = proxy_model_label[width_low:width_high, height_low:height_high]
                proxy_model_label = proxy_model_label.unsqueeze(0)

        if not self.stage == 1:
            return gt_image_name, proxy_model_input, proxy_model_label
        # NOTE: everything below this point is for stage 1 (model training) only!

        # Saving crops
        if c.SAVE_CROPS and self.sweep == False:
            if self.proxy_type != "demosaic":
                save_cropped_tensors(proxy_model_input.detach().cpu(), proxy_model_label.detach().cpu(), gt_image_name, self.name)
            else:
                save_cropped_tensors(pre_pack_input.detach().cpu(), proxy_model_label.detach().cpu(), gt_image_name, self.name)

        # Appending parameter tensor to input tensor
        if self.append_params: # NOTE: this is only necessary if the model has input parameters
            params = self.param_mat[:,index]
            if self.sweep:
                params = self.param_mat[:, param_num]

            # Normalizing param values to [0, 1] range
            #if len(params) != len(self.param_lower_bounds):
                #raise ValueError("ERROR: param possible ranges should be the same length as params")
            if self.param_ranges is not None:
                params = (params + self.param_lower_bounds) / self.param_diff
            
            if self.embedding_type != "none": # Params will be appended later if embedding is necessary
                return gt_image_name, proxy_model_input, params, proxy_model_label

            proxy_model_input = _format_input_with_parameters(proxy_model_input, params)

        if self.proxy_type == "demosaic": # Demosaic model needs pre-packed input
            return gt_image_name, [proxy_model_input, pre_pack_input], proxy_model_label    
        if self.sweep: # Not returning a label tensor for sweep mode
            return gt_image_name, proxy_model_input
        return gt_image_name, proxy_model_input, proxy_model_label

    def __len__(self):
        return len(self.image_name_list)