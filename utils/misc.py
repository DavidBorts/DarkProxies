# Miscellaneous util functions

import os
import numpy as np
import tifffile

# Local files
import Constants as c

def sort_params(proxy_type, params, values=None):
    '''
    Re-arrange a params list into a standard order (returns list)
    '''
    if params is None or params[0].lower() == 'full':
        return c.PARAM_NAMES[proxy_type]

    names_lower = [name.lower() for name in c.PARAM_NAMES[proxy_type]]
    params_lower = [param.lower() for param in params]

    if values is not None:
        sorted_params = []
        sorted_values = []
        for name in names_lower:
            for param, val in zip(params_lower, values):
                if name == param:
                    sorted_params.append(param)
                    sorted_values.append(val)
        return sorted_params, sorted_values

    sorted = [name for name in names_lower if name in params_lower]
    return sorted

def get_possible_values(proxy_type, params):
    '''
    Given a proxy type and a subset of its input params,
    return the corresponding ranges of possible values
    of those parameters
    #NOTE: this returns a list of lists
    '''
    if proxy_type in c.NO_PARAMS:
        return []
    
    all_possible_values = c.POSSIBLE_VALUES[proxy_type]
    all_param_names = c.PARAM_NAMES[proxy_type]

    if params is None or params[0].lower() == 'full':
        return all_possible_values
    
    params_sorted = sort_params(proxy_type, params)

    possible_values = [all_possible_values[all_param_names.index(param)] for param in params_sorted]
    return possible_values

def unroll_possible_values(possible_values):
    '''
    Given a list of tuples of possible values, finds any nested lists and
    unrolls them (useful for paramaters that are list-based in Darktable).

    Example:
    [(0.0, 1.0), [(3.0, 4.0), (4.0, 5.0)]] -> [(0.0, 1.0), (3.0, 4.0), (4.0, 5.0)]
    '''
    def has_list(list_rolled):
        for item in list_rolled:
            if type(item) is list:
                return True
            return False

    unrolled_values = possible_values
    while has_list(unrolled_values):
        unrolled_values_t = []
        for item in unrolled_values:
            if type(item) is list:
                for i in item:
                    unrolled_values_t.append(i)
            else:
                unrolled_values_t.append(item)
        unrolled_values = unrolled_values_t
    return unrolled_values

def write_img_list(name, stage, img_list):

    img_list_path = os.path.join(c.IMAGE_ROOT_DIR,
                                 getattr(c, f"STAGE_{stage}_PATH"),
                                 f"{name}img_list.txt")
    
    with open(img_list_path, 'w') as file:
        file.write('\n'.join(img_list))

def read_img_list(name, stage):

    img_list_path = os.path.join(c.IMAGE_ROOT_DIR,
                                 getattr(c, f"STAGE_{stage}_PATH"),
                                 f"{name}img_list.txt")
    
    with open(img_list_path, 'r') as file:
        img_list = file.readlines()
        return img_list
    
def write_tif(path, img, photometric='rgb'):
    '''
    Use the tifffile library to save a numpy array as a greyscale/RGB TIFF 
    '''
    if type(img) is not np.ndarray:
        raise TypeError(f"This method only supports ndarrays.\nType provided: {str(type(img))}")
    
    if len(img.shape) != 3:
        raise ValueError(f"img must be HxWx3 or HxWx1.\nProvided img is: {str(img.shape)}")

    tifffile.imwrite(path, img, photometric=photometric)

def get_num_channels(proxy_type, possible_params, append_params):
    '''
    TODO: add comment
    '''
    # Bayer mosaics are always unpacked into 4 channels,
    # with no appended parameter channels
    if proxy_type == "demosaic":
        return 4, 0, 12

    num_input_channels = c.NUM_IMAGE_CHANNEL # input tensor
    if proxy_type in c.SINGLE_IMAGE_CHANNEL:
        num_input_channels = 1
    params_size = (len(possible_params), c.IMG_SIZE, c.IMG_SIZE) # tuple size of the appended params
    num_output_channels = num_input_channels

    # Depending on c.EMBEDDING_TYPE, some number of parameter
    # channels might be appended to the input tensor
    #TODO: move this into Models.py
    if append_params:
        num_params = len(possible_params)
        embedding_type = c.EMBEDDING_TYPES[c.EMBEDDING_TYPE]

        if embedding_type == "none":
            num_input_channels += len(possible_params)

        elif embedding_type == "linear_to_channel":
            channels = int(np.ceil(float(num_params) / c.EMBEDDING_RATIO))
            if c.EMBED_TO_SINGLE:
                channels = 1
            params_size = (c.PROXY_MODEL_BATCH_SIZE, channels, int(c.IMG_SIZE/16), int(c.IMG_SIZE/16))
            num_input_channels += channels

        else: # embedding type is "linear_to_value"
            final = int(np.ceil(num_params*1.0 / c.EMBEDDING_RATIO))
            if c.EMBED_TO_SINGLE:
                final = 1
            params_size = (c.PROXY_MODEL_BATCH_SIZE, final, c.IMG_SIZE, c.IMG_SIZE)
            num_input_channels += final
    return num_input_channels, params_size, num_output_channels