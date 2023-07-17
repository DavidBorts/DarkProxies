# Miscellaneous util functions

import os

# Local files
import Constants as c

def sort_params(proxy_type, params):
    '''
    Re-arrange a params list into a standard order (returns list)
    '''
    if params is None or params[0].lower() == 'full':
        return c.PARAM_NAMES[proxy_type]

    names = c.PARAM_NAMES[proxy_type]
    params_upper = [param.upper() for param in params]

    sorted = [name.lower() for name in names if name.upper() in params_upper]
    return sorted

def get_possible_values(proxy_type, params):
    '''
    Given a proxy type and a subset of its input params,
    return the corresponding ranges of possible values
    of those parameters
    '''
    all_possible_values = c.POSSIBLE_VALUES[proxy_type]
    all_param_names = c.PARAM_NAMES[proxy_type]

    if proxy_type in c.NO_PARAMS:
        return []

    if params is None or params[0].lower() == 'full':
        return all_possible_values
    
    params_sorted = sort_params(proxy_type, params)

    possible_values = [all_possible_values[all_param_names.index(param)] for param in params_sorted]
    return possible_values

def write_img_list(name, stage, img_list):

    img_list_path = os.path.join(c.IMAGE_ROOT_DIR,
                                 getattr(c, f"STAGE_{stage}_PATH"),
                                 f"{name}_img_list.txt")
    
    with open(img_list_path, 'w') as file:
        file.write('\n'.join(img_list))

def read_img_list(name, stage):

    img_list_path = os.path.join(c.IMAGE_ROOT_DIR,
                                 getattr(c, f"STAGE_{stage}_PATH"),
                                 f"{name}_img_list.txt")
    
    with open(img_list_path, 'r') as file:
        img_list = file.readlines()
        return img_list