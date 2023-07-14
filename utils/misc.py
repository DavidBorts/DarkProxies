# Miscellaneous util functions

import os

import Constants as c

def get_possible_values(proxy_type, params):
    '''
    Given a proxy type and a subset of its input params,
    return the corresponding ranges of possible values
    of those parameters
    '''
    all_possible_values = c.POSSIBLE_VALUES[proxy_type]
    all_param_names = c.PARAM_NAMES[proxy_type]

    if params is None:
        return all_possible_values

    possible_values = [all_possible_values[all_param_names.index(param)] for param in params]
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