# Main script for data generation, proxy training, and parameter regression

import sys
import os
import argparse
import torch

# Local files
import Generate_data as data
import Constants as c
import Train_proxy
import Parameter_regression
from utils.misc import get_possible_values, write_img_list, read_img_list

def sort_params(proxy_type, params):
    '''
    Re-arrange the params list into a standard order (returns list)
    '''
    if params is None:
        return c.POSSIBLE_VALUES[proxy_type]

    names = c.PARAM_NAMES[proxy_type]
    params_upper = [param.upper() for param in params]

    sorted = [name.lower() for name in names if name.upper() in params_upper]
    return sorted

if __name__ != '__main__':
    raise RuntimeError("This script is only configured to be called directly by the user!")

# argparser args
parser = argparse.ArgumentParser()
parser.add_argument("proxy", help="Name of the Darktable block for which to train a proxy network", 
                    choices = ['colorbalancergb', 'sharpen', 'exposure', 'colorin', 'colorout', 
                               'demosaic', 'filmic', 'colorize', 'bloom', 'soften', 'graduateddensity',
                               "lowpass"])
parser.add_argument("-p", "--params", help="[OPTIONAL] Specify a list of _ separated input parameters\
                    on which to train a proxy, keeping all others fixed (i.e. \
                    -p contrast_radius_brightness)", default=None)
parser.add_argument("-n", "--number", help="Number of training examples to generate for each \
                    source DNG image", required=True, default=0)
parser.add_argument("-c", "--custom", help="[OPTIONAL] A custom name for the given proxy - overrides \
                    default naming scheme set by --params flag", default=None)
parser.add_argument("-d", "--dataset", help="[OPTIONAL] Name of the dataset to train on. Useful \
                    for training multiple proxies on the same images. Note: this automatically \
                    disables any data generation.", default=None)
parser.add_argument("-r", "--regression", help="[OPTIONAL] Number of data points to generate for \
                    parameter regression experiments", default=10)
args = parser.parse_args()
proxy_type = args.proxy
params = args.params
custom = args.custom
dataset_name = args.dataset
num = int(args.number)
num_regression = int(args.regression)

# Global constants
image_root_dir = c.IMAGE_ROOT_DIR

# Sorting params list for consistency
if params is not None:
    params = params.split('_')
    params = sort_params(proxy_type, params)

# Getting proxy name (unique ID for this training job)
name = f'{proxy_type}_'
if params is not None and custom is None:
    name.join(f'{param}_' for param in params)
if custom is not None:
    name = f"{proxy_type}_{custom}_"

if params is None:
    params = c.PARAM_NAMES[proxy_type]

# Stage 1 constants
generate_stage_1 = c.GENERATE_STAGE_1
stage_1_batch_size = c.PROXY_MODEL_BATCH_SIZE
num_epoch = c.PROXY_MODEL_NUM_EPOCH
weight_out_dir = os.path.join(image_root_dir, c.STAGE_1_PATH, name + c.MODEL_WEIGHTS_PATH)

# Stage 2 constants
generate_stage_2 = c.GENERATE_STAGE_2
num_iters = c.PARAM_TUNING_NUM_ITER
param_out_dir = c.STAGE_2_PARAM_PATH

# If an existing dataset is being used, skip data generation
if dataset_name is not None:
    generate_stage_1 = False
    generate_stage_2 = False

# If the given proxy takes input params, it is necesary to find their
# ranges of possible values
possible_values = None # NOTE: possible_values needs to be a list
append_params = proxy_type not in c.NO_PARAMS
if append_params:
    possible_values = get_possible_values(proxy_type, params)
else:
    # Proxy has no input parameters - use inputs of a "sampler" block instead
    #TODO: replace with highlights or temperature to support demosaic?
    if proxy_type == 'demosaic': # Temporary hack: not sweeping for demosaic
        possible_values = [(None, None)]
    else:
        sampler_block, sampler_param = c.SAMPLER_BLOCKS[proxy_type].split('_')
        possible_values = get_possible_values(sampler_block, [sampler_param])

# Creating stage 1 and 2 directories if they do not already exist
stage_1_path = os.path.join(image_root_dir, c.STAGE_1_PATH)
stage_2_path = os.path.join(image_root_dir, c.STAGE_2_PATH)
if not os.path.exists(stage_1_path):
    os.mkdir(stage_1_path)
    print('Directory created at: ' + stage_1_path)
if not os.path.exists(stage_2_path):
    os.mkdir(stage_2_path)
    print('Directory created at: ' + stage_2_path)

# Generating training data 
# (This is done by performing a parameter sweep via Darktable's CLI)
if generate_stage_1:
    print("Generating training data: stage 1")
    gts_1 = data.generate(proxy_type, 
                  params, 
                  1, 
                  possible_values,
                  num,  
                  name)
    print("Training data generated: stage 1")
    print("Writing image list...")
    write_img_list(name, 1, gts_1)
if generate_stage_2:
    print("Generating training data: stage 2")
    gts_2 = data.generate(proxy_type, 
                  params, 
                  2, 
                  possible_values,
                  num_regression, 
                  name)
    print("Training data generated: stage 2")
    print("Writing image list...")
    write_img_list(name, 2, gts_2)

# Stage 1 - proxy training
if c.TRAIN_PROXY:
    print("Begin proxy training (stage 1)")
    use_gpu = torch.cuda.is_available() 
    if dataset_name is not None:
        gt_list = read_img_list(dataset_name, 1)
    else:
        gt_list = read_img_list(name, 1)
    Train_proxy.run_training_procedure( 
        weight_out_dir, 
        stage_1_batch_size, 
        num_epoch, 
        use_gpu, 
        possible_values, 
        proxy_type,
        params,
        append_params,
        name,
        dataset_name,
        gt_list = gt_list
    )
    print(f'{name}: proxy training completed.')

# Stage 2 - parameter regression experiment
if not append_params:
    print(f'{proxy_type} has no input parameters and therefore cannot be used for stage 2.')
    print('Skipping stage 2.')
    quit()
if not c.REGRESS_PROXY:
    print("Done.")
    quit()
print("Begin finetune parameters (stage 2)")
sys.stdout.flush()
use_gpu = torch.cuda.is_available()
Parameter_regression.run_finetune_procedure(
    image_root_dir, 
    param_out_dir,
    weight_out_dir,#model_weight_file,
    possible_values,
    num_iters,
    use_gpu,
    proxy_type,
    params,
    name
)
print(f'{name}: slider regression completed.\n Done.')