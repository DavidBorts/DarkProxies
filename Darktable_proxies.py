# Main script for data generation, proxy training, and hyperparameter finetuning (slider regression)

import sys
import os
import argparse
import torch

# Local files
import Generate_data as data
import Constants as c
import Train_proxy
import Parameter_regression

if __name__ != '__main__':
    raise RuntimeError("This script is only configured to be called directly by the user!")

# argparser args
parser = argparse.ArgumentParser()
parser.add_argument("proxy", help="Name of the Darktable block for which to train a proxy network", 
                    choices = ['colorbalancergb', 'sharpen', 'exposure', 'colorin', 'colorout', 
                               'demosaic'], required=True)
parser.add_argument("-m", "--mono", help="[OPTIONAL] Train a proxy on only one user-specified input \
                    parameter for a given Darktable block, keeping the others fixed", default=None)
parser.add_argument("-n", "--number", help="Number of training examples to generate for each \
                    source DNG image", required=True, default=0)
args = parser.parse_args()
proxy_type = args.proxy
param = args.mono
num = args.number

# Global constants
image_root_dir = c.IMAGE_ROOT_DIR
interactive = c.INTERACTIVE

# Stage 1 constants
generate_stage_1 = c.GENERATE_STAGE_1
stage_1_batch_size = c.PROXY_MODEL_BATCH_SIZE
num_epoch = c.PROXY_MODEL_NUM_EPOCH
if param is not None:
    weight_out_dir = os.path.join(image_root_dir, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)
else:
    weight_out_dir = os.path.join(image_root_dir, c.STAGE_1_PATH, proxy_type + '_' + c.MODEL_WEIGHTS_PATH)

# Stage 2 constants
generate_stage_2 = c.GENERATE_STAGE_2
num_iters = c.PARAM_TUNING_NUM_ITER
param_out_dir = c.STAGE_2_PARAM_PATH

# Adjusting for varying training requirement across different proxies
append_params = proxy_type not in c.NO_PARAMS

# If the given proxy takes input params, it is necesary to find their
# ranges of possible values
possible_values = None
if append_params: 
    if param is not None:
        # Mono-proxies only have one input param unfrozen
        # NOTE: possible_values needs to be a list
        possible_values = [c.POSSIBLE_VALUES[proxy_type][c.PARAM_NAMES[proxy_type].index(param)]]
    else:
        possible_values = c.POSSIBLE_VALUES[proxy_type]
else:
    # Proxy has no input parameters - use inputs of a sampler block instead
    #TODO: replace with highlights or temperature to support demosaic?

    if proxy_type == 'demosaic': # Temporary hack: not sweeping for demosaic
        possible_values = [(None, None)]
    else:
        sampler_block, sampler_param = c.SAMPLER_BLOCKS[proxy_type].split('_')
        possible_values = [c.POSSIBLE_VALUES[sampler_block][c.PARAM_NAMES[sampler_block].index(sampler_param)]]

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
# TODO: does generate() really need interactive as an argument?
if generate_stage_1:
    if c.GENERATE_WITH_CHECKPOINTS:
        print("Generating training data (w/ checkpoints): stage 1")
        data.generate_piecewise(proxy_type, param, 1, possible_values, num)
    else:
        print("Generating training data: stage 1")
        data.generate(proxy_type, param, 1, possible_values, interactive, num)
if generate_stage_2:
    if c.GENERATE_WITH_CHECKPOINTS:
        print("Generating training data (w/ checkpoints): stage 2")
        data.generate_piecewise(proxy_type, param, 2, possible_values, c.STAGE_2_NUM_IMAGES)
    else:
        print("Generating training data: stage 2")
        data.generate(proxy_type, param, 2, possible_values, interactive, c.STAGE_2_NUM_IMAGES)

# Stage 1 - proxy training
if c.TRAIN_PROXY:
    print("Begin proxy training (stage 1)")
    use_gpu = torch.cuda.is_available() 
    Train_proxy.run_training_procedure( 
        weight_out_dir, 
        stage_1_batch_size, 
        num_epoch, use_gpu, 
        possible_values, 
        proxy_type,
        param,
        append_params,
        interactive
    )

# Stage 2 - parameter regression
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
    param,
    interactive
)
if param is not None:
    print(f"{proxy_type}: {param} slider regression completed.\n Done.")
else:
    print(f"{proxy_type} slider regression completed.\n Done.")