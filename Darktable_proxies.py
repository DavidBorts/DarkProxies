# Main script for data generation, proxy training, and hyperparameter finetuning (slider regression)

import sys
import os
import torch
from Dataset import Darktable_Dataset

# Local files
import Darktable_generate_data as data
import Darktable_constants as c
import Darktable_train_proxy
import Darktable_finetune_parameters

# Command-line args
proxy_type, param = sys.argv[1].split('_')  # Type of proxy to train [module, param]
num = int(sys.argv[2])                      # Number of images to generate

# Constants
image_root_dir = c.IMAGE_ROOT_DIR
generate_stage_1 = c.GENERATE_STAGE_1
generate_stage_2 = c.GENERATE_STAGE_2
interactive = c.INTERACTIVE

# Stage 1 constants
stage_1_batch_size = c.PROXY_MODEL_BATCH_SIZE
num_epoch = c.PROXY_MODEL_NUM_EPOCH
weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)

# Stage 2 constants
num_iters = c.PARAM_TUNING_NUM_ITER
param_out_dir = c.STAGE_2_PARAM_PATH

# Adjusting for varying training requirement across different proxies
append_params = proxy_type not in c.NO_PARAMS
possible_values = None
if append_params: 
    possible_values = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)
else:
    #TODO: replace with highlights or temperature to support demosaic?
    if proxy_type == 'colorin':
        possible_values = getattr(c.POSSIBLE_VALUES(), 'exposure_exposure')

    if proxy_type == 'colorout':
        possible_values = getattr(c.POSSIBLE_VALUES(), 'colorbalancergb_contrast')

    # Temporary hack: not sweeping for demosaic
    if proxy_type == 'demosaic':
        possible_values = getattr(c.POSSIBLE_VALUES(), 'demosaic')

if __name__ == "__main__":

    min = possible_values[0][0]
    max = possible_values[0][1]

    # Creating stage 1 and 2 directories if they do not already exist
    stage_1_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH)
    stage_2_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_PATH)
    if not os.path.exists(stage_1_path):
        os.mkdir(stage_1_path)
        print('Directory created at: ' + stage_1_path)
    if not os.path.exists(stage_2_path):
        os.mkdir(stage_2_path)
        print('Directory created at: ' + stage_2_path)

    # Generating input data 
    # (This is done by performing a parameter sweep via Darktable's CLI)
    # TODO: this could be moved into generate() to avoid ugly if statements
    if generate_stage_1:
        if c.GENERATE_WITH_CHECKPOINTS:
            print("Generating training data (w/ checkpoints): stage 1")
            data.generate_piecewise(proxy_type, param, 1, min, max, num)
        else:
            print("Generating training data: stage 1")
            data.generate(proxy_type, param, 1, min, max, interactive, num)
    if generate_stage_2:
        if c.GENERATE_WITH_CHECKPOINTS:
            print("Generating training data (w/ checkpoints): stage 2")
            data.generate_piecewise(proxy_type, param, 2, min, max, num)
        else:
            print("Generating training data: stage 2")
            data.generate(proxy_type, param, 2, min, max, interactive, num)
    
    # Stage 1 - proxy training
    print("Begin proxy training (stage 1)")
    use_gpu = torch.cuda.is_available() 
    Darktable_train_proxy.run_training_procedure(
        image_root_dir, 
        weight_out_dir, 
        stage_1_batch_size, 
        num_epoch, use_gpu, 
        possible_values, 
        proxy_type,
        param,
        append_params,
        interactive
    )

    # Stage 2 - parameter finetuning
    if not append_params:
        print(f'{proxy_type} has no input parameters and therefore cannot be used for stage 2.')
        print('Skipping stage 2.')
        quit()
    print("Begin finetune parameters (stage 2)")
    sys.stdout.flush()
    use_gpu = torch.cuda.is_available()
    Darktable_finetune_parameters.run_finetune_procedure(
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
    print(f"{proxy_type}: {param} proxy finetuned.\n Done.")