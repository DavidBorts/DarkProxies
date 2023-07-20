# Model evaluation helper script
#
# Evaluates any given model on any number of specified parameter values.
# Handles both data generation and evaluation in one script.
#
# How to use: python Eval.py [model_name] [path to .txt file with parameters to evaluate on]

import os
import argparse
import torch

# Local files
from Generate_data import generate_eval
from Train_proxy import run_eval_procedure
from utils.misc import get_possible_values, sort_params
import Constants as c

# Argparser args
parser = argparse.ArgumentParser()
parser.add_argument("proxy", help="Name of the model to evaluate")
parser.add_argument("params_file", help=" Path to the .txt file with the parameters to evaluate on")
#TODO: implement params flag
parser.add_argument("-p", "--params", help="[OPTIONAL] Specify a list of _ separated input parameters\
                    on which to train a proxy, keeping all others fixed (i.e. \
                    -p contrast_radius_brightness)", default=None)
args = parser.parse_args()
name = args.proxy
proxy_type = name.split('_')[0]
params = args.params
if params is None:
    params = c.PARAM_NAMES[proxy_type]
else:
    params = params.split('_')
    params = sort_params(proxy_type, params)
params_file = args.params_file

# Constants
#TODO: add support for proxies w/out params
weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, name + '_' + c.MODEL_WEIGHTS_PATH)
use_gpu = torch.cuda.is_available()
possible_values = get_possible_values(proxy_type, params)

# Generating data for evaluation
params_mat = generate_eval(proxy_type, params, params_file, name, possible_values)

# Evaluating on generated data
run_eval_procedure(weight_out_dir, use_gpu, params_mat, possible_values, proxy_type, params, name)