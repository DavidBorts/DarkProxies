# Model evaluation helper script
#
# Evaluates any given model on any number of specified parameter values.
# Handles both data generation and evaluation in one script.

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
parser.add_argument("-p", "--params", help="Specify a list of comma-separated input parameters\
                    on which to evaluate, keeping all others fixed (i.e. \
                    -p contrast_radius_brightness)")
parser.add_argument("-v", "--values", help="Specify a parallel list to --params of comma-separated\
                    values for the input parameters specified in --params")
args = parser.parse_args()
name = args.proxy
proxy_type = name.split('_')[0]
values = args.values
assert(proxy_type not in c.NO_PARAMS)
values = values.split(',')
params = args.params
params = params.split(',')
params, values = sort_params(proxy_type, params, values=values)

# Constants
weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, name + '_' + c.MODEL_WEIGHTS_PATH)
print(f"Loading weights from: {weight_out_dir}")
use_gpu = torch.cuda.is_available()
possible_values = get_possible_values(proxy_type, params)

# Generating data for evaluation
params_mat = generate_eval(proxy_type, params, values, name, possible_values)

# Evaluating on generated data
run_eval_procedure(weight_out_dir, use_gpu, params_mat, possible_values, proxy_type, params, name)