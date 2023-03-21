# Model evaluation helper script
#
# Evaluates any given model on any number of specified parameter values.
# Handles both data generation and evaluation in one script.
#
# How to use: python Darktable_eval.py [proxy type]_[parameter] [path to .txt file with parameters to evaluate on]

import os
import sys
import torch

# Local files
from Generate_data import generate_eval
from Train_proxy import run_eval_procedure
import Darktable_constants as c

# Command-line arguments
proxy_type, param = sys.argv[1].split('_') # proxy type and parameter type
params_file = sys.argv[2]                  # path to .txt file with parameters to evaluate on

# Constants
weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)
use_gpu = torch.cuda.is_available()
possible_values = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)

# Generating data for evaluation
params_mat = generate_eval(proxy_type, param, params_file)

# Evaluating on generated data
run_eval_procedure(c.IMAGE_ROOT_DIR, weight_out_dir, use_gpu, params_mat, possible_values, proxy_type, param)