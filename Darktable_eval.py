import os
import sys
import torch

# Local files
from Darktable_generate_data import generate_eval
from Darktable_train_proxy import run_eval_procedure
import Darktable_constants as c

proxy_type, param = sys.argv[1].split('_')
input_path = sys.argv[2]
output_path = sys.argv[3]
params_file = sys.argv[4]

# Constants
weight_out_dir = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, proxy_type + '_' + param + '_' + c.MODEL_WEIGHTS_PATH)

# Generating data for eval
params_mat = generate_eval(proxy_type, param, input_path, output_path, params_file)

use_gpu = torch.cuda.is_available()
possible_values = getattr(c.POSSIBLE_VALUES(), proxy_type + '_' + param)

run_eval_procedure(c.IMAGE_ROOT_DIR, weight_out_dir, use_gpu, input_path, output_path, params_mat, possible_values, proxy_type, param)