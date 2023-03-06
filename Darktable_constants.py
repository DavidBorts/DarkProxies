'''
File to store script constants/parameters

 ( An X indicates that a variable should
   generally not be tampered with )

 IMPORTANT NOTE: Before running any code, make sure to replace this
 string with the correct path to darktable-cli.exe
 '''
DARKTABLE_PATH = "/home/dborts/programs/darktable/build/bin/darktable-cli" # CHANGE ME!

# Global constants
IMAGE_ROOT_DIR = '.'                     #   Root directory from which to construct all file paths
INTERACTIVE = True                       #   Toggle interactive prompts between stages
NUM_IMAGE_CHANNEL = 3                    # X Number of channels in each image (3 for RGB)
IMG_SIZE = 736                           # X Dimensions to crop all images to (IMG_SIZE x IMG_SIZE)

# Data generation constants (Stage 0)
GENERATE_STAGE_1 = True                  #   Toggles new data generation for proxy training (stage 1)
GENERATE_STAGE_2 = False                 #   Toggles new data generation for slider regression (stage 2)
GENERATE_WITH_CHECKPOINTS = False        #   If True, progress will be tracked in case of interruption
INPUT_DIR = 'input/'                     #   Name of directories that store training data
OUTPUT_DIR = 'output/'                   #   Name of directories that store ground truth data
PARAM_FILE_DIR = 'param_files'           #   Name of directories with data generation checkpoints
CHECK_DIRS = False                       #   Toggles whether INPUT/OUTPUT_DIR need to be empty before generating
STAGE_1_DNG_PATH = 'images/stage_1/'     # X Path to folder with all DNG files for proxy training
STAGE_2_DNG_PATH = 'images/stage_2/'     # X Path to folder with all DNG files for slider regression
STAGE_3_DNG_PATH = 'images/stage_3/'     # X Path to folder with all DNG files for pipeline regression

# Training the proxies (Stage 1)
STAGE_1_PATH = 'stage_1/'                #   Directory that stores all training data, model weights, and predictions
PROXY_MODEL_BATCH_SIZE = 1               #   Batch size for proxy training
PROXY_MODEL_NUM_EPOCH = 500              #   Number of epochs for which to train
MODEL_WEIGHTS_PATH = 'model_weights/'    #   Name of directories where model weights are stored
SAVE_OUTPUT_FREQ = 50                    #   Frequency at which to save model predictions (in terms of epochs)
OUTPUT_PREDICTIONS_PATH = 'predictions/' #   Name of directories where model predictions are stored
OUTPUT_PREDICTIONS_FORMAT = 'tiff'       #   Format to save model predictions as
SAVE_CROPS = False                       #   If True, cropped versions of all training data are saved
CROP_FORMAT = 'png'                      #   Format to save crops as (Default: png)
CROPPED_INPUT_DIR = 'input_crop/'        #   Directory at which to save cropped training data
CROPPED_OUTPUT_DIR = 'output_crop/'      #   Directory at which to save ground truth data
SWEEP_INPUT_DIR = 'sweep_input'          #   Name of directories that store data for the Darktable_sweep.py script
SWEEP_OUTPUT_DIR = 'sweep_output'        #   Name of directories that store proxy outputs for the Darktable_sweep.py script
EVAL_PATH = 'eval/'                      #   Name of directories that store evluations data and results

# Slider regression on trained proxies (Stage 2)
STAGE_2_PATH = 'stage_2/'                #   Directory that stores all training data and predictions
STAGE_2_PARAM_PATH = 'stage_2/params/'   #   Directory that stores optimized proxy params
ANIMATIONS_DIR = 'animations/'           #   Directory that stores frames for all animations
PARAM_TUNING_NUM_ITER = 100              #   Number of regression iterations per image
CREATE_ANIMATION = True                  #   If True, saves frames from regression to use for videos
ANIMATIONS_FORMAT = 'tiff'               #   Format to save frames as (Default: png)

# Assembling the trained proxies into a differentiable ISP (Stage 3)
STAGE_3_PATH = 'stage_3/'                #   Directory that stores all training data and predictions
STAGE_3_INPUT_DIR = 'input/'             #   Directory that stores all training data
STAGE_3_OUTPUT_DIR = 'output/'           #   Directory that stores all ground truth data
STAGE_3_PARAM_DIR = 'params/'            #   Directory that stores optimized proxy params
CONFIG_FILE = 'pipeline_config.txt'      # X Path to file with pipeline conifugration
PIPELINE_REGRESSION_NUM_ITERS = 2000     #   Number of regression iterations per image
PIPELINE_OUTPUT_FORMAT = 'tiff'          #   Format to save pipeline outputs as (Default: png)
PIPELINE_CREATE_ANIMATION = True         #   If True, saves frames from regression to use for videos
PIPE_REGRESSION_ANIMATION_FORMAT = 'png' #   Format to save frames as (Default: png)

'''
Dictionary to store which Darktable blocks will require which loss functions to train

Format: [stage_1, stage_2]
'''
WHICH_LOSS = {
    'highlights': ['MSE', 'L1'],
    'demosaic': ['MSE', 'L1'],
    'denoise': ['MSE', 'L1'],
    'hazeremoval': ['MSE', 'L1'],
    'exposure': ['MSE', 'L1'],
    'colorin': ['MSE', 'L1'],
    'censorize': ['MSE', 'L1'],
    'lowpass': ['MSE', 'L1'],
    'sharpen': ['Spatial', 'Spatial'],
    'colorbalancergb': ['MSE', 'L1'],
    'colorout': ['MSE', 'L1']
}

# DO NOT MODIFY ANYTHING BELOW THIS LINE!!!
################################################################################################################################

# Class to store the ranges of various Darktable paramters
# TODO: Add support for highlights
# TODO: Rework to support multiple paramters per proxy
class POSSIBLE_VALUES:
    colorbalancergb_contrast =  [(-0.9, 0.9)]
    sharpen_amount = [(0.0, 10.0)]
    exposure_exposure = [(-2.0, 4.0)]
    hazeremoval_strength = [(-1.0, 1.0)]
    denoiseprofile_strength = [(0.001, 1000.0)]
    lowpass_radius = [(0.1, 500.0)]
    censorize_pixelate = [(0.0, 500.0)]
    censorize_noise = [(0.0, 1.0)]
    demosaic = [(None, None)] # Temporary Hack. TODO: remove me!

'''
Dictionary to store which Darktable blocks will require
which intermediary tapouts from the Darktable pipeline as
part of their training data

Each block name has a correponding list of length 2, in
the following format: [input tapout, ground truth tapout]
'''
TAPOUTS = {

    # Highlights needs to be trained on images that have
    # not yet been demosaiced
    'highlights': ['highlights_bayer_in', 'highlights_bayer_out'],

    # Demosaicing needs to be trained on images that have
    # not had any Darktable operations performed on them
    # TODO: replace with ['highlights_bayer_out', 'colorin_in'] ??
    'demosaic': ['temperature_bayer_out', 'colorin_in'],

    # Denoise through colorin all need to be trained on
    # images that have not yet had their colorspaces
    # tranformed by the colorin block
    # TODO: ADD TAPOUT SUPPORT FOR DENOISE, HAZEREMOVAL,
    # CENSORIZE, & LOWPASS
    'denoise': ['demosaic_out', 'denoise_out'],
    'hazeremoval': ['demosaic_out', 'hazeremoval_out'],
    'exposure': ['demosaic_out', 'exposure_out'],
    'colorin': ['colorin_in', 'colorbalancergb_out'],

    # All blocks after colorin and before colorout
    # require no tapouts for training
    'censorize': None,
    'lowpass': None,
    'sharpen': None,
    'colorbalancergb': None,

    # Colorout needs to be trained on images that
    # have not yet been processed for output by the
    # colorout block
    # TODO: Do we need 'colorout_out'??
    'colorout': ['colorbalancergb_out', 'colorout_out']
}

# List of Darktable blocks that do not require input parameters
NO_PARAMS = ['colorin','colorout', 'demosaic']

# Dict of which blocks to sample parameters from for parameter-less
# blocks like colorin and colorout
SAMPLER_BLOCKS = {
    'colorin': 'exposure_exposure',
    'colorout': 'colorbalancergb_contrast',
    'demosaic': None
}
