'''
File to store script constants/parameters

 ( An X indicates that a variable should
   generally not be tampered with )

 IMPORTANT NOTE: Before running any code, make sure to replace the
 following string with the correct path to darktable-cli.exe
 '''
#DARKTABLE_PATH = "/home/dborts/programs/darktable/build/bin/darktable-cli" # CHANGE ME!
DARKTABLE_PATH = "/home/dborts/programs/testing/nt-darktable/darktable/build/bin/darktable-cli"

# Global constants
IMAGE_ROOT_DIR = '.'                     #   Root directory from which to construct all file paths
LOG_DIR  = 'runs'                        #   Directory name in which to dump TensorBoard logs
INTERACTIVE = False                      #   Toggle interactive prompts between stages
MAGICK_COMMAND = "convert"
NUM_IMAGE_CHANNEL = 3                    # X Number of channels in each image (3 for RGB)
IMG_SIZE = 736                           # X Dimensions to crop all images to (IMG_SIZE x IMG_SIZE)
CLIP_OUTPUT = True                       # X Toggle clipping of proxy outputs
CLIP_RANGE = [0.0, 1.0]                  # X  Set lower and upper bounds to clip model outputs to
RESCALE_PARAMS = True                    # X Toggle normalization of input parameters for model training/eval
EMBEDDING_TYPE = 0                       # X Selects method of embedding parameters into a latent vector
EMBEDDING_RATIO = 2                      # X Number of input parameters to map to a single latent parameter/channel
EMBED_TO_SINGLE = False                  # X If True, input paramaters are mapped to a single latent parameter
DOWNSAMPLE_IMAGES = False                # X Toggles downsampling of training data w/ bilinear interpolation
NPF_BASELINE = False                     # X Toggles NPF (Tseng et al. 2022) model architecture
SKIP_CONNECT = True                      # X Toggles skip connections in the neural networks

# Data generation constants (Stage 0)
GENERATE_STAGE_1 = False                  #   Toggles new data generation for proxy training (stage 1)
GENERATE_STAGE_2 = False                 #   Toggles new data generation for slider regression (stage 2)
INPUT_DIR = 'input/'                     #   Name of directories that store training data
OUTPUT_DIR = 'output/'                   #   Name of directories that store ground truth data
STAGE_1_DNG_PATH = 'images/stage_1/'     # X Path to folder with all DNG files for proxy training
STAGE_2_DNG_PATH = 'images/stage_2/'     # X Path to folder with all DNG files for slider regression
STAGE_3_DNG_PATH = 'images/stage_3/'     # X Path to folder with all DNG files for pipeline regression
TAPOUTS = False

# Training the proxies (Stage 1)
TRAIN_PROXY = True                       #   Toggles proxy training
STAGE_1_PATH = 'stage_1/'                #   Directory that stores all training data, model weights, and predictions
PROXY_MODEL_BATCH_SIZE = 2               #   Batch size for proxy training
PROXY_MODEL_NUM_EPOCH = 4500              #   Number of epochs for which to train
MODEL_WEIGHTS_PATH = 'model_weights/'    #   Name of directories where model weights are stored
SAVE_OUTPUT_FREQ = 10                    #   Frequency at which to save model predictions (in terms of epochs)
OUTPUT_PREDICTIONS_PATH = 'predictions/' #   Name of directories where model predictions are stored
# TODO: make sure that crops are only saved once
SAVE_CROPS = False                       #   If True, cropped versions of all training data are saved
CROPPED_INPUT_DIR = 'input_crop/'        #   Directory at which to save cropped training data
CROPPED_OUTPUT_DIR = 'output_crop/'      #   Directory at which to save ground truth data
SWEEP_INPUT_DIR = 'sweep_input'          #   Name of directories that store data for the Sweep.py script
SWEEP_OUTPUT_DIR = 'sweep_output'        #   Name of directories that store proxy outputs for the Sweep.py script
EVAL_PATH = 'eval/'                      #   Name of directories that store evluations data and results

# Slider regression on trained proxies (Stage 2)
REGRESS_PROXY = False                     #   Toggles proxy regression
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
STAGE_3_LOSS_FN = 'MSE'                  #   Loss for Pipeline Regression (MSE, L1, Spatial, or Perceptual)
CONFIG_FILE = 'pipeline_config.txt'      # X Path to file with pipeline conifugration
SAVE_ALL_OUTPUTS = True                  #   Toggles saving of all intermediary model outputs in the pipeline
PIPELINE_REGRESSION_NUM_ITERS = 2000     #   Number of regression iterations per image
PIPELINE_CREATE_ANIMATION = True         #   If True, saves frames from regression to use for animations

WHICH_LOSS = {
    'highlights': ['MSE', 'L1'],
    'demosaic': ['MSE', 'L1'],
    'denoise': ['MSE', 'L1'],
    'hazeremoval': ['MSE', 'L1'],
    'exposure': ['MSE', 'L1'],
    'graduateddensity': ['MSE', 'L1'],
    'colorin': ['MSE', 'L1'],
    'censorize': ['MSE', 'L1'],
    'lowpass': ['MSE', 'L1'],
    'sharpen': ['Spatial', 'Spatial'],
    'colorbalancergb': ['MSE', 'L1'],
    'filmicrgb': ['MSE', 'L1'],
    'bloom': ['MSE', 'L1'],
    'colorize': ['MSE', 'L1'],
    'soften': ['MSE', 'L1'],
    'colorout': ['MSE', 'L1'],
    'temperature': ['MSE', 'L1']
}
'''
Dictionary to store which Darktable blocks will require which loss functions to train

Options: MSE, L1, Spatial, Perceptual

Format: [stage_1, stage_2]
'''

# DO NOT MODIFY ANYTHING BELOW THIS LINE!!!
################################################################################################################################

# TODO: Add support for highlights
POSSIBLE_VALUES = {
    'colorbalancergb': [(-1.0, 1.0), (0.0, 1.0), (-0.9, 0.9)],
    'sharpen': [(0.0, 99.0), (0.0, 2.0), (0.0, 100.0)],# FIXME: amount was 0.0 - 10.0 ??
    'exposure': [(-1.0, 1.0), (-2.0, 4.0), (0.0, 100.0), 
                 (-18.0, 18.0)],# FIXME: exposure is actually -18.0 - 18.0??
    'graduateddensity':[(-8.0, 8.0), (0.0, 100.0), (-180.0, 180.0), (0.0, 1.0), (0.0, 1.0)],
    'hazeremoval': [(-1.0, 1.0), (0.0, 1.0)],
    'denoiseprofile': [(0.0, 12.0), (1.0, 30.0), (0.001, 1000.0), 
                       (0.0, 1.8), (-1000.0, 100.0), (0.0, 20.0), 
                       (0.0, 10.0), (0.001, 1000.0)],
    'lowpass': [(0.1, 20.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],# NOTE: radius truncated to 20.0, not 500.0
    'filmicrgb': [],
    'bloom': [(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)],
    'colorize': [(0.0, 1.0), (0.0, 1.0), (0.0, 100.0), (0.0, 100.0)],
    'soften': [(0.0, 100.0), (0.0, 100.0), (-2.0, 2.0), (0.0, 100.0)],
    'temperature': [(0.0, 8.0), (0.0, 8.0), (0.0, 8.0)],
    'highlights': [None]
}
'''
Dictionary to store the range of possible values for every
parameter of all supported Darktable blocks
'''

PARAM_NAMES= {
    'colorbalancergb': ['vibrance', 'grey_fulcrum', 'contrast'],
    'sharpen': ['radius', 'amount', 'threshold'],
    'exposure': ['black', 'exposure', 'deflicker_percentile', 
                 'deflicker_target_level'],
    'graduateddensity': ['density', 'hardness', 'rotation',
                         'hue', 'saturation'],
    'hazeremoval': ['strength', 'distance'],
    'denoiseprofile': ['radius', 'nbhood', 'strength', 'shadows', 
                       'bias', 'scattering', 'central_pixel_weight', 
                       'overshooting'],
    'lowpass': ['radius', 'contrast', 'brightness', 'saturation'],
    'filmic': [],
    'bloom': ['size', 'threshold', 'strength'],
    'colorize': ['hue', 'saturation', 'source_lightness_mix',
                 'lightness'],
    'soften': ['size', 'saturation', 'brightness','amount'],
    'temperature': ['red', 'green', 'blue'],
    'highlights': [None]
}
'''
Dictionary to store which parameter name maps to which index
in POSSIBLE_VALUES for each proxy type

NOTE: each list only includes the params supported by this codebase,
and not necessarily every param in the module (some are deprecated or
not otherwise useful to learn)
'''

NO_PARAMS = ['colorin','colorout', 'demosaic']
'''
List of Darktable blocks that do not require input parameters
'''

SINGLE_IMAGE_CHANNEL = []
'''
List of Darktable blocks that only have one image channel
(in other words, this is all blocks up to demosaic)
'''

SAMPLER_BLOCKS = {
    'colorin': 'exposure_exposure',
    'colorout': 'colorbalancergb_contrast',
    'demosaic': 'highlights_strength'
}
'''
Dict of which blocks to sample parameters from for parameter-less
blocks like colorin and colorout
'''

EMBEDDING_TYPES = ["none", 
                   "linear_to_channel",
                   "linear_to_value"]
'''
Types of parameter embedding used by proxy neural networks
(set by the integer EMBEDDING_TYPE constant)

0, none: (DEFAULT) no embedding - each param is its own channel
1, linear_to_channel: learn a linear layer to map params to n channels
2, linear_to_value: learn a linear layer to map params to n float value
'''