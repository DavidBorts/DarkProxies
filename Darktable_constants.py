# File to store constants

INTERACTIVE = True
NUM_IMAGE_CHANNEL = 3
CROP_SIZE = 256
IMG_SIZE = 736 # was 1504

# Data generation (Stage 0)
GENERATE_DATA = True
DARKTABLE_DIR = "C:/Program Files/darktable/bin/darktable-cli.exe"
#DARKTABLE_DIR = "/opt/darktable/bin/darktable-cli"
IMAGE_ROOT_DIR = '.'
INPUT_DIR = 'input/'
STAGE_1_DNG_PATH = 'images/stage_1/'
STAGE_2_DNG_PATH = 'images/stage_2/'
OUTPUT_DIR = 'output/'
OUTPUT_PREDICTIONS_DIR = 'predictions/'

# Training the proxy 
STAGE_1_PATH = 'stage_1/'
PROXY_MODEL_BATCH_SIZE = 1 # Should be 20
PROXY_MODEL_NUM_EPOCH = 600 # Should be 1000, was 300
MODEL_WEIGHTS_PATH = 'model_weights/' #= '/model_weights/'
SAVE_OUTPUT_FREQ = 50 # Should be 50
OUTPUT_PREDICTIONS_PATH = 'predictions/'
SAVE_CROPS = False
CROPPED_INPUT_DIR = 'input_crop/'
CROPPED_OUTPUT_DIR = 'output_crop/'
SWEEP_INPUT_DIR = 'sweep_input'
SWEEP_OUTPUT_DIR = 'sweep_output'

# Fine-tuning parameters using trained proxy (Stage 2)
STAGE_2_PATH = 'stage_2/'
PARAM_PATH = 'stage_2/params/'
PARAM_TUNING_BATCH_SIZE = 1
PARAM_TUNING_NUM_ITER = 100
class POSSIBLE_VALUES:
    def __init__(self):
        self.colorbalancergb_contrast =  [(-0.9, 0.9)]
        self.sharpen_amount = [(0.0, 10.0)]
        self.exposure_exposure = [(3.0, 4.0)]#[(-2.0, 4.0)]
        self.highlights = [(0.05, 0.4)] # highlights clipping threshold
        self.hazeremoval_strength = [(-1.0, 1.0)]
        self.denoiseprofile_strength = [(0.001, 1000.0)]
        self.lowpass_radius = [(0.1, 500.0)]
        self.censorize_pixelate = [(0.0, 500.0)]
        self.censorize_noise = [(0.0, 1.0)]
    
    def get_colorbalancergb_contrast(self):
        return self.colorbalancergb_contrast
    
    def get_sharpen_amount(self):
        return self.sharpen_amount
    
    def get_exposure_exposure(self):
        return self.exposure_exposure
    
    def get_highlights(self):
        return self.highlights

    def get_hazeremoval(self):
        return self.hazeremoval_strength
    
    def get_denoiseprofile(self):
        return self.denoiseprofile_strength
    
    def get_lowpass(self):
        return self.lowpass_radius

