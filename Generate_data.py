# Data generation methods

import os
import numpy as np

# Local files
import Constants as c
import PyDarktable as dt
from Tapouts import tmp2tiff
from npy_convert import convert
from utils.latin_hypercube_sampling import lhs

class ParamSamplerIterator():
    '''
    Iterator for sampled parameter values
    '''
    def __init__(self, sampler, params, list):
        self.sampler = sampler
        self.params = params
        self.list = list
        self.num = len(sampler)
        self.idx = 0
    
    def __next__(self):

        vals = None
        if self.idx < self.num:

            if len(self.params) > 1:
                vals = np.squeeze(self.list[:, self.idx])
            else:
                return [self.list[self.idx]]

            self.idx += 1
            return vals
        raise StopIteration

class ParamSampler():
    '''
    Iterable wrapper class to sample the parameter space of any 
    proxy, no matter its dimensionality
    '''
    
    def __init__(self, proxy_type, params, possible_values, num):
        self.proxy_type = proxy_type
        self.params = params
        self.possible_values = possible_values
        self.num = int(num)

        if len(self.params) > 1:
            self.list = lhs(self.possible_values, self.num)
        else:
            self.list = np.linspace(self.possible_values[0][0], self.possible_values[0][1], self.num)

    def __iter__(self):
        return ParamSamplerIterator(self, self.params, self.list)
    
    def __len__(self):
        return self.num

def generate(proxy_type, params, stage, possible_values, interactive, num, name):

    # Getting stage paths
    stage_path = getattr(c, 'STAGE_' + str(stage) + '_PATH')

    # Getting image directory paths
    input_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, name + c.INPUT_DIR)
    output_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, name + c.OUTPUT_DIR)

    # Creating given input and output directories if they do not already exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Directory created at: ' + output_path)
    
    # Checking that given input and output directories are empty
    if ((len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0) and c.CHECK_DIRS):
        print(f'ERROR: directories already contain data for stage {stage}.')

        # Checking if user wants to skip to stage 2 data generation
        if stage == 1 and interactive:
            skip = None
            while skip != 'n' and skip != 'y':
                skip = input('Do you want to skip to generating data for stage 2? (y/n)\n')
            if skip == 'n':
                print('quitting')
                quit()
            else:
                return

        # Checking if user wants to skip to proxy training
        if stage == 2 and interactive:
            skip = None
            while skip != 'n' and skip != 'y':
                skip = input('Do you want to skip to proxy training? (y/n)\n')
            if skip == 'n':
                print('quitting')
                quit()
            else:
                return

    # Information about tapout requirement for the given proxy
    tapouts = c.TAPOUTS[proxy_type]

    # Which parameter spaces to sample
    # (by default, proxies with input parameters sample their own
    # paramater spaces. Howvever, for proxies without input paramaters,
    # the parameter space of a different processing block, called a 
    # "sampler block" in this codebase, is sampled instead to augment 
    # training data. This effects which rendering parameters are passed
    # to darktable-cli)
    proxy_type_gt = proxy_type
    params_gt = params
    if tapouts is not None and proxy_type != "demosaic":#TODO: temporary hack, remove me!!
        proxy_type_gt, params_gt = c.SAMPLER_BLOCKS[proxy_type].split('_')
        params_gt = [params_gt] # NOTE: params_gt, like params, must be a list

    # Getting DNG images
    if stage == 1:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    else:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_DNG_PATH)
    src_images = sorted(os.listdir(dng_path))

    # Temporary hack: not sweeping parameter spaces for demosaic
    if possible_values[0][0] is None or possible_values[0][1] is None:
        generate_single(proxy_type, dng_path, src_images, input_path, output_path, tapouts)
        print(f"Training data generated: stage {stage}")
        return
    
    sampler = ParamSampler(proxy_type_gt, params_gt, possible_values, num)
    if params_gt is None:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(len(src_images))], axis=1)
    else:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(len(src_images))])
    filename = name + 'params.npy'
    convert(samples_concatenated, os.path.join(c.IMAGE_ROOT_DIR, stage_path, filename), ndarray=True)
    print("Params file saved.")

    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # No need to render input image separately when using ISP tap-outs
        if tapouts is not None:
            # Getting path of the input image
            input_file_path = os.path.join(input_path, image.split('.')[0])
            input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

            # Checking if input image already exists
            # (this can happen if a previous data generation job was interrupted)
            input_exists = os.path.exists(input_file_path)
            
            # Assembling a dictionary of all of the original params for the source image
            # (used to render proxy input)
            original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

            # Rendering an unchanged copy of the source image for model input
            if not input_exists:
                dt.render(src_path, input_file_path, original_params)

        # Parameter value sweep
        # (to generate ground truth images)
        for values in sampler:

            # Getting path of the ground truth image
            gt_file_path = os.path.join(output_path, f'{image}_{proxy_type}')
            gt_file_path = (repr(gt_file_path).replace('\\\\', '/')).strip("'") # Dealing with Darktable CLI pickiness
            for val in values:
                gt_file_path += '_' + str(val)
            gt_file_path += '.tif'

            # Skip ground truth images that already exist
            if os.path.exists(gt_file_path):
                continue
            
            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type_gt, params_gt, values, temperature_params, raw_prepare_params)

            # Rendering the output image
            dt.render(src_path, gt_file_path, params_dict)

            # Checking if input & ground truth images need to be replaced with tapouts
            if tapouts is not None:
                
                # Getting file path of the tapout
                input_tapout_path = tapouts[0] + '.tmp'
                gt_tapout_path = tapouts[1] + '.tmp'
                #print('input tapout path: ' + input_tapout_path)
                #print('ground truth tapout path: ' + gt_tapout_path)

                # Getting path of the input image
                input_file_path = os.path.join(input_path, image.split('.')[0])
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

                # Deleting final output image
                os.remove(gt_file_path)
                print('replacing: ' + gt_file_path)

                # Read in the tapouts and save them as tiffs
                tmp2tiff(input_tapout_path, input_file_path)
                tmp2tiff(gt_tapout_path, gt_file_path)
    print(f"Training data generated: stage {stage}")

# Renders a single input -> ground truth pair for a given source image in src_images, 
# instead of sweeping over a range of parameter values
def generate_single(proxy_type, dng_path, src_images, input_path, output_path, tapouts):
    
    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)

        # Extracting necessary params from the source image
        raw_prepare_params, _ = dt.read_dng_params(src_path)

        # Getting path of the input image
        input_file_path = os.path.join(input_path, image.split('.')[0])
        input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
        
        # Assembling a dictionary of all of the original params for the source image
        # (used to render proxy input)
        original_params = dt.get_params_dict(None, None, None, dt.TemperatureParams(), raw_prepare_params)

        # Rendering an unchanged copy of the source image for model input
        dt.render(src_path, input_file_path, original_params)

        # Rendering the output image
        output_file_path = os.path.join(output_path, f'{image}_{proxy_type}')
        output_file_path = (repr(output_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
        dt.render(src_path, output_file_path, original_params)

        # If the given proxy is colorin, colorout, or any proxy that appears before colorin,
        # The rendered image needs to be replaced with an intermediary tapout from
        # the Darktable CLI (This is because colorin converts to a different
        # colorspace, which would leave the rendered image in a different colorspace
        # from what the given proxy requires as input)
        if tapouts is not None:
            
            # Getting file path of the tapouts
            tapout_path_input = tapouts[0] + '.tmp'
            tapout_path_gt = tapouts[1] + '.tmp'

            # Deleting final output image
            os.remove(input_file_path)
            os.remove(output_file_path)

            # Read in the tapout and save as a tiff
            tmp2tiff(tapout_path_input, input_file_path, color='minisblack')
            tmp2tiff(tapout_path_gt, output_file_path)

#TODO: UPDATE ME TO ONLY GENERATE ONE INPUT IMAGE PER DNG!!
def generate_eval(proxy_type, param, params_file):
    
    # Getting path of the params matrix file
    eval_path = os.path.join(c.IMAGE_ROOT_DIR, c.EVAL_PATH)
    params_mat = os.path.join(eval_path, f'{proxy_type}_{param}_eval_params.npy')

    # Getting paths of input and output directories
    input_path = os.path.join(eval_path, f'{proxy_type}_{param}_input')
    output_path = os.path.join(eval_path, f'{proxy_type}_{param}_output')

    # Creating any directories that don't already exist
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
        print('Directory created at: ' + eval_path)
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)  
        print('Directory created at: ' + output_path) 

    # Checking that given input and output directories are empty
    if ((len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0) and c.CHECK_DIRS):
        print(f'ERROR: {proxy_type}: {param} eval directories already contain data.')
        return params_mat

    # List of all slider values
    vals = []

    # Getting path of source DNG files
    dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    src_images = os.listdir(dng_path)

    # Reading through the params .txt file and generating data accordingly
    with open(params_file, 'r') as file:
        values = file.readlines()

        for value in values:
            for image in src_images:

                # Getting path of individual source DNG file
                src_path = os.path.join(dng_path, image)

                # Extracting necessary params from the source image
                raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering the input image
                input_file_path = os.path.join(input_path, f'{image}_{proxy_type}_{param}')
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
                dt.render(src_path, input_file_path, original_params)

                # Assembling a dictionary of all of the parameters to apply to the source DNG
                # Temperature and rawprepare params must be maintained in order to produce expected results
                params_dict = dt.get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)
                vals.append(float(value)) # Adding to params list

                # Rendering the output image
                output_file_path = os.path.join(output_path, f'{image}_{proxy_type}_{param}') #f'./stage_1/contrast_input/contrast_{contrast}.tif'
                output_file_path = (repr(output_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
                dt.render(src_path, output_file_path, params_dict)

    # Converting param list to numpy array and saving to file
    convert(vals, params_mat)

    print("Data for evaluation generated.")
    return params_mat

#TODO: Re-work to support tapouts
def generate_pipeline(param_file, proxy_order, input_path, label_path, dng_path=None):
    
    # Reading in the list of input param values
    params_list = None
    with open(param_file, 'r') as file:
        params_list = file.readlines()

    # Checking that input_path and label_path exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Input directory created at: ' + input_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)
        print('Ground truth label directory created at: ' + label_path)

    # Getting path of source DNG images
    if dng_path == None:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_DNG_PATH)
    src_images = os.listdir(dng_path)
    
    for image_num in range(len(src_images)):

        # Getting path of individual source DNG file
        image = src_images[image_num]
        src_path = os.path.join(dng_path, image)

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

         # Getting path of the input image
        input_file_path = os.path.join(input_path, image.split('.')[0])
        input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
            
        # Assembling a dictionary of all of the original params for the source image
        # (used to render proxy input)
        original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

        # Rendering an unchanged copy of the source image for model input
        dt.render(src_path, input_file_path, original_params)

        # Rendering ground truth labels for each set of pipeline parameters
        for index in range(len(params_list)):
            param_id = params_list[index] # Used to give ground truth images unique names
            #params = param_id.split(',')
            # NOTE: param_values is a list of lists
            param_values = [[param for param in proxy.split('_')] for proxy in params_list[index].split(',')]

            # Assembling a dictionary of all of the parameters to apply to the source DNG in order
            # to render a given ground truth label image
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)
        
            # Filling the dictionary
            for idx, proxy_type, params in enumerate(proxy_order):
                params_list = params.split('_')
                params_dict = dt.get_params_dict(proxy_type, params_list, param_values[idx], None, None, dict=params_dict)
            print('Assembled params dict: \n' + params_dict)

            # Rendering the ground truth image
            label_file_path = os.path.join(label_path, f'{image}_pipeline_{param_id}')
            label_file_path = (repr(label_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
            dt.render(src_path, label_file_path, params_dict)  

    print('Pipeline images generated.')

def generate_finetune(proxy_type, param, finetune, param_finetune, possible_values, possible_values_finetune, num):
    '''
    Generate data on which to finetune a given proxy on the 
    outputs of upstream image processing blocks. (This is 
    to prevent distortionate effects during pipeline-wide
    regression, by increasing the domain of images that
    each proxy is trained on).

    Inputs:
        [proxy_type]: Which proxy to finetune
        [param]: if None, train proxy on full parameter space, else train only on [param]
        [finetune]: Which (upstream) proxy to finetune on
        [param_finetune]: if None, finetune proxy on fulle parameter space of [finetune],
                        else finetune only on [param_finetune]
        [possible_values]: (list of tuples of floats) ranges of each input parameter 
                            for [proxy_type]
        [possible_values_finetune]: (list of tuples of floats) ranges of each input 
                                    parameter for [finetune]
        [num]: (int) Number of datapoints to sample from input space
                    (since both [proxy_type] and [finetune] have params to sweep, num**2 
                     total images will be generated)
    '''

    # Demosaic and colorin do not need to be finetuned
    # TODO: allow colorin to be finetuned once highlights is implemented
    if proxy_type == "demosaic" or proxy_type == "colorin":
        raise ValueError(f"{proxy_type} proxy does not need to be finetuned, as it is at the beginning of the pipleine.")
    
    if proxy_type == "colorout":
        raise ValueError("colorout proxy does not need to be finetuned, given its relative simplicity.")

    # Getting image directory paths
    input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, (proxy_type + '_' + c.INPUT_DIR))
    output_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, (proxy_type + '_' + c.OUTPUT_DIR))
    if param is not None:
        input_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, (proxy_type + '_' + param + '_' + c.INPUT_DIR))
        output_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, (proxy_type + '_' + param + '_' + c.OUTPUT_DIR))

    # Creating given input and output directories if they do not already exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Directory created at: ' + output_path)
    
    # Checking that given input and output directories are empty
    if ((len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0) and c.CHECK_DIRS):
        # TODO: check if {param} can accept None
        raise RuntimeError(f'ERROR: {proxy_type}: {param} directories already contain data for stage 3.')

    # Tapout settings for the given proxy
    tapouts = c.TAPOUTS[proxy_type]

    # Getting DNG images
    dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    src_images = sorted(os.listdir(dng_path))

    # If the given proxy has multiple parameters enabled, 
    # use Latin Hypercube Sampling (LHS)
    # TODO: use the same LHS samples for all images?
    sampler = ParamSampler(proxy_type, param, possible_values, num)#TODO: will this work for proxies that do not need input params??
    if param is None:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(num*len(src_images))], axis=1)
        convert(samples_concatenated, os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, f'{proxy_type}_params.npy'), ndarray=True)
    else:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(num*len(src_images))])
        convert(samples_concatenated, os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_3_PATH, f'{proxy_type}_{param}_params.npy'), ndarray=True)
    print("Params file saved.")

    # If the given finetuning block has multiple parameters
    # emabled, use Latin Hypercube Sampling (LHS)
    sampler_finetune = ParamSampler(finetune, param_finetune, possible_values_finetune, num)

    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)
        
        # Keeping track of the first input images generated
        first_input = np.zeros(len(sampler_finetune))

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Fintetune block value sweep
        for n, values_finetune in enumerate(sampler_finetune):

            if first_input[n] == 0 and tapouts is None:
            # No need to render input images separately when tapouts are being used

                first_input[n] = 1

                # Getting path of the input image
                input_file_path = os.path.join(input_path, image.split('.')[0])
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") # Dealing with Darktable CLI pickiness
                for val in values_finetune:
                    input_file_path += '_' + str(val) #TODO: will this format correctly?
                input_file_path += '.tif'

                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(finetune, c.PARAM_NAMES[finetune], values_finetune, temperature_params, raw_prepare_params)

                # Rendering input image
                dt.render(src_path, input_file_path, original_params)

            # Proxy block value sweep
            for values in sampler:
        
                # Getting path of the ground truth image
                gt_file_path = os.path.join(output_path, f'{image}_{proxy_type}')
                gt_file_path = (repr(gt_file_path).replace('\\\\', '/')).strip("'") # Dealing with Darktable CLI pickiness
                for val in values_finetune:
                    gt_file_path += '_' + str(val)
                gt_file_path += '_to'
                for val in values:
                    gt_file_path += '_' + str(val)
                gt_file_path += '.tif'

                # Assembling a dictionary of all of the parameters to apply to the source DNG
                # (Temperature and rawprepare params must be maintained in order to produce expected results)
                params_dict = dt.get_params_dict(proxy_type, c.PARAM_NAMES[proxy_type], values, temperature_params, raw_prepare_params)

                # Rendering the output image
                dt.render(src_path, gt_file_path, params_dict)   

                # Checking if input & ground truth images need to be replaced with tapouts
                if tapouts is not None:
                    
                    # Getting file path of the tapout
                    input_tapout_path = tapouts[0] + '.tmp'
                    gt_tapout_path = tapouts[1] + '.tmp'
                    #print('input tapout path: ' + input_tapout_path)
                    #print('ground truth tapout path: ' + gt_tapout_path)

                    # Getting path of the input image
                    input_file_path = os.path.join(input_path, image.split('.')[0])
                    input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") # Dealing with Darktable CLI pickiness
                    for val in values_finetune:
                        input_file_path += '_' + str(val) #TODO: will this format correctly?
                    input_file_path += '.tif'

                    # Deleting final output image
                    os.remove(gt_file_path)
                    print('replacing: ' + gt_file_path)

                    # Read in the tapouts and save them as tiffs
                    tmp2tiff(input_tapout_path, input_file_path)
                    tmp2tiff(gt_tapout_path, gt_file_path)
    print("Finetuning data generated.")