# Data generation methods

import os
import numpy as np

# Local files
import Constants as c
import PyDarktable as dt
from utils.npy_convert import convert
from utils.lhs import lhs
from utils.misc import get_possible_values, sort_params
from utils.extract_RAW import get_cfa

class ParamSamplerIterator():
    '''
    Iterator for sampled parameter values
    '''
    def __init__(self, sampler, params, list, possible_values):
        self.sampler = sampler
        self.params = params
        self.possible_values = possible_values
        self.list = list
        self.num = len(sampler)
        self.idx = 0
    
    def __next__(self):

        vals = None
        if self.idx < self.num:

            if len(self.possible_values) > 1:
                vals = np.squeeze(self.list[:, self.idx])
            else:
                vals = [self.list[self.idx]]

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

        if len(self.possible_values) > 1:
            self.list = lhs(self.possible_values, self.num)
        else:
            self.list = np.linspace(self.possible_values[0][0], self.possible_values[0][1], self.num)

    def __iter__(self):
        return ParamSamplerIterator(self, self.params, self.list, self.possible_values)
    
    def __len__(self):
        return self.num

def generate(proxy_type, params, stage, possible_values, num, name):

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

    # Which parameter spaces to sample
    # (by default, proxies with input parameters sample their own
    # paramater spaces. Howvever, for proxies without input paramaters,
    # the parameter space of a different processing block, called a 
    # "sampler block" in this codebase, is sampled instead to augment 
    # training data. This effects which rendering parameters are passed
    # to darktable-cli)
    #TODO: is this still necessary?
    proxy_type_gt = proxy_type
    params_gt = params
    if proxy_type in c.NO_PARAMS and proxy_type != "demosaic" and proxy_type != "exposure":#TODO: temporary hack, remove me!!
        proxy_type_gt, params_gt = c.SAMPLER_BLOCKS[proxy_type].split('_')
        params_gt = [params_gt] # NOTE: params_gt, like params, must be a list

    # Getting DNG images
    if stage == 1:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    else:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_DNG_PATH)
    src_images = sorted(os.listdir(dng_path))
    
    sampler = ParamSampler(proxy_type_gt, params_gt, possible_values, num)
    if len(possible_values) > 1:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(len(src_images))], axis=1)
    else:
        samples_concatenated = np.concatenate([sampler.list.copy() for _ in range(len(src_images))])
    filename = name + 'params.npy'
    convert(samples_concatenated, os.path.join(c.IMAGE_ROOT_DIR, stage_path, filename), ndarray=True)
    print("Params file saved.")

    gt_imgs = []

    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Getting path of the input image
        input_file_path = os.path.join(input_path, image.split('.')[0])
        input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

        # Parameter value sweep
        # (to generate ground truth images)
        for values in sampler:

            # Checking if input image already exists
            # (this can happen if a previous data generation job was interrupted)
            input_exists = os.path.exists(input_file_path)

            # Getting path of the ground truth image
            gt_file_path = f'{image}_{proxy_type}'
            param_string = ''
            for val in values:
                param_string += '_' + str(val)[0:4]
            if len(param_string) > 40:
                param_string = param_string[0:41]
            gt_file_path += param_string
            gt_file_path += '.tif'
            gt_imgs.append(gt_file_path)
            gt_file_path = os.path.join(output_path, gt_file_path)
            gt_file_path = (repr(gt_file_path).replace('\\\\', '/')).strip("'") # Dealing with Darktable CLI pickiness

            # Skip ground truth images that already exist
            if os.path.exists(gt_file_path):
                continue
            
            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type_gt, params_gt, values, temperature_params, raw_prepare_params)

            # Rendering the output image
            tapouts = dt.render(src_path, gt_file_path, params_dict, True, proxy_type)

            # Converting PFM tapouts to usable TIFF files
            os.remove(gt_file_path)
            if not input_exists:
                dt.pfm_to_tif(tapouts[0], input_file_path)
            os.remove(tapouts[0])
            dt.pfm_to_tif(tapouts[1], gt_file_path)
            os.remove(tapouts[1])
    print(f"Training data generated: stage {stage}")
    return gt_imgs

def generate_eval(proxy_type, params, values, name, possible_values):
    
    # Getting path of the params matrix file
    eval_path = os.path.join(c.IMAGE_ROOT_DIR, c.EVAL_PATH)
    params_mat = os.path.join(eval_path, f'{name}_eval_params.npy')

    # Getting paths of input and output directories
    input_path = os.path.join(eval_path, f'{name}_input')
    output_path = os.path.join(eval_path, f'{name}_output')

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

    # Getting path of source DNG files
    dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    src_images = os.listdir(dng_path)

    # Writing param .npy file
    values_list_ndarray = np.zeros((len(possible_values), 1))
    values_float = [float(value) for value in values]
    values_list_ndarray[:, :] = np.array(values_float)
    samples_concatenated = np.concatenate([values_list_ndarray for _ in range(len(src_images))], axis=1)
    convert(samples_concatenated, params_mat, ndarray=True)
    print("Params file saved.")

    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Getting path of the input image
        input_file_path = os.path.join(input_path, image.split('.')[0])
        input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

        # Checking if input image already exists
        # (this can happen if a previous data generation job was interrupted)
        input_exists = os.path.exists(input_file_path)

        # Assembling a dictionary of all of the parameters to apply to the source DNG
        # Temperature and rawprepare params must be maintained in order to produce expected results
        params_dict = dt.get_params_dict(proxy_type, params, values.split(','), temperature_params, raw_prepare_params)

        # Rendering the output image
        gt_file_path = os.path.join(output_path, f'{image}_{name}')
        for val in values:
            gt_file_path += '_' + str(val)
        gt_file_path = (repr(gt_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
        tapouts = dt.render(src_path, gt_file_path, params_dict, True, proxy_type)
        # Converting PFM tapouts to usable TIFF files
        os.remove(gt_file_path)
        if not input_exists:
            dt.pfm_to_tif(tapouts[0], input_file_path)
        os.remove(tapouts[0])
        dt.pfm_to_tif(tapouts[1], gt_file_path)
        os.remove(tapouts[1])

    print("Data for evaluation generated.")
    return params_mat

def generate_pipeline(proxy_order, input_path, label_path, dng_path=None):

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

    cfa = None
    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)
        cfa = get_cfa(src_path)

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Getting possible values & sampling
        proxy_type_list = []
        param_names_list = []
        sampled_values_list = []
        for proxy_name, params in proxy_order:
            proxy_type = proxy_name.split('_')[0]
            proxy_type_list.append(proxy_type)
            params = params.split(',')
            if proxy_type not in c.NO_PARAMS:
                param_names_list.append(sort_params(proxy_type, params))
                sampled_values_list.append(lhs(get_possible_values(proxy_type, params), 1))
            else:
                param_names_list.append(None)
                sampled_values_list.append(None)

        # Assembling a dictionary of all of the parameters to apply to the source DNG in order
        # to render a given ground truth label image
        # Temperature and rawprepare params must be maintained in order to produce expected results
        params_dict = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)
        
        # Filling the dictionary with the sampled params
        for proxy_type, param_names, sampled_values in zip(proxy_type_list, param_names_list, sampled_values_list):
            print(f"sample: {str(proxy_type)} - {str(param_names)} - {str(sampled_values)}")
            if param_names is not None:
                params_dict = dt.get_params_dict(proxy_type, param_names, sampled_values, None, None, dict=params_dict)
        print('Assembled params dict: \n' + str(params_dict))

        # Rendering an image with Darktable
        input_file_path = os.path.join(input_path, f'{image}_pipeline')
        input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
        label_file_path = os.path.join(label_path, f'{image}_pipeline')
        label_file_path = (repr(label_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
        tapouts = dt.render(src_path, label_file_path, params_dict, True, proxy_type_list)

        # Converting PFM tapouts to usable TIFF files
        #TODO: don't use sRGB DT renders as GT
        #os.remove(label_file_path)
        #dt.pfm_to_tif(tapouts[1], label_file_path)
        os.remove(tapouts[1])
        dt.pfm_to_tif(tapouts[0], input_file_path)
        os.remove(tapouts[0])
    print('Pipeline images generated.')
    return cfa

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
    print("Finetuning data generated.")