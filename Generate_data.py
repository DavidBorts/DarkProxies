# Data generation scripts

import os
import numpy as np
#import numpy.random as random

# Local files
import Constants as c
import PyDarktable as dt
from Tapouts import tmp2tiff
from npy_convert import convert, merge

def generate(proxy_type, param, stage, min, max, interactive, num):

    # Getting stage paths
    stage_path = getattr(c, 'STAGE_' + str(stage) + '_PATH')

    # Getting image directory paths
    input_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.INPUT_DIR))
    output_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.OUTPUT_DIR))

    # Creating given input and output directories if they do not already exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Directory created at: ' + output_path)

    
    # Checking that given input and output directories are empty
    if ((len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0) and c.CHECK_DIRS):
        print(f'ERROR: {proxy_type}: {param} directories already contain data for stage {stage}.')

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
    
    # List of all slider values
    vals = []

    # Information about tapout requirement for the given proxy
    tapouts = c.TAPOUTS[proxy_type]

    # Which parameter spaces to sample
    # (by default, proxies with input parameters sample their own
    # paramater spaces. Howvever, for proxies without input paramaters,
    # the parameter space of a different processing block, called a
    # "sampler block" in this codebase, is sampled instead to augment
    # training data. This effects which rendering parameters are
    # passed to darktable-cli)
    proxy_type_gt = proxy_type
    param_gt = param
    #if tapouts is not None:
    if tapouts is not None and proxy_type != "demosaic":#TODO: temporary hack, remove me!!
        proxy_type_gt, param_gt = c.SAMPLER_BLOCKS[proxy_type].split('_')

    # Getting DNG images
    if stage == 1:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    else:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_DNG_PATH)
    src_images = sorted(os.listdir(dng_path))

    # Temporary hack: not sweeping parameter spaces for demosaic
    if min is None or max is None:
        generate_single(proxy_type, dng_path, src_images, input_path, output_path, tapouts)
        print(f"Training data generated: stage {stage}")
        return

    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)
        
        # Keeping track of the first input image generated
        first_input = None

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Parameter value sweep
        # Alternate approach -> for value in random.uniform(min, max, int(num)):
        for value in np.linspace(min, max, int(num)):
            
            # Making sure that the same input image is not generated multiple times
            if first_input is None and tapouts is None:

                # Getting path of the input image
                input_file_path = os.path.join(input_path, image.split('.')[0])
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

                first_input = input_file_path
                
                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering an unchanged copy of the source image for model input
                dt.render(src_path, input_file_path, original_params)

            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type_gt, param_gt, value, temperature_params, raw_prepare_params)
            if proxy_type == "colorin" or proxy_type == "colorout":
                params_dict = dt.get_params_dict('colorbalancergb', 'contrast', float(0.0), None, None, dict=params_dict)
            vals.append(float(value)) # Adding to params list

            # Rendering the output image
            output_file_path = os.path.join(output_path, f'{image}_{proxy_type}_{param}')
            output_file_path = (repr(output_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
            dt.render(src_path, output_file_path, params_dict)

            # Checking if input & ground truth images need to be replaced with tapouts
            if tapouts is not None:
                
                # Getting file path of the tapout
                input_tapout_path = tapouts[0] + '.tmp'
                gt_tapout_path = tapouts[1] + '.tmp'
                print('input tapout path: ' + input_tapout_path)
                print('ground truth tapout path: ' + gt_tapout_path)

                # Getting path of the input image
                input_file_path = os.path.join(input_path, image.split('.')[0])
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif'

                # Deleting final output image
                os.remove(output_file_path)
                print('replacing: ' + output_file_path)

                # Read in the tapouts and save them as tiffs
                tmp2tiff(input_tapout_path, input_file_path)
                tmp2tiff(gt_tapout_path, output_file_path)

    # Converting param list to numpy array and saving to file
    convert(vals, os.path.join(c.IMAGE_ROOT_DIR, stage_path, f'{proxy_type}_{param}_params.npy'))

    print(f"Training data generated: stage {stage}")

def generate_piecewise(proxy_type, param, stage, min, max, num):

    # Getting stage paths
    stage_path = getattr(c, 'STAGE_' + str(stage) + '_PATH')

    # Getting image directory paths
    input_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.INPUT_DIR))
    output_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.OUTPUT_DIR))
    params_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.PARAM_FILE_DIR))
    merged_params_path = os.path.join(c.IMAGE_ROOT_DIR, stage_path, f'{proxy_type}_{param}_params.npy')

    # Information about tapout requirement for the given proxy
    tapouts = c.TAPOUTS[proxy_type]

    # Creating given input, output, and checkpoint file directories if they do not already exist
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print('Directory created at: ' + input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('Directory created at: ' + output_path)
    if not os.path.exists(params_path):
        os.mkdir(params_path)
        print('Directory created at: ' + params_path)    

    # Iterating over each source DNG file
    if stage == 1:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    else:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_DNG_PATH)
    src_images = os.listdir(dng_path)
    imgNum = 1
    for image in src_images:

        # Making sure that the given source image has not already
        # had data generated
        if imgNum <= len(os.listdir(params_path)):
            imgNum += 1
            continue

        # List of all slider values
        vals = []

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)
        
        # Keeping track of the first input image generated
        first_input = None

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Parameter value sweep
        #Alternate approach -> for value in random.uniform(min, max, int(num)):
        for value in np.linspace(min, max, int(num)):

            # Tracking duplicate images
            skip_input = False
            skip_output = False

            # Making sure that the same input image is not generated multiple times
            if first_input is None and tapouts is None:

                # Getting path of the input image and confirming that it does not yet exist
                input_file_path = os.path.join(input_path, image.split('.')[0])
                input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
                if os.path.exists(input_file_path):
                    print(f'{input_file_path} already exists. Skipping.')
                    first_input = input_file_path
                    skip_input = True

                first_input = input_file_path # Path of the original input image to be copied later
                
                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering an unchanged copy of the source image for model input
                if not skip_input:
                    dt.render(src_path, input_file_path, original_params)

            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)
            vals.append(float(value)) # Adding to params list

            # Rendering the output image and confirming that it does not yet exist
            output_file_path = os.path.join(output_path, f'{image}_{proxy_type}_{param}')
            output_file_path = (repr(output_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
            if os.path.exists(output_file_path):
                    print(f'{output_file_path} already exists. Skipping.')
                    skip_output = True
            if not skip_output:
                dt.render(src_path, output_file_path, params_dict)

                # Checking if ground truth image needs to be replaced with tapout
                if tapouts is not None:
                    
                    # Getting file path of the tapout
                    input_tapout_path = tapouts[0] + '.tmp'
                    gt_tapout_path = tapouts[1] + '.tmp'
                    print('input tapout path: ' + input_tapout_path)
                    print('ground truth tapout path: ' + gt_tapout_path)

                    # Getting path of the input image
                    input_file_path = os.path.join(input_path, image.split('.')[0])
                    input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif'

                    # Deleting final output image
                    os.remove(output_file_path)
                    print('replacing: ' + output_file_path)

                    # Read in the tapouts and save them as tiffs
                    tmp2tiff(input_tapout_path, input_file_path)
                    tmp2tiff(gt_tapout_path, output_file_path)

        # Converting param list to numpy array and saving to checkpoint file (once for each source image)
        convert(vals, os.path.join(params_path, f'{image}_{proxy_type}_{param}.npy'))

        # Keeping track of checkpoint progress
        imgNum += 1

    # Merging all of the checkpoint files together into one .npy file
    param_files = os.listdir(params_path)
    if len(os.listdir(params_path)) == len(os.listdir(dng_path)):
        merge(param_files, params_path, merged_params_path)
    else:
        print("ERROR: number of checkpoints not equal to number of source images.\n Quitting.")
        quit()
    
    print(f"Training data generated: stage {stage}")

# Renders a single input -> ground truth pair for a given source image in src_images, 
# instead of sweeping over a range of parameter values
def generate_single(proxy_type, dng_path, src_images, input_path, output_path, tapouts):
    
    for image in src_images:

        # Getting path of individual source DNG file
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
            tmp2tiff(tapout_path_input, input_file_path)
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

#TODO: ADD SUPPORT FOR MULTIPLE PARAMS PER PROXY
def generate_pipeline(param_file, input_path, label_path, dng_path=None):
    
    # Reading in proxy order
    proxy_order = []
    with open(os.path.join(c.IMAGE_ROOT_DIR, c.CONFIG_FILE), 'r') as file:
        lines = file.readlines()
        for line in lines:
            proxy, enable = line.split(' ')
            if int(enable) == 1:
                proxy_order.append(proxy)
    
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

    # Information about tapout requirement for the given proxy
    tapouts = c.TAPOUTS[proxy_type]
    
    for image_num in range(len(src_images)):

        # Getting path of individual source DNG file
        image = src_images[image_num]
        src_path = os.path.join(dng_path, image)

        # Keeping track of the first input image generated
        first_input = None

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Making sure that the same input image is not generated multiple times
        if first_input is None:

            # Getting path of the input image
            input_file_path = os.path.join(input_path, image.split('.')[0])
            input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness

            first_input = input_file_path
            
            # Assembling a dictionary of all of the original params for the source image
            # (used to render proxy input)
            original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

            # Rendering an unchanged copy of the source image for model input
            dt.render(src_path, input_file_path, original_params)

            # If the given proxy is colorin, or any proxy that appears before colorin,
            # The rendered image needs to be replaced with an intermediary tapout from
            # the Darktable CLI (This is because colorin converts to a different
            # colorspace, which would leave the rendered image in a different colorspace
            # from what the given proxy requires as input)
            if tapouts is not None:
                
                # Getting file path of the tapout
                tapout_path = tapouts[0] + '.tmp'

                # Deleting final output image
                os.remove(input_file_path)

                # Read in the tapout and save as a tiff
                tmp2tiff(tapout_path, input_file_path)
        
        # Rendering ground truth labels for each set of pipeline parameters
        for index in range(len(params_list)):
            param_id = params_list[index] # Used to give label images unique names
            params = param_id.split(',')

            # Assembling a dictionary of all of the parameters to apply to the source DNG in order
            # to render a given ground truth label image
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

            # Iterating over every proxy in the pipeline and adding its params to the dict
            for proxy_num in range(len(proxy_order)):

                # Getting current proxy and param values
                proxy_type, param = proxy_order[proxy_num].split('_')
                param_value = params[proxy_num]

                # Adding to the params dict
                params_dict = dt.get_params_dict(proxy_type, param, param_value, None, None, dict=params_dict)    
            print('Assembled params dict: \n' + params_dict)

            # Rendering the ground truth image
            label_file_path = os.path.join(label_path, f'{image}_pipeline_{param_id}')
            label_file_path = (repr(label_file_path).replace('\\\\', '/')).strip("'") + '.tif' # Dealing with Darktable CLI pickiness
            dt.render(src_path, label_file_path, params_dict)

            # Checking if ground truth image needs to be replaced with tapout
            if tapouts is not None:
                
                # Getting file path of the tapout
                tapout_path = tapouts[1] + '.tmp'

                # Deleting final output image
                os.remove(label_file_path)

                # Read in the tapout and save as a tiff
                tmp2tiff(tapout_path, label_file_path)

    print('Pipeline images generated.')
