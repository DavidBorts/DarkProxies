
import os
import numpy as np
import shutil
import numpy.random as random

# Local files
import Darktable_constants as c
import PyDarktable as dt
from npy_convert import convert

def generate(proxy_type, param, stage, min, max, interactive, num=20):

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
    if (len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0):
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

    # Iterating over each source DNG file
    if stage == 1:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    else:
        dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_2_DNG_PATH)
    src_images = os.listdir(dng_path)
    for image in src_images:

        # Getting path of individual source DNG file
        src_path = os.path.join(dng_path, image)
        
        # Keeping track of the first input image generated
        first_input = None

        # Extracting necessary params from the source image
        raw_prepare_params, temperature_params = dt.read_dng_params(src_path)

        # Parameter value sweep
        #for value in random.uniform(min, max, int(num)):
        for value in np.linspace(min, max, int(num)):
            
            # Getting path of the input image
            input_file_path = os.path.join(input_path, f'{image}_{proxy_type}_{param}')
            input_file_path = (repr(input_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness

            # Making sure that the same input image is not generated multiple times
            if first_input is None:
                first_input = input_file_path # Path of the original input image to be copied later
                
                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering an unchanged copy of the source image for model input
                dt.render(src_path, input_file_path, original_params)
            else:
                _ = shutil.copyfile(first_input, input_file_path) # Copying the first input image to get the others

            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)
            vals.append(float(value)) # Adding to params list

            # Rendering the output image
            output_file_path = os.path.join(output_path, f'{image}_{proxy_type}_{param}') #f'./stage_1/contrast_input/contrast_{contrast}.tif'
            output_file_path = (repr(output_file_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
            dt.render(src_path, output_file_path, params_dict)

    # Converting param list to numpy array and saving to file
    if stage == 1:
        convert(vals, os.path.join(c.IMAGE_ROOT_DIR, stage_path, f'{proxy_type}_{param}_params.npy'))

    print(f"Training data generated: stage {stage}")

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
    if (len(os.listdir(input_path)) > 0 or len(os.listdir(output_path)) > 0):
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
