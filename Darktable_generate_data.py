
import os
import numpy as np
import shutil
import numpy.random as random

# Local files
import Darktable_constants as c
import PyDarktable as dt

def generate(proxy_type, param, stage, min, max, interactive, num=20):

    # Getting stage paths
    stage_path = getattr(c, 'STAGE_' + str(stage) + '_PATH')
    '''
    if stage is 1:
        stage_path = c.STAGE_1_PATH
    elif stage is 2:
        stage_path = c.STAGE_2_PATH
    else:
        print('ERROR: stage must be either 1 or 2.')
        quit()
    '''

    # Getting image directory paths
    input_dir = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.INPUT_DIR))
    output_dir = os.path.join(c.IMAGE_ROOT_DIR, stage_path, (proxy_type + '_' + param + '_' + c.OUTPUT_DIR))

    # Checking that given input and output directories exist
    if (not os.path.isdir(input_dir) or not os.path.isdir(output_dir)):
        print(f'ERROR: {proxy_type}_{param}_input or {proxy_type}_{param}_output directories for stage {stage} do not exist.')
        print(f'Input directory given: {input_dir}')
        print(f'Output directory given: {output_dir}')
        quit()

    # Checking that given input and output directories are empty
    if (len(os.listdir(input_dir)) > 0 or len(os.listdir(output_dir)) > 0):
        print(f'ERROR: {proxy_type}: {param} directories already contain data for stage {stage}.')

        # Checking if user wants to skip to stage 2 data generation
        if stage == 1 and interactive:
            skip = None
            while skip != 'n' and skip != 'y':
                skip = input('Do you want to skip to generating data for stage 2? (y/n)')
            if skip == 'n':
                print('quitting')
                quit()
            else:
                return

        # Checking if user wants to skip to proxy training
        if stage == 2 and interactive:
            skip = None
            while skip != 'n' and skip != 'y':
                skip = input('Do you want to skip to proxy training? (y/n)')
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
            input_path = os.path.join(input_dir, f'{image}_{proxy_type}_{param}')
            input_path = (repr(input_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness

            # Making sure that the same input image is not generated multiple times
            if first_input is None:
                first_input = input_path # Path of the original input image to be copied later
                
                # Assembling a dictionary of all of the original params for the source image
                # (used to render proxy input)
                original_params = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

                # Rendering an unchanged copy of the source image for model input
                dt.render(src_path, input_path, original_params)
            else:
                _ = shutil.copyfile(first_input, input_path) # Copying the first input image to get the others

            # Assembling a dictionary of all of the parameters to apply to the source DNG
            # Temperature and rawprepare params must be maintained in order to produce expected results
            params_dict = dt.get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)
            vals.append(float(value)) # Adding to params list

            # Rendering the output image
            output_path = os.path.join(output_dir, f'{image}_{proxy_type}_{param}') #f'./stage_1/contrast_input/contrast_{contrast}.tif'
            output_path = (repr(output_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
            dt.render(src_path, output_path, params_dict)

    # Converting param list to numpy array and saving to file
    if stage == 1:
        vals = np.array(vals)
        vals = np.expand_dims(vals, axis=0)
        with open(os.path.join(c.IMAGE_ROOT_DIR, stage_path, f'{proxy_type}_{param}_params.npy'), 'wb') as f:
            np.save(f, vals)

    print(f"Training data generated: stage {stage}")

def generate_eval(proxy_type, param, input_dir, output_dir, params_file):

    # Getting path of the params matrix file
    print('input dir: ' + os.path.dirname(input_dir))
    params_mat = os.path.join(os.path.dirname(input_dir), f'{proxy_type}_{param}_eval_params.npy')

    # Checking that given input and output directories are empty
    if (len(os.listdir(input_dir)) > 0 or len(os.listdir(output_dir)) > 0):
        print(f'ERROR: {proxy_type}: {param} eval directories already contain data.')
        return params_mat

    # List of all slider values
    vals = []

    # Getting path of source DNG files
    dng_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_DNG_PATH)
    src_images = os.listdir(dng_path)

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
                input_path = os.path.join(input_dir, f'{image}_{proxy_type}_{param}')
                input_path = (repr(input_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
                dt.render(src_path, input_path, original_params)

                # Assembling a dictionary of all of the parameters to apply to the source DNG
                # Temperature and rawprepare params must be maintained in order to produce expected results
                params_dict = dt.get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)
                vals.append(float(value)) # Adding to params list

                # Rendering the output image
                output_path = os.path.join(output_dir, f'{image}_{proxy_type}_{param}') #f'./stage_1/contrast_input/contrast_{contrast}.tif'
                output_path = (repr(output_path).replace('\\\\', '/')).strip("'") + f'_{value}.tif' # Dealing with Darktable CLI pickiness
                dt.render(src_path, output_path, params_dict)

    # Converting param list to numpy array and saving to file
    vals = np.array(vals)
    vals = np.expand_dims(vals, axis=0)
    with open(params_mat, 'wb') as f:
        np.save(f, vals)

    print("Data for evaluation generated.")
    return params_mat
