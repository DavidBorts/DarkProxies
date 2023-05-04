import sys
import os
import numpy as np

def write(ndarray, output_file_path):

    # Writing to .npy file
    with open(output_file_path, 'wb') as f:
        np.save(f, ndarray)

def convert(vals, output_file_path, ndarray=False):

    # Converting param list to numpy array
    if not ndarray:
        vals = np.array(vals)
    if len(np.shape(vals)) < 2:
        vals = np.expand_dims(vals, axis=0)

    # Writing to .npy file
    write(vals, output_file_path)

def merge(param_files, params_path, output_file_path):

    # Loading in all of the .npy files as
    # ndarrays in a list
    param_arrays = []
    for param_file in param_files:
        param_array = np.load(os.path.join(params_path, param_file))
        param_arrays.append(param_array)
    
    # Concatenating all of the param arrays along the
    # horizontal axis
    merged_params = np.hstack(param_arrays)

    with open(output_file_path, 'wb') as f:
        np.save(f, merged_params)

if __name__ == '__main__':

    # Command-line arguments
    params_file = sys.argv[1]
    output_file_path = sys.argv[2]

    # Array to store all the param values in the file
    vals = []

    # Reading params from file and appending to vals[]
    with open(params_file, 'r') as file:
        values = file.readlines()

        for value in values:
            vals.append(float(value))
    
    # Converting params_file to a .npy file
    convert(vals, output_file_path)