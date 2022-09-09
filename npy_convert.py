import numpy as np
import sys

def convert(vals, output_file_path):

    # Converting param list to numpy array
    vals = np.array(vals)
    vals = np.expand_dims(vals, axis=0)

    # Saving to .npy file
    with open(output_file_path, 'wb') as f:
        np.save(f, vals)

if __name__ == '__main__':

    # Command-line arguments
    params_file = sys.argv[1]
    output_file_path = sys.argv[2]

    # Array to store all the para, values in the file
    vals = []

    # Reading params from file and appending to vals[]
    with open(params_file, 'r') as file:
        values = file.readlines()

        for value in values:
            vals.append(float(value))
    
    # Converting params_file to a .npy file
    convert(vals, output_file_path)