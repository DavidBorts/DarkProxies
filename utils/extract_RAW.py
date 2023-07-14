'''
Various helper scripts to extract RAW file metadata using RawPy
(if called directly, this file will print out the CFAs of every
DNG file in the provided directory)
'''

import sys
import os
import rawpy

# Code adapted from: https://stackoverflow.com/a/67280050/20394190
def get_cfa(raw_file_path):
    '''
    Get RAW image color filter array (CFA)

    inputs:
        [raw_file_path]: string path to RAW file

    returns:
        [cfa]: string, corresponding to the image's CFA in raster order
               len(cfa) = 4
    '''
    raw = rawpy.imread(raw_file_path)

    cfa = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
    return cfa

# Helper script to print out the CFAs of all RAW images in a given directory
if __name__ == "__main__":
    dng_dir = sys.argv[1]

    dng_images = os.listdir(dng_dir)

    print("Getting cfa's:")
    for img in dng_images:
        print("image name: " + img)
        print(get_cfa(os.path.join(dng_dir, img)))
