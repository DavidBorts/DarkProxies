'''
File with various scripts for tapout images saved by Darktable.

More precisely, scripts in this file can be used to load in 
Darktable tapouts and convert them to .tiff files for use as
training data.
'''

import os
import numpy as np
import tifffile

'''
Load an ImageStack TMP file, an up to 4-dimensional array of integer or
floating point data (this is the format used for Darktable's tapouts).

Format specification is here:
https://github.com/abadams/ImageStack/blob/master/src/FileTMP.cpp

The output is permutted into the numpy C-contiguous convention: (frames, height, width, channels).
Channels changes the fastest in memory, frames the slowest.
'''
def loadTMP(filename):
    with open(filename, 'rb') as f:
        buffer = f.read()

    type_code_list = \
    [
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.uint8),
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
        np.dtype(np.uint32),
        np.dtype(np.int32),
        np.dtype(np.uint64),
        np.dtype(np.int64),
    ]

    offset = 0
    tmp_dims = np.frombuffer(buffer, count=4, dtype=np.int32)
    offset += tmp_dims.nbytes
    type_code = np.frombuffer(buffer, count=1, offset=offset, dtype=np.int32)
    offset += type_code.nbytes
    data_type = type_code_list[int(type_code)]

    # The TMP file writes the dimensions as:
    #   [width height frames channels]
    # Rewrite it in the numpy order:
    #   (channels, frames, height, width)
    a2_dims = np.flipud(tmp_dims)

    # Buffer stores the data as: (channels, frames, height, width).
    a2 = np.frombuffer(buffer, offset=offset, dtype=data_type)
    a2.shape = a2_dims

    # Permute the dimensions so that the data is returned in the numpy C-contiguous convention:
    #   (frames, height, width, channels).
    return a2.transpose((1, 2, 3, 0))

'''
Reads in TMP files (tapouts) generated by *modified* Darktable and converts them to .TIFF
'''
def tmp2tiff(input_file, output_file):

    # Read in .TMP file as ndarray
    im = loadTMP(os.path.join('/tmp', input_file))
    #print(f"{input_file} has shape: {im.shape}")
    print('Reading tmp tapout from: ' + input_file)

    # Getting channel information
    output_channels = im.shape[-1]
    output_channels = min(output_channels, 3)
    #print(f"clipping to {output_channels} channel(s)")

    # Clipping image
    im = im[0, :, :, :output_channels]
    print('dtype: ' + str(im.dtype))
    print("max tapout value: " + str(np.amax(im)))
    #im = (im>>16).astype(np.float16) # Does this work??
    im = im.astype(np.float16)
    print("new max tapout value:" + str(np.amax(im)))

    #print(f'Writing: {output_file}')
    tifffile.imwrite(output_file, im)
