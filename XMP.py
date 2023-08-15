import argparse

# local files
import Constants as c
import PyDarktable as dt

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help="Path to the source DNG image for which to \
                    generate the XMP file")
parser.add_argument("destination", help="Path to destination folder to which the \
                    XMP file will be written")
parser.add_argument("blocks", help="comma-seprated list of Darktable blocks for \
                    which to set parameters. All other non-core blocks will be \
                    disabled", default="")
parser.add_argument("params", help="list of parameters to set for the provided \
                    blocks. parameters within the same block are seprated by commas, \
                    while parameters for different blocks are separated by semicolons. \
                    Example: Block1Param1,Block1Param2;Block2Param1", default="")
args = parser.parse_args()
source = args.source
destination = args.destination
blocks = args.blocks
params_list = args.params

# Separating out blocks and params into lists
blocks = blocks.split(',')
params_list = params_list.split(';')
params_list = [params.split(',') for params in params_list] #NOTE: params_list is a list of lists
params_list = [ [float(param) for param in params] for params in params_list]
# params_list = [float(param) for params in params_list for param in params]

# Extracting necessary params from the source image
raw_prepare_params, temperature_params = dt.read_dng_params(source)
params_dict = dt.get_params_dict(None, None, None, temperature_params, raw_prepare_params)

# Assemble param dictionary
for idx, elements in enumerate(zip(blocks, params_list)):
    block, params = elements
    names = c.PARAM_NAMES[block]
    params_dict = dt.get_params_dict(block, names, params, None, None, params_dict)

# create and write XMP file
xmp = dt.get_pipe_xmp(**params_dict)
with open(destination, 'w') as f:
    f.write(xmp)