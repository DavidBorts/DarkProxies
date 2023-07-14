'''
This file contains scripts to finetune existing proxies on 
inputs that have been pre-processed by earlier proxies in 
the pipeline.

This serves to expand the domain of input images that proxies
are trained on, mitigating the distortionate effects that 
take place when regressing across an entire pipeline

Example usage:
python Finetune_proxy.py [proxy_type] --mono [param_name (OPTIONAL)] [sampler_type]
'''
import numpy as np
import torch
import argparse

# Local files
import Constants as c
import Dataset
from Train_proxy import run_training_procedure
from Generate_data import generate_finetune

def run_finetune_procedure(proxy_type, param, possible_values, finetune, param_finetune, possible_values_finetune):
    
    raise NotImplementedError

def finetune_proxy():
    raise NotImplementedError

if __name__ != '__main__':
    raise RuntimeError("This scirpt is only configured to be called directly by the user!")

# argparser args
parser = argparse.ArgumentParser()
parser.add_argument("proxy", help="Name of the Darktable block for which to finetune a proxy network", 
                    choices = ['colorbalancergb', 'sharpen', 'exposure', 'colorin', 'colorout', 
                               'demosaic'], required=True)
parser.add_argument("finetune", help="Name of the Darktable block on which to finetune the proxy", 
                    choices= ['colorbalancergb', 'sharpen', 'exposure'], required=True)
parser.add_argument("-m", "--mono", help="[OPTIONAL] Train [proxy] on only user-specified input \
                    parameters, keeping the others fixed", default=None)
parser.add_argument("-f", "--monof", help="[OPTIONAL] Finetune [proxy] on only user-specified \
                    input parameters of [finetune], keeping the others fixed", default=None)
parser.add_argument("-n", "--number", default=0, help="Number of training examples to generate for each \
                    source DNG image", required=True)
args = parser.parse_args()
proxy_type = args.proxy
param = args.mono
finetune = args.finetune
param_finetune = args.monof
num = args.number

# Some proxies have no input params
append_params = proxy_type not in c.NO_PARAMS

# If the given proxy takes input params, it is necesary to find their
# ranges of possible values
possible_values = None
if append_params: 
    if param is not None:
        # Mono-proxies only have one input param unfrozen
        # NOTE: possible_values needs to be a list
        possible_values = [c.POSSIBLE_VALUES[proxy_type][c.PARAM_NAMES[proxy_type].index(param)]]
    else:
        possible_values = c.POSSIBLE_VALUES[proxy_type]
else:
    # Proxy has no input parameters
    #TODO: replace with highlights or temperature to support demosaic?

    if proxy_type == 'demosaic': # Temporary hack: not sweeping for demosaic
        possible_values = [(None, None)]
    else:
        sampler_block, sampler_param = c.SAMPLER_BLOCKS[proxy_type].split('_')
        possible_values = [c.POSSIBLE_VALUES[sampler_block][c.PARAM_NAMES[sampler_block].index(sampler_param)]]

# Finding the ranges of possible values for the given sampler block
possible_values_finetune = c.POSSIBLE_VALUES[finetune]

# Generating data
generate_finetune(proxy_type, 
                  param, 
                  finetune, 
                  param_finetune, 
                  possible_values, 
                  possible_values_finetune, 
                  num)

# Finetuning proxy
run_finetune_procedure(proxy_type,
                       param,
                       possible_values,
                       finetune,
                       param_finetune,
                       possible_values_finetune
                       )