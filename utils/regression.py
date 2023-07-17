import math
import random
import numpy as np

def initial_guess(possible_values):
    '''
    Returns an initial guess vector. The values of the initial guess are determined randomly uniformly.
    '''
    x_0 = []
    for vals in possible_values:
        if type(vals) is list:
            x_0.append(vals[random.randint(0,len(vals)-1)])
        elif type(vals) is tuple:
            x_0.append(random.uniform(vals[0], vals[1]))
        else:
            raise TypeError('Possible values must be given as list (discrete) or tuple (continuous)')
    return np.array(x_0)
    
def project_param_values(unprojected_param_values, possible_values, finalize, dtype):
    '''
    Projects parameter values so that they lie in the valid range.
    Inputs:
        unprojected_param_values: Numpy array of parameter values before projection.
        possible_values: List of possible values that parameters can take.
        finalize: Applies to discrete parameters. Set to True if you want to round discrete parameters to 
        to nearest valid integer. Set to False during training so that discrete parameters lie within
        smallest and largest integer.
        dtype: Datatype of the tensor. Either torch.FloatTensor, or torch.cuda.FloatTensor.
    Outputs:
        projected_param_values: Numpy array of parameter values after projection.
    '''
    def _project_onto_discrete(value, vals_list):
        array = np.array(vals_list)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    def _project_onto_discrete_range(value, vals_list):
        if value < vals_list[0]:
            value = vals_list[0]
        if value > vals_list[-1]:
            value = vals_list[-1]
        return value
    
    def _project_onto_continuous(value, range_tuple):
        if value < range_tuple[0]:
            value = range_tuple[0]
        if value > range_tuple[1]:
            value = range_tuple[1]
        return value
    
    projected_param_values = []
    for i in range(len(possible_values)):
        vals = possible_values[i]
        if type(vals) == tuple:
            projected_param_values.append(_project_onto_continuous(unprojected_param_values[i], vals))
        elif finalize:
            projected_param_values.append(_project_onto_discrete(unprojected_param_values[i], vals))
        else:
            projected_param_values.append(_project_onto_discrete_range(unprojected_param_values[i], vals))
    return np.array(projected_param_values, dtype=np.float64)

def decide(i, num_iters):
    '''
    Decide whether or not to save the given iteration
    as a frame in an animation
    '''
    if i == 0 or i == (num_iters - 1):
        return True
    
    save_output = i % (math.pow(i, 1.75) * (1 / num_iters))
    
    if int(save_output) == 0:
        return True
    else:
        return False