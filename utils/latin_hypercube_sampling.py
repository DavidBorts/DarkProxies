# Helper functions to perform Latin Hypercube Sampling (LHS)
import numpy as np

def lhs(possible_values, num):
    '''
    Latin Hypercube Sampling

    inputs:
    [possible_values]: list of tuples of size 2, each containing the min
                       and max values of the corresponding parameter
    [num]: number of samples to take per parameter

    returns:
    [samples]: ndarray of shape [len(possible_values), num], where each
               column is a sampled parameter vector
    '''

    dim = len(possible_values)
    if dim < 2:
        raise ValueError("This method is only implemented for \
                         parameter spaces of dimension 2 or greater.")
    
    # List of lists to store sampled values
    values = []

    for values in possible_values:

        min = values[0]
        max = values[1]
        bin_size = (max-min)/num

        # Binning the param
        bins_low = np.arange(min, max, bin_size)
        bins_high = np.arange(min+bin_size, max+bin_size, bin_size)

        # Sampling random value from each bin
        sampled_vals = np.random.uniform(bins_low, bins_high)
        values.append(sampled_vals)
    
    # Randomly pairing together the sampled variables
    samples = np.zeros([dim, num])
    for i, param in enumerate(values):
        samples[i, :] = np.random.shuffle(np.asarray(param))
    
    return samples