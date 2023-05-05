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

    for vals in possible_values:

        min = vals[0]
        max = vals[1]
        bin_size = (max-min)/float(num)

        # Binning the param
        bins_low = np.arange(min, max, bin_size)
        bins_high = np.arange(min+bin_size, max+bin_size, bin_size)
        if len(bins_high) > len(bins_low):
            bins_high = bins_high[:-1]
        if len(bins_low) > len(bins_high):
            bins_low = bins_low[:-1]

        # Sampling random value from each bin
        sampled_vals = np.random.uniform(bins_low, bins_high)
        values.append(sampled_vals)
    
    # Randomly pairing together the sampled variables
    samples = np.zeros([dim, int(num)])
    for i, param in enumerate(values):
        param_shuffled = np.asarray(param).copy()	
        np.random.shuffle(param_shuffled)	
        samples[i, :] = param_shuffled
    
    return samples