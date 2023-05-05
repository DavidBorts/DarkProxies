# Miscellaneous util functions

import Constants as c

def get_possible_values(proxy_type, params):
    '''
    Given a proxy type and a subset of its input params,
    return the corresponding ranges of possible values
    of those parameters
    '''
    all_possible_values = c.POSSIBLE_VALUES[proxy_type]
    all_param_names = c.PARAM_NAMES[proxy_type]

    if params is None:
        return all_possible_values

    possible_values = [all_possible_values[all_param_names.index(param)] for param in params]
    return possible_values