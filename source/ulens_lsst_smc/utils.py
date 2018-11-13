import numpy as np
import configparser


# Contains:
#  get_dicts_from_file()
#  find_nearest_value()
#  find_index_of_nearest_value()

def get_dicts_from_file(file_name):
    """
    Read 3 dicts from config file. The values in dicts are floats.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(file_name)
    
    section = 'parameters'
    param = dict()
    if section in config.sections():
        for var in config[section]:
            param[var] = config.getfloat(section, var)

    section = 'source_fluxes'
    s_fluxes = dict()
    if section in config.sections():
        for var in config[section]:
            s_fluxes[var] = config.getfloat(section, var)

    section = 'blending_fluxes'
    b_fluxes = dict()
    if section in config.sections():
        for var in config[section]:
            b_fluxes[var] = config.getfloat(section, var)

    return param, b_fluxes, s_fluxes

def find_nearest_value(array, value):
    """
    Take a np.ndarray and find the element that is closeset to given value.
    """
    index = find_index_of_nearest_value(array, value)
    return array[index]

def find_index_of_nearest_value(array, value):
    """
    In a np.ndarray find an index of element
    that is closest to the given value.
    """
    return (np.abs(array - value)).argmin()
