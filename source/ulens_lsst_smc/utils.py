import numpy as np
import configparser


# Contains:
#  get_dicts_from_file()
#  parse_dict()
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

def parse_dict(input_dict):
    """
    XXX
    """
    parameters = {}
    source_flux = {}
    blending_flux = {}
    for (key, value) in input_dict.items():
        if key.startswith("source_flux_"):
            source_flux[key[-1]] = value
        elif key.startswith("blending_flux_"):
            blending_flux[key[-1]] = value
        else:
            parameters[key] = value
    return (parameters, source_flux, blending_flux)

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

def xyz_to_ra_dec(x, y, z):
    """
    Transforming Cartesian coordinates to equatorial coordinates
    """
    deg = np.pi/180.0
    ra_0 = 16.25*deg # SMC center
    dec_0 = -72.42*deg # SMC center
    # Heliocentric distance:
    dist = np.sqrt(x**2+y**2+z**2)
    # Calculating equatorial coordinates
    sindec = (y/dist)*np.cos(dec_0)+(z/dist)*np.sin(dec_0)
    dec = np.arcsin(sindec)
    sinalfcosdec = -x/dist
    cosalfcosdec = -(y/dist)*np.sin(dec_0)+(z/dist)*np.cos(dec_0)
    alf = np.arctan2(sinalfcosdec,cosalfcosdec)+ra_0
    # radians to hours/degrees
    dec /= deg
    alf *= 12.0/np.pi
    return alf,dec,dist
