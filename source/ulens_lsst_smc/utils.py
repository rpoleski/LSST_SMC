import numpy as np
import configparser


# Contains:
#  get_dicts_from_file()
#  parse_dict()
#  find_nearest_value()
#  find_index_of_nearest_value()
#  random_power_law()
#  random_broken_power_law()
#  cumulative_IMF()

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

def random_power_law(alpha, min_, max_, u=None):
    """XXX"""
    slope = 1. - alpha
    lim_0 = pow(min_, slope)
    lim_1 = pow(max_, slope)
    if u is None:
        u = np.random.rand()
    u = u * (lim_1 - lim_0) + lim_0
    return pow(u, 1./slope)

def random_broken_power_law(alpha_1, alpha_2, x_0, x_1, x_2):
    """XXX"""
    slope_1 = 1. - alpha_1
    slope_2 = 1. - alpha_2
    int_1 = (pow(x_1, slope_1) - pow(x_0, slope_1)) / slope_1
    int_1 *= pow(x_1, alpha_1-alpha_2)
    int_2 = (pow(x_2, slope_2) - pow(x_1, slope_2)) / slope_2
    border = int_1/(int_1 + int_2)
    u = np.random.rand()
    if u < border:
        u /= border
        return random_power_law(alpha_1, x_0, x_1, u)
    else:
        u = (u - border) / (1. - border)
        return random_power_law(alpha_2, x_1, x_2, u)

def cumulative_IMF(mass):
    """
    Cumulative distribution function of initial mass function.
    """
    IMF_A1, IMF_A2, IMF_A3 = 0.3, 1.3, 2.3
    IMF_M0, IMF_M1, IMF_M2, IMF_M3 = 0.01, 0.08, 0.5, 150.0

    tmp = pow(IMF_M1, IMF_A1-IMF_A2) * (pow(IMF_M1,1.0-IMF_A1) - pow(IMF_M0,1.0-IMF_A1)) / (1.0-IMF_A1)
    tmp += (pow(IMF_M2,1.0-IMF_A2) - pow(IMF_M1,1.0-IMF_A2)) / (1.0-IMF_A2)
    tmp += pow(IMF_M2, IMF_A3-IMF_A2) * (pow(IMF_M3,1.0-IMF_A3) - pow(IMF_M2,1.0-IMF_A3)) / (1.0-IMF_A3)

    # IMF normalization
    IMF_B = 1.0/tmp
    IMF_A = IMF_B * pow(IMF_M1, IMF_A1-IMF_A2)
    IMF_C = IMF_B * pow(IMF_M2, IMF_A3-IMF_A2)

    IMF_CDF1 = IMF_A * (pow(IMF_M1,1.0-IMF_A1) - pow(IMF_M0,1.0-IMF_A1)) / (1.0-IMF_A1)
    IMF_CDF2 = 1.0 - IMF_C * (pow(IMF_M3,1.0-IMF_A3) - pow(IMF_M2,1.0-IMF_A3)) / (1.0-IMF_A3)

    if mass < IMF_M0:
        return 0.0
    elif mass <= IMF_M1:
        tmp = pow(mass, 1.0-IMF_A1) - pow(IMF_M0, 1.0-IMF_A1)
        tmp = tmp * IMF_A / (1.0-IMF_A1)
        return tmp
    elif mass <= IMF_M2:
        tmp = pow(mass, 1.0-IMF_A2) - pow(IMF_M1, 1.0-IMF_A2)
        tmp = tmp * IMF_B / (1.0-IMF_A2)
        tmp = tmp + IMF_CDF1
        return tmp
    elif mass <= IMF_M3:
        tmp = pow(mass, 1.0-IMF_A3) - pow(IMF_M2, 1.0-IMF_A3)
        tmp = tmp * IMF_C / (1.0-IMF_A3)
        tmp = tmp + IMF_CDF2
        return tmp
    else:
        return 1.0
