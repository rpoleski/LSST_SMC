import numpy as np
import matplotlib.pyplot as plt
import json

from ulens_lsst_smc.ulenslsst import UlensLSST
from ulens_lsst_smc.utils import parse_dict


#opsim_data_file = '../data/15_baseline2018a_16.56875_-72.89689_EXPANDED_v1.dat'
opsim_data_file = '../data/baseline2018a_16.56875_-72.89689_EXPANDED_v3.dat'

parameters_file = '../data/simulated_events_001.json'

chi2_detection_limit = 100.

# settings end here

dtypes = {'formats': ('f8', 'f8', 'S1'), 'names': ('MJD', '5s', 'f')}
opsim_data = np.loadtxt(opsim_data_file, unpack=True, dtype=dtypes)
full_JD = opsim_data[0] + 2400000.5 # OpSim data are in MJD
filters = np.array([t.decode('UTF-8') for t in opsim_data[2]])
opsim_data = [full_JD, opsim_data[1], filters]

with open(parameters_file) as file_:
    parameters_all = json.load(file_)
    
for parameters_ in parameters_all:
    print("############")
    dicts = parse_dict(parameters_)
    parameters = dicts[0]
    f_source = dicts[1]
    f_blend = dicts[2]
    
    ulens = UlensLSST(opsim_data, parameters, f_source, f_blend)
    ulens.find_detection_time()
    print("Event detection:", ulens.detection_time, ulens.detection_band)
    if ulens.detection_time is None:
        continue

    ulens.add_follow_up()

    print("Delta chi2 for binary lens:", ulens.delta_chi2_BL_PL, flush=True)
    if ulens.delta_chi2_BL_PL is not None:
        if ulens.delta_chi2_BL_PL > chi2_detection_limit:
            print("Planet is detected.")
            print("Parameters:")
            print("t_0: {:.2f} u_0: {:.3f} t_E: {:.3f}".format(
                *[parameters[p] for p in ['t_0', 'u_0', 't_E']]))
            print("rho: {:.4f} s: {:.3f} q: {:.5f} alpha: {:.2f}".format(
                *[parameters[p] for p in ['rho', 's', 'q', 'alpha']]))
