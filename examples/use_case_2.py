import numpy as np

from ulens_lsst_smc.ulenslsst import UlensLSST
from ulens_lsst_smc.utils import get_dicts_from_file


opsim_data_file = '../data/15_baseline2018a_16.56875_-72.89689_EXPANDED_v1.dat'

parameters_file = '../data/params_text.cfg'

# settings end here

dtypes = {'formats': ('f8', 'f8', 'S1'), 'names': ('MJD', '5s', 'f')}
opsim_data = np.loadtxt(opsim_data_file, unpack=True, dtype=dtypes)
full_JD = opsim_data[0] + 2400000.5 # OpSim data are in MJD
filters = np.array([t.decode('UTF-8') for t in opsim_data[2]])
opsim_data = [full_JD, opsim_data[1], filters]

dicts = get_dicts_from_file(parameters_file)
parameters = dicts[0]
f_source = dicts[1]
f_blend = dicts[2]

ulens = UlensLSST(opsim_data, parameters, f_source, f_blend)

ulens.find_detection_time()
print(ulens.detection_time)

ulens.add_follow_up()

