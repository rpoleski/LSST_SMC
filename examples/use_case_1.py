import numpy as np

import UlensLSST
from utils import get_dicts_from_file


opsim_data_file = ???

parameters_file = ???

# settings end here

dtypes = {'formats': ('f8', 'f8', 'S1'), 'names': ('MJD', '5s', 'f')}
opsim_data = np.loadtxt(opsim_data_file, unpack=True, dtype=dtypes)
full_JD = opsim_data[0] + 2400000.5 # OpSim data are in MJD
filters = [t.decode('UTF-8') for t in opsim_data[2]]
opsim_data = [full_JD, opsim_data[1], filters]

dicts = get_dicts_from_file(parameters_file) # This also has to be written.
parameters = dicts[0]
f_source = parameters[1]
f_blend = parameters[2]

ulens = UlensLSST(opsim_data, parameters, f_source, f_blend)

ulens.simulate_lightcurve()
ulens.find_detection_time()
ulens.add_follow_up()
d_chi2 = ulens.get_PSPL_delta_chi2()

print(d_chi2)
