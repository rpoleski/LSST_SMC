import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ulens_lsst_smc.ulenslsst import UlensLSST
from ulens_lsst_smc.utils import parse_dict


opsim_data_file_1 = '../data/15_baseline2018a_16.56875_-72.89689_EXPANDED_v1.dat'
opsim_data_file_2 = '../data/baseline2018a_16.56875_-72.89689_EXPANDED_v3.dat'

# Below we copy-paste from simulation:
parameters = {"t_0": 2462776.427496204, "u_0": 0.002581080984985995, "t_E": 88.51904690894331, "rho": 0.0008723830817228773, "s": 1.1059326739098874, "q": 0.001709377976772477, "alpha": 350.09511924152406, "source_flux_u": 0.13161428424714847, "blending_flux_u": 4.79954509730335e-06, "source_flux_g": 0.29116646164855614, "blending_flux_g": 0.0001091440493875484, "source_flux_r": 0.34969823517494997, "blending_flux_r": 0.0004211143677613329, "source_flux_i": 0.35810435800565765, "blending_flux_i": 0.001032761521219054, "source_flux_z": 0.3473819188894729, "blending_flux_z": 0.001583434284274017, "source_flux_y": 0.3398491994583522, "blending_flux_y": 0.0019195527118489471}

# settings end here

def read_opsim(file_name):
    """
    Read opsim data file and change formats.
    """
    dtypes = {'formats': ('f8', 'f8', 'S1'), 'names': ('MJD', '5s', 'f')}
    opsim_data = np.loadtxt(file_name, unpack=True, dtype=dtypes)
    full_JD = opsim_data[0] + 2400000.5 # OpSim data are in MJD
    filters = np.array([t.decode('UTF-8') for t in opsim_data[2]])
    opsim_data = [full_JD, opsim_data[1], filters]
    return opsim_data

opsim_data_1 = read_opsim(opsim_data_file_1)
opsim_data_2 = read_opsim(opsim_data_file_2)

dicts = parse_dict(parameters)
parameters = dicts[0]
f_source = dicts[1]
f_blend = dicts[2]

ulens_1 = UlensLSST(opsim_data_1, parameters, f_source, f_blend)
ulens_2 = UlensLSST(opsim_data_2, parameters, f_source, f_blend)

ulens_1.find_detection_time()
ulens_2.find_detection_time()

ulens_1.add_follow_up()
ulens_2.add_follow_up()

print(ulens_1.delta_chi2_BL_PL)
print(ulens_2.delta_chi2_BL_PL)

kwargs = {
    'subtract_2460000': True}

gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
plt.figure()
plt.subplot(gs[0])
ulens_1.plot_data(kwargs_data=kwargs, kwargs_model=kwargs)
plt.subplot(gs[1])
ulens_2.plot_data(kwargs_data=kwargs, kwargs_model=kwargs)

plt.show()
