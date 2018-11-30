import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ulens_lsst_smc.ulenslsst import UlensLSST
from ulens_lsst_smc.utils import parse_dict


opsim_data_file_1 = '../data/baseline2018a_16.56875_-72.89689_EXPANDED_v3.dat'

file_out = None
fig_args = {"left":0.06, "bottom":0.12, "right":.995, "top":.995}

# Below we copy-paste from simulation:
parameters = {"t_0": 2463130.433540467, "u_0": 0.014881549272810046, "t_E": 88.74192060581716, "rho": 0.0011432863542499274, "s": 1.0441383023261566, "q": 0.0007853866082341254, "alpha": 314.85795841740924, "source_flux_u": 0.1477456476065312, "blending_flux_u": 4.79954509730335e-06, "source_flux_g": 0.3571146904574803, "blending_flux_g": 0.0001091440493875484, "source_flux_r": 0.47060249558579836, "blending_flux_r": 0.0004211143677613329, "source_flux_i": 0.5052254652934122, "blending_flux_i": 0.001032761521219054, "source_flux_z": 0.5045459632722024, "blending_flux_z": 0.001583434284274017, "source_flux_y": 0.5023524070172148, "blending_flux_y": 0.0019195527118489471}
t_start = 3116. 
t_stop = 3144.
y_lim = [20.9, 18.1]
size = (10, 3.7)
file_out = '../latex/plot_2.png'

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

dicts = parse_dict(parameters)
parameters = dicts[0]
f_source = dicts[1]
f_blend = dicts[2]

ulens_1 = UlensLSST(opsim_data_1, parameters, f_source, f_blend)
ulens_1.find_detection_time()
ulens_1.add_follow_up()
print(ulens_1.delta_chi2_BL_PL)

kwargs = {
    'subtract_2460000': True}
kwargs_model = {'t_start': 2460000.+t_start, 't_stop': 2460000.+t_stop}
kwargs_model.update(kwargs)

plt.figure(figsize=size)

ulens_1.plot_data(kwargs_data=kwargs, kwargs_model=kwargs_model)
plt.xlim(t_start, t_stop)
plt.ylim(*y_lim)

plt.subplots_adjust(**fig_args)

if file_out is None:
    plt.show()
else:
    plt.savefig(file_out)
