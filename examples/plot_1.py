import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ulens_lsst_smc.ulenslsst import UlensLSST
from ulens_lsst_smc.utils import parse_dict


opsim_data_file_1 = '../data/baseline2018a_16.56875_-72.89689.dat'
opsim_data_file_2 = '../data/baseline2018a_16.56875_-72.89689_EXPANDED_v3.dat'

file_out = None
fig_args = {"left":0.06, "bottom":0.08, "right":.995, "top":.995}

# Below we copy-paste from simulation:
parameters = {"t_0": 2460998.103228466, "u_0": 0.050822751022266965, "t_E": 145.63994066924565, "rho": 0.0005071493178190735, "s": 0.9642818673294677, "q": 0.00011988059592119628, "alpha": 146.71606262815484, "source_flux_u": 0.08684171855959938, "blending_flux_u": 0.03522237950558175, "source_flux_g": 0.1955208070977924, "blending_flux_g": 0.11030176475766786, "source_flux_r": 0.24707720378487355, "blending_flux_r": 0.17840043711647197, "source_flux_i": 0.25944931350887507, "blending_flux_i": 0.21858384178345014, "source_flux_z": 0.25552235838458154, "blending_flux_z": 0.23977827529130874, "source_flux_y": 0.2517196376158122, "blending_flux_y": 0.2526652959967369}

t_start = 815.
t_stop = 1110.
y_lim = [23.45, 19.75]
size = (10, 6)
file_out = '../latex/plot_1.png'

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
kwargs_model = {'t_start': 2460000.+t_start, 't_stop': 2460000.+t_stop}
kwargs_model.update(kwargs)

gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
plt.figure(figsize=size)

plt.subplot(gs[0])
ulens_1.plot_data(kwargs_data=kwargs, kwargs_model=kwargs_model)
plt.xlim(t_start, t_stop)
plt.ylim(*y_lim)

plt.subplot(gs[1])
ulens_2.plot_data(kwargs_data=kwargs, kwargs_model=kwargs_model)
plt.xlim(t_start, t_stop)
plt.ylim(*y_lim)

plt.subplots_adjust(**fig_args)

if file_out is None:
    plt.show()
else:
    plt.savefig(file_out)
