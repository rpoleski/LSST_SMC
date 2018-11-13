"""
Take info on when SMC is visible, simulate uniform observations during that
time and check if LSST has taken data (to simulate weather).
"""
import numpy as np
import sys

from ulens_lsst_smc import utils


file_vis = sys.argv[1]
file_LSST_epochs = sys.argv[2]
file_LSST_5sigma = sys.argv[3]

requested_cadence = 1. / 24 # i.e. 1 hr
minimum_dt = 0.5 * requested_cadence 
# We need at least 30 minutes of target visibility to get 1 datapoint.

max_diff = 1. / 24 # We accept epoch, if there is at least 1 LSST epoch
# taken within 1 hour of the checked epoch.

visible = np.loadtxt(file_vis, unpack=True, usecols=(0))

visible_night_begin = visible.astype(int)

unique_nights = np.unique(visible_night_begin)

jds_all_nights = []
for night in unique_nights:
    mask = (visible_night_begin == night)
    night_jds = visible[mask]

    dt = np.max(night_jds) - np.min(night_jds)
    if dt < minimum_dt:
        continue

    # Below we optimaly place observations in given night.
    n_obs = int(dt/requested_cadence) + 1
    time_diff = dt - (n_obs - 1) * requested_cadence
    first_jd = night_jds[0] + time_diff / 2.
    requested_jds = first_jd + requested_cadence * np.arange(n_obs)

    jds = [utils.find_nearest_value(night_jds, jd) for jd in requested_jds]
    jds_all_nights.extend(jds)

LSST_epochs = np.loadtxt(file_LSST_epochs, unpack=True, usecols=(0))
LSST_epochs += 2400000.5

(LSST_5sigma_time, LSST_5sigma_value) = np.loadtxt(
        file_LSST_5sigma , unpack=True, usecols=(0, 1))
LSST_5sigma_time += 2400000.5

for jd in jds_all_nights:
    if np.any((LSST_epochs < jd + max_diff) & (LSST_epochs > jd - max_diff)):
        index = utils.find_index_of_nearest_value(LSST_5sigma_time, jd)
        print("{:.5f} {:.3f}".format(jd, LSST_5sigma_value[index]))
