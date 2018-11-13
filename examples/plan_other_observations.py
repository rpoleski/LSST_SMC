"""
Transforms Chilean observations to non-Chilean observations.
"""
import numpy as np
import sys

from ulens_lsst_smc import utils


file_in = sys.argv[1]

jd_shift = 0.5 # JD for non-Chilean observations is shifted by
# half a day relative to Chilean observations.

selection_probability = 0.5 # We assume data are taken over half of
# as many nights as in Chile.

(times, five_sigma_depth) = np.loadtxt(file_in, unpack=True)

times_shifted = times + jd_shift
times_shifted_int = np.rint(times_shifted)

unique_nights = np.unique(times_shifted_int)

mask = (np.random.random_sample(len(unique_nights)) < selection_probability)
selected_nights = unique_nights[mask]

form = "{:.5f} {:.3f}"
for night in selected_nights:
    selected = np.where(night == times_shifted_int)[0]
    index = np.random.choice(selected, size=1)[0]
    print(form.format(times_shifted[index], five_sigma_depth[index]))
