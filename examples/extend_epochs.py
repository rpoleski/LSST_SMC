"""
Reads a list of epochs and tries to fill the gaps by adding observations
during engeering runs.
1st parameter - input
2nd parameter - output
"""
import sys
import numpy as np


in_file = sys.argv[1]
out_file = sys.argv[2]

min_skip_dt = 7. # This is minimum length of gap in days that we will treat.

time = np.loadtxt(in_file, unpack=True)
time_min = np.min(time)
time_max = np.max(time)

diff = time[1:] - time[:-1]
indexes = np.where(diff > min_skip_dt)[0]
gaps_begin = time[indexes]
gaps_end = time[indexes+1]

add = []
for i in range(len(indexes)):
    begin = gaps_begin[i]
    end = gaps_end[i]
    # -10 to 10 without 0 and in random order:
    for j in [-7, -4, 4, -3, 5, -6, 8, 9, -8, -1, -10, 1, -2, 7, -5, 10, 6, 3, -9, 2]:
        dt = int(round(365.25 * j))
        shifted_begin = begin + dt
        shifted_end = end + dt
        if shifted_begin < time_min or shifted_end > time_max:
            continue
        overlap = False
        for k in range(len(indexes)):
            if i == k:
                continue
            if gaps_begin[k] > shifted_begin and gaps_begin[k] < shifted_end:
                overlap = True
            if gaps_end[k] > shifted_begin and gaps_end[k] < shifted_end:
                overlap = True
        if overlap:
            continue
        mask = ((time > shifted_begin) & (time < shifted_end))
        if np.sum(mask) == 0:
            continue
        add.extend(time[mask] - dt)
        break

time_out = np.concatenate( (time, add) )
time_out.sort()
np.savetxt(out_file, time_out.T)
