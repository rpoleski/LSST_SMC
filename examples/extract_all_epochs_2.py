"""
Extracts database with all the epochs in given LSST simulation.
1st parameter - input npz file
2nd parameter - output file
"""
import sys
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


file_name = sys.argv[1]
file_out = sys.argv[2]

data = np.load(file_name)

column_use = 'observationStartMJD'

radec_all = []
for i_data in data['metricValues']:
    if i_data is None:
        continue
    for (ra, dec) in zip(i_data['fieldRA'], i_data['fieldDec']):
        radec_all.append( (ra, dec) )
radec_all = np.array(list(set(radec_all)))
radec_all = radec_all[np.argsort(radec_all, axis=0)[:,0]]

# extract unique fields:
done = set()
data_out = []
for i_data in data['metricValues']:
    if i_data is None:
        continue
    for (ra_deg, dec_deg) in set(zip(i_data['fieldRA'], i_data['fieldDec'])):
        if (ra_deg, dec_deg) in done:
            print("SKIP")
            continue
        done.update( (ra_deg, dec_deg) )
        mask = (i_data['fieldRA'] == ra_deg) & (i_data['fieldDec'] == dec_deg)
        data_out.extend(i_data[column_use][mask])
        
np.savetxt(file_out, np.sort(np.array(data_out)).T)
