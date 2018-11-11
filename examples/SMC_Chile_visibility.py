import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun


limit_target = 30.
limit_sun = -15.

file_out = "SMC_Chile_visibility_v1.dat"

target = SkyCoord("00:52:44.8 -72:49:43", unit=(u.hourangle, u.deg))

observatory = EarthLocation(
    lat=-30.240722*u.deg, lon=-70.73658*u.deg, height=2715*u.m)

time_jd = np.arange(2459853., 2463505., 1./(24 * 12)) # 10 years and 5min spacing

times = Time(time_jd, format='jd')

alt_az = AltAz(obstime=times, location=observatory)

target_altaz = target.transform_to(alt_az).alt.value
mask_1 = target_altaz > limit_target

sun_altaz = get_sun(times).transform_to(alt_az).alt.value
mask_2 = sun_altaz < limit_sun

mask = mask_1 * mask_2
data_out = np.array([time_jd[mask], target_altaz[mask], sun_altaz[mask]]).T

np.savetxt(file_out, data_out, "%.5f")
