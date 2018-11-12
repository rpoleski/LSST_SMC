"""
Extracts database with all the epochs in given LSST simulation.
1st parameter - path to OpSim db file
2nd parameter - too name for output file
"""
import sys
import numpy as np 
from lsst.sims.maf import db, metrics, metricBundles, slicers
from astropy import units as u
from astropy.coordinates import SkyCoord


database = sys.argv[1]
out_file = sys.argv[2]
out_dir = sys.argv[2]

gal_l_min = 0.
gal_l_max = 360.
gal_b_min = -89.
gal_b_max = 89.
diameter = 3.5
step = diameter / np.sqrt(2) # This would be enough on a 2D plane.
step *= 0.85

gal_l_all = np.linspace(gal_l_min, gal_l_max, (gal_l_max-gal_l_min)/step+1)
gal_b_all = np.linspace(gal_b_min, gal_b_max, (gal_b_max-gal_b_min)/step+1)
(gal_l, gal_b) = np.meshgrid(gal_l_all, gal_b_all)

c = SkyCoord(gal_l.flatten(), gal_b.flatten(), unit=u.deg, frame='galactic')
userRA = c.fk5.ra.value
userDec = c.fk5.dec.value

columns = ['observationStartMJD', 'filter', 'fiveSigmaDepth']

metric = metrics.PassMetric(cols=columns) 
slicer = slicers.UserPointsSlicer(userRA, userDec)
sqlconstraint = ''
MJDmetric = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                                            fileRoot=out_file)
bundleDict = {'MJDmetric': MJDmetric}
opsdb = db.OpsimDatabase(database)
group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=out_dir)
group.runAll()
