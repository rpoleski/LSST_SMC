import numpy as np
import pickle

from ulens_lsst_smc import utils

class SimulatedEvents(object):
    """
    Simulated microlensing events.

    Arguments :
        microlensing_file: 
            XXX
        isochrone_old_file:
            XXX
        isochrone_young_file:
            XXX
        model_old_file:
            XXX
        model_young_file:
            XXX
    """
    def __init__(
                self, microlensing_file,
                isochrone_old_file, isochrone_young_file,
                model_old_file, model_young_file):

        self._microlensing_file = microlensing_file

        self._isochrone_old = self._read_isochrone(isochrone_old_file)
        self._isochrone_young = self._read_isochrone(isochrone_young_file)

        with open(model_old_file, 'rb') as file_:
            self._GMM_old = pickle.load(file_, encoding='latin1')
        with open(model_young_file, 'rb') as file_:
            self._GMM_young = pickle.load(file_, encoding='latin1')

        self.frac_young = 0.4
        self.n_samples = 10
        self.flag = None
        self.ra = None
        self.dec = None
        self.dist = None

    def _read_isochrone(self, file_name):
        """
        Reads isochrone file
        """
        use_cols = (2,23,24,25,26,27,28)
        data = np.loadtxt(file_name, unpack=True, usecols=use_cols)
        out = {'M_initial': data[0]}
        for (i, band) in enumerate(['u', 'g', 'r', 'i', 'z', 'Y']):
            out[band] = data[i]
        return out

    def generate_coords(self):
        n_young = int(self.frac_young*self.n_samples) # number of young stars
        n_old = self.n_samples-n_young # number of old stars
        self._flag = np.ones(self.n_samples,dtype=int)
        self._flag[0:n_young] = 0
        # old stars
        coords_old, ident_old = self._GMM_old.sample(n_samples=n_old)
        ra_old, dec_old, dist_old = utils.xyz_to_ra_dec(coords_old[:,0], coords_old[:,1], coords_old[:,2])
        # young stars
        coords_yng, ident_yng = self._GMM_young.sample(n_samples=n_young)
        ra_yng, dec_yng, dist_yng = utils.xyz_to_ra_dec(coords_yng[:,0],coords_yng[:,1],coords_yng[:,2])
        self.ra = np.concatenate((ra_yng,ra_old))
        self.dec = np.concatenate((dec_yng,dec_old))
        self.dist = np.concatenate((dist_yng,dist_old))
        
