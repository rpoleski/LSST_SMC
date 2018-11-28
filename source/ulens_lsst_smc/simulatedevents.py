import numpy as np
import math
import pickle

from ulens_lsst_smc import utils


class SimulatedEvents(object):
    """
    Simulated microlensing events.

    Arguments :
        microlensing_file: 
            File used to simulate basic parameters.
            Format not yet specified - XXX.

        isochrone_old_file:
            File with isochrone for _old_ population.
            Assumes format of PARSEC/CMD.

        isochrone_young_file:
            File with isochrone for _young_ population.
            Assumes format of PARSEC/CMD.

        model_old_file:
            Pickled file with gaussian mixture model for old population.

        model_young_file:
            Pickled file with gaussian mixture model for young population.
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
        self.n_samples = 1000
        self._flag = None
        self.ra = None
        self.dec = None
        self.dist = None

        self.survey_start = 2459853.
        self.survey_stop = 2463505.

        # Ranges of properties of simulated planets:
        self._min_s = 0.3
        self._max_s = 3.0
        self._min_q = 1.e-4
        self._max_q = 0.03

        self._min_log_s = math.log10(self._min_s)
        self._max_log_s = math.log10(self._max_s)
        self._min_log_q = math.log10(self._min_q)
        self._max_log_q = math.log10(self._max_q)

        self._planet_parameters = None

    def _read_isochrone(self, file_name):
        """
        Reads isochrone file
        """
        use_cols = (2, 23, 24, 25, 26, 27, 28, 4, 5)
        data = np.loadtxt(file_name, unpack=True, usecols=use_cols)
        out = {'M_initial': data[0]}
        for (i, band) in enumerate(['u', 'g', 'r', 'i', 'z', 'Y']):
            out[band] = data[i+1]
        out['radius'] = 10**(.5*data[7] - 2.*(data[8]-3.761))
        # This is relative to R_Sun. 3.761 is log_10(T_eff_Sun).
        return out

    def generate_coords(self):
        """
        Generates RA, Dec coordinates of events.
        Sets .ra, .dec. and .dist properties.
        """
        n_young = int(self.frac_young*self.n_samples) # number of young stars
        n_old = self.n_samples-n_young # number of old stars
        self._flag = np.ones(self.n_samples,dtype=int)
        self._flag[0:n_young] = 0
        # old stars
        (coords_old, ident_old) = self._GMM_old.sample(n_samples=n_old)
        (ra_old, dec_old, dist_old) = utils.xyz_to_ra_dec(
                coords_old[:,0], coords_old[:,1], coords_old[:,2])
        # young stars
        (coords_yng, ident_yng) = self._GMM_young.sample(n_samples=n_young)
        (ra_yng, dec_yng, dist_yng) = utils.xyz_to_ra_dec(
                coords_yng[:,0], coords_yng[:,1], coords_yng[:,2])
        self.ra = np.concatenate( (ra_yng, ra_old) )
        self.dec = np.concatenate( (dec_yng, dec_old) )
        self.dist = np.concatenate( (dist_yng, dist_old) )

    def generate_fluxes(self):
        """XXX"""
        pass

    def generate_microlensing_parameteres(self):
        """
        Generates t_0 and u_0
        """
        self._t_0 = np.random.uniform(self.survey_start, self.survey_stop,
                                     self.n_samples)
        self._u_0 = np.random.uniform(-1., 1., self.n_samples)

    def _Suzuki2016_Udalski2018(self, log_s, log_q):
        """
        Calculate planet rate assuming Galactic bulge properties.
        """
        s = 10**log_s
        q = 10**log_q

        A = 0.61
        m = 0.49
        q_br = 1.7e-4
        if q > q_br:
            n = -0.93
        else:
            n = 0.73

        return A * (q/q_br)**n * s**m

    def add_planets(self):
        """
        Add parameters (s, q, alpha). The population of planet is taken from
        `Suzuki et al. (2016)
        <http://adsabs.harvard.edu/abs/2016ApJ...833..145S>`_ and
        `Udalski et al. (2018)
        <http://adsabs.harvard.edu/abs/2018AcA....68....1U>`_.

        XXX - do we modify it?
        """
        log_s = np.random.uniform(
            self._min_log_s, self._max_log_s, self.n_samples)
        log_q = np.random.uniform(
            self._min_log_q, self._max_log_q, self.n_samples)
        rates = np.ones(self.n_samples)
        rates *= (self._max_log_q - self._min_log_q)
        rates *= (self._max_log_s - self._min_log_s)
        for (i, (logs, logq)) in enumerate(zip(log_s, log_q)):
            rates[i] *= self._Suzuki2016_Udalski2018(logs, logq)

        # If the rate for given (s,q) is >1 than we add multiple planets
        # to the list and later assign them to random events.
        # This way total number of planets is conserved and we have only
        # 0 or 1 planets per system.
        to_add = []
        for (rate, logs, logq) in zip(rates, log_s, log_q):
            n_add = int(rate)
            for _ in range(n_add):
                to_add.append([logs, logq])
            if rate-n_add > np.random.uniform():
                to_add.append([logs, logq])

        if len(to_add) > self.n_samples:
            raise ValueError('to many planets')

        order = np.random.permutation(self.n_samples)
        alpha = np.random.uniform(0., 360., len(to_add))

        if self._planet_parameters is None:
            self._planet_parameters = [{} for _ in range(self.n_samples)]

        for i in range(len(to_add)):
            params = to_add[i]
            self._planet_parameters[order[i]] = {
                's': 10**params[0], 'q': 10**params[1], 'alpha': alpha[i]}
