import numpy as np
import math
import pickle
import json
from scipy.integrate import simps

from MulensModel import Utils

from ulens_lsst_smc import utils


class SimulatedEvents(object):
    """
    Simulated microlensing events.

    Drawing random masses from the IMF (assuming that the IMF can be
    approximated as a power law f(M) ~ M^{-a} in the mass range [M1,M2].
    Kroupa (2001) IMF: a = 2.3 for 0.5 < M < 150.0
    Kalirai et al. (2012) SMC IMF: a = 1.9 for 0.37 < M < 0.93
    http://adsabs.harvard.edu/abs/2013ApJ...763..110K
    Here we use 2.3, i.e., same as in Mroz & Poleski 2018.
    The minimum mass is 0.65. The maximum mass is 0.92 for old population
    and 3.5 for young population.

    Arguments :
        microlensing_file: 
            File used to simulate t_E distribution. The first column of this
            file is log10(t_E), the second is weight.

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

        mu_rel_file:
            File with distribution of proper motions.
    """
    def __init__(
                self, microlensing_file,
                isochrone_old_file, isochrone_young_file,
                model_old_file, model_young_file, mu_rel_file=None):

        self._microlensing_file = microlensing_file

        self._bands = ['u', 'g', 'r', 'i', 'z', 'Y']
        self._isochrone_old = self._read_isochrone(isochrone_old_file)
        self._isochrone_young = self._read_isochrone(isochrone_young_file)
        self._mu_rel_file = mu_rel_file

        with open(model_old_file, 'rb') as file_:
            self._GMM_old = pickle.load(file_, encoding='latin1')
        with open(model_young_file, 'rb') as file_:
            self._GMM_young = pickle.load(file_, encoding='latin1')

        self.frac_young = 0.4
        self.n_samples = 1000

        self.survey_start = 2459853.
        self.survey_stop = 2463505.

        self._source_min_mass = 0.65
        self._source_max_mass_old = 0.92
        self._source_max_mass_young = 3.5
        self._imf_slope = 2.3
        self._distance_modulus = 18.99 # includes extinction
        self._lens_mass = 0.2 # very rough approximation
        self._seeing_FWHM = 1. # arcsec - used to estimate blending
        self._m_total_SMC = 1.e9 # total mass of SMC in M_Sun

        # Ranges of properties of simulated planets:
        self._min_s = 0.3
        self._max_s = 3.0
        self._min_q = 1.e-4
        self._max_q = 0.03

        self._min_log_s = math.log10(self._min_s)
        self._max_log_s = math.log10(self._max_s)
        self._min_log_q = math.log10(self._min_q)
        self._max_log_q = math.log10(self._max_q)

        self._flag = None
        self._ra = None
        self._dec = None
        self._dist = None
        self._source_mass = None # These are initial masses of sources.
        self._source_flux = None
        self._blending_flux = None
        self._planet_parameters = None

    def _read_isochrone(self, file_name):
        """
        Reads isochrone file
        """
        use_cols = (2, 23, 24, 25, 26, 27, 28, 4, 5)
        data = np.loadtxt(file_name, unpack=True, usecols=use_cols)
        out = {'M_initial': data[0]}
        for (i, band) in enumerate(self._bands):
            out[band] = data[i+1]
        out['radius'] = 10**(.5*data[7] - 2.*(data[8]-3.761))
        # This is relative to R_Sun. 3.761 is log_10(T_eff_Sun).
        return out

    def generate_coords(self):
        """
        Generates RA, Dec coordinates of events.
        Sets .ra, .dec. and .dist properties.
        """
        self._n_young = int(self.frac_young*self.n_samples) # number of young stars
        self._n_old = self.n_samples - self._n_young # number of old stars
        self._flag = np.ones(self.n_samples,dtype=int)
        self._flag[0:self._n_young] = 0
        # old stars
        (coords_old, ident_old) = self._GMM_old.sample(n_samples=self._n_old)
        (ra_old, dec_old, dist_old) = utils.xyz_to_ra_dec(
                coords_old[:,0], coords_old[:,1], coords_old[:,2])
        # young stars
        (coords_yng, ident_yng) = self._GMM_young.sample(n_samples=self._n_young)
        (ra_yng, dec_yng, dist_yng) = utils.xyz_to_ra_dec(
                coords_yng[:,0], coords_yng[:,1], coords_yng[:,2])
        self._ra = np.concatenate( (ra_yng, ra_old) )
        self._dec = np.concatenate( (dec_yng, dec_old) )
        self._dist = np.concatenate( (dist_yng, dist_old) )
        self._coords = np.concatenate( (coords_yng, coords_old) )

    def _random_from_IMF(self, source_max_mass):
        """
        Drawing random source masses from the IMF
        """
        u = np.random.rand(self.n_samples)
        out = utils.random_power_law(self._imf_slope, self._source_min_mass,
                source_max_mass, u)
        return out

    def _generate_source_mass(self):
        """just generate masses of sources"""
        mass_old = self._random_from_IMF(self._source_max_mass_old)
        mass_young = self._random_from_IMF(self._source_max_mass_young)
        self._source_mass = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            if self._flag[i] == 0:
                self._source_mass[i] = mass_young[i]
            else:
                self._source_mass[i] = mass_old[i]

    def _generate_source_and_lens_flux(self):
        """generates fluxes of source and lens"""
        old_mags = {}
        young_mags = {}
        lens_flux = {}
        for band in self._bands:
            old_mags[band] = np.interp(self._source_mass,
                self._isochrone_old['M_initial'],
                self._isochrone_old[band])
            young_mags[band] = np.interp(self._source_mass,
                self._isochrone_young['M_initial'],
                self._isochrone_young[band])
            old_mags[band] += self._distance_modulus
            young_mags[band] += self._distance_modulus
            lens_mag = np.interp(self._lens_mass,
                self._isochrone_young['M_initial'],
                self._isochrone_young[band])
            lens_flux[band] = Utils.get_flux_from_mag(lens_mag +
                self._distance_modulus)

        old_radii = np.interp(self._source_mass,
            self._isochrone_old['M_initial'],
            self._isochrone_old['radius'])
        young_radii = np.interp(self._source_mass,
            self._isochrone_young['M_initial'],
            self._isochrone_young['radius'])

        self._source_flux = [{} for _ in range(self.n_samples)]
        self._blending_flux = [{} for _ in range(self.n_samples)]
        self._source_radius = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            if self._flag[i] == 0:
                mags = young_mags
                self._source_radius[i] = young_radii[i]
            else:
                mags = old_mags
                self._source_radius[i] = old_radii[i]
            for band in self._bands:
                self._source_flux[i][band] = Utils.get_flux_from_mag(mags[band][i])
                self._blending_flux[i][band] = lens_flux[band]

    def _generate_additional_blending_flux(self):
        """
        simulate how much mass is within seeing and then simulate masses
        in addition to lens and source
        """
        distance_min = 45. # kpc
        distance_max = 80. # kpc

        distances = np.arange(distance_min, distance_max)
        coords_temp = np.zeros( (len(distances), 3) )
        coords_temp[:, 2] = distances
        factor = np.pi * (0.5*self._seeing_FWHM * distances / 206265.)**2
        factor *= self._m_total_SMC
        for i in range(self.n_samples):
            coords_temp[:, 0] = self._coords[i, 0]
            coords_temp[:, 1] = self._coords[i, 1]
            densities_old = np.exp(self._GMM_old.score_samples(coords_temp))
            densities_young = np.exp(self._GMM_young.score_samples(coords_temp))
            densities_old *= factor
            densities_young *= factor
            old_mass = (1.-self.frac_young) * simps(densities_old, distances)
            young_mass = self.frac_young * simps(densities_young, distances)
            total_mass = old_mass + young_mass

            masses = []
            mass_diff = total_mass - self._source_mass[i] - self._lens_mass
            while mass_diff > 0.08:
                if mass_diff > 0.5:
                    mass = utils.random_broken_power_law(1.3, 2.3, 0.08, 0.5, mass_diff)
                else:
                    mass = utils.random_power_law(1.3, 0.08, mass_diff)
                masses.append(mass)
                mass_diff -= mass
                # Update blending fluxes:
                if np.random.rand() < self.frac_young:
                    if mass < self._source_max_mass_young:
                    # More massive objects are already WD etc.
                        for band in self._bands:
                            mag = np.interp(mass, self._isochrone_young['M_initial'],
                                self._isochrone_young[band]) + self._distance_modulus
                            self._blending_flux[i][band] += Utils.get_flux_from_mag(mag)
                else:
                    if mass < self._source_max_mass_old:
                        for band in self._bands:
                            mag = np.interp(mass, self._isochrone_old['M_initial'],
                                self._isochrone_old[band]) + self._distance_modulus
                            self._blending_flux[i][band] += Utils.get_flux_from_mag(mag)

    def generate_fluxes(self):
        """
        Generate source and blending fluxes.
        """
        self._generate_source_mass()
        self._generate_source_and_lens_flux()
        self._generate_additional_blending_flux()

    def generate_microlensing_parameters(self):
        """
        Generates internal parameters: t_0, u_0, t_E, theta_E, and rho.
        """
        self._t_0 = np.random.uniform(self.survey_start, self.survey_stop,
                                     self.n_samples)
        self._u_0 = np.random.uniform(-1., 1., self.n_samples)

        (log_t_E, weights) = np.loadtxt(self._microlensing_file, unpack=True)
        CDF = np.cumsum(weights)
        CDF /= CDF[-1]
        prob = np.random.uniform(0., 1., self.n_samples)
        log_t_E_rand = np.interp(prob, CDF, log_t_E)
        self._t_E = pow(10., log_t_E_rand)

        if self._mu_rel_file is None:
            mu_rel = np.random.normal(0.281, 0.135, self.n_samples)
            mu_rel[mu_rel < 0.01] = 0.01
        else:
            (mu_rel, weights) = np.loadtxt(self._mu_rel_file, unpack=True)
            CDF_mu_rel = np.cumsum(weights)
            CDF_mu_rel /= CDF_mu_rel[-1]
            prob = np.random.uniform(0., 1., self.n_samples)
            mu_rel = np.interp(prob, CDF_mu_rel, mu_rel)
        self._theta_E = self._t_E * mu_rel / 365.25 # mas

        radius = self._source_radius * 0.696e6 # km
        theta_star = radius / (65.e3 * 149.6e6) # arcsec; Here we assume 65kpc
        # because it's a source distance.
        theta_star *= 1000. # mas
        self._rho = theta_star / self._theta_E

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

    def save(self, file_name):
        """
        Saves a list of dicts to a json file
        """
        with open(file_name, 'w') as file_:
            file_.write("[\n")
            for i in range(self.n_samples):
                out = {'t_0': self._t_0[i], 'u_0': self._u_0[i],
                       't_E': self._t_E[i], 'rho': self._rho[i]}
                for (key, value) in self._planet_parameters[i].items():
                    out[key] = value
                for band_ in self._bands:
                    band = band_.lower()
                    out["source_flux_" + band] = self._source_flux[i][band_]
                    out["blending_flux_" + band] = self._blending_flux[i][band_]
                json.dump(out, file_)
                if i < self.n_samples - 1:
                    file_.write(",\n")
            file_.write("\n]\n")
