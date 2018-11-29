import numpy as np
import math
import os
import scipy.optimize as op
import matplotlib.pyplot as plt

import MulensModel as MM

from ulens_lsst_smc import utils


class UlensLSST(object):
    """
    Model and simulated observed light curve of microlensing event in LSST
    and follow-up data.

    Arguments :
        opsim_data: *list* of three 1-D *np.ndarrays*
            Data read from OpSim database: JD (full format),
            5-sigma detection threshold, and filter.

        parameters: *dict*
            event parameters - see MulensModel.ModelParameters

        source_fluxes: *dict*
            Flux of sources for each band as *float* and in
            MulensModel conventions (zero-point of 22 mag).

        blending_fluxes: *float*
            Blending flux for each band as *float* and in
            MulensModel conventions (zero-point of 22 mag).

        coords: *str* or *MM.Coordinates*, optional
            Event coordinates. SMC center is assumed if not provided.

        follow_up_file_1: *str*, optional
            Name of a 2-column file with MJD and 5 sigma depth
            (which will be degraded). If not provided, then it defaults
            to baseline2018a file. This is used to simulate Chilean follow-up.

        follow_up_file_2: *str*, optional
            Name of a 2-column file with MJD and 5 sigma depth
            (which will be degraded). If not provided, then it defaults
            to baseline2018a file (different from the one above).
            This is used to simulate non-Chilean follow-up.

    Attributes :
    """

    def __init__(self, opsim_data, parameters, source_flux, blending_flux,
                 coords=None, follow_up_file_1=None, follow_up_file_2=None):
        if not isinstance(opsim_data, list):
            raise TypeError('opsim_data has to be a list')
        if len(opsim_data) != 3:
            raise ValueError(
                'wrong length of opsim_data {:}'.format(len(opsim_data)))
        for value in opsim_data:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    'wrong type of argument: {:}\n{:}'.format(
                    type(value), value))
        self._JD = opsim_data[0]
        self._5_sigma_detection = opsim_data[1]
        self._filter = opsim_data[2]

        if not isinstance(parameters, dict):
            raise TypeError("parameters must be of dict type")

        self.source_flux = source_flux
        self.blending_flux = blending_flux

        if coords is None:
            coords = "00:52:44.8 -72:49:43"
        self._parameters = parameters
        self._model = MM.Model(self._parameters, coords=coords)

        #self.bands = ['u', 'g', 'r', 'i', 'z', 'y'] # XXX - CHECK THAT
        self.bands = ['r', 'g', 'i', 'z', 'y', 'u']
        self._simulated_flux = {b: None for b in self.bands}
        self._sigma_flux = {b: None for b in self.bands}
        self._binary_chi2_sum = 0.
        self._LSST_PSPL_chi2 = None

        self.detection_time = None
        self.detection_band = None
        self._detected = None
        self._follow_up = None
        self._event_PSPL = None
        self._follow_up_Chilean = None
        self._follow_up_nonChilean = None

        if follow_up_file_1 is not None:
            self._Chilean_follow_up_file = follow_up_file_1
        else:
            temp = os.path.abspath(__file__)
            for i in range(3):
                temp = os.path.dirname(temp)
            self._Chilean_follow_up_file = os.path.join(
                    temp, 'data', 'baseline2018a_followup_epochs_v1.dat')
        self._Chilean_follow_up_data = None
        # XXX there should be lazy-loading for this file and it also should be class variable.

        if follow_up_file_2 is not None:
            self._nonChilean_follow_up_file = follow_up_file_2
        else:
            temp = os.path.abspath(__file__)
            for i in range(3):
                temp = os.path.dirname(temp)
            self._nonChilean_follow_up_file = os.path.join(
                    temp, 'data', 'baseline2018a_followup_epochs_v2.dat')
        self._nonChilean_follow_up_data = None

        self._band_follow_up = 'i'
        self._d_5sigma = 0.53 # This degrades LSST accuracy. It should be 1.17
        # for a 2.5-m telescope - see Ivezic+ 0805.2366v5 footnote 16.
        # We assume a 4-m follow-up telescope.
        self._dt_shift = 0.5 # Follow-up starts half a day after detection.
        self._t_E_factor = 3. # How many t_E after t_0 we stop follow-up?

    def _LSST_uncertainties(self, mag, five_sigma_mag, band):
        """
        Calculate LSST photometric uncertainties. Uses
        Ivezic et al. 2008 (v5 https://arxiv.org/abs/0805.2366)
        """
        sigma_sys = 0.005
        if band == "u":
            gamma = 0.038
        else:
            gamma = 0.039

        x = 10**(0.4*(mag-five_sigma_mag))
        sigma = np.sqrt(sigma_sys**2 + (0.04-gamma)*x + gamma*x**2)
        return sigma

    def _calculate_magnification(self, times):
        """
        Calculate magnification for given times (np.ndarray).
        For binary lenses, we're using VBBL.
        XXX - FSPL in some cases? 
        """
        if self._model.n_lenses == 2:
            factor = 10.
            params = self._model.parameters
            t_1 = params.t_0 - factor * params.t_E
            t_2 = params.t_0 + factor * params.t_E
            self._model.set_magnification_methods([t_1, 'VBBL', t_2])
            self._model.set_default_magnification_method(
                'point_source_point_lens')

        magnification = self._model.magnification(times)
        return magnification

    def _simulate_flux(self, times, five_sigma_mag, band):
        """
        Simulate light curve in flux units for a single band.
        Light curve includes gaussioan noise. Input fluxes
        (source_flux, blend_flux) are in units used in MulensModel.

        Returns simulated light curve and uncertainties
        """
        magnification = self._calculate_magnification(times)
        
        source_flux = self.source_flux[band]
        blending_flux = self.blending_flux[band]
        model_flux = source_flux * magnification + blending_flux
        model_mag = MM.Utils.get_mag_from_flux(model_flux)
        sigma_mag = self._LSST_uncertainties(model_mag, five_sigma_mag, band)
        temp = MM.Utils.get_flux_and_err_from_mag(model_mag, sigma_mag)
        sigma_flux = temp[1]
        
        simulated = model_flux + np.random.normal(scale=sigma_flux) # XXX negative flux
        
        if self._model.n_lenses == 2:
            diff = (model_flux - simulated) / sigma_flux
            self._binary_chi2_sum += np.sum(diff**2)

        return (simulated, sigma_flux)

    def _get_detection_date_band(self, times, flux):
        """
        Find detection date using simple algorithm: 3 consecutive points are
        >3 sigma above the mean of previous points. Works on single band data.
        Returns None if event is not detected.
        """
        n_consecutive = 3 # Number of consecutive points required.
        n_sigma_limit = 3. # Significance of the points
        d_time = 10. # Delay in days.
        n_min = 10 # Minimum number of points to calculate mean and sigma.
        
        n_max = len(times) - n_consecutive
        
        for n in range(n_min, n_max):
            mask = (times < times[n] - d_time) # XXX This can be improved.
            if np.sum(mask) < n_min:
                continue
            earlier = flux[mask]
            mean = np.mean(earlier)
            sigma = np.std(earlier)
            significance = (flux[n:n+n_consecutive] - mean) / sigma
            if np.all(significance > n_sigma_limit):
                return times[n+n_consecutive-1]
        return None

    def find_detection_time(self):
        """
        Find detection date based on each band.
        Returns None if event is not detected at all.
        """
        self._detected = False

        for band in self.bands:
            mask = (self._filter == band)
            if not np.any(mask):
                continue
            times = self._JD[mask]
            five_sigma_mag = self._5_sigma_detection[mask]
            if self._simulated_flux[band] is None:
                sim = self._simulate_flux(times, five_sigma_mag, band)
                self._simulated_flux[band] = sim[0]
                self._sigma_flux[band] = sim[1]

            date = self._get_detection_date_band(
                    times, self._simulated_flux[band])
            if date is None:
                continue
            if (self.detection_time is None) or (date < self.detection_time):
                self.detection_time = date
                self.detection_band = band
                self._detected = True

    def _set_event_PSPL(self):
        """
        Sets internal variable to MulensModel.Event instance
        that uses PSPL model.
        """
        datasets = []
        for band in self.bands:
            mask = (self._filter == band)
            if not np.any(mask):
                continue
            times = self._JD[mask]
            flux = self._simulated_flux[band]
            sigma_flux = self._sigma_flux[band]
            plot = {}
            data = MM.MulensData([times, flux, sigma_flux], phot_fmt='flux',
                bandpass=band, plot_properties={'label': 'LSST '+band})
            datasets.append(data)

        if self._follow_up_Chilean is not None:
            datasets.append(self._follow_up_Chilean)

        if self._follow_up_nonChilean is not None:
            datasets.append(self._follow_up_nonChilean)

        model = MM.Model({p: self._parameters[p] for p in ['t_0', 'u_0', 't_E']})
        self._event_PSPL = MM.Event(datasets, model)

    def _fit_point_lens(self):
        """
        fits point lens model to simulated data
        """

        def chi2_fun(theta, event, parameters_to_fit):
            """
            for a given event set attributes from parameters_to_fit
            (list of str) to values from theta list
            """
            for (key, val) in enumerate(parameters_to_fit):
                setattr(event.model.parameters, val, theta[key])
            chi2 = event.get_chi2()
            if chi2 < chi2_fun.best_chi2:
                chi2_fun.best_chi2 = chi2
            return chi2
        chi2_fun.best_chi2 = 1.e10

        def jacobian(theta, event, parameters_to_fit):
            """
            Calculate chi^2 gradient (also called Jacobian).
            """
            for (key, val) in enumerate(parameters_to_fit):
                setattr(event.model.parameters, val, theta[key])
            return event.chi2_gradient(parameters_to_fit)

        if self._event_PSPL is None:
            self._set_event_PSPL()

        parameters_to_fit = ["t_0", "u_0", "t_E"]
        initial_guess = [self._parameters[p] for p in parameters_to_fit]

        failed = False
        try:
            result = op.minimize(
                chi2_fun, x0=initial_guess,
                args=(self._event_PSPL, parameters_to_fit),
                method='Newton-CG', jac=jacobian, tol=3.e-4)
        except:
            failed = True

        if failed:
            try:
                result = op.minimize(
                    chi2_fun, x0=initial_guess,
                    args=(self._event_PSPL, parameters_to_fit),
                    method='Newton-CG', jac=jacobian, tol=3.e-4)
            except:
                pass
# XXX what if fit failed (i.e., .success is False)?

        self._LSST_PSPL_chi2 = chi2_fun.best_chi2

    def add_follow_up(self):
        """
        Simulate follow-up observations.
        """
        if self._detected is not True:
            return

        self._add_follow_up_Chilean()
        self._add_follow_up_nonChilean()

    def _add_follow_up_Chilean(self):
        """
        Add follow-up from Chilean observatories
        """
        if self._Chilean_follow_up_data is None:
            temp = np.loadtxt(self._Chilean_follow_up_file, unpack=True)
            self._Chilean_follow_up_data = {
                'jd': temp[0],
                '5sigma_depth': temp[1]}

        start = self.detection_time + self._dt_shift
        stop = self._parameters['t_0'] + self._t_E_factor * self._parameters['t_E']
        mask = (self._Chilean_follow_up_data['jd'] > start)
        mask *= (self._Chilean_follow_up_data['jd'] < stop)
        times = self._Chilean_follow_up_data['jd'][mask]

        mag_5sig = self._Chilean_follow_up_data['5sigma_depth'][mask]
        mag_5sig -= self._d_5sigma
        sim = self._simulate_flux(times, mag_5sig, self._band_follow_up)

        self._follow_up_Chilean = MM.MulensData([times, sim[0], sim[1]],
                phot_fmt='flux', bandpass=self._band_follow_up,
                plot_properties={"label": "follow-up i", "zorder": -1000})

    def _add_follow_up_nonChilean(self):
        """
        Add follow-up from observatories that are outside Chile.
        """
        if self._nonChilean_follow_up_data is None:
            temp = np.loadtxt(self._nonChilean_follow_up_file, unpack=True)
            self._nonChilean_follow_up_data = {
                'jd': temp[0],
                '5sigma_depth': temp[1]}

        start = self.detection_time + self._dt_shift
        stop = self._parameters['t_0'] + self._t_E_factor * self._parameters['t_E']
        mask = (self._nonChilean_follow_up_data['jd'] > start)
        mask *= (self._nonChilean_follow_up_data['jd'] < stop)
        times = self._nonChilean_follow_up_data['jd'][mask]

        mag_5sig = self._nonChilean_follow_up_data['5sigma_depth'][mask]
        mag_5sig -= self._d_5sigma
        sim = self._simulate_flux(times, mag_5sig, self._band_follow_up)

        self._follow_up_nonChilean = MM.MulensData([times, sim[0], sim[1]],
                phot_fmt='flux', bandpass=self._band_follow_up,
                plot_properties={"label": "follow-up i", "zorder": -500})

    @property
    def delta_chi2_BL_PL(self):
        """
        Delta chi2 between point source and binary lens models.

        Returns None for single lens models.
        """
        if self._model.n_lenses == 1:
            return None

        if self._LSST_PSPL_chi2 is None:
            self._fit_point_lens()

        return self._LSST_PSPL_chi2 - self._binary_chi2_sum

    def plot_data(self, kwargs_data=None, kwargs_model=None):
        """
        Plot simulated data. Use plt.savefig() or plt.show() afterwards.
        """
        if kwargs_data is None:
            kwargs_data = {}
        if kwargs_model is None:
            kwargs_model = {}

        self._event_PSPL.plot_data(**kwargs_data)

        self._model.set_datasets(self._event_PSPL.datasets)
        for band in self.bands:
            self._model.set_limb_coeff_u(band, 0.)
        self._model.plot_lc(**kwargs_model)
        plt.legend()
