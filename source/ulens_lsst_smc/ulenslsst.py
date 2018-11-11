import numpy as np
import math

import MulensModel as MM

import utils


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

    Attributes :
    """

    def __init__(self, opsim_data, parameters, source_flux, blending_flux,
                 coords=None):
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
        self._model = MM.Model(parameters, coords=coords)

        self.bands = ['u', 'g', 'r', 'i', 'z', 'y'] # XXX - CHECK THAT
        self._simulated_flux = {b: None for b in self.bands}
        self._sigma_flux = {b: None for b in self.bands}
        self._binary_chi2 = {b: None for b in self.bands}

        self._detection_time = None
        self._detected = None
        self._follow_up = None

        file_name = "SMC_Chile_visibility_v1.dat"
        self._visibility = np.loadtxt(file_name, unpack=True, usecols=(0))

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
        sigma_flux = MM.Utils.get_flux_and_err_from_mag(model_mag, sigma_mag)[1]
        
        simulated = model_flux + np.random.normal(scale=sigma_flux)
        
        if self._model.n_lenses == 2:
            diff = (model_flux - simulated) / sigma_flux
            self._binary_chi2[band] = diff**2

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
            mask = (times < times[n] - d_time)
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
        dates = []
        for band in self.bands:
            mask = (self._filter == band)
            # XXX if not np.any(mask) - give warning?
            times = self._JD[mask]
            five_sigma_mag = self._5_sigma_detection[mask]
            if self._simulated_flux[band] is None:
                sim = self._simulate_flux(times, five_sigma_mag, band)
                self._simulated_flux[band] = sim[0]
                self._sigma_flux[band] = sim[1]
            #(simulated_flux, sigma_flux) = self._simulate_flux(
                    #times, five_sigma_mag, band)

            date = self._get_detection_date_band(times, self._simulated_flux[band])
            if date is not None:
                dates.append(date)

        if len(dates) == 0:
            self._detection_time = None
            self._detected = False
        else:
            self._detection_time = min(dates)
            self._detected = True

    def add_follow_up(self):
        """
        Simulate follow-up observations.
        """
        if self._detected is not True:
            return

        self._add_follow_up_Chile()
        #self._add_follow_up_outside_Chile() # XXX

    def _add_follow_up_Chile(self):
        """
        Add follow-up from Chilean observatories
        """
        requested_cadence = 1./24 # i.e. 1 hr
        minimum_dt = 0.5 * requested_cadence
        
        visible = self._visibility[self._visibility > self._detection_time]
        visible_night_begin = visible.astype(int) 
        # This gives begin of night, because day fractions are in range (0.49, 0.94).

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
            
            #print(*jds)
