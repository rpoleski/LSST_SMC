import numpy as np


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

    def _read_isochrone(self, file_name):
        """
        Reads isochrone file
        """
        use_cols = (2,23,24,25,26,27)
        data = np.loadtxt(file_name, unpack=True, usecols=use_cols)
        out = {'M_initial': data[0]}
        for (i, band) in enumerate(['u', 'g', 'r', 'i', 'z']):
            out[band] = data[i]
        return out

