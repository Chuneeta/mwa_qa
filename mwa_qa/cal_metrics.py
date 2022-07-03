from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
from collections import OrderedDict
import numpy as np
import json

class CalMetrics(object):
	def __init__(self, calfile, metafits=None, pol='X'):
		"""
        Object that takes in .fits containing the calibration solutions file readable by astropy
        and initializes them as global varaibles
        - calfile : .fits file containing the calibration solutions
        - metafits : Metafits with extension *.metafits or _ppds.fits containing information
                     on an observation done with MWA
        - pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
        with the given pol is provided. Default is X.
        """
		self.calfile = calfile
		self.Csoln = rc.Csoln(calfile, metafits = metafits, pol = pol)
		self.Metafits = rm.Metafits(metafits, pol)

    def skewness_across_baselines(self, uv_cut, norm=True):
        """
        Evaluates the Pearson skewness 3 * (mean - median) / std across the variances 
        averaged over baseliness shorter than the given length
        - uv_cut : Baseline cut in metres, will use only baselines shorter than the given value
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
        _, variances = self.variance_for_baselines_less_than(uv_cut, norm = norm)
        skewness = (3 * ( np.nanmean(variances, axis = 1) - np.nanmedian(variances, axis = 1))) / np.nanstd(variances, axis=1)
        return skewness

	def get_receivers(self, n = 16):
		"""
		Returns the receivers connected to the various tiles in the array
		- n : Number of receivers in the array. Optional, enabled if metafits is not provided.
			  Default is 16.
		"""
		if self.Metafits.metafits is None:
			receivers = np.arange(1, n + 1)
		else:
			receivers = self.Metafits.receivers()
		return receivers

	def _initialize_metrics_dict(self):
		"""
		Initializes the metric dictionary with some of the default parameters
		"""
		metrics = OrderedDict()
		_, freqs, _ = self.Csoln.freqs_info()
		_, tile_ids, tile_flags = self.Csoln.tile_info()
		metrics['pols'] = ['XX', 'XY', 'YX', 'YY']
		metrics['freqs'] = freqs
		metrics['tile_ids'] = tile_ids
		metrics['uvcut'] = 40 # temporarily, will be adjusted
		return metrics 

	def run_metrics(self):
		metrics = OrderedDict()
		metrics['mean_freq'] = np.nanmean(self.Csoln.amplitudes(), axis = 2)
		metrics['median_freq'] = np.nanmedian(self.Csoln.amplitudes(), axis = 2)
		metrics['variance_freq'] = np.nanvar(self.Csoln.amplitudes(), axis = 2)
		metrics['rms_freq'] = np.sqrt(np.nanmean(np.abs(gains) ** 2, axis = 2))
		metrics['mean_time'] = np.nanmean(self.Csoln.amplitudes(), axis = 0)
		metrics['median_time'] = np.nanmedian(self.Csoln.amplitudes(), axis = 0)
		metrics['variance_time'] = np.nanvar(self.Csoln.amplitudes(), axis = 0)
		metrics['rms_time'] = np.sqrt(np.nanmean(np.abs(gains) ** 2, axis = 0))
		metrics['skewness_across_baselines'] = self.skewness_across_baselines(metrics['uvcut'])
		return metrics

	def write_metrics(self):
		pass
