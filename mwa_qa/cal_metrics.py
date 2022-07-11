from lib2to3.pgen2.token import AMPER
from sys import api_version
from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

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

	def variance_for_tilepair(self, tile_pair, norm = True):
		"""
		Returns variance across frequency for the given tile pair
		- tile_pair : Tile pair or tuple of tile numbers e.g (102, 103)
		- norm : boolean, If True returns normalized gains else unormalized gains.
				 Default is set to True.
		"""
		gains = self.Csoln.gains_for_tilepair(tile_pair, norm = norm)
		return np.nanvar(gains, axis = 2)

	def variance_for_baselines_less_than(self, uv_cut, norm=True):
		"""
		Returns bls shorter than the specified cut and the variances calculated across frequency for
		each of the antenna pair
		- baseline_cut : Baseline cut in metres, will use only baselines shorter than the given value
		- norm : boolean, If True returns normalized gains else unormalized gains.
				 Default is set to True.
		"""
		baseline_dict = self.Metafits.get_baselines_less_than(uv_cut)
		bls = list(baseline_dict.keys())
		_sh = self.Csoln.gains().shape
		variances = np.zeros((_sh[0], len(bls), _sh[3]))
		for i , bl in enumerate(bls):
			variances[:, i, :] = self.variance_for_tilepair(bl, norm = norm)[:, :, :]
		return bls, variances

	def skewness_across_uvcut(self, uv_cut, norm=True):
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
			receivers = list(np.arange(1, n + 1))
		else:
			receivers = self.Metafits.receivers()
		return receivers

	def _initialize_metrics_dict(self):
		"""
		Initializes the metric dictionary with some of the default parameters
		"""
		self.metrics = OrderedDict()
		_, freqs, _ = self.Csoln.freqs_info()
		_, tile_ids, tile_flags = self.Csoln.tile_info()
		tile_numbers = [int(tid.strip('Tile')) for tid in tile_ids]
		hdr = self.Csoln.header(0)
		pols = ['XX', 'XY', 'YX', 'YY']
		receivers = self.get_receivers()
		self.metrics['pols'] = pols
		self.metrics['uvcut'] = hdr['UVW_MIN']
		self.metrics['niter'] = hdr['MAXITER']
		for i, p in enumerate(pols):
			self.metrics[p] = OrderedDict()
			for j, tn in enumerate(tile_numbers):
				self.metrics[p][tn] = OrderedDict()
			for r in receivers:
				self.metrics[p]['R{}'.format(r)] = OrderedDict()
				self.metrics[p]['R{}'.format(r)]['Tiles'] = [int(tid.strip('Tile')) for tid in self.Metafits.tiles_for_receiver(r)]
	
	def run_metrics(self):
		self._initialize_metrics_dict()
		pols = self.metrics['pols']
		_, tile_ids, tile_flags = self.Csoln.tile_info()
		tile_numbers = [int(tid.strip('Tile')) for tid in tile_ids]
		receivers = self.get_receivers()
		for i, p in enumerate(pols):
			for j, tn in enumerate(tile_numbers):
				gain_amplitudes = self.Csoln.amplitudes()[:, j, :, i]
				self.metrics[p][tn]['mean_amp_freq'] = np.nanmean(gain_amplitudes, axis = 1).tolist()  
				self.metrics[p][tn]['median_amp_freq'] = np.nanmedian(gain_amplitudes, axis = 1).tolist()
				self.metrics[p][tn]['var_amp_freq'] = np.nanvar(gain_amplitudes, axis = 1).tolist() 
				self.metrics[p][tn]['rms_amp_freq'] = np.sqrt(np.nanmean(gain_amplitudes ** 2, axis = 1)).tolist()
			# skewness of the variance across frequency avergaed over short baselines 
			#self.metrics[p]['var_skewness_uvcut'] = self.skewness_across_uvcut(self.metrics['uvcut'])
		for r in receivers:
				gains_rcv = self.Csoln.gains_for_receiver(r)
				gains_rcv_amplitudes = np.abs(gains_rcv[:, :, :, i])
				chi_sq = np.nansum(gains_rcv_amplitudes - np.nanmean(gains_rcv_amplitudes, axis = 1)) ** 2 / np.nanmean(gains_rcv_amplitudes, axis = 0)
				self.metrics[p]['R{}'.format(r)]['mean_chi_sq'] = np.nanmean(chi_sq, axis = 1).tolist()
				self.metrics[p]['R{}'.format(r)]['var_chi_sq'] = np.nanvar(chi_sq, axis = 1).tolist()
	
	def write_to(self, outfile=None):
		if outfile is None:
			outfile = self.calfile.replace('.fits', '_cal_metrics.json')
		ju.write_metrics(self.metrics, outfile)
