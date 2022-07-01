from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
import numpy as np

class Metric(object):
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

	def variance(self, norm=True):
		"""
		Returns he variance of amplitude of calibration solutions across frequency
		- norm :
		"""
		gains = self.Csoln.amplitudes(norm = norm)
		return np.nanvar(gains, axis = 2)

	def variance_for_tilepair(self, tile_pair, norm = True):
		"""
		Returns variance across frequency for the given tile pair
		 - tile_pair : Tile pair or tuple of tile numbers e.g (102, 103)
	     - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		gains = self.Csoln.gains_for_tilepair(tile_pair, norm = norm)
		return np.nanvar(gains, axis = 2)

	def variance_across_tiles(self, norm=True):
		"""
		Returns varaince of amplitude of the gain solutions across all tiles
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		amps = self.Csoln.amplitudes()
		return np.nanvar(amps, axis = 1)

	def variance_across_baselines(self, baseline_cut, norm=True):
		"""
		Returns variance of amplitude of the gain solutions across all the baselines less than
		the given cut
		- baseline_cut : Baseling length in metres which should be excluded in the computation
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		baseline_dict = self.Metafits.get_baselines_less_than(baseline_cut)
		tile_pairs = list(baseline_dict.keys())
		_sh = self.Csoln.gains().shape
		ggains = np.zeros((_sh[0], len(tile_pairs), _sh[2], _sh[3]))
		for i, tl in enumerate(tile_pairs):
			ggains[:, i, :, :] = self.Csoln.gains_for_tilepair(tl, norm = norm)
		return np.nanvar(np.abs(ggains), axis = 1)

	def variance_for_baselines_less_than(self, baseline_cut, norm=True):
		"""
		Returns bls shorter than the specified cut and the variances calculated across frequency for
		each of the antenna pair
		- baseline_cut : Baseline cut in metres, will use only baselines shorter than the given value
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		baseline_dict = self.Metafits.get_baselines_less_than(baseline_cut)
		bls = list(baseline_dict.keys())
		_sh = self.Csoln.gains().shape
		variances = np.zeros((_sh[0], len(bls), _sh[3]))
		for i , bl in enumerate(bls):
			variances[:, i, :] = self.variance_for_tilepair(bl, norm = norm)[:, :, :]
		return bls, variances

	def skewness_across_baselines(self, baseline_cut, norm=True):
		"""
		Evaluates the Pearson skewness 3 * (mean - median) / std across the variances 
		averaged over baseliness shorter than the given length
		- baseline_cut : Baseline cut in metres, will use only baselines shorter than the given value
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		_, variances = self.variance_for_baselines_less_than(baseline_cut, norm = norm)
		skewness = (3 * ( np.nanmean(variances, axis = 1) - np.nanmedian(variances, axis = 1))) / np.nanstd(variances, axis=1)
		return skewness
	
	def gains_for_receiver(self, receiver, norm=True):
		"""
		Returns the tile IDs associated with the receiver and their corresponding gains solutions
		as a 2D numpy array
		- receiver : Receiver Number, (1 - 16)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		gains_dict = self.Csoln.gains_for_receiver(receiver, norm = norm)
		tile_ids = np.array(list(gains_dict.keys()))
		# gains from the from the first key
		gains00 = gains_dict[tile_ids[0]]
		_sh = gains00.shape
		gains = np.zeros((_sh[0], len(tile_ids), _sh[2], _sh[3]), dtype=complex)
		for i, tid in enumerate(tile_ids):
			gains[:, i, :, :] = gains_dict[tid]
		return tile_ids, gains

	def mean_for_receiver(self, receiver, norm=True):
		"""
		Evaluates mean across the amplitude of the gain solutions for tiles connected to the given receiver
		Returns the tile IDs connected with the receiver and the evaluated mean across those tiles
		- receiver : Receiver Number, (1 - 16)
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		tile_ids, gains = self.gains_for_receiver(receiver, norm = norm)
		return tile_ids, np.nanmean(np.abs(gains), axis = 1)

	def median_for_receiver(self, receiver, norm=True):
		"""
        Evaluates median across amplitude of the gain solutions for tiles connected to the given receiver
		Returns the tile IDs connected with the receiver and the evaluated median across those tiles
        - receiver : Receiver Number, (1 - 16)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
		tile_ids, gains = self.gains_for_receiver(receiver, norm = norm)
		return tile_ids, np.nanmedian(np.abs(gains), axis = 1)

	def rms_for_receiver(self, receiver, norm=True):
		"""
        Evaluates the root mean square (rms) across amplitude of the gain solutions for tiles connected to the given receiver
		Returns the tile IDs connected with the receiver and the evaluated rms across those tiles
        - receiver : Receiver Number, (1 - 16)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
		tile_ids, gains = self.gains_for_receiver(receiver, norm = norm)
		return tile_ids, np.sqrt(np.nanmean(np.abs(gains) ** 2, axis = 1))

	def var_for_receiver(self, receiver, norm=True):
		"""
        Evaluates median across amplitude of the gain solutions for tiles connected to the given receiver
        Returns the tile IDs connected with the receiver and the evaluated variance across those tiles
		- receiver : Receiver Number, (1 - 16)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
		tile_ids, gains = self.gains_for_receiver(receiver, norm = norm)
		return tile_ids, np.nanvar(np.abs(gains), axis = 1)
