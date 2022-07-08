from mwa_qa import read_metafits as rm
from collections import OrderedDict
from scipy import signal
from astropy.io import fits
import numpy as np
import copy

class Csoln(object):
	def __init__(self, calfile, metafits=None, pol='X'):
		"""
		Object takes in a calfile in fits format and extracts bit and pieces of the required informations
 		- calfile : Fits file readable by astropy containing calibration solutions (support for hyperdrive
				   output only for now) and related information
		- metafits : Metafits with extension *.metafits containing information corresponding to the observation
					 for which the calibration solutions is derived
		- pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
                with the given pol is provided. Default is 'X'
		"""
		self.calfile = calfile
		self.Metafits = rm.Metafits(metafits, pol)

	def data(self, hdu):
		"""
		Returns the data stored in the specified HDU column of the image
		hdu : hdu column, ranges from 1 to 6
			  1 - the calibration solution
			  2 - the start time, end time and average time
			  3 - tiles information (antenna, tilename, flag)
			  4 - chanblocks (index, freq, flag)
			  5 - calibration results (timeblock, chan, convergence)
			  6 - weights used for each baseline
			  for more details refer to https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
		"""
		return fits.open(self.calfile)[hdu].data

	def header(self, hdu):
		"""
		Returns the header of the specified HDU column
		hdu : hdu column, ranges from 0 to 6
			  0 - header information on the paramters used for the calibration process
              1 - header information on the calibration solutions
			  2 - header information on the timeblocks
              3 - header information on the tiles (antenna, tilename, flag)
              4 - header information on the chanblocks (index, freq, flag)
              5 - header information on the calibration results (timeblock, chan, convergence)
              6 - header information on the weights used for each baseline
              for more details refer to https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
        """
		return fits.open(self.calfile)[hdu].header

	def gains_real(self):
		"""
		Returns the real part of the calibration solutions
		"""
		cal_solutions = self.data(1)
		return cal_solutions[:, :, :, ::2]

	def gains_imag(self):
		"""
		Returns the imaginary part of the calibration solutions
		"""
		cal_solutions = self.data(1)
		return cal_solutions[:, :, :, 1::2]
	
	def gains(self):
		"""
        Combines the real and imaginary parts to form the 4 polarization (xx, xy, yx and yy)
        """
		return self.gains_real() + self.gains_imag() * 1j

	def gains_shape(self):
		"""
		Returns shape of the array containing the gain soultions
		"""
		return self.gains().shape

	def tile_info(self):
		"""
		Returns the info on the tiles index, tile ID and flags
		"""
		tiles_info = self.data(3)
		tile_inds = [tl[0] for tl in tiles_info]
		tile_ids = [tl[1] for tl in tiles_info]
		tile_flags = [tl[2] for tl in tiles_info]
		return tile_inds, tile_ids, tile_flags

	def freqs_info(self):
		"""
		Returns the frequency index, frequency array and frequency flags 
		"""
		freqs_info = self.data(4)
		freq_inds = [fq[0] for fq in freqs_info]
		freqs = [fq[1] for fq in freqs_info]
		freq_flags = [fq[2] for fq in freqs_info]
		return freq_inds, freqs, freq_flags

	def gains_ind_for(self, tile_id):
		"""
		Returns index of the gain solutions fot the given tile ID
		- tile_id : Tile ID e.g Tile 103
		"""
		tile_inds, tile_ids, _ = self.tile_info()
		ind = np.where(np.array(tile_ids) == tile_id)
		return np.array(tile_inds)[ind[0]]		

	def _check_ref_tile(self, tile_id):
		"""
		Checks if the given reference antenna is flagged due to non-convergence or any 
		malfunctioning reports
		- tile_ind : Index of the reference tile
		"""
		tile_inds, tile_ids, tile_flags = self.tile_info()
		ind = self.gains_ind_for(tile_id)
		flag = np.array(tile_flags)[ind]
		assert flag == 0,  "{} seems to be flagged, therefore does not have calibration solutions, choose a different tile"	

	def _normalized_data(self, data, ref_tile_id=None):
		"""
		Normalizes the gain solutions for each timeblock given a reference tile
		- data : the input array of shape( tiles, freq, pols) containing the solutions
		- ref_tile_id: Tile ID of the reference tile e.g Tile 103. Default is set to the last antenna of the telescope.
						For example for MWA128T, the reference antennat is Tile 168
		"""
		if ref_tile_id is None:
			_, tile_ids, _ = self.tile_info()
			ref_ind = -1
			ref_tile_id = tile_ids[ref_ind]
		else:
			ref_ind = self.gains_ind_for(ref_tile_id)
		self._check_ref_tile(ref_tile_id)
		refs = []
		for ref in data[ref_ind].reshape((-1, 2, 2)):
			refs.append(np.linalg.inv(ref))
		refs = np.array(refs)
		div_ref = []
		for tile_i in data:
			for (i, ref) in zip(tile_i, refs):
				div_ref.append(i.reshape((2, 2)).dot(ref))
		return np.array(div_ref).reshape(data.shape)

	def normalized_gains(self, ref_tile_id=None):
		"""
		Returns the normalized gain solutions using the given reference antenna
		- ref_tile_id: Tile ID of the reference tile e.g Tile 103. Default is set to the last antenna of the telescope.
                       For example for MWA128T, the reference antennat is Tile 168
		"""
		gains = self.gains()
		ngains = copy.deepcopy(gains)
		for t in range(len(ngains)):
			ngains[t] = self._normalized_data(gains[t], ref_tile_id)
		return ngains

	def _select_gains(self, norm):
		"""
		Return normalized if norm is True else unnomalized gains 
		- norm : boolean, If True returns normalized gains else unormalized gains.
		"""
		if norm:
			return self.normalized_gains()
		else:
			return self.gains()

	def amplitudes(self, norm=True):
		"""
		Returns amplitude of the normalized gain solutions
		- norm : boolean, If True returns normalized gains else unormalized gains.
        		 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		return np.abs(gains)

	def phases(self, norm=True):
		"""
		Returns phases in degrees of the normalized gain solutions
		- norm : boolean, If True returns normalized gains else unormalized gains.
				 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		return np.angle(gains) * 180 / np.pi

	def gains_for_tile(self, tile_id, norm=True):
		"""
		Returns gain solutions for the given tile ID
		- tile_id : Tile ID e.g Tile103
		- norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		gains = self._select_gains(norm = norm)
		ind = self.gains_ind_for(tile_id)
		return gains[:, ind, :, :] 

	def gains_for_tilepair(self, tilepair, norm=True):
		"""
		Evaluates conjugation of the gain solutions for antenna pair (tile0, tile1)
		- tile_pair : tuple of tile numbers such as (11, 13)
		"""
		tile0, tile1 = 'Tile{:03g}'.format(tilepair[0]), 'Tile{:03g}'.format(tilepair[1])
		gains_t0 = self.gains_for_tile(tile0, norm = norm)
		gains_t1 = self.gains_for_tile(tile1, norm = norm)
		return gains_t0 * np.conj(gains_t1)
	
	def gains_for_receiver(self, receiver, norm=True):
		"""
		Returns the dictionary of gains solutions for all the tiles (8 tiles) connected to the given reciver
		"""
		assert not self.Metafits.metafits is None, "metafits file associated with this observation is required to extract the receiver information"
		tile_ids = self.Metafits.tiles_for_receiver(receiver)
		gains_receiver = OrderedDict()
		for tile_id in tile_ids:
			gains_receiver[tile_id] = self.gains_for_tile(tile_id, norm = norm)
		return gains_receiver

	def blackmanharris(self, n):
		return signal.windows.blackmanharris(n)
		
	def delays(self):
    	#Evaluates geometric delay (fourier conjugate of frequency)
		_, freqs, _ = self.freqs_info()
		freqs = np.array(freqs) * 1e-9
		df = freqs[1] - freqs[0]
		delays = np.fft.fftfreq(freqs.size, df)
		return delays

	def _filter_nans(self, data):
		nonans_inds = np.where(~np.isnan(data))[0]
		nans_inds = np.where(np.isnan(data))[0]
		return nonans_inds, nans_inds

	def gains_fft(self):
		gains = self.gains()
		fft_data = np.zeros(gains.shape, dtype=gains.dtype)
		_sh = gains.shape
		_, freqs, _ = self.freqs_info()
		window = self.blackmanharris(len(freqs))
		for t in range(_sh[0]):
			for i in range(_sh[1]):
				for j in range(_sh[3]):
					try:
						nonans_inds, nans_inds = self._filter_nans(gains[t, i, :, j])
						d_fft = np.fft.fft(gains[t, i, nonans_inds, j] * window[nonans_inds])
						fft_data[t, i, nonans_inds, j] = d_fft
						fft_data[t, i, nans_inds, j] = np.nan
					except ValueError:
						fft_data[t, i, :, j] = np.nan
		return fft_data
	
