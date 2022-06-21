from mwa_clysis import read_metafits as rm
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

	def _read_data(self):
		"""
		Reads the fits file and returns the data/ gains solutions.
		Returns a 4D array (time, tiles, freq, pol)
		"""
		return fits.open(self.calfile)[1].data

	def real(self):
		"""
		Returns the real part of the calibration solutions
		"""
		data = self._read_data()
		return data[:, :, :, ::2]

	def imag(self):
		"""
		Returns the imaginary part of the calibration solutions
		"""
		data = self._read_data()
		return data[:, :, :, 1::2]
	
	def gains(self):
		"""
        Combines the real and imaginary parts to form the 4 polarization (xx, xy, yx and yy)
        """
		return self.real() + self.imag() * 1j

	def header(self):
		"""
		Reads and return the header column hdu[0]
        """
		return fits.open(self.calfile)[0].header

	def _check_ref_tile_data(self, tile_ind):
		"""
		Checks if the given reference antenna is flagged due to non-convergence or any 
		malfunctioning reports
		- tile_ind : Index of the reference tile
		"""
		gains = self.gains()
		assert not np.isnan(gains[:, tile_ind, :, :]).all(), "The specified reference antenna seems to be flagged. Choose a different reference antenna"

	def _normalized_data(self, data, ref_tile_id=None):
		"""
		Normalizes the gain solutions for each timeblock given a reference tile
		- data : the input array of shape( tiles, freq, pols) containing the solutions
		- ref_tile_id: Tile ID of the reference tile e.g Tile 103. Default is set to the last antenna of the telescope.
						For example for MWA128T, the reference antennat is Tile 168
		"""
		if ref_tile_id is None:
			ref_tile_ind = -1
		else:
			ref_tile_ind = self.Metafits.get_tile_ind(ref_tile_id)[0]
		self._check_ref_tile_data(ref_tile_ind)
		refs = []
		for ref in data[ref_tile_ind].reshape((-1, 2, 2)):
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

	def amplitudes(self):
		"""
		Returns amplitude of the normalized gain solutions
		"""
		gains = self.normalized_gains()
		return np.abs(gains)

	def phases(self):
		"""
		Returns phases in degrees of the normalized gain solutions
		"""
		gains = self.normalized_gains()
		return np.angle(gains) * 180 / np.pi

	def 
	def gains_for_receiver(self, receiver):
		"""
		Returns the gains solutions for all the tiles (8 tiles) connected to the given reciver
		"""
		pass
		
