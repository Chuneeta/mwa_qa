from astropy.io import fits
import numpy as np

class Csoln(object):
	def __init__(self, calfile, pol):
		"""
		Object takes in a calfile in fits format and extracts bit and pieces of the required informations
 		- calfile: Fits file readable by astropy containing calibration solutions (support for hyperdrive
				   output only for now) and related information
		- pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
                with the given pol is provided.
		"""
		self.calfile = calfile
		self.pol = pol.upper()

	def _read_gains(self):
		"""
		Reads the fits file and returns the data/ gains solutions.
		Returns a 4D array (time, tiles, frqe, pol)
		"""

	def _combine_real_imag(self):
		gains = self._read_gains()
		gains = gains[:. :, :, ::2] + gains[:, :, :, ::2] * 1j
