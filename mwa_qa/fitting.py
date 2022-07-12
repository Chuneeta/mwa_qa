from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
import numpy as np

class Fit(object):
	def __init__(self, calfile, metafits=None, pol='X'):
		"""
		Object that takes in .fits containing the calibration solutions file readable by astropy
		and initializes them as global varaibles
		- calfile : .fits file containing the calibration solutions
		- metafits : Metafits with extension *.metafits or _ppds.fits containing information 
                     on an observation done with MWA
		- pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
                with the given pol is provided.
		"""
		self.calfile = calfile
		self.Csoln = rc.Csoln(calfile, metafits = metafits, pol = pol)
		self.Metafits = rm.Metafits(metafits, pol = pol)

	def _get_gains_for_receiver(self, receiver, norm=True):
		"""
		Returns the tile ID s connected to the given receivers and a 4D numpy array
		containing the gain solutions
		- receiver : Receiver ID, ranges from 1 to 16
		"""
		gains = self.Csoln.gains_for_receiver(receiver, norm = norm)
		tile_ids = list(gains.keys())
		_sh = gains[tile_ids[0]].shape
		dtype = gains[tile_ids[0]].dtype
		gains_array = np.zeros((_sh[0], len(tile_ids), _sh[2], _sh[3]), dtype=dtype)
		for i in range(len(tile_ids)):
			gains_array[:, i, :, :] = gains[tile_ids[i]]
		return tile_ids, gains_array

	def average_per_receiver(self, receiver, norm=True):
		"""
		Computes a simple/unweighted average of the gains across tiles connected to the given
		receiver. 
		Returns the tile IDs connectd to the receiver and the average numpy array.
		- receiver : Receriver ID, ranges from 1 to 16
		"""
		tile_ids, gains_array = self._get_gains_for_receiver(receiver, norm = norm)
		return tile_ids, np.nanmean(np.abs(gains_array), axis = 1)	
		
	def polynomial_fit(self):
		pass
