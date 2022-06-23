from mwa_clysis import read_metafits as rm
from mwa_clysis import read_csolutions as rc
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

	def gains_for_receiver(self, receiver, norm):
		"""
		Returns the tile IDs associated with the receiver and their corresponding gains solutions
		as a 2D numpy array
		- receiver : Receiver Number, (1 - 16)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
		"""
		gains_dict = self.Csoln.gains_for_receiver(receiver, norm = norm)
		tile_ids = np.array(list(gains_dict.keys()))
		# gains from the from the fira=st key
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


