from mwa_qa import read_metafits as rm
from astropy.io import fits
import numpy as np

# speed of light 
c = 299_792_458

class UVfits(object):
	def __init__(self, uvfits, metafits=None, pol='X'):
		self.uvfits = uvfits
		self.Metafits = rm.Metafits(metafits = metafits, pol = pol)
		self._dgroup = self._read_dgroup()

	def _read_dgroup(self):
		return fits.open(self.uvfits)[0].data

	def _header(self):
		return fits.open(self.uvfits)[0].header

	def group_count(self):
		hdr = self._header()
		return hdr['GCOUNT']

	def baselines(self):
		gcount = self.group_count()
		baselines = [self._dgroup[i][3] for i in range(gcount)]
		return np.array(baselines)

	def _encode_baseline(self, ant1, ant2):
		if ant2 > 255:
			return ant1 * 2048 + ant2 + 65_536
		else:
			return ant1 * 256 + ant2

	def _decode_baseline(self, bl):
		if bl < 65_535:
			ant2 = bl % 256
			ant1 = (bl - ant2) / 256
		else:
			ant2 = (bl - 65_536) % 2048
			ant1 = (bl - ant2 - 65_536) / 2048
		return (int(ant1), int(ant2))

	def antpairs(self):
		baselines = self.baselines()
		antpairs = [self._decode_baseline(bl) for bl in baselines]
		return antpairs

	def uvw(self):
		gcount = self.group_count()
		uvw = np.zeros((3, gcount))
		for i in range(gcount):
			uvw[0, i] = self._dgroup[i][0] * c
			uvw[1, i] = self._dgroup[i][1] * c
			uvw[2, i] = self._dgroup[i][2] * c
		return uvw

	def data(self):
		pass
