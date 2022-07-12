from mwa_qa import read_metafits as rm
from astropy.io import fits
import numpy as np

# speed of light 
c = 299_792_458
pol_dict = {'XX' : 0, 'YY' : 1, 'XY' : 2, 'YX' : 3}

class UVfits(object):
	def __init__(self, uvfits, metafits=None, pol='X'):
		self.uvfits = uvfits
		self.Metafits = rm.Metafits(metafits = metafits, pol = pol)
		self._dgroup = self._read_dgroup()
		self.Ntiles = len(self._tile_info())
		bls = self.baselines()
		self.Nbls = len(np.unique(np.array(bls)))
		self.Ntimes = int(len(bls) / self.Nbls)
		self.Nfreqs = self._dgroup[0][5].shape[2]
		self.Npols = self._dgroup[0][5].shape[3]

	def _read_dgroup(self):
		return fits.open(self.uvfits)[0].data

	def _header(self):
		return fits.open(self.uvfits)[0].header

	def _tile_info(self):
		return fits.open(self.uvfits)[1].data

	def tile_ids(self):
		tile_info = self._tile_info()
		return [tile_info[i][0] for i in range(self.Ntiles)]
			
	def tile_numbers(self):
		tile_ids = self.tile_ids()
		return [int(tl.strip('Tile')) for tl in tile_ids]

	def tile_labels(self):
		tile_info = self._tile_info()
		return [tile_info[i][2] for i in range(self.Ntiles)]

	def group_count(self):
		hdr = self._header()
		return hdr['GCOUNT']

	def baselines(self):
		gcount = self.group_count()
		baselines = [self._dgroup[i][3] for i in range(gcount)]
		return baselines

	def _encode_baseline(self, tile1_label, tile2_label):
		if tile2_label > 255:
			return tile1_label * 2048 + tile2_label + 65_536
		else:
			return tile1_label * 256 + tile2_label

	def _decode_baseline(self, bl):
		if bl < 65_535:
			ant2_label = bl % 256
			ant1_label = (bl - ant2_label) / 256
		else:
			ant2_label = (bl - 65_536) % 2048
			ant1_label = (bl - ant2_label - 65_536) / 2048
		return (int(ant1_label), int(ant2_label))

	def _label_to_tile(self, label):
		tile_numbers = np.array(self.tile_numbers())
		tile_labels = np.array(self.tile_labels())
		ind = np.where(tile_labels == label)[0][0]
		return tile_numbers[ind]

	def _tile_to_label(self, tile_number):
		tile_numbers = np.array(self.tile_numbers())
		tile_labels = np.array(self.tile_labels())
		ind = np.where(tile_numbers == tile_number)[0][0]
		return tile_labels[ind]

	def _indices_for_tilepair(self, tilepair):
		bls = np.array(self.baselines())
		bl = self._encode_baseline(self._tile_to_label(tilepair[0]), self._tile_to_label(tilepair[1]))
		return np.where(bls == bl)[0]

	def antpairs(self):
		baselines = self.baselines()
		tile_numbers = self.tile_numbers()
		tile_labels = self.tile_labels()
		tilepairs = []
		for bl in baselines:
			tile_labels = self._decode_baseline(bl)
			tilepairs.append((self._label_to_tile(tile_labels[0]), self._label_to_tile(tile_labels[1])))
		return tilepairs

	def uvw(self):
		gcount = self.group_count()
		uvw = np.zeros((3, gcount))
		for i in range(gcount):
			uvw[0, i] = self._dgroup[i][0] * c
			uvw[1, i] = self._dgroup[i][1] * c
			uvw[2, i] = self._dgroup[i][2] * c
		return uvw

	def pols(self):
		# Npols=4 --> ('XX', 'XY', 'YX', 'YY')
		# Npols=2  --> ('XX', 'YY')
		if self.Npols == 2:
			return ['XX', 'YY']
		if self.Npols == 4:
			return ['XX', 'XY', 'YX', 'YY']
		else:
			raise (ValueError, "currently support only 2 and 4 polarizations")

	def data_for_tilepair(self, tilepair):
		inds = self._indices_for_tilepair(tilepair)
		pols = self.pols()
		# data shape (times, freqs, pol)
		data = np.zeros((len(inds), self.Nfreqs, self.Npols), dtype=np.complex128)
		for i, ind in enumerate(inds):
			for j, p in enumerate(pols):
				data[i, :, j] = self._dgroup[i][5][0, 0, :, pol_dict[p], 0] + self._dgroup[i][5][0, 0, :, pol_dict[p], 1] * 1j 
		return data

	def data_for_tilepairpol(self, tilepairpol):
		data_tilepair = self.data_for_tilepair((tilepairpol[0], tilepairpol[1]))
		pols = np.array(self.pols())
		return data_tilepair[:, :, np.where(pols == tilepairpol[2])[0][0]]

		
