from astropy.io import fits
from collections import OrderedDict
import numpy as np
import pylab

pol_dict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}
_pol_color = ['red', 'red', 'blue', 'blue']
_pol_alpha = [1, 0.2, 0.2, 1]

class Cal(object):
	def __init__(self, calfile=None, metafits=None):
		self.calfile = calfile
		self.metafits = metafits

	def read_data(self):
		hdu = fits.open(self.calfile)
		data = hdu[1].data
		# Only looking at the first timeblock.
		i_timeblock = 0
		data = data[i_timeblock, :, :, ::2] + data[i_timeblock, :, :, 1::2] * 1j
		return data

	def get_normalized_data(self):
		data = self.read_data()
		# the last antenna/tile is usually taken as refernce antenna
		i_tile_ref = -1
		refs = []
		for ref in data[i_tile_ref].reshape((-1, 2, 2)):
			refs.append(np.linalg.inv(ref))
		refs = np.array(refs)
		j_div_ref = []
		for tile_j in data:
			for (j, ref) in zip(tile_j, refs):
				j_div_ref.append(j.reshape((2, 2)).dot(ref))
		data = np.array(j_div_ref).reshape(data.shape)
		return data

	def read_metadata(self):
		hdu = fits.open(self.metafits)
		mdata = hdu[1].data
		return mdata

	def read_metaheader(self):
		hdu = fits.open(self.metafits)
		mhdr = hdu[0].header
		return mhdr

	def get_nchans(self):
		mhdr = self.read_metaheader()
		nchans = mhdr['NCHANS']
		return nchans

	def get_freqs(self):
		mhdr = self.read_metaheader()
		coarse_ch = mhdr['CHANNELS']
		nchans = self.get_nchans()
		start = float(coarse_ch.split(',')[0]) * 1.28
		stop = float(coarse_ch.split(',')[-1]) * 1.28
		freqs = np.linspace(start, stop, nchans) 
		return freqs		

	def get_obsdate(self):
		mhdr = self.read_metaheader()
		obsdate = mhdr['DATE-OBS']
		return obsdate

	def extract_tiles(self):
		mdata = self.read_metadata()
		tiles = np.array([])
		tiles = np.unique([np.append(tiles, mdata[i][3]) for i in range(len(mdata))])
		tiles_dict = {x:i for i, x in enumerate(tiles)}
		return tiles_dict

	def get_tile_numbers(self):
		tiles_dict = self.extract_tiles()
		tiles = list(tiles_dict)
		tnums = [int(tl.strip('Tile')) for tl in tiles]
		return tnums

	def get_tile_pos(self):
		mdata = self.read_metadata()
		tnums = self.get_tile_numbers()
		npos = len(tnums)
		pos = OrderedDict()
		for i in range(0, npos):
			pos[tnums[i]] = [mdata[i * 2][9], mdata[i * 2][10], mdata[i * 2][11]]
		return pos
	
	def get_baselines(self):
		tile_pos_dict = self.get_tile_pos()
		tile_pos = np.array(list(tile_pos_dict.values()))
		tnums = list(tile_pos_dict.keys())
		npos = len(tnums)
		bls = OrderedDict()
		for i in range(npos):
			for j in range(i+1, npos):
				bls[(tnums[i], tnums[j])] = np.sqrt((tile_pos[i, 0] - tile_pos[j, 0]) ** 2 + (tile_pos[i, 1] - tile_pos[j, 1]) ** 2)
		return bls 

	def get_bls_greater_than(self, cut_bl):
		bls_dict = self.get_baselines()
		bls = {key: value for key, value in bls_dict.items() if value > cut_bl}
		return bls

	def get_bls_less_than(self, cut_bl):
		bls_dict = self.get_baselines()
		bls = {key: value for key, value in bls_dict.items() if value < cut_bl}
		return bls

	def get_tile_pos_fromdict(self, bls_dict):
		bls = list(bls_dict.keys())
		tiles = [tile for blt in bls for tile in blt]
		tiles = np.unique(np.array(tiles))
		tile_pos_dict = self.get_tile_pos()
		print (tile_pos_dict)
		tile_pos = {key: value for key, value in tile_pos_dict.items() if key in tiles}
		return tile_pos

	def get_amps_phases(self):
		data = self.get_normalized_data()
		amps = np.abs(data)
		phases = np.angle(data)
		return amps, phases

	def _get_max(self, data):
		return np.nanmax(data)

	def _get_min(self, data):
		return np.nanmin(data)

	def get_amp_min_max(self):
		amps, _ = self.get_amps_phases()
		pols = list(pol_dict.keys())
		min_max = 	OrderedDict()
		for i in range(len(pols)):
			min_max[pols[i]] = (self._get_min(amps), self._get_max(amps))
		return min_max

	def generate_flags(self, tiles=[], fq_chans=[], pols=''):
		amps, phs = self.get_amps_phases()
		flags = np.zeros((amps.shape), dtype=bool)
		tile_nums = np.array(self.get_tile_numbers())
		for tl in tiles:
			ind = np.where(tile_nums == tl)[0][0]
			for ch in fq_chans:
				if pols == "":
					flags[ind, ch, :] = 1
		return flags

	def plot_soln_amps(self, ant='', pols='', save=None, figname=None):
		amps, _ = self.get_amps_phases()
		_sh1, _sh2, _sh3 = amps.shape
		freqs = np.arange(0, _sh2)
		tiles_dict = self.extract_tiles()
		nants = len(amps)
		if pols ==  '':
			pols = ['XX', 'XY', 'YX', 'YY']
		if ant == '':
			fig = pylab.figure(figsize=(16, 16))
			ax = fig.subplots(8, 16)
			for i in range(nants):
				for p in pols:
					ax[i // 16, i % 16].scatter(freqs, amps[i, :, pol_dict[p]].flatten(), s=0.5, c='{}'.format(_pol_color[pol_dict[p]]), alpha=_pol_alpha[pol_dict[p]], marker='.')
					ax[i // 16, i % 16].set_ylim(0, 2)
					ax[i // 16, i % 16].set_xlabel(list(tiles_dict.keys())[i].strip('Tile'), size=8, labelpad=0.9)
					ax[i // 16, i % 16].set_aspect('auto')
					ax[i // 16, i % 16].grid(ls='dashed')
					ax[i // 16, i % 16].xaxis.tick_top()
					ax[i // 16, i % 16].tick_params(labelsize=5)
					if i%16 != 0:
						ax[i // 16, i % 16].tick_params(left=False, right=False , labelleft=False ,labelbottom=False, bottom=False)
			pylab.subplots_adjust(right=0.99, left=0.02, top=0.95, bottom=0.05, wspace=0, hspace=0.5)

		else:
			ax = pylab.subplot(111)
			for p in pols:
				ax.scatter(freqs, amps[tiles_dict['Tile{}'.format(str(ant).zfill(3))], :, pol_dict[p]].flatten(), s=0.5, c='{}'.format(_pol_color[pol_dict[p]]), alpha=_pol_alpha[pol_dict[p]], marker='.')
				ax.set_ylim(0, 2)
			pylab.tick_params(labelsize=12)
			pylab.grid(ls='dashed')
			pylab.title('Tile{}'.format(format(str(ant).zfill(3))), size=15)
			pylab.xlabel('Frequency (MHz)', fontsize=14)
			pylab.ylabel('Amplitude', fontsize=14)

		if save:
			if not figname is None: figname = calfile.replace('.fits', '_amps.png')
			pylab.savefig(figname)
		else:
			pylab.show()

	def plot_soln_phs(self, ant='', pols='', save=None, figname=None):
		_, phases = self.get_amps_phases()
		phases = phases * 180 / np.pi
		_sh1, _sh2, _sh3 = phases.shape
		freqs = np.arange(0, _sh2)
		tiles_dict = self.extract_tiles()
		nants = len(phases)
		if pols ==  '':
			pols = ['XX', 'XY', 'YX', 'YY']
		if ant == '':
			fig = pylab.figure(figsize=(16, 16))
			ax = fig.subplots(8, 16)
			for i in range(nants):
				for p in pols:
					ax[i // 16, i % 16].scatter(freqs, phases[i, :, pol_dict[p]].flatten(), s=0.5, c='{}'.format(_pol_color[pol_dict[p]]), alpha=_pol_alpha[pol_dict[p]], marker='.')
					ax[i // 16, i % 16].set_ylim(-180, 180)
					ax[i // 16, i % 16].set_xlabel(list(tiles_dict.keys())[i].strip('Tile'), size=8, labelpad=0.9)
					ax[i // 16, i % 16].set_aspect('auto')
					ax[i // 16, i % 16].grid(ls='dashed')
					ax[i // 16, i % 16].xaxis.tick_top()
					ax[i // 16, i % 16].tick_params(labelsize=5)
					if i%16 != 0:			
						ax[i // 16, i % 16].tick_params(left=False, right=False , labelleft=False ,labelbottom=False, bottom=False)
			pylab.subplots_adjust(right=0.99, left=0.02, top=0.95, bottom=0.05, wspace=0, hspace=0.5)
	
		else:
			ax = pylab.subplot(111)
			for p in pols:
				ax.scatter(freqs, amps[tiles_dict['Tile{}'.format(str(ant).zfill(3))], :, pol_dict[p]].flatten(), s=0.5, c='{}'.format(_pol_color[pol_dict[p]]), alpha=_pol_alpha[pol_dict[p]], marker='.')
				ax.set_ylim(0, 2)

			pylab.tick_params(labelsize=12)
			pylab.grid(ls='dashed')
			pylab.title('Tile{}'.format(format(str(ant).zfill(3))), size=15)
			pylab.xlabel('Frequency (MHz)', fontsize=14)
			pylab.ylabel('Phase', fontsize=14)
		
		if save:
			if not figname is None: figname = calfile.replace('.fits', '_amps.png')
			pylab.savefig(figname)
		else:
			pylab.show()
