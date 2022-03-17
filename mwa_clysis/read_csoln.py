from astropy.io import fits
import numpy as np
import pylab

pol_dict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}
_pol_color = ['red', 'red', 'blue', 'blue']
_pol_alpha = [1, 0.2, 0.2, 1]

class Cal(object):
	def __init__(self, calfile, metafits):
		self.calfile = calfile
		self.metafits = metafits

	def read_soln(self):
		hdu = fits.open(self.calfile)
		data = hdu[1].data
		return data

	def read_metadata(self):
		hdu = fits.open(self.metafits)
		mdata = hdu[1].data
		return mdata
	
	def extract_tiles(self):
		mdata = self.read_metadata()
		tiles = np.array([])
		tiles = np.unique([np.append(tiles, mdata[i][3]) for i in range(len(mdata))])
		tiles_dict = {x:i for i, x in enumerate(tiles)}
		return tiles_dict

	def get_tile_numbers(self):
		tiles_dict = self._extract_tiles()
		tiles = list(tiles_dict)
		tnums = [int(tl.strip('Tile')) for tl in tiles]
		return tnums

	def get_amps_phases(self):
		data = self.read_soln()
		amps = np.abs(data[0, :, :, ::2] + data[0, :, :, 1::2] * 1j)
		phases = np.angle(data[0, :, :, ::2] + data[0, :, :, 1::2] * 1j)
		return amps, phases

	def get_real_imag(self):
		data = self.read_soln()
		d_real = data[0, :, :, ::2]
		d_imag = data[0, :, :, 1::2]
		return d_real, d_imag

	def plot_soln_amps(self, ant='', pols='', save=None, figname=None):
		amps, _ = self.get_amps_phases()
		_sh1, _sh2, _sh3 = amps.shape
		freqs = np.arange(0, _sh2)
		tiles_dict = self._extract_tiles()
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
		tiles_dict = self._extract_tiles()
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
