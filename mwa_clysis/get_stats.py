from collections import OrderedDict
import numpy as np
from mwa_clysis import read_csoln as rs
import pylab

pols = ['XX', 'XY', 'YX', 'YY']

class Stats(object):
	def __init__(self, calfile=None, metafits=None):
		self.calfile = calfile
		self.metafits = metafits
		self.cal = rs.Cal(self.calfile, self.metafits)

	def eval_mean(self):
		amps, _ = self.cal.get_amps_phases()
		mean = np.nanmean(amps, axis=0)
		return mean

	def eval_median(self):
		amps, _ = self.cal.get_amps_phases()
		median = np.nanmedian(amps, axis=0)
		return median
	
	def eval_rms(self):
		amps, _ = self.cal.get_amps_phases()
		rms = np.sqrt(np.nanmean(amps ** 2, axis=0))
		return rms

	def eval_var(self):
		amps, _ = self.cal.get_amps_phases()
		var = np.nanvar(amps, axis = 0)
		return var

	def plot_stats(self, pols=[], save=None):
		fig = pylab.figure(figsize=(12, 8))
		ax = fig.subplots(2, 2)
		mean = self.eval_mean()
		median = self.eval_median()
		var = self.eval_var()
		rms = self.eval_rms()
		freqs = self.cal.get_freqs()
		modes = [mean, median, var, rms]
		modes_str = ['mean', 'median', 'var', 'rms']
		plot_colors = ['cornflowerblue', 'indianred', 'mediumorchid', 'olive']
		for i in range(4):
			for j, p in enumerate(pols):
				ax[i // 2, i % 2].plot(freqs, modes[i][:, rs.pol_dict[p.upper()]], '.-', color=plot_colors[j], label=p)
			ax[i //2, i % 2].grid(ls='dashed')
			ax[i // 2, i % 2].set_ylabel(modes_str[i], fontsize=12)
			if i == 2 or i == 3:
				ax[i // 2, i % 2].set_xlabel('Frequency(MHz)', fontsize=12)
			if i == 1:
				ax[i // 2, i % 2].legend(bbox_to_anchor=(0.9,1.2), loc="upper right", fancybox=True, ncol=2)
		pylab.suptitle('Gain Amplitude', size=15)
		if save:
			figname = self.calfile.replace('.fits', '_stats.png')
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def fit_polynomial(self, pol, tile, deg):
		amps, _ = self.cal.get_amps_phases()
		fq_chans = np.arange(0, amps.shape[1])
		freqs = self.cal.get_freqs() 
		tiles_dict = self.cal.extract_tiles()
		ydata = amps[tiles_dict['Tile{0:03d}'.format(tile)], :, rs.pol_dict[pol.upper()]]	
		nn_inds = np.where(~np.isnan(ydata))
		poly = np.polyfit(freqs[nn_inds], ydata[nn_inds], deg=deg)
		return poly

	def get_fit_params(self, pol, deg=3):
		fit_params = OrderedDict()
		tiles = self.cal.get_tile_numbers()
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		fq_chans = np.arange(0, amps.shape[0])
		for i in range(len(tiles)):
			try:
				poly = self.fit_polynomial(pol, tiles[i], deg)
				fit_err = np.sqrt(np.nansum((np.polyval(poly, freqs) - amps[i, :, rs.pol_dict[pol]]) ** 2))
				# the last parameter is the error in the polynomial fitting
				fit_params['Tile{0:03d}'.format(tiles[i])] = np.append(poly, fit_err)
			except TypeError:
				print ('WARNING: Data for tile{} seems to be flagged'.format(tiles[i]))
		return fit_params

	def plot_fit_soln(self, pol, deg=3, save=None):
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		tiles = self.cal.get_tile_numbers()
		fig = pylab.figure(figsize=(16, 16))
		ax = fig.subplots(8, 16)
		for i, tl in enumerate(tiles):
			try:
				ax[i // 16, i % 16].plot(freqs, np.polyval(fit_params['Tile{:03d}'.format(tl)][:-1], freqs), 'k-', linewidth=1)
				ax[i // 16, i % 16].text(170, 1.3, '{:.4f}'.format(fit_params['Tile{:03d}'.format(tl)][-1]), color='green', fontsize=6)
			except KeyError:
				print ('WARNING: Omitting Tile{:03d}'.format(tl))
			ax[i // 16, i % 16].scatter(freqs, amps[i, :, rs.pol_dict[pol]].flatten(), s=0.5, c='red', alpha=0.7, marker='.')
			ax[i // 16, i % 16].set_aspect('auto')
			ax[i // 16, i % 16].grid(ls='dashed')
			ax[i // 16, i % 16].set_ylim(0.6, 1.5)
			#ax[i // 16, i % 16].xaxis.tick_top()
			ax[i // 16, i % 16]
			ax[i // 16, i % 16].tick_params(labelsize=5)
			if i%16 != 0:
				ax[i // 16, i % 16].tick_params(left=False, right=False , labelleft=False ,labelbottom=False, bottom=False)
			pylab.subplots_adjust(right=0.99, left=0.02, top=0.95, bottom=0.05, wspace=0, hspace=0.5)

		pylab.suptitle('Polynomial Fitting (n = {}) to {}'.format(deg, pol))
		if save:
			figname = self.calfile.replace('.fits', '_{}_fit.png'.format(pol))
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def plot_fit_err(self, pol, deg=3, save=None):
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		tiles = [int(tl.strip('Tile')) for tl in fit_params.keys()] 
		fig = pylab.figure()
		pylab.plot(tiles, np.array([*fit_params.values()])[:, -1], '.-', color='olive')
		pylab.xlabel('Tile Number')
		pylab.ylabel('Fit error')
		pylab.title('Amplitude -- {}'.format(pol))
		pylab.grid(ls='dotted')
		if save:
			figname = self.calfile.replace('.fits', '_{}_fit_err.png'.format(pol))
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def calc_fit_chisq(self, pol, deg=3):
		# Calculating chi square with respect to reference tile (last tile)
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		tiles = self.cal.get_tile_numbers()
		ref_fit = np.polyval(fit_params['Tile168'][:-1], freqs)
		chi_sqs = {}
		for tl in tiles:
			try:
				diff = (np.polyval(fit_params['Tile{:03d}'.format(tl)][:-1], freqs) - ref_fit) ** 2
				chisq = np.nansum(diff / ref_fit) 
				chi_sqs[tl] = chisq
			except KeyError:
				print ('WARNING: Omitting Tile{:03d}'.format(tl))
		del chi_sqs[168]
		return chi_sqs

