from collections import OrderedDict
import numpy as np
from mwa_clysis import read_csoln as rs
import collections
import pandas as pd
import pylab
import seaborn as sns

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
			ax[i // 2, i % 2].set_ylabel(modes_str[i].upper(), fontsize=12)
			if i == 2 or i == 3:
				ax[i // 2, i % 2].set_xlabel('Frequency(MHz)', fontsize=12)
			if i == 1:
				ax[i // 2, i % 2].legend(bbox_to_anchor=(0.9,1.2), loc="upper right", fancybox=True, ncol=2)
			ax[i // 2, i % 2].tick_params(labelsize=12)
		pylab.suptitle('Gain Amplitude', size=15)
		if save:
			figname = self.calfile.replace('.fits', '_stats.png')
			pylab.savefig(figname, dpi=300)
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
		poly  = np.polyfit(freqs[nn_inds], ydata[nn_inds], deg=deg)
		return poly

	def get_fit_params(self, pol, deg=3):
		pol = pol.upper()
		fit_params = OrderedDict()
		tiles = self.cal.get_tile_numbers()
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		fq_chans = np.arange(0, amps.shape[0])
		for i in range(len(tiles)):
			try:
				poly = self.fit_polynomial(pol, tiles[i], deg)
				fit_err = np.sqrt(np.nansum((np.polyval(poly, freqs) - amps[i, :, rs.pol_dict[pol]]) ** 2))
				chisq = np.nansum(((np.polyval(poly, freqs) - amps[i, :, rs.pol_dict[pol]]) ** 2) / amps[i, :, rs.pol_dict[pol]])  
				# the last parameter is the error in the polynomial fitting
				fit_params['Tile{0:03d}'.format(tiles[i])] = np.append(poly, fit_err)
				fit_params['Tile{0:03d}'.format(tiles[i])] = np.append(fit_params['Tile{0:03d}'.format(tiles[i])], chisq)
			except TypeError:
				print ('WARNING: Data for tile{} seems to be flagged'.format(tiles[i]))
		return fit_params

	
	def plot_fit_soln_tile(self, pol, tile, deg=3, save=None):
		# enusring pol in in upper casse letter
		pol = pol.upper()
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		tiles = self.cal.extract_tiles()
		tile_ind = tiles['Tile{:03d}'.format(tile)]
		mn_mx_dict = self.cal.get_amp_min_max()
		min_val = mn_mx_dict[pol.upper()][0]
		max_val = mn_mx_dict[pol.upper()][1]
		fig = pylab.figure(figsize=(8, 6))
		ax = pylab.subplot(111)
		try:
			ax.plot(freqs, np.polyval(fit_params['Tile{:03d}'.format(tile)][:-2], freqs), 'k-', linewidth=2)
			ax.text(170, max_val + 0.05, '{:.4f}'.format(fit_params['Tile{:03d}'.format(tile)][-2]), color='black', fontsize=10, bbox={
        'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
			ax.text(175, max_val + 0.05, '{:.4f}'.format(fit_params['Tile{:03d}'.format(tile)][-1]), color='black', fontsize=10, bbox={
        'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
			ax.set_ylim(min_val + 0.5, max_val + 0.15)
		except (KeyError, ValueError):
			print ('WARNING: Omitting Tile{:03d}'.format(tile))
		ax.scatter(freqs, amps[tile_ind, :, rs.pol_dict[pol]].flatten(), s=10, c='red', alpha=0.9, marker='.')
		ax.set_aspect('auto')
		ax.grid(ls='dashed')
		ax.tick_params(labelsize=5)
		ax.set_ylabel('Amplitude', fontsize=12)
		ax.set_xlabel('Frequency (MHz)', fontsize=12)
		ax.set_title('Tile {}'.format(tile), size=15)
		ax.tick_params(labelsize=10)
		if save:
			figname = self.calfile.replace('.fits', '_{}_fit.png'.format(pol))
			figname = figname.replace('.png', '_{0:03d}.png'.format(tile))
			pylab.savefig(figname)
		else:
			pylab.show()

	def plot_fit_soln(self, pol, deg=3, save=None):
		# enusring pol in in upper case letter
		pol = pol.upper()
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		amps, _ = self.cal.get_amps_phases()
		freqs = self.cal.get_freqs()
		tiles = self.cal.get_tile_numbers()
		mn_mx_dict = self.cal.get_amp_min_max()
		min_val = mn_mx_dict[pol.upper()][0]
		max_val = mn_mx_dict[pol.upper()][1]
		fig = pylab.figure(figsize=(16, 16))
		ax = fig.subplots(8, 16)
		for i, tl in enumerate(tiles):
			try:
				ax[i // 16, i % 16].plot(freqs, np.polyval(fit_params['Tile{:03d}'.format(tl)][:-2], freqs), 'k-', linewidth=1)
				ax[i // 16, i % 16].set_ylim(min_val + 0.5, max_val + 0.2)
			except (KeyError, ValueError):
				print ('WARNING: Omitting Tile{:03d}'.format(tl))
			ax[i // 16, i % 16].scatter(freqs, amps[i, :, rs.pol_dict[pol]].flatten(), s=0.5, c='red', alpha=0.7, marker='.')
			ax[i // 16, i % 16].set_aspect('auto')
			ax[i // 16, i % 16].grid(ls='dashed')
			ax[i // 16, i % 16].tick_params(labelsize=10)
			ax[i // 16, i % 16].set_title('Tile {}'.format(tl))
			if i%16 != 0:
				ax[i // 16, i % 16].tick_params(left=False, right=False , labelleft=False ,labelbottom=True, bottom=True)
			pylab.subplots_adjust(right=0.99, left=0.02, top=0.95, bottom=0.05, wspace=0, hspace=0.5)

		pylab.suptitle('Polynomial Fitting (n = {}) to {}'.format(deg, pol))
		if save:
			figname = self.calfile.replace('.fits', '_{}_fit.png'.format(pol))
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def plot_fit_err(self, pol, deg=3, save=None):
		fit_params = self.get_fit_params(pol=pol.upper(), deg=deg)
		print (np.array([*fit_params.values()])[:, -2])
		print (np.array([*fit_params.values()])[:, -1])
		tiles = [int(tl.strip('Tile')) for tl in fit_params.keys()] 
		fig = pylab.figure()
		ax1 = pylab.subplot(211)		
		ax1.plot(tiles, np.array([*fit_params.values()])[:, -2], '.-', color='blue', linewidth=2, alpha=0.7)
		ax1.set_ylabel('Fit error')
		ax1.set_title('Amplitude -- {}'.format(pol))
		ax1.grid(ls='dotted')
		ax2 = pylab.subplot(212)
		ax2.plot(tiles, np.array([*fit_params.values()])[:, -1], '.-', color='blue', linewidth=2, alpha=0.7)
		ax2.set_xlabel('Tile Number')
		ax2.set_ylabel('Chi square error')
		ax2.grid(ls='dotted')

		pylab.subplots_adjust(hspace=0.3)

		if save:
			figname = self.calfile.replace('.fits', '_{}_fit_err.png'.format(pol))
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def calc_fit_chisq_wrt_tiles(self, pol, deg=3):
		# Calcuting chi square with respect to a reference antenna
		# chi^2 = np.sum (O_I - E_i)^2
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		keys = list(fit_params.keys())
		tiles = [int(tl.strip('Tile')) for tl in keys]
		freqs = self.cal.get_freqs()
		polyfits = collections.OrderedDict()
		# constructing polyfit
		for key in keys:
			polyfits[key] = np.polyval(fit_params[key], freqs)
		# calculating chi square
		chi_sqs = {}
		for i, k1 in enumerate (keys):
			chis = np.ones((len(tiles)))
			for j, k2 in enumerate(keys):
				# skip diagonals				
				if i != j:
					chis[j] = np.nansum(((polyfits[k2] - polyfits[k1]) ** 2) / polyfits[k1])
			chi_sqs[tiles[i]] = chis
		# Creating panda dataframe
		df = pd.DataFrame(data = chi_sqs)
		df.index = tiles
		return df

	def plot_fit_chisq_wrt_tiles(self, pol, deg=3, save=None):
		# Plotting chisq fit
		pylab.figure(figsize = (10, 10))
		chi_df = self.calc_fit_chisq_wrt_tiles(pol=pol, deg=deg)
		heatmap = sns.heatmap(chi_df, cmap='coolwarm', annot=True, fmt='.1f')
		heatmap.set_title('Chi Square', fontdict={'fontsize':14}, pad=12)
		heatmap.set_xlabel('Tile Number', fontdict={'fontsize':6}, labelpad=12)
		heatmap.set_ylabel('Tile Number', fontdict={'fontsize':6}, labelpad=12)
		if save:
			figname= self.calfile.replace('.fits', '_{}_chisq.png'.format(pol))
			pylab.savefig(figname, dpi=300)
		else:
			pylab.show()

	def f2etas(self, freqs):
    	#Evaluates geometric delay (fourier conjugate of frequency)
   		#freqs: Frequencies in GHz; type:numpy.ndarray 
		df = freqs[1] - freqs[0]
		etas = np.fft.fftshift(np.fft.fftfreq(freqs.size, df))
		return etas

	def filter_nans(self, data, freqs):
		inds = np.where(~np.isnan(data))
		data_filtered = data[inds[0]]
		freqs_filtered = freqs[inds[0]]
		return data_filtered, freqs_filtered

	def fft_data(self):
		data = self.cal.get_normalized_data()
		tile_dict = self.cal.extract_tiles()
		tiles = list(tile_dict.keys())
		tile_inds = list(tile_dict.values())
		freqs = self.cal.get_freqs()
		fft_data_dict = OrderedDict()
		for i in range(len(tile_inds)):
			fft_data_dict[tiles[i]] = OrderedDict()
			fft_data_dict[tiles[i]] = OrderedDict()
			fft_data_dict[tiles[i]]['xx'] = []
			fft_data_dict[tiles[i]]['yy'] = []
			data_xx = data[tile_inds[i], :, 0]
			data_yy = data[tile_inds[i], :, 3]
			filtered_data_xx, filtered_freqs_xx = self.filter_nans(data_xx, freqs * 1e-3) 
			filtered_data_yy, filtered_freqs_yy = self.filter_nans(data_yy, freqs * 1e-3)
			# xx polarization
			try:
				fft_data_xx = np.fft.fftshift(np.fft.fft(filtered_data_xx))
				etas_xx = self.f2etas(filtered_freqs_xx)
				fft_data_dict[tiles[i]]['xx'] = [etas_xx, fft_data_xx]			
				# yy polarization
				fft_data_yy = np.fft.fft(filtered_data_yy)
				etas_yy = self.f2etas(filtered_freqs_yy)
				fft_data_dict[tiles[i]]['yy'] = [etas_yy, fft_data_yy]		
			except ValueError:
				pass

		return fft_data_dict

