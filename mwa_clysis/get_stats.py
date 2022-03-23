from collections import OrderedDict
import numpy as np
from mwa_clysis import read_csoln as rs
import pylab

pols= ['XX', 'XY', 'YX', 'YY']

class Stats(object):
	def __init__(self, calfile, metafits):
		self.calfile = calfile
		self.metafits = metafits
		self.cal = rs.Cal(self.calfile, self.metafits)

#	def eval_mean(self):
#		data = np.array(self.cal.get_real_imag())
#		mean = np.nanmean(data, axis=2)
#		return mean
	
#	def eval_median(self):
#		data = np.array(self.cal.get_real_imag())
#		median = np.nanmedian(data, axis=2)
#		return median

#	def eval_std(self): 
#		data = np.array(self.cal.get_real_imag())
#		std = np.nanmedian(data, axis=2)
#		return std

	def eval_mean(self):
		amps, _ = self.cal.get_amps_phases()
		mean = np.nanmean(amps, axis=0)
		return mean

	def eval_median(self):
		amps, _ = self.cal.get_amps_phases()
		mean = np.nanmean(amps, axis=0)
		return mean
	
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
		plot_colors = ['cornflowerblue', 'indianred', 'mediumorchid', 'olivedrab']
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

	def plot_stats1(self, save=None):
		mean = self.eval_mean()
		median = self.eval_median()
		std = self.eval_std()
		tiles = self.cal.get_tile_numbers()
		
		# plotting mean
		fig = pylab.figure(figsize=(12, 5))
		ax = fig.subplots(2, 2)
		for i in range(4):
			ax[i // 2, i % 2].plot(tiles, mean[0, :, i], '.-', color='orange', alpha=0.6, label='real')
			ax[i // 2, i % 2].plot(tiles, mean[1, :, i], '.-',color='green', alpha=0.6, label='imag')
			mmean_r, mmean_i = np.nanmean(mean[0, :, i]), np.nanmean(mean[1, :,i])
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mmean_r, color='orange', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mmean_i, color='green', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].set_ylabel(pols[i])
			ax[i // 2, i % 2].grid(ls='dotted')
			if  i > 1:
				ax[i // 2, i % 2].set_xlabel('Tile Number')
			if i == 1:
				ax[i // 2, i % 2].legend(bbox_to_anchor=(0.9,1.2), loc="upper right", fancybox=True, ncol=2)
			if i == 0 or i == 3:
				ax[i // 2, i % 2].set_ylim(-1.4, 1.4)
				y_coord = 1.1
			else:
				ax[i // 2, i % 2].set_ylim(-0.3, 0.3)
				y_coord = 0.22			
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mmean_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mmean_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Mean', size=15)
		pylab.subplots_adjust(hspace=0.2)
		if save:
			figname = self.calfile.replace('.fits', '_mean.png')
			pylab.savefig(figname)
			pylab.close()

		# plotting median
		fig = pylab.figure(figsize=(12, 5))
		ax = fig.subplots(2, 2)
		for i in range(4):
			ax[i // 2, i % 2].plot(tiles, median[0, :, i], '.-', color='orange', alpha=0.6, label='real')
			ax[i //2, i % 2].plot(tiles, median[1, :, i], '.-',color='green', alpha=0.6, label='imag')
			mmedian_r, mmedian_i = np.nanmean(median[0, :, i]), np.nanmedian(median[1, :,i])
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mmedian_r, color='orange', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mmedian_i, color='green', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].set_ylabel(pols[i])
			ax[i // 2, i % 2].grid(ls='dotted')
			if  i > 1:
				ax[i // 2, i % 2].set_xlabel('Tile Number')
			if i == 1:
				ax[i // 2, i % 2].legend(bbox_to_anchor=(0.9,1.2), loc="upper right", fancybox=True, ncol=2)
			if i == 0 or i == 3:
				ax[i // 2, i % 2].set_ylim(-1.4, 1.4)
				y_coord = 1.1
			else:
				ax[i // 2, i % 2].set_ylim(-0.3, 0.3)
				y_coord = 0.22
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mmedian_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mmedian_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Median', size=15)
		pylab.subplots_adjust(hspace=0.2)
		if save:
			figname = self.calfile.replace('.fits', '_median.png')
			pylab.savefig(figname)
			pylab.close()		

		# plottng std
		fig = pylab.figure(figsize=(12, 5))
		ax = fig.subplots(2, 2)
		for i in range(4):
			ax[i // 2, i % 2].plot(tiles, std[0, :, i], '.-', color='orange', alpha=0.6, label='real')
			ax[i //2, i % 2].plot(tiles, std[1, :, i], '.-',color='green', alpha=0.6, label='imag')
			mstd_r, mstd_i = np.nanstd(std[0, :, i]), np.nanstd(std[1, :,i])
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mstd_r, color='orange', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].plot(tiles, np.ones(len(tiles)) * mstd_i, color='green', ls='dashed', linewidth=2)
			ax[i // 2, i % 2].set_ylabel(pols[i])
			ax[i // 2, i % 2].grid(ls='dotted')
			if  i > 1:
				ax[i // 2, i % 2].set_xlabel('Tile Number')
			if i == 1:
				ax[i // 2, i % 2].legend(bbox_to_anchor=(0.9,1.2), loc="upper right", fancybox=True, ncol=2)
			if i == 0 or i == 3:
				ax[i // 2, i % 2].set_ylim(-1.4, 1.4)
				y_coord = 1.1
			else:
				ax[i // 2, i % 2].set_ylim(-0.3, 0.3)
				y_coord = 0.22
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mstd_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mstd_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Standard deviation', size=15)
		pylab.subplots_adjust(hspace=0.2)
		if save:
			figname = self.calfile.replace('.fits', '_std.png')
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def display_stats(self, mode='mean', write_to=None, outfile=None):
		mean = self.eval_mean()
		median = self.eval_median
		std = self.eval_std
		tiles = self.cal.get_tile_numbers()
		mode_dict = {'mean' : 0, 'median' : 1, 'std' : 2}
		modes = [mean, median, std]
		arr = modes[mode_dict[mode]]
		str_out = "#{}\n".format(mode.upper())
		str_out += "#Tile  XX_R    XX_I    XY_R    XY_I    YX_R   YX_I    YX_R    YX_I    YY_R  YY_I\n"
		for i in range(len(tiles)):
			str_out +=  " {}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}\n".format(tiles[i], arr[0, i, 0], arr[1, i, 0], arr[0, i, 1], arr[1, i, 1], arr[0, i, 2], arr[1, i, 2], arr[0, i, 3], arr[1, i, 3])
		# displaying output
		print (str_out)     

		# writing to file
		if not write_to is None:
			std_out = open(outfile, 'wb')
			std_out.write(str_out)
			std_out.close()

	def fit_polynomial(self, pol, tile, deg):
		amps, _ = self.cal.get_amps_phases()
		fq_chans = np.arange(0, amps.shape[1]) 
		tiles_dict = self.cal.extract_tiles()
		ydata = amps[tiles_dict['Tile{0:03d}'.format(tile)], :, rs.pol_dict[pol.upper()]]	
		nn_inds = np.where(~np.isnan(ydata))
		poly = np.polyfit(fq_chans[nn_inds], ydata[nn_inds], deg=deg)
		return poly

	def get_fit_params(self, pol, deg=3):
		fit_params = OrderedDict()
		tiles = self.cal.get_tile_numbers()
		amps, _ = self.cal.get_amps_phases()
		fq_chans = np.arange(0, amps.shape[1])
		for i in range(len(tiles)):
			try:
				poly = self.fit_polynomial(pol, tiles[i], deg)
				fit_err = np.sqrt(np.nansum((np.polyval(poly, fq_chans) - amps[i, :, rs.pol_dict[pol]]) ** 2))
				# the last parameter is the error in the polynomial fitting
				fit_params['Tile{0:03d}'.format(tiles[i])] = np.append(poly, fit_err)
			except TypeError:
				print ('WARNING: Data for tile{} seems to be flagged'.format(tiles[i]))
		return fit_params

	def plot_fit_soln(self, pol, deg=3, save=None):
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		amps, _ = self.cal.get_amps_phases()
		fq_chans = np.arange(0, amps.shape[1])
		tiles = self.cal.get_tile_numbers()
		fig = pylab.figure(figsize=(16, 16))
		ax = fig.subplots(8, 16)
		for i, tl in enumerate(tiles):
			try:
				ax[i // 16, i % 16].plot(fq_chans, np.polyval(fit_params['Tile{:03d}'.format(tl)][:-1], fq_chans), 'k-', linewidth=1)
				ax[i // 16, i % 16].text(100, 1.3, '{:.4f}'.format(fit_params['Tile{:03d}'.format(tl)][-1]), color='green', fontsize=6)
			except KeyError:
				print ('WARNING: Omitting Tile{:03d}'.format(tl))
			ax[i // 16, i % 16].scatter(fq_chans, amps[i, :, rs.pol_dict[pol]].flatten(), s=0.5, c='red', alpha=0.7, marker='.')
			ax[i // 16, i % 16].set_aspect('auto')
			ax[i // 16, i % 16].grid(ls='dashed')
			ax[i // 16, i % 16].xaxis.tick_top()
			ax[i // 16, i % 16].tick_params(labelsize=5)
			ax[i // 16, i % 16].set_ylim(0.5, 1.5)
			if i%16 != 0:
				ax[i // 16, i % 16].tick_params(left=False, right=False , labelleft=False ,labelbottom=False, bottom=False)
			pylab.subplots_adjust(right=0.99, left=0.02, top=0.95, bottom=0.05, wspace=0, hspace=0.5)

		pylab.suptitle('Polynomial Fitting (n = {}) to {}'.format(deg, pol))
		if save:
			figname = calfile.replace('.fits', '_{}_fit.png')
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

	def plot_fit_err(self, pol, deg=3, save=None):
		fit_params = self.get_fit_params(pol=pol, deg=deg)
		tiles = [int(tl.strip('Tile')) for tl in fit_params.keys()] 
		fig = pylab.figure()
		pylab.plot(tiles, np.array([*fit_params.values()])[:, -1], '.-', color='maroon')
		pylab.xlabel('Tile Number')
		pylab.ylabel('Fit error')
		pylab.grid(ls='dotted')
		if save:
			figname = calfile.replace('.fits', '_{}_fit_err.png')
			pylab.savefig(figname)
			pylab.close()
		else:
			pylab.show()

