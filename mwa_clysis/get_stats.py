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

	def eval_mean(self):
		data = np.array(self.cal.get_real_imag())
		mean = np.nanmean(data, axis=2)
		return mean
	
	def eval_median(self):
		data = np.array(self.cal.get_real_imag())
		median = np.nanmedian(data, axis=2)
		return median

	def eval_std(self): 
		data = np.array(self.cal.get_real_imag())
		std = np.nanmedian(data, axis=2)
		return std

	def plot_stats(self):
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
				y_coord = 0.2			
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mmean_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mmean_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Mean', size=15)
		pylab.subplots_adjust(hspace=0.2)
		
		# plottng median
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
				y_coord = 0.2
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mmedian_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mmedian_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Median', size=15)
		pylab.subplots_adjust(hspace=0.2)
		
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
				y_coord = 0.2
			x1_coord = 10
			x2_coord = 35
			ax[i // 2, i % 2].text(x1_coord, y_coord, '{:.4f}'.format(mstd_r), bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 3})
			ax[i // 2, i % 2].text(x2_coord, y_coord, '{:.4f}'.format(mstd_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		pylab.suptitle('Standard deviation', size=15)
		pylab.subplots_adjust(hspace=0.2)

		pylab.show()

	def plot_stats1(self):
		mean = self.eval_mean()
		median = self.eval_median()
		tiles = np.arange(11, 139)
		fig = pylab.figure(figsize=(10, 10))
		ax = fig.subplots(4, 2)
		for i in range(4):
			mmean_r, mmean_i = np.nanmean(mean[0, :, i]), np.nanmean(mean[1, :,i])
			mmedian_r, mmedian_i = np.nanmedian(mean[0, :, i]), np.nanmean(median[1, :,i])
			ax = fig.subplots(2, 2)
			ax[i, 0].plot(tiles, mean[1, :, i], '.',color='green', alpha=0.9, label='imag')
			ax[i, 0].plot(tiles, np.nanmean(mean[0, :, i]) * np.ones(len(tiles)), color='black', alpha=0.5)
			ax[i, 1].plot(tiles, median[0, :, i], '.', color='red', label='real')
			ax[i, 1].plot(tiles, median[1, :, i], '.', color='green', alpha=0.9, label='imag')
			ax[i, 1].plot(tiles, np.nanmean(median[0, :, i]) * np.ones(len(tiles)), color='black', alpha=0.5)
			ax[i, 0].grid(ls='dotted')
			ax[i, 1].grid(ls='dotted')

			# adding mean values
			if i == 1 or i == 2:
				y_coord = 0.2
			else:
				y_coord = 1.2
			x1_coord = 10
			x2_coord = 35
			ax[i, 0].text(x1_coord, y_coord, '{:.4f}'.format(mmean_r), bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 3})
			ax[i, 0].text(x2_coord, y_coord, '{:.4f}'.format(mmean_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})
			ax[i, 1].text(x1_coord, y_coord, '{:.4f}'.format(mmedian_r), bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 3})
			ax[i, 1].text(x2_coord, y_coord, '{:.4f}'.format(mmedian_i), bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 3})

		# set title
		ax[0, 0].set_title('Mean', size=15)
		ax[0, 1].set_title('Median', size=15)
		# set limits
		ax[0, 0].set_ylim(-1.7, 1.7)		
		ax[0, 1].set_ylim(-1.7, 1.7)
		ax[1, 0].set_ylim(-0.3, 0.3)
		ax[1, 1].set_ylim(-0.3, 0.3)
		ax[2, 0].set_ylim(-0.3, 0.3)
		ax[2, 1].set_ylim(-0.3, 0.3)
		ax[3, 0].set_ylim(-1.7, 1.7)
		ax[3, 1].set_ylim(-1.7, 1.7)	

		# set ylabels		
		ax[0, 0].set_ylabel('XX', rotation=90, fontsize=12, labelpad=0.5)	 	
		ax[1, 0].set_ylabel('XY', rotation=90, fontsize=12, labelpad=0.5)
		ax[2, 0].set_ylabel('YX', rotation=90, fontsize=12, labelpad=0.5)
		ax[3, 0].set_ylabel('YY', rotation=90, fontsize=12, labelpad=0.5)
		ax[3, 0].set_xlabel('Tile Number', fontsize=12)
		ax[3, 1].set_xlabel('Tile Number', fontsize=12)

		# legend
		ax[1, 1].legend(bbox_to_anchor=(0., 0.5, 0., 0.), ncol = 1)
		pylab.subplots_adjust(wspace=0.2, hspace=0)
		pylab.show()

	def display_stats(self, mode='mean', write_to=None, outfile=None):
		mean, median = self.cal_stats()
		tiles = self.cal.get_tile_numbers()
		arr = mean if mode == 'mean' else median 
		str_out = "{}\n".format(mode.upper())
		str_out += " Tile    XX_R    XX_I    XY_R    XY_I    YX_R    YX_I    YX_R    YX_I    YY_R    YY_I\n"
		for i in range(len(tiles)):
			str_out +=  "{}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}\n".format(tiles[i], arr[0, i, 0], arr[1, i, 0], arr[0, i, 1], arr[1, i, 1], arr[0, i, 2], arr[1, i, 2], arr[0, i, 3], arr[1, i, 3])
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

	def get_polyfit(self, pol, tiles='', deg=3):	
		fit_params = OrderedDict()
		if tiles == '':
			tiles = self.cal.get_tile_numbers()
		for i in range(len(tiles)):
			poly = self.fit_polynomial(pol, tiles[i], deg)	
			fit_params['Tile{0:03d}'.format(tiles[i])] = poly
		return fit_params
