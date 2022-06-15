from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from mwa_clysis import read_csoln as rs
import numpy as np
import pylab
import argparse
import pickle
import os, sys
import re

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dict', nargs='+', type=str, help='Dictionary containing the fft of the calibration solutions in pickle format')
parser.add_argument('-p', '--pol', type=str, help='Polarization information (either xx or yy)')
parser.add_argument('-s', '--save', action='store_true', help='If true, saves the various plots')
parser.add_argument('-i', '--individual', action='store_true', help='If inidividual will plot for each file else only a combine output will be provided')
parser.add_argument('-m', '--metafits', nargs='+', type=str, help='Metafits containing info on the observation')
parser.add_argument('--inset', action='store_true', help='If True, will inset inset on the position of the tiles')
args = parser.parse_args()

dict_files = args.data_dict
# initializing the 2d array
dirname = os.path.dirname(dict_files[0])
dict_fft = pickle.load(open(dict_files[0], 'rb'))
tile_keys = list(dict_fft.keys())
ntiles = len(tile_keys)
etas = dict_fft[tile_keys[0]][args.pol.lower()]['etas'][0]
fft_2d = np.zeros((len(dict_files), ntiles, len(etas)), dtype=complex)
_sh = fft_2d.shape
obsdates, gps, lsts = [], [], []
for i, dcf in enumerate(dict_files):
	cfl_gps = re.findall(r'\d+', dcf.split('/')[-1])
	dict_fft = pickle.load(open(dcf, 'rb'))
	if args.metafits is None:
		if len(cfl_gps) == 1:
			mfl = '{}/{}.metafits'.format(dirname, cfl_gps[0])
		else:
			raise Exception("The string contains two integers values. Need only one 1")
	else:
		mfl = args.metafits
	cal = rs.Cal(calfile=None, metafits=mfl)
	obsdates.append(cal.get_obsdate())
	lsts.append(round(cal.get_lst() * 0.25, 2))
	gps.append(cfl_gps[0])
	for j, tk in enumerate(tile_keys):
		data_tile = dict_fft[tk][args.pol.lower()]
		data_fft = data_tile['fft']
		if len(data_fft) == 0:
			print ('WARNING: No data found for {}, flagged musthave been applied',format(tk))
			fft_2d[i, j, :] = np.nan
		else:
			fft_2d[i, j, :] = data_fft[0]

# plotting individual plots
if args.individual:
	tile_pos_dict = cal.get_tile_pos()
	tile_pos = np.array(list(tile_pos_dict.values()))
	for i in range(_sh[0]): # across files
		print (i)
		for j in range(_sh[1]): # across tiles
			print ('OBS: {} ... Plotting 1D fft for {} '.format(gps[i], tile_keys[j]))
			fig, ax1 = pylab.subplots(figsize = (9, 7))
			ax1.semilogy(etas, np.abs(fft_2d[i, j, :]), linewidth=2)
			ax1.set_title('{} --{}'.format(tile_keys[j], args.pol.upper()), size=15)
			ax1.set_xlabel('Delays (ns)', fontsize=13)
			ax1.set_ylabel('Amplitude', fontsize=13)
			ax1.set_ylim(10**-2, 10**4.4)
			ax1.grid(ls='dotted')
			if args.inset:
                #inset for the tile position
				ax2 = pylab.axes([0, 0, 0.5, 0.5])
				ip = InsetPosition(ax1, [0.62, 0.62, 0.35, 0.35])
				ax2.set_axes_locator(ip)
				ax2.scatter(tile_pos[:, 0], tile_pos[:, 1], marker='o', edgecolor='dodgerblue')
				ax2.grid(ls='dotted')
				x0 = tile_pos_dict[int(tile_keys[j].strip('Tile'))][0]
				y0 = tile_pos_dict[int(tile_keys[j].strip('Tile'))][1]
				ax2.scatter([x0], [y0], color='magenta', marker = 'o')
				ax2.set_xlabel('East-West (m)')
				ax2.set_ylabel('North-South (m)', labelpad=-4)             
                # inset for the zoom version  of tile position
				ax3 = pylab.axes([0, 0, 0.5, 0.5])
				ip = InsetPosition(ax1, [0.2, 0.78, 0.2, 0.2])
				ax3.set_axes_locator(ip)
				mark_inset(ax2, ax3, loc1=2, loc2=3, fc="none", ec='0.5')
				ax3.scatter(tile_pos[:, 0], tile_pos[:, 1], marker='o', edgecolor='dodgerblue')
				ax3.scatter([x0], [y0], color='magenta', marker = 'o')
				x0_lim = [x0 - 200, x0 + 200]
				y0_lim = [y0 - 200, y0 + 200]
				ax3.set_xlim(min(x0_lim), max(x0_lim))
				ax3.set_ylim(min(y0_lim), max(y0_lim))
				ax3.grid(ls='dotted')
			if args.save:
				outfile = '{}/{}_fft_1d_{}_{}.png'.format(dirname, cfl_gps[i], args.pol.lower(), tile_keys[j].lower())
				print ('Saving 1D individual plot ...{} '.format(outfile))
				pylab.savefig(outfile, dpi=300)
				pylab.close()
			else:
				pylab.show()

else:
	for i, tk in enumerate(tile_keys):
		vmin, vmax = -1.0, 2.5
		fig = pylab.figure(figsize=(10, 7))
		pylab.imshow(np.log10(np.abs(fft_2d[:, i, :])), aspect='auto', extent=(etas[0], etas[-1], 0, len(lsts)), vmin=vmin, vmax=vmax)
		pylab.colorbar()
		pylab.title('{} -- {}'.format(tk, args.pol.upper()), size=15)
		pylab.ylabel('LST (hrs)', fontsize=13)
		pylab.xlabel('Delays (ns)', fontsize=13)
		pylab.xlim(-10000, 10000)
		pylab.yticks(np.arange(len(lsts)), obsdates[::-1], size=6)
		if args.save:
			outfile = '{}/fft_2d_{}_{}.png'.format(dirname,tk, args.pol.lower())
			print ('Saving 2D combined plot ...{} '.format(outfile))
			pylab.savefig(outfile, dpi=300)
			pylab.close()
		else:
			pylab.show()

                                                                                                                                    
