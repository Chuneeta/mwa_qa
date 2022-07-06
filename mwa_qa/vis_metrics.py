from mwa_qa import read_uvfits as ru
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

class VisMetrics(object):
	def __init__(self, uvfits):
		self.uvfits = uvfits
		self.uvf = ru.UVfits(uvfits)

	def autos_for_tile(self, tile):
		auto_data = self.uvf.data_for_tilepair((tile, tile))
		if len(auto_data) == 0:
			print ('WARNING: No data found for Tile {}, maybe it is flagged'.format(tile))
		return auto_data

	def autos(self):
		tile_numbers = self.uvf.tile_numbers()
		auto_array = np.zeros((len(tile_numbers), self.uvf.Ntimes, self.uvf.Nfreqs, self.uvf.Npols), dtype=np.complex64)
		for i, tl in enumerate(tile_numbers):
			auto_data = self.autos_for_tile(tl)
			if len(auto_data) > 0:
				auto_array[i, :, :, :] = auto_data
			else:
				# fill array with Nans
				auto_array[i, :, :, :] = np.nan
		return auto_array

	def _initialize_metrics_dict(self):
		self.metrics = OrderedDict()
		self.metrics['Ntiles'] = self.uvf.Ntiles
		self.metrics['Ntimes'] = self.uvf.Ntimes
		self.metrics['Nfreqs'] = self.uvf.Nfreqs
		self.metrics['Npols'] = self.uvf.Npols
		# initializing each pol key as a dictionary
		pols = self.uvf.pols()
		tile_numbers = self.uvf.tile_numbers()
		for p in pols:
			self.metrics[p] = OrderedDict([('autos', OrderedDict()), ('cross', OrderedDict())])
			for tn in tile_numbers:
				self.metrics[p]['autos'][tn] = OrderedDict()
				#self.metrics[p]['cross'][tn] = OrderedDict()

	def run_metrics(self):
		self._initialize_metrics_dict()
		autos = self.autos()
		# averaged over time
		avg_autos = np.nanmean(autos, axis = 1)
		amps_avg_autos = np.abs(avg_autos)
		pols = self.uvf.pols()
		tile_numbers = self.uvf.tile_numbers()
		for i, p in enumerate(pols):
			for j, tn in enumerate(tile_numbers):
				self.metrics[p]['autos'][tn]['mean_amp_freq'] = np.nanmean(amps_avg_autos[j, :, i])
				self.metrics[p]['autos'][tn]['median_amp_freq'] = np.nanmedian(amps_avg_autos[j, :, i]) 
				self.metrics[p]['autos'][tn]['rms_amp_freq'] = np.sqrt(np.nanmean(amps_avg_autos[j, :, i] ** 2))
				self.metrics[p]['autos'][tn]['var_amp_freq'] = np.nanvar(amps_avg_autos[j, :, i])

	def write_to(self, outfile=None):
		if outfile is None:
			outfile = self.uvfits.replace('.uvfits', '_vis_metrics.json')
		ju.write_metrics(self.metrics, outfile)
