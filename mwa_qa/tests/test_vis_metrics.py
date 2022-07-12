from mwa_qa import vis_metrics as vm
from mwa_qa import read_uvfits as ru
import numpy as np
import unittest
import os

uvfits = '../../test_files/1061315688_cal.uvfits'

class TestVisMetrics(unittest.TestCase):
	def test__init__(self):
		vis = vm.VisMetrics(uvfits)
		self.assertTrue(isinstance(vis.uvf, ru.UVfits))

	def test_autos_for_tile(self):
		vis = vm.VisMetrics(uvfits)
		auto_data = vis.autos_for_tile(11)
		self.assertEqual(auto_data.shape, (27, 768, 4))
		np.testing.assert_almost_equal(auto_data[0, 100, :], np.array([24948.32226562   -4.35759449j,  1646.88220215+1351.92565918j,
         													663.93841553 -743.69134521j, 28362.50585938   -8.16024017j]))
		auto_data = vis.autos_for_tile(105)
		self.assertEqual(len(auto_data), 0)

	def test_autos(self):
		vis = vm.VisMetrics(uvfits)
		autos = vis.autos()
		np.testing.assert_almost_equal(autos[0, 0, 100, :], np.array([24948.32226562   -4.35759449j,  1646.88220215+1351.92565918j,
                                                            663.93841553 -743.69134521j, 28362.50585938   -8.16024017j]))
	def test_initialize_metrics_dict(self):
		vis = vm.VisMetrics(uvfits)
		vis._initialize_metrics_dict()
		expected = ['Ntiles', 'Ntimes', 'Nfreqs', 'Npols', 'XX', 'XY', 'YX', 'YY']
		self.assertEqual(list(vis.metrics.keys()), expected)
		self.assertEqual(list(vis.metrics['XX']), ['autos', 'cross'])
		self.assertEqual(list(vis.metrics['XY']), ['autos', 'cross'])
		self.assertEqual(list(vis.metrics['YX']), ['autos', 'cross'])
		self.assertEqual(list(vis.metrics['YY']), ['autos', 'cross'])
		expected = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 103, 104, 105, 106, 107, 108, 111, 112, 113, 114, 115, 116, 117, 118, 121, 122, 123, 124, 125, 126, 127, 128, 131, 132, 133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 151, 152, 153, 154, 155, 156, 157, 158, 161, 162, 163, 164, 165, 166, 167, 168]
		self.assertEqual(list(vis.metrics['XX']['autos']), expected)
		self.assertEqual(list(vis.metrics['XY']['autos']), expected)
		self.assertEqual(list(vis.metrics['YX']['autos']), expected)
		self.assertEqual(list(vis.metrics['YY']['autos']), expected)

	def test_run_metrics(self):
		vis = vm.VisMetrics(uvfits)
		vis.run_metrics()
		self.assertEqual(list(vis.metrics['XX']['autos'][11].keys()), ['mean_amp_freq', 'median_amp_freq', 'rms_amp_freq', 'var_amp_freq'])
		np.testing.assert_almost_equal(vis.metrics['XX']['autos'][11]['mean_amp_freq'], 943.5318914370374, decimal = 2)
		np.testing.assert_almost_equal(vis.metrics['XX']['autos'][11]['median_amp_freq'], 885.4816159565241, decimal = 2)
		np.testing.assert_almost_equal(vis.metrics['XX']['autos'][11]['rms_amp_freq'], 967.5731931584965, decimal = 2)
		np.testing.assert_almost_equal(vis.metrics['XX']['autos'][11]['var_amp_freq'], 45945.45396017565, decimal = 2)

	def test_write_to(self):
		vis = vm.VisMetrics(uvfits)
		vis._initialize_metrics_dict()
		outfile = uvfits.replace('.uvfits', '_vis_metrics.json')
		vis.write_to()
		self.assertTrue(os.path.exists(outfile))
		os.system('rm -rf {}'.format(outfile))
		outfile = 'metrics.json'
		vis.write_to(outfile = outfile)
		self.assertTrue(os.path.exists(outfile))
		os.system('rm -rf {}'.format(outfile))

		
