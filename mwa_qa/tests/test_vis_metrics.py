from mwa_qa import vis_metrics as vm
from mwa_qa import read_uvfits1 as ru
import numpy as np
import unittest
import os

uvfits = '../../test_files/1061315688_cal.uvfits'


class TestVisMetrics(unittest.TestCase):
    def test__init__(self):
        vis = vm.VisMetrics(uvfits)
        self.assertTrue(isinstance(vis.uvf, ru.UVfits))

    def test_autos_for_antnum(self):
        vis = vm.VisMetrics(uvfits)
        auto_data = vis.autos_for_antnum(1)
        self.assertEqual(auto_data.shape, (27, 768, 4))
        np.testing.assert_almost_equal(auto_data[0, 100, :], np.array([24948.32226562 - 4.35759449j,  1646.88220215+1351.92565918j,
                                                                       663.93841553 - 743.69134521j, 28362.50585938 - 8.16024017j]))
        auto_data = vis.autos_for_antnum(77)
        self.assertEqual(len(auto_data), 0)

    def test_autos(self):
        vis = vm.VisMetrics(uvfits)
        autos = vis.autos()
        np.testing.assert_almost_equal(autos[0, 0, 100, :], np.array([24948.32226562 - 4.35759449j,  1646.88220215+1351.92565918j,
                                                                      663.93841553 - 743.69134521j, 28362.50585938 - 8.16024017j]))

    def test_initialize_metrics_dict(self):
        vis = vm.VisMetrics(uvfits)
        vis._initialize_metrics_dict()
        keys = ['Nants', 'Ntimes', 'Nfreqs', 'Npols']
        keys.append(np.arange(1, 129).tolist())
        self.assertEqual(list(vis.metrics.keys()), keys)
        self.assertEqual(vis.metrics['Nants'], 128)
        self.assertEqual(vis.metrics['Ntimes'], 26)
        self.assertEqual(vis.metrics['Nfreqs'], 768)
        self.assertEqual(vis.metrics['Npols'], 4)
        self.assertEqual(list(vis.metrics[1].keys()), ['XX', 'XY', 'YX', 'YY'])
        self.assertEqual(list(vis.metrics[128].keys()), [
                         'XX', 'XY', 'YX', 'YY'])
        self.assertEqual(list(vis.metrics[1]['XX'].keys()), ['autos', 'cross'])
        self.assertEqual(list(vis.metrics[1]['XY'].keys()), ['autos', 'cross'])
        self.assertEqual(list(vis.metrics[1]['YX'].keys()), ['autos', 'cross'])
        self.assertEqual(list(vis.metrics[1]['YY'].keys()), ['autos', 'cross'])

    def test_run_metrics(self):
        vis = vm.VisMetrics(uvfits)
        vis.run_metrics()
        self.assertEqual(list(vis.metrics['XX']['autos']['Tile011'].keys()), [
                         'mean_amp_freq', 'median_amp_freq', 'rms_amp_freq', 'var_amp_freq'])
        np.testing.assert_almost_equal(
            vis.metrics[1]['XX']['autos']['mean_amp_freq'], 943.5318914370374, decimal=2)
        np.testing.assert_almost_equal(
            vis.metrics[1]['XX']['autos']['median_amp_freq'], 885.4816159565241, decimal=2)
        np.testing.assert_almost_equal(
            vis.metrics[1]['XX']['autos']['rms_amp_freq'], 967.5731931584965, decimal=2)
        np.testing.assert_almost_equal(
            vis.metrics[1]['XX']['autos']['var_amp_freq'], 45945.45396017565, decimal=2)

    def test_write_to(self):
        vis = vm.VisMetrics(uvfits)
        vis._initialize_metrics_dict()
        outfile = uvfits.replace('.uvfits', '_vis_metrics.json')
        vis.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        vis.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
