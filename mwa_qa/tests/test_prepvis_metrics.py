from mwa_qa.prepvis_metrics import PrepvisMetrics
from collections import OrderedDict
import numpy as np
import unittest
import os

uvfits_path = '/Users/ridhima/Documents/mwa/calibration/mwa_qa/test_files/'
uvfits = os.path.join(uvfits_path, '1061315688_cal.uvfits')
uvfits_hex = os.path.join(uvfits_path, '1320411840_hyp_cal.uvfits')


class TestPrepvisMetrics(unittest.TestCase):
    def test__init__(self):
        vis = PrepvisMetrics(uvfits)
        self.assertTrue(vis.uvfits_path, uvfits)

    def test_autos(self):
        vis = PrepvisMetrics(uvfits)
        autos = vis.autos()
        np.testing.assert_almost_equal(autos[0, 0, 100, :],
                                       np.array([np.nan+0.j, np.nan+0.j,
                                                 np.nan+0.j, np.nan+0.j]))

    def test_initialize_metrics_dict(self):
        vis = PrepvisMetrics(uvfits)
        vis._initialize_metrics_dict()
        self.assertEqual(list(vis.metrics.keys()), [
            'NANTS', 'NTIMES', 'NFREQS', 'NPOLS', 'OBSID', 'XX', 'YY'])
        self.assertEqual(vis.metrics['NANTS'], 128)
        self.assertEqual(vis.metrics['NFREQS'], 768)
        self.assertEqual(vis.metrics['NPOLS'], 4)
        self.assertEqual(vis.metrics['NTIMES'], 27)
        self.assertTrue(isinstance(
            vis.metrics['XX'], OrderedDict))
        self.assertTrue(isinstance(
            vis.metrics['YY'], OrderedDict))

    def test_run_metrics(self):
        vis = PrepvisMetrics(uvfits)
        vis.run_metrics()
        self.assertEqual(list(vis.metrics.keys()), [
            'NANTS', 'NTIMES', 'NFREQS', 'NPOLS', 'OBSID', 'XX', 'YY'])
        self.assertEqual(list(vis.metrics['XX'].keys()),
                         ['RMS_AMP_ANT', 'RMS_AMP_FREQ', 'MXRMS_AMP_ANT',
                          'MNRMS_AMP_ANT', 'MXRMS_AMP_FREQ', 'MNRMS_AMP_FREQ',
                          'POOR_ANTENNAS', 'NPOOR_ANTENNAS'])
        self.assertEqual(list(vis.metrics['YY'].keys()),
                         ['RMS_AMP_ANT', 'RMS_AMP_FREQ', 'MXRMS_AMP_ANT',
                          'MNRMS_AMP_ANT', 'MXRMS_AMP_FREQ', 'MNRMS_AMP_FREQ',
                          'POOR_ANTENNAS', 'NPOOR_ANTENNAS'])
        np.testing.assert_equal(vis.metrics
                                ['XX']['MXRMS_AMP_ANT'], np.nan)
        np.testing.assert_equal(vis.metrics
                                ['XX']['MNRMS_AMP_ANT'], np.nan)
        self.assertTrue(np.all(vis.metrics['XX']['RMS_AMP_ANT']))
        np.testing.assert_equal(vis.metrics
                                ['XX']['MXRMS_AMP_FREQ'], np.nan)
        np.testing.assert_equal(vis.metrics
                                ['XX']['MXRMS_AMP_FREQ'], np.nan)
        self.assertTrue(np.all(vis.metrics['XX']['RMS_AMP_FREQ']))
        self.assertTrue(np.all(vis.metrics['XX']['RMS_AMP_FREQ']))
        np.testing.assert_equal(vis.metrics
                                ['XX']['POOR_ANTENNAS'], np.array([]))
        self.assertEqual(vis.metrics['XX']['NPOOR_ANTENNAS'], 0)

    def test_write_to(self):
        vis = PrepvisMetrics(uvfits)
        vis._initialize_metrics_dict()
        outfile = uvfits.replace('.uvfits', '_prepvis_metrics.json')
        vis.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        vis.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
