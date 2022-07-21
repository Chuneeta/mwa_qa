from collections import OrderedDict
from mwa_qa import read_csolutions as rc
from mwa_qa import read_metafits as rm
from mwa_qa import cal_metrics as cm
from mwa_qa.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits')
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')


class TestCalMetrics(unittest.TestCase):
    def test_init__(self):
        m = cm.CalMetrics(calfile, metafits=metafits)
        self.assertEqual(m.calfile, calfile)
        self.assertEqual(m.Csoln.calfile, calfile)
        self.assertEqual(m.Csoln.Metafits.metafits, metafits)
        self.assertEqual(m. Csoln.Metafits.pol, 'X')

    # def test_variance_for_antpair(self):
    #    m = cm.CalMetrics(calfile, metafits=metafits)
    #    variance = m.variance_for_antpair((102, 104))
    #    self.assertEqual(variance.shape, (1, 1, 4))
    #    np.testing.assert_almost_equal(variance[0, 0, :], np.array(
    #        [6.15063689e-02, 6.10921185e-06, 1.24056403e-05, 8.76188245e-02]), decimal=6)

    # def test_variance_for_baselines_less_than(self):
    #    m = cm.CalMetrics(calfile, metafits=metafits)
    #    bls, variances = m.variance_for_baselines_less_than(10)
    #   self.assertEqual(len(bls), 23)
    #    self.assertEqual(bls, [(23, 21), (22, 21), (27, 26), (27, 43), (27, 41), (26, 25), (25, 43), (14, 36), (13, 12), (17, 16), (
    #        17, 15), (94, 47), (84, 83), (38, 37), (37, 42), (44, 45), (43, 41), (42, 41), (42, 45), (47, 45), (64, 63), (64, 66), (64, 65)])
     #   self.assertEqual(variances.shape, (1, len(bls), 4))
     #   np.testing.assert_almost_equal(variances[0, 0, :], np.array(
     #       [1.69654699e-02, 8.44322446e-06, 1.13070715e-05, 2.04571904e-02]), decimal=6)

    # def test_skewness_across_baselines(self):
     #   m = cm.CalMetrics(calfile, metafits=metafits)
     #   skewness = m.skewness_across_baselines(10)
     #   np.testing.assert_almost_equal(skewness, np.array(
      #      [[1.39739935, 1.1925527, 0.08251331, 1.06225206]]))

    # def test_get_receivers(self):
    #	m = cm.CalMetrics(calfile, metafits = metafits)
    #	receivers = m.get_receivers()
    #	expected = [10,10,10,10,10,10,10,10,7,7,7,7,7,7,7,7,16,16,16,16,16,16,16,16,15,15,15,15,15,15,15,15,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,12,12,12,12,12,12,12,12,11,11,11,11,11,11,11,11,14,14,14,14,14,14,14,14,13,13,13,13,13,13,13,13,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5]
    #	self.assertEqual(receivers, expected)
    #	m = cm.CalMetrics(calfile)
    #	receivers = m.get_receivers()
    #	self.assertEqual(receivers, list(np.arange(1, 17)))

    def test_flagged_baselines_percent(self):
        m = cm.CalMetrics(calfile)
        percent = m.flagged_baselines_percent()
        self.assertEqual(percent, 1.5625)

    def test_flagged_channels_percent(self):
        m = cm.CalMetrics(calfile)
        percent = m.flagged_channels_percent()
        self.assertEqual(percent, 15.625)

    def test_non_converging_percent(self):
        m = cm.CalMetrics(calfile)
        percent = m.non_converging_percent()
        self.assertEqual(percent, 15.625)
    # def test_initialize_metrics_dict(self):
    #    m = cm.CalMetrics(calfile)
    #    metrics = m._initialize_metrics_dict()
    #    self.assertTrue(isinstance(metrics, OrderedDict))
        #self.assertEqual(list(metrics.keys()), ['pols', 'freqs', 'tile_ids'])

    def test_run_metrics(self):
        pass

    def test_write_metrics(self):
        pass
