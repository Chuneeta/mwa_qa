from mwa_clysis import read_csolutions as rc
from mwa_clysis import read_metafits as rm
from mwa_clysis import metrics as m
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
metafits = os.path.join(DATA_PATH, 'test.metafits')

class TestMetric(unittest.TestCase):
	def test_init__(self):
		mt = m.Metric(calfile)
		self.assertEqual(mt.calfile, calfile)
		mt = m.Metric(calfile, metafits = metafits, pol='X')
		self.assertTrue(isinstance(mt.Csoln, rc.Csoln))
		self.assertTrue(isinstance(mt.Metafits, rm.Metafits))
		self.assertEqual(mt.Csoln.calfile, calfile)
		self.assertEqual(mt.Metafits.metafits, metafits)
		self.assertEqual(mt.Metafits.pol, 'X')

	def test_variance(self):
		mt = m.Metric(calfile, metafits = metafits)
		variance = mt.variance()
		self.assertEqual(variance.shape, (1, 3, 4))
		np.testing.assert_almost_equal(variance[0, 0, :], np.array([0.00363333, 0.00079564, 0.00083943, 0.00451215]))

	def test_variance_for_tilepair(self):
		mt = m.Metric(calfile, metafits = metafits)
		variance = mt.variance_for_tilepair((102, 104))
		self.assertEqual(variance.shape, (1, 1, 4))
		np.testing.assert_almost_equal(variance[0, 0, :], np.array([2.1214463e-02, 2.7344371e-20, 2.7173171e-20, 2.0473793e-02]))	

	def test_variance_across_tiles(self):
		mt = m.Metric(calfile, metafits = metafits)
		variance = mt.variance_across_tiles()
		self.assertEqual(variance.shape, (1, 768, 4))
		np.testing.assert_almost_equal(variance[0, 100, :], np.array([0.00396506, 0.00106192, 0.00126686, 0.00946672]))

	def test_variance_across_baselines(self):
		mt = m.Metric(calfile, metafits = metafits)
		var = mt.variance_across_baselines(250.)
		self.assertEqual(var.shape, (1, 768, 4))
		np.testing.assert_almost_equal(var[0, 100, :], np.array([0., 0., 0., 0.]))
		var = mt.variance_across_baselines(600.)
		np.testing.assert_almost_equal(var[0, 100, :], np.array([3.83073253e-03, 1.69371562e-07, 5.45765118e-06, 1.23210255e-01]))

	def test_variance_for_baselines_less_than(self):
		mt = m.Metric(calfile, metafits = metafits)
		bls, variances = mt.variance_for_baselines_less_than(250)
		self.assertEqual(bls, [(103, 102)])
		self.assertEqual(variances.shape, (1, 1, 4))
		np.testing.assert_almost_equal(variances, np.array([[[1.20606404e-02, 2.27002206e-20, 2.50099609e-20, 1.36930794e-02]]]))

	def test_skewness_across_baselines(self):
		mt = m.Metric(calfile, metafits = metafits)
		skewness = mt.skewness_across_baselines(600)
		self.assertEqual(skewness.shape, (1, 4))
		np.testing.assert_almost_equal(skewness, np.array([[0., 0., 0., 0.]]))

	def test_gains_for_receiver(self):
		mt = m.Metric(calfile, metafits = metafits)
		tile_ids, gains = mt.gains_for_receiver(10, norm = True)
		self.assertEqual(len(tile_ids), 3)
		self.assertTrue((tile_ids == np.array(['Tile104', 'Tile103', 'Tile102'])).all())
		self.assertEqual(gains.shape[1], len(tile_ids))
		expected = np.array([-0.65445423+0.9396148j ,  0.04411516-0.05776248j,
        					0.05096334+0.02108187j, -0.24661255+1.201722j  ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :] , expected)

	def test_mean_for_receiver(self):
		mt = m.Metric(calfile, metafits = metafits)
		tile_ids, mean_gains = mt.mean_for_receiver(10)
		self.assertEqual(len(tile_ids), 3)
		self.assertTrue((tile_ids == np.array(['Tile104', 'Tile103', 'Tile102'])).all())
		self.assertEqual(mean_gains.shape, (1, 768, 4))
		expected = np.array([1.0574108 , 0.04586671, 0.04706821, 1.0922133])
		np.testing.assert_almost_equal(mean_gains[0, 100, :], expected)

	def test_median_for_receiver(self):
		mt = m.Metric(calfile, metafits = metafits)
		tile_ids, median_gains = mt.median_for_receiver(10)
		self.assertEqual(len(tile_ids), 3)
		self.assertTrue((tile_ids == np.array(['Tile104', 'Tile103', 'Tile102'])).all())
		self.assertEqual(median_gains.shape, (1, 768, 4))
		expected = np.array([1.0271622 , 0.06491827, 0.05515167, 1.0498743])
		np.testing.assert_almost_equal(median_gains[0, 100, :], expected)

	def test_rms_for_receiver(self):
		mt = m.Metric(calfile, metafits = metafits)
		tile_ids, rms_gains = mt.rms_for_receiver(10)
		self.assertEqual(len(tile_ids), 3)
		self.assertTrue((tile_ids == np.array(['Tile104', 'Tile103', 'Tile102'])).all())
		self.assertEqual(rms_gains.shape, (1, 768, 4))
		expected = np.array([1.0592841 , 0.05626436, 0.05901079, 1.0965384 ])
		np.testing.assert_almost_equal(rms_gains[0, 100, :], expected)

	def test_var_for_receiver(self):
		mt = m.Metric(calfile, metafits = metafits)
		tile_ids, var_gains = mt.var_for_receiver(10)
		self.assertEqual(len(tile_ids), 3)
		self.assertTrue((tile_ids == np.array(['Tile104', 'Tile103', 'Tile102'])).all())
		self.assertEqual(var_gains.shape, (1, 768, 4))
		expected = np.array([0.00396506, 0.00106192, 0.00126686, 0.00946672])
		np.testing.assert_almost_equal(var_gains[0, 100, :], expected)


