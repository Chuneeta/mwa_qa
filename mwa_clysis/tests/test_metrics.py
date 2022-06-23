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


