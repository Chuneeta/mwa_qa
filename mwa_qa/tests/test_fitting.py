from mwa_clysis import read_csolutions as rc
from mwa_clysis import read_metafits as rm
from mwa_clysis import metrics as m
from mwa_clysis import fitting as f
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
metafits = os.path.join(DATA_PATH, 'test.metafits')

class TestFit(unittest.TestCase):
	def test__init__(self):
		fit = f.Fit(calfile)
		self.assertEqual(fit.calfile, calfile)
		self.assertTrue(isinstance(fit.Csoln, rc.Csoln))
		self.assertTrue(isinstance(fit.Metafits, rm.Metafits))
		fit = f.Fit(calfile, metafits = metafits)
		self.assertEqual(calfile, fit.Csoln.calfile)
		self.assertEqual(metafits, fit.Csoln.Metafits.metafits)
		self.assertEqual(metafits, fit.Metafits.metafits)
		self.assertEqual('X', fit.Csoln.Metafits.pol)
		self.assertEqual('X', fit.Metafits.pol)

	def test_get_gains_for_receiver(self):
		fit = f.Fit(calfile, metafits = metafits)
		tile_ids, gains_array = fit._get_gains_for_receiver(10)
		self.assertTrue((tile_ids, ['Tile104', 'Tile103', 'Tile102']))
		self.assertEqual(gains_array.shape, (1, 3, 768, 4))
		expected = np.array([-0.65445423+0.9396148j ,  0.04411516-0.05776248j,
                            0.05096334+0.02108187j, -0.24661255+1.201722j  ])
		np.testing.assert_almost_equal(gains_array[0, 0, 100, :], expected)

	def test_average_per_receiver(self):
		fit = f.Fit(calfile, metafits = metafits)
		tile_ids, average = fit.average_per_receiver(10)
		self.assertTrue((tile_ids, ['Tile104', 'Tile103', 'Tile102'])) 
		self.assertEqual(average.shape, (1, 768, 4))
		expected = np.array([1.0574108 , 0.04586671, 0.04706821, 1.0922133])
		np.testing.assert_almost_equal(average[0, 100, :], expected)
