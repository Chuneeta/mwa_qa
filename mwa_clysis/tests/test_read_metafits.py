from mwa_clysis import read_metafits as rm
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import astropy
import os

metafits = os.path.join(DATA_PATH, 'test.metafits')
hdu = astropy.io.fits.open(metafits)[1].data
hdr = astropy.io.fits.open(metafits)[0].header

class TestMetafits(unittest.TestCase):
	def test_init__(self):
		m = rm.Metafits(metafits, 'X')
		self.assertEqual(m.metafits, metafits)
		m = rm.Metafits(metafits, 'X')
		self.assertEqual(m.pol, 'X')
		m = rm.Metafits(metafits, 'y')
		self.assertEqual(m.pol, 'Y')

	def test_check_data(self):
		m = rm.Metafits(metafits, 'X')


if __name__=='__main__':
	unittest.main()
