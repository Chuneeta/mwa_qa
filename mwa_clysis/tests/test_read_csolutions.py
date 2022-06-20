from mwa_clysis import read_csolutions as rc
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import logging
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
hdu = astropy.io.fits.open(calfile)[1].data
hdr = astropy.io.fits.open(calfile)[0].header

class TestCsoln(unittest.TestCase):
	def test_init__(self):
		c = rc.Csoln(calfile, 'X')
		self.assertTrue(c.calfile == calfile)
		self.assertTrue(c.pol == 'X')
		c = rc.Csoln(calfile, 'y')
		self.assertTrue(c.pol, 'Y')
