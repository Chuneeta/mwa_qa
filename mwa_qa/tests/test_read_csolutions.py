from mwa_clysis import read_csolutions as rc
from mwa_clysis import read_metafits as rm
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
metafits = os.path.join(DATA_PATH, 'test.metafits')
hdu = astropy.io.fits.open(calfile)[1].data
hdr = astropy.io.fits.open(calfile)[0].header

class TestCsoln(unittest.TestCase):
	def test_init__(self):
		c = rc.Csoln(calfile)
		self.assertTrue(c.calfile == calfile)
		c = rc.Csoln(calfile, metafits=metafits)
		self.assertTrue(c.calfile == calfile)
		self.assertTrue(isinstance(c.Metafits, rm.Metafits))

	def test_read_data(self):
		c = rc.Csoln(calfile)
		data = c._read_data()
		np.testing.assert_almost_equal(data, hdu)
		self.assertTrue(data.shape == (1, 3, 768, 8))

	def test_real(self):
		c = rc.Csoln(calfile)
		real_part = c.real()
		self.assertTrue(real_part.shape == (1, 3, 768, 4))
		expected = np.array([-0.03619801,  0.02484267,  0.00525203,  0.23554221])
		np.testing.assert_almost_equal(real_part[0, 0, 100, :], expected)
		
	def test_imag(self):
		c = rc.Csoln(calfile)
		imag_part = c.imag()
		self.assertTrue(imag_part.shape == (1, 3, 768, 4))
		expected = np.array([0.8295782 , 0.00862688, 0.00732957, 0.8980131 ])
		np.testing.assert_almost_equal(hdu[0, 0, 100, 1::2], expected)

	def test_gains(self):
		c = rc.Csoln(calfile)
		gains = c.gains()
		self.assertTrue(gains.shape == (1, 3, 768, 4))
		expected = np.array([-0.03619801+0.8295782j ,  0.02484267+0.00862688j,
        					0.00525203+0.00732957j,  0.23554221+0.8980131j ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)

	def test_gains_shape(self):
		c = rc.Csoln(calfile)
		gains_shape = c.gains_shape()
		self.assertEqual(gains_shape, (1, 3, 768, 4))

	def header(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header()
		self.assertEqual(cal_hdr, hdr)

	def test_check_ref_tile_data(self):
		c = rc.Csoln(calfile, metafits=metafits)
		with self.assertRaises(Exception):
			self_check_ref_tile_data(2)

	def test_normalized_data(self):
		c = rc.Csoln(calfile)
		gains = c.gains()
		ndata = c._normalized_data(gains[0])
		self.assertTrue(np.allclose(ndata[-1, :, 0][~np.isnan(ndata[-1, :, 0])], 1.0))
		self.assertTrue(np.allclose(ndata[-1, :, 3][~np.isnan(ndata[-1, :, 3])], 1.0))

	def test_normalized_gains(self):
		c = rc.Csoln(calfile)
		ngains = c.normalized_gains()
		self.assertTrue(ngains.shape == (1, 3, 768, 4))
		self.assertTrue(np.isnan(ngains[0, 0, 0, :]).all())
		expected = np.array([-0.6544542 +0.9396147j ,  0.04411516-0.05776248j,
        				0.05096334+0.02108187j, -0.24661253+1.2017221j ])
		np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)

	def test_select_gains(self):
		c = rc.Csoln(calfile)
		gains = c._select_gains(norm = True)
		expected = np.array([-0.6544542 +0.9396147j ,  0.04411516-0.05776248j,
                        	0.05096334+0.02108187j, -0.24661253+1.2017221j ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)
		gains = c._select_gains(norm = False)
		expected = np.array([-0.03619801+0.8295782j ,  0.02484267+0.00862688j,
                      		0.00525203+0.00732957j,  0.23554221+0.8980131j ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)

	def test_amplitudes(self):
		c = rc.Csoln(calfile)
		amps = c.amplitudes()
		self.assertTrue(amps.shape == (1, 3, 768, 4))
		self.assertTrue(np.isnan(np.abs(amps[0, 0, 0, :])).all())
		expected = np.array([1.1450703 , 0.07268185, 0.05515167, 1.2267656 ])
		np.testing.assert_almost_equal(amps[0, 0, 100, :], expected, decimal = 6) 
		amps = c.amplitudes(norm = False)
		expected = np.array([0.83036757, 0.02629794, 0.00901701, 0.92838985])		
		np.testing.assert_almost_equal(amps[0, 0, 100, :], expected, decimal = 6)

	def test_phases(self):
		c = rc.Csoln(calfile)
		phases = c.phases()
		self.assertTrue(phases.shape == (1, 3, 768, 4))
		self.assertTrue(np.isnan(np.abs(phases[0, 0, 0, :])).all())
		expected = np.array([124.85773 , -52.629803,  22.47328 , 101.597 ])
		np.testing.assert_almost_equal(phases[0, 0, 100, :], expected, decimal = 3)
		phases = c.phases(norm = False)
		expected = np.array([92.498474, 19.150087, 54.376358, 75.30281 ])
		np.testing.assert_almost_equal(phases[0, 0, 100, :], expected, decimal = 6)

	def test_gains_for_tile(self):
		c = rc.Csoln(calfile, metafits=metafits)
		gains = c.gains_for_tile('Tile104')
		self.assertTrue(gains.shape == (1, 1, 768, 4))
		expected = np.array([-0.6544542 +0.9396147j ,  0.04411516-0.05776248j,
        					0.05096334+0.02108187j, -0.24661253+1.2017221j ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)
		gains = c.gains_for_tile('Tile104', norm = False)
		expected = np.array([-0.03619801+0.8295782j ,  0.02484267+0.00862688j,
        			0.00525203+0.00732957j,  0.23554221+0.8980131j ])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)

	def test_gains_for_receiver(self):
		c = rc.Csoln(calfile, metafits=metafits)
		gains_dict = c.gains_for_receiver(10)
		self.assertTrue(len(list(gains_dict.keys())) == 3)
		self.assertEqual(list(gains_dict.keys()), ['Tile104', 'Tile103', 'Tile102'])
		expected = np.array([-0.6544542 +0.9396147j ,  0.04411516-0.05776248j,
                            0.05096334+0.02108187j, -0.24661253+1.2017221j ])
		np.testing.assert_almost_equal(gains_dict['Tile104'][0, 0, 100, :], expected)
		gains_dict = c.gains_for_receiver(10, norm = False)
		expected = np.array([-0.03619801+0.8295782j ,  0.02484267+0.00862688j,
                    		0.00525203+0.00732957j,  0.23554221+0.8980131j ])
		np.testing.assert_almost_equal(gains_dict['Tile104'][0, 0, 100, :], expected)
		
	def test_gains_for_tilepair(self):
		c = rc.Csoln(calfile, metafits=metafits)
		gains_01 = c.gains_for_tilepair((102, 104))
		self.assertEqual(gains_01.shape, (1, 1, 768, 4))
		expected = np.array([-6.5445423e-01-9.3961477e-01j, -3.2572828e-10-9.1925689e-11j,
                             3.9268040e-11+9.4926615e-11j, -2.4661255e-01-1.2017220e+00j])
		np.testing.assert_almost_equal(gains_01[0, 0, 100, :], expected)
