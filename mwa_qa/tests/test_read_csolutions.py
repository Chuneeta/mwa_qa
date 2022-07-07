from mwa_qa import read_csolutions as rc
from mwa_qa import read_metafits as rm
from mwa_qa.data import DATA_PATH
import unittest
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits') 
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')
hdu = astropy.io.fits.open(calfile)
exp_gains = hdu[1].data[:, :, :, ::2] + hdu[1].data[:, :, :, 1::2] * 1j
_sh = hdu[1].data.shape

class TestCsoln(unittest.TestCase):
	def test_init__(self):
		c = rc.Csoln(calfile)
		self.assertTrue(c.calfile == calfile)
		c = rc.Csoln(calfile, metafits=metafits)
		self.assertTrue(c.calfile == calfile)
		self.assertTrue(isinstance(c.Metafits, rm.Metafits))

	def test_data_hdu1(self):
		c = rc.Csoln(calfile)
		data = c.data(1)
		self.assertTrue(data.shape == hdu[1].data.shape)
		np.testing.assert_almost_equal(data, hdu[1].data)
		self.assertTrue(data.shape == hdu[1].data.shape)
		
	def test_data_hdu2(self):
		c = rc.Csoln(calfile)
		data = c.data(2)
		self.assertTrue(data.shape == hdu[2].data.shape)
		self.assertTrue((data == hdu[2].data).all())

	def test_data_hdu3(self):
		c = rc.Csoln(calfile)
		data = c.data(3)
		self.assertTrue(data.shape == hdu[3].data.shape)
		self.assertTrue((data == hdu[3].data).all())

	def test_data_hdu4(self):
		c = rc.Csoln(calfile)
		data = c.data(4)
		self.assertTrue(data.shape == hdu[4].data.shape)
		self.assertTrue((data == hdu[4].data).all())

	def test_data_hdu5(self):
		c = rc.Csoln(calfile)
		data = c.data(5)
		self.assertTrue(data.shape == hdu[5].data.shape)
		np.testing.assert_almost_equal(np.array(data), np.array(hdu[5].data))

	def test_data_hdu6(self):
		c = rc.Csoln(calfile)
		data = c.data(6)
		self.assertTrue(data.shape == hdu[6].data.shape)
		np.testing.assert_almost_equal(np.array(data), np.array(hdu[6].data))

	def header_hdu0(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(0)
		self.assertEqual(cal_hdr, hdu[0].header)

	def header_hdu1(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(1)
		self.assertEqual(cal_hdr, hdu[1].header)

	def header_hdu2(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(2)
		self.assertEqual(cal_hdr, hdu[2].header)

	def header_hdu3(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(3)
		self.assertEqual(cal_hdr, hdu[3].header)

	def header_hdu4(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(4)
		self.assertEqual(cal_hdr, hdu[4].header)

	def header_hdu5(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(5)
		self.assertEqual(cal_hdr, hdu[5].header)

	def header_hdu6(self):
		c = rc.Csoln(calfile)
		cal_hdr = c.header(6)
		self.assertEqual(cal_hdr, hdu[6].header)

	def test_real(self):
		c = rc.Csoln(calfile)
		real_part = c.gains_real()
		self.assertEqual(real_part.shape, (_sh[0], _sh[1], _sh[2], 4))
		np.testing.assert_almost_equal(real_part[0, 0, 100, :], hdu[1].data[:, :, :, ::2][0, 0, 100, :])
		
	def test_imag(self):
		c = rc.Csoln(calfile)
		imag_part = c.gains_imag()
		self.assertEqual(imag_part.shape, (_sh[0], _sh[1], _sh[2], 4))
		np.testing.assert_almost_equal(imag_part[0, 0, 100, :], hdu[1].data[:, :, :, 1::2][0, 0, 100, :])

	def test_gains(self):
		c = rc.Csoln(calfile)
		gains = c.gains()
		self.assertEqual(gains.shape, (_sh[0], _sh[1], _sh[2], 4))
		np.testing.assert_almost_equal(gains[0, 0, 100, :], exp_gains[0, 0, 100, :])

	def test_gains_shape(self):
		c = rc.Csoln(calfile)
		gains_shape = c.gains_shape()
		self.assertEqual(gains_shape, (_sh[0], _sh[1], _sh[2], 4))

	def test_freq_info(self):
		c = rc.Csoln(calfile, metafits)
		freq_inds, freqs, freq_flags = c.freqs_info()
		np.testing.assert_almost_equal(freq_inds, np.arange(0, 768))
		self.assertEqual(freqs[0], 167055000.0)
		self.assertEqual(freqs[-1], 197735000.0)
		inds = np.where(np.array(freq_flags) == 0)
		self.assertEqual(len(inds[0]), 648)
		inds = np.where(np.array(freq_flags) == 1)
		self.assertEqual(len(inds[0]), 768 - 648)

	def test_tile_info(self):
		c = rc.Csoln(calfile)
		tile_inds, tile_ids, tile_flags = c.tile_info()
		np.testing.assert_almost_equal(tile_inds, np.arange(0, 128))
		expected_tile_ids = [tl[1] for tl in hdu[3].data]
		self.assertEqual(expected_tile_ids, tile_ids)
		expected_tile_flags = np.zeros((128))
		expected_tile_flags[76] = 1
		np.testing.assert_almost_equal(tile_flags, expected_tile_flags)

	def test_gains_ind_for(self):
		c = rc.Csoln(calfile)
		ind = c.gains_ind_for('Tile011')
		self.assertTrue(ind == 0)

	def test_check_ref_tile(self):
		c = rc.Csoln(calfile)
		with self.assertRaises(Exception):
			self._check_ref_tile('Tile105')	

	def test_normalized_data(self):
		c = rc.Csoln(calfile)
		ngains = c._normalized_data(exp_gains[0, :, :, :])
		expected = np.array([ 0.71117848-0.70117164j, -0.03601556-0.02672433j,
					        0.03182042+0.02749634j, -0.72570327-1.00192601j])
		np.testing.assert_almost_equal(ngains[0, 100, :], expected)

	def test_normalized_gains(self):
		c = rc.Csoln(calfile)
		ngains = c.normalized_gains()
		expected = np.array([ 0.71117848-0.70117164j, -0.03601556-0.02672433j,
                            0.03182042+0.02749634j, -0.72570327-1.00192601j])
		np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)

	def test_select_gains(self):
		c = rc.Csoln(calfile)
		ngains = c._select_gains(norm = True)
		expected = np.array([ 0.71117848-0.70117164j, -0.03601556-0.02672433j,
                            0.03182042+0.02749634j, -0.72570327-1.00192601j])
		np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)
		ngains = c._select_gains(norm = None)
		np.testing.assert_almost_equal(ngains[0, 0, 100, :], exp_gains[0, 0, 100, :])

	def test_amplitudes(self):
		c = rc.Csoln(calfile)
		amps = c.amplitudes()
		np.testing.assert_almost_equal(amps[0, 0, 100, :], np.array([0.99870742, 0.04484763, 0.04205458, 1.23713418]))

	def test_phases(self):
		c = rc.Csoln(calfile)
		phases = c.phases()
		np.testing.assert_almost_equal(phases[0, 0, 100, :], np.array([ -44.59405224, -143.42377521,   40.83062598, -125.91612395]))

	def test_gains_for_tile(self):
		c = rc.Csoln(calfile)
		gains = c.gains_for_tile('Tile011')
		expected = np.array([ 0.71117848-0.70117164j, -0.03601556-0.02672433j,
        					0.03182042+0.02749634j, -0.72570327-1.00192601j])
		np.testing.assert_almost_equal(gains[0, 0, 100, :], expected)

	def test_gains_for_tilepair(self):
		c = rc.Csoln(calfile)
		gains_tilepair = c.gains_for_tilepair((11,12))
		expected = np.array([ 3.48835869e-01+0.95405085j, -1.12010126e-03+0.00262549j,
        					2.59994960e-03+0.00177372j,  1.22932459e+00+0.74457111j])
		np.testing.assert_almost_equal(gains_tilepair[0, 0, 100, :], expected)

	def test_gains_for_receiver(self):
		c = rc.Csoln(calfile)
		with self.assertRaises(Exception):
			c.gains_for_receiver(1)
		c = rc.Csoln(calfile, metafits = metafits)
		gains_dict = c.gains_for_receiver(1)
		self.assertEqual(list(gains_dict.keys()), ['Tile014', 'Tile013', 'Tile012', 'Tile011', 'Tile018', 'Tile017',
       'Tile016', 'Tile015'])
		np.testing.assert_almost_equal(gains_dict['Tile014'][0, 0, 100, :], c.gains_for_tile('Tile014')[0, 0, 100, :])

	def test_generate_blackmaneharris(self):
		c = rc.Csoln(calfile)
		n = 768
		bm_filter = c.blackmanharris(n)
		self.assertEqual(len(bm_filter), n)
		expected = signal.windows.blackmanharris(n)
		np.testing.assert_almost_equal(bm_filter, expected)

	def test_delays(self):
		c = rc.Csoln(calfile)
		delays = c.delays()
		self.assertEqual(len(delays), 768) 
		self.assertEqual(np.max(delays), 12467.447916666668)
		self.assertEqual(np.min(delays), 0.0)

	def test_filter_nans(self):
		pass
