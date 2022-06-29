from mwa_clysis import read_metafits as rm
from mwa_clysis.data import DATA_PATH
import unittest
import numpy as np
import logging
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

	def test_read_data(self):
		m = rm.Metafits(metafits, 'X')
		data = m._read_data()
		self.assertTrue((data == hdu).all())

	def test_check_data(self):
		m = rm.Metafits(metafits, 'X')
		data = m._read_data()
		with self.assertRaises(Exception):
			m._check_data(data[0])
		with self.assertRaises(Exception):
			tempered_data = data[0:3:2]
			m._check_data(tempered_data)
		with self.assertRaises(Exception):
			m.check_data(data[0:3])

	def test_pol_index(self):	
		m = rm.Metafits(metafits, 'X')
		ind = m._pol_index(hdu, 'X')
		self.assertEqual(ind, 1)	

	def test_mdata(self):
		m = rm.Metafits(metafits, 'X')
		data = m.mdata()
		self.assertTrue((data == hdu[1::2]).all())

	def test_mhdr(self):
		m = rm.Metafits(metafits, 'X')
		mhdr = m.mhdr()
		self.assertTrue(mhdr == hdr)

	def test_nchans(self):
		m = rm.Metafits(metafits, 'X')
		nchans = m.nchans()
		self.assertEqual(nchans, 768)
	
	def test_frequencies(self):
		m = rm.Metafits(metafits, 'X')
		frequencies = m.frequencies()
		self.assertEqual(len(frequencies), hdr['NCHANS'])
		expected_frequencies = np.linspace(131 * 1.28, 154 * 1.28, 768)
		self.assertTrue((frequencies == expected_frequencies).all())

	def test_obs_time(self):
		m = rm.Metafits(metafits, 'X')
		obs_time = m.obs_time()
		self.assertEqual(obs_time, '2013-08-23T17:20:00')

	def test_int_time(self):
		m = rm.Metafits(metafits, 'X')
		int_time = m.int_time()
		self.assertEqual(int_time, 0.5)

	def test_exposure(self):
		m = rm.Metafits(metafits, 'X')
		exposure = m.exposure()
		self.assertEqual(exposure, 112)

	def test_start_gpstime(self):
		m = rm.Metafits(metafits, 'X')
		start_gpstime = m.start_gpstime()
		self.assertEqual(start_gpstime, 1061313616)

	def test_stop_gpstime(self):
		m = rm.Metafits(metafits, 'X')
		stop_gpstime = m.stop_gpstime()
		self.assertEqual(stop_gpstime, 1061313728)

	def test_eor_field(self):
		m = rm.Metafits(metafits, 'X')
		eor_field = m.eor_field()
		self.assertEqual(eor_field, 'EoR0')

	def test_az_alt(self):
		m = rm.Metafits(metafits, 'X')
		az_alt = m.az_alt()
		self.assertTrue(az_alt, (90.0, 83.19119999999999))

	def test_ha(self):
		m = rm.Metafits(metafits, 'X')
		ha = m.ha()
		self.assertEqual(ha, ' 00:30:27.04')

	def test_lst(self):
		m = rm.Metafits(metafits, 'X')
		lst = m.lst()
		self.assertEqual(lst, 348.83449394268)

	def test_phase_centre(self):
		m = rm.Metafits(metafits, 'X')
		phase_centre = m.phase_centre()
		self.assertEqual(phase_centre, (0, -27.0))

	def test_pointing(self):
		m = rm.Metafits(metafits, 'X')
		pointing = m.pointing()
		self.assertEqual(pointing, (356.2677733102371, -26.57752518599214))

	def test_delays(self):
		m = rm.Metafits(metafits, 'X')
		delays = m.delays()
		self.assertEqual(delays, '0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3')

	def test_tile_ids(self):
		m = rm.Metafits(metafits, 'X')
		tile_ids = m.tile_ids()
		self.assertEqual(tile_ids, ['Tile104', 'Tile103', 'Tile102'])

	def test_get_tile_ind(self):
		m = rm.Metafits(metafits, 'X')
		tile_ind = m.get_tile_ind('Tile104')
		self.assertEqual(tile_ind, 0)

	def test_tile_number(self):
		m = rm.Metafits(metafits, 'X')
		tile_numbers = m.tile_numbers()
		self.assertEqual(tile_numbers, [104, 103, 102])

	def test_tile_pos(self):
		m = rm.Metafits(metafits, 'X')
		tile_pos = m.tile_pos()
		expected = np.array([[-101.52999878, -585.67498779,  375.21200562],
       						[ 415.00799561, -575.55700684,  373.37399292],
       						[ 604.56799316, -489.94299316,  372.90701294]])
		np.testing.assert_almost_equal(tile_pos, expected)

	def test_get_tile_pos(self):
		m = rm.Metafits(metafits, 'X')
		tile_pos = m.get_tile_pos('Tile103')
		self.assertTrue(len(tile_pos) == 1)
		sh = tile_pos.shape
		self.assertTrue(sh[1] == 3)
		self.assertTrue(type(tile_pos) == np.ndarray)
		expected = np.array([ 415.00799561, -575.55700684,  373.37399292]) 
		np.testing.assert_almost_equal(tile_pos[0], expected)

	def test_baseline_lengths(self):
		m = rm.Metafits(metafits, 'X')
		baseline_lengths = m.baseline_lengths()
		self.assertEqual(len(list(baseline_lengths.keys())), 3)
		self.assertEqual(list(baseline_lengths.keys()), [(104, 103), (104, 102), (103, 102)])
		self.assertEqual(list(baseline_lengths.values()), [516.6370807265803, 712.5580601060333, 207.99700000582237])

	def test_get_baselines_greater_than(self):
		m = rm.Metafits(metafits, 'X')
		bls = m.get_baselines_greater_than(250)
		self.assertEqual(len(list(bls.keys())), 2)
		self.assertEqual(list(bls.keys()), [(104, 103), (104, 102)])
		self.assertEqual(list(bls.values()), [516.6370807265803, 712.5580601060333])

	def test_get_baselines_less_than(self):
		m = rm.Metafits(metafits, 'X')
		bls = m.get_baselines_less_than(250)
		self.assertEqual(len(list(bls.keys())), 1)
		self.assertEqual(list(bls.keys()), [(103, 102)])
		self.assertEqual(list(bls.values()), [207.99700000582237])

	def test_cable_flavors(self):
		m = rm.Metafits(metafits, 'X')
		ctypes, clengths = m._cable_flavors()
		self.assertTrue(len(ctypes) == 3)
		self.assertTrue(len(clengths) == 3)
		self.assertEqual(ctypes, ['LMR400', 'RG6', 'LMR400'])
		self.assertEqual(clengths, [524.0, 150.0, 400.0])

	def test_get_cable_length(self):
		m = rm.Metafits(metafits, 'X')
		clength = m.get_cable_length('Tile103')
		self.assertEqual(clength, 150.0) 

	def test_get_cable_type(self):
		m = rm.Metafits(metafits, 'X')
		ctype = m.get_cable_type('Tile103')
		self.assertEqual(ctype, 'RG6')

	def test_receivers(self):
		m = rm.Metafits(metafits, 'X')
		receivers = m.receivers()
		self.assertEqual(receivers, [10, 10, 10])

	def test_get_receiver_for(self):
		m = rm.Metafits(metafits, 'X')  
		receiver = m.get_receiver_for('Tile103')
		self.assertEqual(receiver, 10)
	
	def test_get_tiles_for_receiver(self):
		m = rm.Metafits(metafits, 'X')
		tiles = m.get_tiles_for_receiver(10)
		expected = np.array(['Tile104', 'Tile103', 'Tile102'])
		self.assertTrue((tiles == expected).all())
	
	def test_btemps(self):
		m = rm.Metafits(metafits, 'X')
		btemps = m.btemps()
		self.assertEqual(len(btemps), 3)
		self.assertEqual(round(float(btemps[0]), 2), 17.84)

if __name__=='__main__':
	unittest.main()
