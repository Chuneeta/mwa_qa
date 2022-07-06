from mwa_qa import read_uvfits as ru
from astropy.io import fits
import numpy as np
import unittest

uvfits = '../../test_files/1061315688_cal.uvfits'
hdu = fits.open(uvfits)
data0, data1 = hdu[0].data, hdu[1].data
hdr0, hdr1 = hdu[0].header, hdu[1].header

class TestUVfits(unittest.TestCase):
	def test__init__(self):
		uvf = ru.UVfits(uvfits)
		self.assertEqual(uvf.uvfits, uvfits)
		self.assertEqual(len(uvf._dgroup), len(data0))
		self.assertEqual(uvf.Ntiles, len(data1))
		self.assertEqual(uvf.Nbls, 8128)
		_sh = uvf._dgroup[0][5].shape
		self.assertEqual(uvf.Nfreqs, _sh[2])
		self.assertEqual(uvf.Npols, _sh[3])

	def test_read_dgroup(self):
		uvf = ru.UVfits(uvfits)
		dgroup = uvf._read_dgroup()
		self.assertEqual(len(dgroup), len(data0))
		self.assertTrue((dgroup == data0).all())

	def test_header(self):
		uvf = ru.UVfits(uvfits)
		header = uvf._header()
		self.assertEqual(header, hdr0)

	def test_tile_info(self):
		uvf = ru.UVfits(uvfits)
		tile_info = uvf._tile_info()
		self.assertTrue((tile_info ==  data1).all())

	def test_tile_ids(self):
		uvf = ru.UVfits(uvfits)
		tile_ids = uvf.tile_ids()
		self.assertEqual(len(tile_ids), uvf.Ntiles)
		self.assertEqual(tile_ids[0], 'Tile011')
		self.assertEqual(tile_ids[-1], 'Tile168')

	def test_tile_numbers(self):
		uvf = ru.UVfits(uvfits)
		tile_numbers = uvf.tile_numbers()
		self.assertEqual(len(tile_numbers), uvf.Ntiles)
		self.assertEqual(tile_numbers[0], 11)
		self.assertEqual(tile_numbers[-1], 168)

	def test_tile_labels(self):
		uvf = ru.UVfits(uvfits)
		tile_labels = uvf.tile_labels()
		self.assertEqual(len(tile_labels), uvf.Ntiles)
		self.assertEqual(tile_labels[0], 1)
		self.assertEqual(tile_labels[-1], 128)

	def test_group_count(self):
		uvf = ru.UVfits(uvfits)
		gcount = uvf.group_count()
		self.assertEqual(gcount, len(uvf._dgroup))

	def test_baselines(self):
		uvf = ru.UVfits(uvfits)
		baselines = uvf.baselines()
		self.assertEqual(len(baselines), 219456)
		self.assertEqual(len(np.unique(np.array(baselines))), 8128)
		self.assertEqual(baselines[0], 257.)
		self.assertEqual(baselines[-1], 32896.)

	def test_encode_baseline(self):
		uvf = ru.UVfits(uvfits)
		bl = uvf._encode_baseline(1, 1)
		self.assertEqual(bl, 257)

	def test_decode_baseline(self):
		uvf = ru.UVfits(uvfits)
		ant_labels = uvf._decode_baseline(257)
		self.assertEqual(ant_labels, (1, 1))

	def test_label_to_tile(self):
		uvf = ru.UVfits(uvfits)
		tile_number = uvf._label_to_tile(1)
		self.assertEqual(tile_number, 11)
	
	def test_tile_to_label(self):
		uvf = ru.UVfits(uvfits)
		tile_label = uvf._tile_to_label(11)
		self.assertEqual(tile_label, 1)

	def test_indices_for_tilepair(self):
		uvf = ru.UVfits(uvfits)
		ind = uvf._indices_for_tilepair((11, 11))
		expected = np.array([     0,   8128,  16256,  24384,  32512,  40640,  48768,  56896,
        					65024,  73152,  81280,  89408,  97536, 105664, 113792, 121920,
       						130048, 138176, 146304, 154432, 162560, 170688, 178816, 186944,
       						195072, 203200, 211328])
		np.testing.assert_equal(ind, expected)

	def antpairs(self):
		pass

	def test_uvw(self):
		uvf = ru.UVfits(uvfits)
		uvw = uvf.uvw()
		self.assertEqual(uvw.shape, (3, 219456))
		np.testing.assert_almost_equal(uvw[:, 0], np.array([0., 0., 0.]))
		np.testing.assert_almost_equal(uvw[:, 1], np.array([-54.25750702,  -5.52497496,  -2.51820893]))

	def test_pols(self):
		uvf = ru.UVfits(uvfits)
		pols = uvf.pols()
		self.assertEqual(pols, ['XX', 'XY', 'YX', 'YY'])
		# need to check the raises

	def test_data_for_tilepair(self):
		uvf = ru.UVfits(uvfits)
		data = uvf.data_for_tilepair((11, 11))
		expected = data0[0][5][0, 0, :, 0, 0] + data0[0][5][0, 0, :, 0, 1] * 1j
		np.testing.assert_almost_equal(data[0, :, 0], expected)

	def test_data_for_tilepairpol(self):
		uvf = ru.UVfits(uvfits)
		data = uvf.data_for_tilepairpol((11, 11, 'XX'))
		expected = data0[0][5][0, 0, :, 0, 0] + data0[0][5][0, 0, :, 0, 1] * 1j
		np.testing.assert_almost_equal(data[0, :], expected)
