from mwa_clysis import read_csoln as rs
from mwa_clysis.data import DATA_PATH
import nose.tools as nt
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
metafits = os.path.join(DATA_PATH, 'test.metafits')
hdr = astropy.io.fits.open(metafits)[0].header

class Test_Cal():
	def test__init__(self):
		cal = rs.Cal(calfile=calfile,metafits= metafits)
		nt.assert_equal(cal.calfile, calfile)
		nt.assert_equal(cal.metafits, metafits)

	def test_read_data(self):
		cal = rs.Cal(calfile, metafits)
		data = cal.read_data()
		nt.assert_equal(data.shape, (3, 768, 4))
		nt.assert_equal(data.dtype, np.complex64)

	def test_normalize_data(self):
		cal = rs.Cal(calfile, metafits)
		data_norm = cal.normalize_data()
		nt.assert_equal(data_norm.shape, (3, 768, 4))
		nt.assert_equal(data_norm.dtype, np.complex64)
		np.testing.assert_almost_equal(data_norm[0][10], np.array([-0.62358   +9.6666819e-01j,  0.04937318+7.2630122e-04j, -0.05393033-8.6201683e-02j, -0.13457748+1.1978822e+00j]))

	def test_read_metadata(self):
		cal = rs.Cal(calfile, metafits)
		mdata = cal.read_metadata()
		nt.assert_equal(mdata.shape, (6,))
		nt.assert_equal(type(mdata[0][3]), str)

	def test_read_metaheader(self):
		cal = rs.Cal(calfile, metafits)
		mhdr = cal.read_metaheader()
		nt.assert_true(type(mhdr), astropy.io.fits.header.Header)

	def test_get_nchans(self):
		cal = rs.Cal(calfile, metafits)
		nchans = cal.get_nchans()
		nt.assert_equal(nchans, hdr['NCHANS'])

	def test_get_freqs(self):
		cal = rs.Cal(calfile, metafits)
		freqs = cal.get_freqs()
		nt.assert_equal(len(freqs), hdr['NCHANS'])
		nt.assert_equal(freqs.dtype, np.float64)
	
	def test_obsdate(self):
		cal = rs.Cal(calfile, metafits)
		obsdate = cal.get_obsdate()
		nt.assert_equal(obsdate, hdr['DATE-OBS'])
	
	def test_extract_tiles(self):
		cal = rs.Cal(calfile, metafits)
		tiles = cal.extract_tiles()
		nt.assert_equal(list(tiles.keys()), ['Tile102', 'Tile103', 'Tile104'])
		nt.assert_equal(list(tiles.values()), [0, 1, 2])

	def test_get_tile_numbers(self):
		cal = rs.Cal(calfile, metafits)
		tile_nums = cal.get_tile_numbers()
		nt.assert_equal(tile_nums, [102, 103, 104])

	def test_get_amps_phases(self):
		cal = rs.Cal(calfile, metafits)
		amps, phs = cal.get_amps_phases()
		nt.assert_true(amps.dtype, np.float32)
		nt.assert_true(phs.dtype, np.float32)
		nt.assert_equal(amps.shape, (3, 768, 4))
		nt.assert_equal(phs.shape, (3, 768, 4))
		np.testing.assert_almost_equal(amps[0][10], np.array([1.1503475, 0.0493785, 0.1016819, 1.2054181]))
		np.testing.assert_almost_equal(phs[0][10], np.array([ 2.1437063 ,  0.01470938, -2.1298482 ,  1.6826733]))

	def test_plot_amps(self):
		pass

	def test_plot_phs(self):
		pass
		
