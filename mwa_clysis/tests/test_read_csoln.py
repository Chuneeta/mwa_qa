from mwa_clysis import read_csoln as rs
from mwa_clysis.data import DATA_PATH
import nose.tools as nt
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'hypsols_1061313616.fits')
metafits = os.path.join(DATA_PATH, '1061313616.metafits')
hdr = astropy.io.fits.open(metafits)[0].header
tile_nums = [ 11,  12,  13,  14,  15,  16,  17,  18,  21,  22,  23,  24,  25,
        26,  27,  28,  31,  32,  33,  34,  35,  36,  37,  38,  41,  42,
        43,  44,  45,  46,  47,  48,  51,  52,  53,  54,  55,  56,  57,
        58,  61,  62,  63,  64,  65,  66,  67,  68,  71,  72,  73,  74,
        75,  76,  77,  78,  81,  82,  83,  84,  85,  86,  87,  88,  91,
        92,  93,  94,  95,  96,  97,  98, 101, 102, 103, 104, 105, 106,
       107, 108, 111, 112, 113, 114, 115, 116, 117, 118, 121, 122, 123,
       124, 125, 126, 127, 128, 131, 132, 133, 134, 135, 136, 137, 138,
       141, 142, 143, 144, 145, 146, 147, 148, 151, 152, 153, 154, 155,
       156, 157, 158, 161, 162, 163, 164, 165, 166, 167, 168]

class Test_Cal():
	def test__init__(self):
		cal = rs.Cal(calfile=calfile,metafits= metafits)
		nt.assert_equal(cal.calfile, calfile)
		nt.assert_equal(cal.metafits, metafits)

	def test_read_data(self):
		cal = rs.Cal(calfile, metafits)
		data = cal.read_data()
		nt.assert_equal(data.shape, (128, 768, 4))
		nt.assert_equal(data.dtype, np.complex64)

	def test_normalize_data(self):
		cal = rs.Cal(calfile, metafits)
		data_norm = cal.normalize_data()
		nt.assert_equal(data_norm.shape, (128, 768, 4))
		nt.assert_equal(data_norm.dtype, np.complex64)
		np.testing.assert_almost_equal(data_norm[0][10], np.array([ 0.7314371-6.4764827e-01j, -0.09676012-5.9422664e-04j, 0.04099014+9.1381989e-02j, -0.61782414-1.0156155e+00j]))

	def test_read_metadata(self):
		cal = rs.Cal(calfile, metafits)
		mdata = cal.read_metadata()
		nt.assert_equal(mdata.shape, (256,))
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
		tiles_exp = ['Tile{:03d}'.format(tl) for tl in tile_nums]
		nt.assert_equal(list(tiles.keys()), tiles_exp)
		nt.assert_equal(list(tiles.values()), list(np.arange(0, len(tile_nums))))

	def test_get_amps_phases(self):
		cal = rs.Cal(calfile, metafits)
		amps, phs = cal.get_amps_phases()
		nt.assert_true(amps.dtype, np.float32)
		nt.assert_true(phs.dtype, np.float32)
		nt.assert_equal(amps.shape, (128, 768, 4))
		nt.assert_equal(phs.shape, (128, 768, 4))
		np.testing.assert_almost_equal(amps[0][10], np.array([0.9769588 , 0.09676195, 0.10015418, 1.188773]))
		np.testing.assert_almost_equal(phs[0][10], np.array([-0.72471595, -3.1354516 ,  1.149142  , -2.1173146 ]))

	def test_plot_amps(self):
		pass

	def test_plot_phs(self):
		pass
		
