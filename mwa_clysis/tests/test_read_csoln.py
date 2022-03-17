import nose.tools as nt
import numpy as np
import read_csoln as rs
import os

DATA_PATH = '/Users/ridhima/Documents/curtin/calsol_analysis/data/)'
calfile = os.path.join(DATA_PATH, 'hypsols_1061313616.fits')
metafits = os.path.join(DATA_PATH, '1061313616.metafits')


def test_class_Cal():
	def test_init_(self):
		cal = rs.Cal(calfile, metafits)
		nt.assert_equal(cal.calfile,  calfile)
		nt.assert_equal(self.metafits, metafits)

	def test_read_soln():
		cal = rs.Cal(calfile, metafits)
		data = cal.read_soln()
		nt.assert_equal(type(data), 'numpy.ndarray')
		nt.assert_equal(data.shape, (1, 128, 768, 8))

	def test_read_mdata():
		cal = rs.Cal(calfile, metafits)
		
	
	def test_extract_tile_number(self):
		cal = rs.Cal(calfile, metafits)
		tiles_dict = cal.extract_tile_number()
		nt.assert_equal(len(tiles_dict), 128)
		tiles_t = np.sort(['Tile{}'.format(i) for i in range(11, 139)])
		tiles_t_dict = {x:i for i, x in enumerate(tiles)}
		np.assert_equal(tiles_dict, tiles_t_dict)

	def test_get_amps_phases(self):
		cal = rs.Cal(calfile, metafits)
		amps, phases = cal.get_amps_phases()
		nt.assert_equal(amps.shape, (128, 768, 4))
		nt.assert_equal(phases.shape, (128, 768, 4))
		nt.assert_equal(amps.dtype, 'float32')
		nt.assert_equal(phases.dtypw, 'float32')

	def plot_soln_amps(self):
		pass
