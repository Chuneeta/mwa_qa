from mwa_clysis import get_stats as gs
from mwa_clysis.data import DATA_PATH
import nose.tools as nt
import collections
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061313616.fits')
metafits = os.path.join(DATA_PATH, 'test.metafits')

class Test_Stats():
	def test__init__(self):
		stats = gs.Stats()
		nt.assert_false(stats.calfile)
		nt.assert_false(stats.metafits)
		stats = gs.Stats(calfile, metafits)
		nt.assert_equal(stats.calfile, calfile)
		nt.assert_equal(stats.metafits, metafits)

	def test_eval_mean(self):
		stats = gs.Stats(calfile, metafits)
		mean = stats.eval_mean()
		nt.assert_equal(mean.dtype, np.float32)
		nt.assert_equal(mean.shape, (768, 4))	
		np.testing.assert_almost_equal(mean[10], np.array([1.1031526 , 0.03025187, 0.03609714, 1.1285845]))

	def test_eval_median(self):
		stats = gs.Stats(calfile, metafits)
		median = stats.eval_median()
		nt.assert_equal(median.dtype, np.float32)
		nt.assert_equal(median.shape, (768, 4))
		np.testing.assert_almost_equal(median[10], np.array([1.1503475 , 0.04137709, 0.00660951, 1.1803353]))

	def test_eval_rms(self):
		stats = gs.Stats(calfile, metafits)
		rms = stats.eval_rms()
		nt.assert_equal(rms.dtype, np.float32)
		nt.assert_equal(rms.shape, (768, 4))
		np.testing.assert_almost_equal(rms[10], np.array([1.1055671 , 0.03719454, 0.05882997, 1.1322874]))

	def test_eval_var(self):
		stats = gs.Stats(calfile, metafits)
		var = stats.eval_var()
		nt.assert_equal(var.dtype, np.float32)
		nt.assert_equal(var.shape, (768, 4))
		np.testing.assert_almost_equal(var[10], np.array([0.00533303, 0.00046826, 0.00215796, 0.00837184]))

	def test_plot_stats(self):
		pass
	
	def test_fit_polynomial(self):
		stats = gs.Stats(calfile, metafits)
		pol, tile, deg = 'XX', 103, 3
		poly = stats.fit_polynomial(pol, tile, deg)
		nt.assert_equal(len(poly), deg + 1)
		nt.assert_equal(poly.dtype, np.float64)
		np.testing.assert_almost_equal(poly, np.array([-3.60099327e-06,  1.95413094e-03, -3.56095087e-01,  2.28547963e+01]))
		poly = stats.fit_polynomial(pol.lower(), tile, deg)
		np.testing.assert_almost_equal(poly, np.array([-3.60099327e-06,  1.95413094e-03, -3.56095087e-01,  2.28547963e+01]))	

	def test_get_fit_params(self):
		stats = gs.Stats(calfile, metafits)
		pol, tile, deg = 'XX', 103, 3
		fit_params = stats.get_fit_params(pol, deg=deg)
		nt.assert_equal(type(fit_params), collections.OrderedDict)
		nt.assert_equal(len(list(fit_params.keys())), 3)
		nt.assert_equal(list(fit_params.keys()), ['Tile102', 'Tile103', 'Tile104'])
		nt.assert_equal(len(list(fit_params.values())[0]), deg + 2)
		np.testing.assert_almost_equal(list(fit_params.values())[0], np.array([1.07878090e-05, -5.88788389e-03,  1.06816955e+00, -6.33249086e+01, 2.06101671e+00]))

	def test_plot_fit_soln(self):
		pass

	def test_plot_fit_err(self):
		pass

	def test_cal_fit_chisq(self):
		pass 
