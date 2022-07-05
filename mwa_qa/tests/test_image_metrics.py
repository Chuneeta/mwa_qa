from mwa_qa import image_metrics as im
from collections import OrderedDict
from astropy.io import fits
import unittest
import os
import numpy as np

image_xx = '../../test_files/1061315688_calibrated-XX-image.fits'
image_xy = '../../test_files/1061315688_calibrated-XY-image.fits'
image_yx = '../../test_files/1061315688_calibrated-XYi-image.fits'
image_yy = '../../test_files/1061315688_calibrated-YY-image.fits'
image_v = '../../test_files/1061315688_calibrated-V-image.fits'
images = [image_xx, image_xy, image_yx, image_yy, image_v]
pols = ['XX', 'XY', 'YX', 'YY', 'V']
hdu = fits.open(images[0])
_d = hdu[0].data
_sh = _d.shape

class TestImgMetrics(unittest.TestCase):
	def test_init__(self):
		m = im.ImgMetrics(images = images, pols = pols)
		self.assertEqual(m.images, images)
		self.assertEqual(m.pols, pols)
		self.assertEqual(m.images[0], images[0])

	def test_check_object(self):
		m = im.ImgMetrics(images = [], pols = pols)
		with self.assertRaises(Exception):
			m._check_object()
		m = im.ImgMetrics(images = images, pols = pols[0:3])
		with self.assertRaises(Exception):
			m._check_object()

	def test_get_index(self):
		m = im.ImgMetrics(images = images, pols = pols)
		ind = m._get_index('XX')
		self.assertTrue(ind == 0)

	def test_initilaize_metrics_dict(self):
		m = im.ImgMetrics(images = images, pols = pols)
		m._initialize_metrics_dict([40, 40])
		self.assertTrue(m.metrics, OrderedDict)
		keys = list(m.metrics.keys())
		self.assertEqual(keys, ['noise_box', 'XX', 'XY', 'YX', 'YY', 'V', 'XX_YY', 'V_XX', 'V_YY']) 

	def test_metric_keys(self):
		m = im.ImgMetrics(images = images, pols = pols)
		m._initialize_metrics_dict([40, 40])
		self.assertEqual(m.metrics['noise_box'], [40, 40])
		self.assertTrue(isinstance(m.metrics['XX'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['XY'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['YX'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['YY'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['XX_YY'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['V_XX'], OrderedDict))
		self.assertTrue(isinstance(m.metrics['V_YY'], OrderedDict))

	def test_run_metrics(self):
		m = im.ImgMetrics(images = images, pols = pols)
		m.run_metrics()
		dxx = fits.open(images[0])[0].data
		dyy = fits.open(images[3])[0].data
		dv = fits.open(images[4])[0].data
		rms_xx = np.sqrt(np.nansum(dxx[0, 0, :, :] ** 2) / (_sh[2] * _sh[3]))
		rms_yy = np.sqrt(np.nansum(dyy[0, 0, :, :] ** 2) / (_sh[2] * _sh[3]))
		rms_v = np.sqrt(np.nansum(dv[0, 0, :, :] ** 2) / (_sh[2] * _sh[3]))
		std_xx = np.sqrt(np.nansum((dxx[0, 0, :, :] - np.nanmean(_d[0, 0, :, :])) ** 2) / (_sh[2] * _sh[3]))
		std_yy = np.sqrt(np.nansum((dyy[0, 0, :, :] - np.nanmean(_d[0, 0, :, :])) ** 2) / (_sh[2] * _sh[3]))
		std_v = np.sqrt(np.nansum((dv[0, 0, :, :] - np.nanmean(_d[0, 0, :, :])) ** 2) / (_sh[2] * _sh[3]))
		rms_xx_box = np.sqrt(np.nansum(dxx[0, 0, 0:100, 0:100] ** 2) / (100 * 100))
		rms_yy_box = np.sqrt(np.nansum(dyy[0, 0, 0:100, 0:100] ** 2) / (100 * 100))
		rms_v_box = np.sqrt(np.nansum(dv[0, 0, 0:100, 0:100] ** 2) / (100 * 100))	
		std_xx_box = np.sqrt(np.nansum((dxx[0, 0, 0:100, 0:100] - np.nanmean(_d[0, 0, 0:100, 0:100])) ** 2) / (100 * 100))
		std_yy_box = np.sqrt(np.nansum((dyy[0, 0, 0:100, 0:100] - np.nanmean(_d[0, 0, 0:100, 0:100])) ** 2) / (100 * 100))
		std_v_box = np.sqrt(np.nansum((dv[0, 0, 0:100, 0:100] - np.nanmean(_d[0, 0, 0:100, 0:100])) ** 2) / (100 * 100))
		np.testing.assert_almost_equal(m.metrics['XX']['std_all'], std_xx)
		np.testing.assert_almost_equal(m.metrics['XX']['rms_all'], rms_xx)
		np.testing.assert_almost_equal(m.metrics['XX']['rms_box'], rms_xx_box)
		np.testing.assert_almost_equal(m.metrics['XX']['std_box'], std_xx_box)
		np.testing.assert_almost_equal(m.metrics['XX_YY']['rms_ratio_all'], rms_xx / rms_yy)
		np.testing.assert_almost_equal(m.metrics['XX_YY']['std_ratio_all'], std_xx / std_yy)
		np.testing.assert_almost_equal(m.metrics['XX_YY']['rms_ratio_box'], rms_xx_box / rms_yy_box)
		np.testing.assert_almost_equal(m.metrics['XX_YY']['std_ratio_box'], std_xx_box / std_yy_box, decimal = 2)
		np.testing.assert_almost_equal(m.metrics['V_XX']['rms_ratio_all'], rms_v / rms_xx)
		np.testing.assert_almost_equal(m.metrics['V_XX']['std_ratio_all'], std_v / std_xx)
		np.testing.assert_almost_equal(m.metrics['V_XX']['rms_ratio_box'], rms_v_box / rms_xx_box)
		np.testing.assert_almost_equal(m.metrics['V_XX']['std_ratio_box'], std_v_box / std_xx_box, decimal = 2)
		np.testing.assert_almost_equal(m.metrics['V_YY']['rms_ratio_all'], rms_v / rms_yy)
		np.testing.assert_almost_equal(m.metrics['V_YY']['std_ratio_all'], std_v / std_yy)
		np.testing.assert_almost_equal(m.metrics['V_YY']['rms_ratio_box'], rms_v_box / rms_yy_box)
		np.testing.assert_almost_equal(m.metrics['V_YY']['std_ratio_box'], std_v_box / std_yy_box, decimal = 2)

	def test_write_to(self):
		m = im.ImgMetrics(images = images, pols = pols)
		m.run_metrics()
		outfile = images[0].replace('.fits', '_metrics.json')
		m.write_to()
		self.assertTrue(os.path.exists(outfile))
		os.system('rm -rf {}'.format(outfile))
		outfile = 'metrics.json'
		m.write_to(outfile = outfile)
		self.assertTrue(os.path.exists(outfile))
		os.system('rm -rf {}'.format(outfile))
