from mwa_qa import image_utils as iu
from astropy.io import fits
import unittest
import os
import numpy as np

image = '../../test_files/1061315688_calibrated-XX-image.fits'
hdu = fits.open(image)
_d = hdu[0].data
_sh = _d.shape

class TestImage(unittest.TestCase):
    def test_data(self):
        data = iu.data(image)
        self.assertEqual(data.shape, (4096, 4096))

    def test_header(self):
        hdr = iu.header(image)
        self.assertTrue(hdr['SIMPLE'])
        self.assertEqual(hdr['NAXIS1'], _sh[2])
        self.assertEqual(hdr['NAXIS2'], _sh[3])

    def test_image_size(self):
        sh = iu.image_size(image)
        self.assertEqual(sh, (_sh[2], _sh[3]))

    def test_mean(self):
        mean = iu.mean(image)
        expected_mean  = np.nansum(hdu[0].data) / (_sh[2] * _sh[3])
        self.assertEqual(mean, expected_mean)

    def test_rms(self):
        rms = iu.rms(image)
        expected_rms = np.sqrt(np.nansum(_d[0, 0, :, :] ** 2) / (_sh[2] * _sh[3]))
        self.assertEqual(rms, expected_rms)

    def test_std(self):
        std = iu.std(image)
        expected_std = np.sqrt(np.nansum((_d[0, 0, :, :] - np.nanmean(_d[0, 0, :, :])) ** 2) / (_sh[2] * _sh[3]))
        np.testing.assert_almost_equal(std, expected_std)

    def test_rms_for(self):
        rms = iu.rms_for(image, 40, 40)
        expected_rms = np.sqrt(np.nansum(_d[0, 0, 0:40, 0:40] ** 2) / (40 * 40))
        np.testing.assert_almost_equal(rms, expected_rms)

    def test_std_for(self):
        std = iu.std_for(image, 40, 40)
        expected_std = np.sqrt(np.nansum((_d[0, 0, 0:40, 0:40] - np.nanmean(_d[0, 0, 0:40, 0:40])) ** 2) / (40 * 40))
        np.testing.assert_almost_equal(std, expected_std)

