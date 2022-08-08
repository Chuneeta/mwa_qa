from mwa_qa.read_image import Image
from astropy.io import fits
import unittest
import numpy as np

image = '../../test_files/1061315688_calibrated-XX-image.fits'
hdu = fits.open(image)
data = hdu[0].data
hdr = hdu[0].header
bmaj = 0.0353 if hdr['BMAJ'] == 0. else hdr['BMAJ']
bmin = 0.0318 if hdr['BMIN'] == 0. else hdr['BMIN']
bmaj_px = bmaj / np.abs(hdr['CDELT1'])
bmin_px = bmin / np.abs(hdr['CDELT2'])
barea = bmaj * bmin * np.pi / 4 / np.log(2)
bnpix = barea / (np.abs(hdr['CDELT1']) * np.abs(hdr['CDELT2']))
_sh = data.shape

srcpos = (6.4549166666666675, -26.04)


class TestImage(unittest.TestCase):
    def test__init__(self):
        img = Image(image)
        self.assertEqual(img.fitspath, image)
        self.assertEqual(img.pix_box, [100, 100])
        self.assertEqual(img.image_ID, 1061315688)
        self.assertEqual(img.obsdate, hdr['DATE-OBS'])
        self.assertEqual(img.image_size, [_sh[2], _sh[3]])
        self.assertEqual(img.xcellsize, np.abs(hdr['CDELT1']))
        self.assertEqual(img.ycellsize, np.abs(hdr['CDELT2']))
        self.assertEqual(img.beam_major, bmaj)
        self.assertEqual(img.beam_minor, bmin)
        self.assertEqual(img.beam_parallactic_angle, hdr['BPA'])
        self.assertEqual(img.beam_major_px, bmaj_px)
        self.assertEqual(img.beam_minor_px, bmin_px)
        self.assertEqual(img.beam_area, barea)
        self.assertEqual(img.beam_npix, bnpix)
        self.assertEqual(img.mean, np.nanmean(data))
        self.assertEqual(img.rms, np.sqrt(np.nanmean(data ** 2)))
        self.assertEqual(img.std, np.nanstd(data))
        self.assertEqual(img.polarization, hdr['CRVAL4'])
        self.assertEqual(img.mean_across_box, np.nanmean(
            data[0, 0, 0: 100, 0: 100]))
        self.assertEqual(img.rms_across_box, np.sqrt(
            np.nanmean(data[0, 0, 0: 100, 0: 100] ** 2)))
        self.assertEqual(img.std_across_box, np.nanstd(
            data[0, 0, 0: 100, 0: 100]))

    def test_src_pix(self):
        img = Image(image)
        src_pix = img.src_pix(srcpos)
        self.assertEqual(src_pix, (1527, 2121))

    def test_src_flux(self):
        img = Image(image)
        src_flux = img.src_flux(srcpos)
        np.testing.assert_almost_equal(
            src_flux[0], 9.296637, decimal=4)
        np.testing.assert_almost_equal(
            src_flux[1], 8.935091385417698, decimal=4)
        np.testing.assert_almost_equal(
            src_flux[2], 2.7584093, decimal=4)
        src_flux = img.src_flux(srcpos, deconvol=True)
        np.testing.assert_almost_equal(
            src_flux[0], 10.905096047396066, decimal=4)
        np.testing.assert_almost_equal(
            src_flux[1], 10.507473433991805, decimal=4)
        np.testing.assert_almost_equal(
            src_flux[2], 0.00593149258694896, decimal=4)

    def test_select_region(self):
        img = Image(image)
        select = img._select_region(srcpos, beam_const=1)
        self.assertEqual(select.dtype, 'bool')
        self.assertEqual(select.shape, (_sh[2], _sh[3]))

    def test_fit_gaussian(self):
        img = Image(image)
        gauss_par = img.fit_gaussian(srcpos, 1)
        np.testing.assert_almost_equal(
            gauss_par[0], 10.905096047396066, decimal=4)
        np.testing.assert_almost_equal(
            gauss_par[1], 10.507473433991805, decimal=4)
        np.testing.assert_almost_equal(
            gauss_par[2], 0.00593149258694896, decimal=4)
