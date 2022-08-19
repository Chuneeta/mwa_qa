from mwa_qa.image_metrics import ImgMetrics
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
images = [image_xx, image_yy, image_v]
hdu = fits.open(images[0])
data = hdu[0].data
hdr = hdu[0].header
bmaj = 0.0353 if hdr['BMAJ'] == 0. else hdr['BMAJ']
bmin = 0.0318 if hdr['BMIN'] == 0. else hdr['BMIN']
bmaj_px = bmaj / np.abs(hdr['CDELT1'])
bmin_px = bmin / np.abs(hdr['CDELT2'])
barea = bmaj * bmin * np.pi
bnpix = barea / (np.abs(hdr['CDELT1']) * np.abs(hdr['CDELT2']))
_sh = data.shape


class TestImgMetrics(unittest.TestCase):
    def test_init__(self):
        m = ImgMetrics(images=images)
        self.assertTrue(len(m.images), 3)
        self.assertEqual(m.images[0].fitspath, image_xx)
        self.assertEqual(m.images[0].pix_box, [100, 100])
        self.assertEqual(m.images[0].image_ID, 1061315688)
        self.assertEqual(m.images[0].obsdate, hdr['DATE-OBS'])
        self.assertEqual(m.images[0].image_size, [_sh[2], _sh[3]])
        self.assertEqual(m.images[0].xcellsize, np.abs(hdr['CDELT1']))
        self.assertEqual(m.images[0].ycellsize, np.abs(hdr['CDELT2']))
        self.assertEqual(m.images[0].beam_major, bmaj)
        self.assertEqual(m.images[0].beam_minor, bmin)
        self.assertEqual(m.images[0].beam_parallactic_angle, hdr['BPA'])
        self.assertEqual(m.images[0].beam_major_px, bmaj_px)
        self.assertEqual(m.images[0].beam_minor_px, bmin_px)
        self.assertEqual(m.images[0].beam_area, barea)
        self.assertEqual(m.images[0].beam_npix, bnpix)
        self.assertEqual(m.images[0].mean, np.nanmean(data))
        self.assertEqual(m.images[0].rms, np.sqrt(np.nanmean(data ** 2)))
        self.assertEqual(m.images[0].std, np.nanstd(data))
        self.assertEqual(m.images[0].polarization, hdr['CRVAL4'])
        self.assertEqual(m.images[0].mean_across_box, np.nanmean(
            data[0, 0, 0: 100, 0: 100]))
        self.assertEqual(m.images[0].rms_across_box, np.sqrt(
            np.nanmean(data[0, 0, 0: 100, 0: 100] ** 2)))
        self.assertEqual(m.images[0].std_across_box, np.nanstd(
            data[0, 0, 0: 100, 0: 100]))

    def test_check_object(self):
        m = ImgMetrics(images=[])
        with self.assertRaises(Exception):
            m._check_object()

    def test_initilaize_metrics_dict(self):
        m = ImgMetrics(images=images)
        m._initialize_metrics_dict()
        self.assertTrue(m.metrics, OrderedDict)
        keys = list(m.metrics.keys())
        self.assertEqual(keys, ['PIX_BOX', 'IMAGE_SIZE', 'XX', 'YY', 'V'])

    def test_metric_keys(self):
        m = ImgMetrics(images=images)
        m._initialize_metrics_dict()
        self.assertEqual(m.metrics['PIX_BOX'], [100, 100])
        self.assertEqual(m.metrics['IMAGE_SIZE'], [4096, 4096])
        self.assertTrue(isinstance(m.metrics['XX'], OrderedDict))
        self.assertTrue(isinstance(m.metrics['YY'], OrderedDict))
        self.assertTrue(isinstance(m.metrics['V'], OrderedDict))

    def test_run_metrics(self):
        m = ImgMetrics(images=images)
        m.run_metrics()
        keys = list(m.metrics.keys())
        self.assertEqual(keys, ['PIX_BOX', 'IMAGE_SIZE', 'XX', 'YY', 'V'])
        self.assertEqual(m.metrics['PIX_BOX'], [100, 100])
        self.assertEqual(m.metrics['IMAGE_SIZE'], [4096, 4096])
        self.assertEqual(list(m.metrics['XX'].keys()), [
                         'IMAGENAME', 'IMAGE_ID',
                         'OBS-DATE', 'PKS0023_026',
                         'MEAN_ALL', 'RMS_ALL',
                         'MEAN_BOX', 'RMS_BOX'])
        self.assertEqual(list(m.metrics['YY'].keys()), [
                         'IMAGENAME', 'IMAGE_ID',
                         'OBS-DATE', 'PKS0023_026',
                         'MEAN_ALL', 'RMS_ALL',
                         'MEAN_BOX', 'RMS_BOX'])
        self.assertEqual(list(m.metrics['V'].keys()), [
                         'IMAGENAME', 'IMAGE_ID',
                         'OBS-DATE', 'PKS0023_026',
                         'MEAN_ALL', 'RMS_ALL',
                         'MEAN_BOX', 'RMS_BOX'])
        self.assertEqual(m.metrics['XX']['IMAGENAME'], m.images[0].fitspath)
        self.assertEqual(m.metrics['XX']['MEAN_ALL'], m.images[0].mean)
        self.assertEqual(m.metrics['XX']['RMS_ALL'], m.images[0].rms)
        self.assertEqual(m.metrics['XX']['MEAN_BOX'],
                         m.images[0].mean_across_box)
        self.assertEqual(m.metrics['XX']['RMS_BOX'],
                         m.images[0].rms_across_box)
        self.assertEqual(m.metrics['XX']['IMAGE_ID'], m.images[0].image_ID)
        self.assertEqual(m.metrics['XX']['OBS-DATE'], m.images[0].obsdate)
        self.assertTrue(isinstance(
            m.metrics['XX']['PKS0023_026'], OrderedDict))
        self.assertEqual(list(m.metrics['XX']['PKS0023_026'].keys()), [
                         'PEAK_FLUX', 'INT_FLUX'])
        self.assertEqual(m.metrics['XX']['PKS0023_026']
                         ['PEAK_FLUX'], np.float32(9.296637))
        self.assertEqual(m.metrics['XX']['PKS0023_026']
                         ['INT_FLUX'], 1.9123037369524847)
        self.assertEqual(m.metrics['YY']['IMAGENAME'], m.images[1].fitspath)
        self.assertEqual(m.metrics['YY']['MEAN_ALL'], m.images[1].mean)
        self.assertEqual(m.metrics['YY']['RMS_ALL'], m.images[1].rms)
        self.assertEqual(m.metrics['YY']['MEAN_BOX'],
                         m.images[1].mean_across_box)
        self.assertEqual(m.metrics['YY']['RMS_BOX'],
                         m.images[1].rms_across_box)
        self.assertEqual(m.metrics['YY']['IMAGE_ID'], m.images[1].image_ID)
        self.assertEqual(m.metrics['YY']['OBS-DATE'], m.images[1].obsdate)
        self.assertTrue(isinstance(
            m.metrics['YY']['PKS0023_026'], OrderedDict))
        self.assertEqual(list(m.metrics['YY']['PKS0023_026'].keys()), [
                         'PEAK_FLUX', 'INT_FLUX'])
        self.assertEqual(m.metrics['YY']['PKS0023_026']
                         ['PEAK_FLUX'], np.float32(10.095679))
        self.assertEqual(m.metrics['YY']['PKS0023_026']
                         ['INT_FLUX'], 2.0718385039782916)
        self.assertEqual(m.metrics['V']['IMAGENAME'], m.images[2].fitspath)
        self.assertEqual(m.metrics['V']['MEAN_ALL'], m.images[2].mean)
        self.assertEqual(m.metrics['V']['RMS_ALL'], m.images[2].rms)
        self.assertEqual(m.metrics['V']['MEAN_BOX'],
                         m.images[2].mean_across_box)
        self.assertEqual(m.metrics['V']['RMS_BOX'],
                         m.images[2].rms_across_box)
        self.assertEqual(m.metrics['V']['IMAGE_ID'], m.images[2].image_ID)
        self.assertEqual(m.metrics['V']['OBS-DATE'], m.images[2].obsdate)
        self.assertTrue(isinstance(
            m.metrics['V']['PKS0023_026'], OrderedDict))
        self.assertEqual(list(m.metrics['V']['PKS0023_026'].keys()), [
                         'PEAK_FLUX', 'INT_FLUX'])
        self.assertEqual(m.metrics['V']['PKS0023_026']
                         ['PEAK_FLUX'], np.float32(0.023942769))
        self.assertEqual(m.metrics['V']['PKS0023_026']
                         ['INT_FLUX'], 0.008755629179907626)

    def test_write_to(self):
        m = ImgMetrics(images=images)
        m.run_metrics()
        outfile = images[0].replace('.fits', '_metrics.json')
        m.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        m.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
