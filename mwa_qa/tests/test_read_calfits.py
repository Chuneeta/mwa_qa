from mwa_qa.read_calfits import CalFits
from mwa_qa.data import DATA_PATH
from scipy import signal
import unittest
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits')
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')
calfile_poly = os.path.join(DATA_PATH, 'test_1061315688_poly.fits')
hdu = astropy.io.fits.open(calfile)
exp_gains = hdu[1].data[:, :, :, ::2] + hdu[1].data[:, :, :, 1::2] * 1j
_sh = hdu[1].data.shape


class TestCalFits(unittest.TestCase):
    def test__init__(self):
        c = CalFits(calfile, metafits)
        self.assertEqual(c.calfits_path, calfile)
        self.assertEqual(c.metafits_path, metafits)
        self.assertEqual(c.Metafits.metafits, metafits)
        self.assertEqual(c.Metafits.pol, 'X')
        self.assertEqual(c.gain_array.shape, (1, 128, 768, 4))
        self.assertEqual(c.start_time, hdu['TIMEBLOCKS'].data[0][0])
        self.assertEqual(c.end_time, hdu['TIMEBLOCKS'].data[0][1])
        self.assertEqual(c.average_time, hdu['TIMEBLOCKS'].data[0][2])
        self.assertEqual(c.Ntime, 1)
        self.assertEqual(c.uvcut, hdu['PRIMARY'].header['UVW_MIN'])
        self.assertEqual(c.obsid, hdu['PRIMARY'].header['OBSID'])
        self.assertEqual(c.s_thresh, hdu['PRIMARY'].header['S_THRESH'])
        self.assertEqual(c.m_thresh, hdu['PRIMARY'].header['M_THRESH'])
        np.testing.assert_equal(np.array(c.antenna), np.arange(128))
        expected_flags = np.zeros((128))
        expected_flags[76] = 1
        np.testing.assert_equal(np.array(c.antenna_flags), expected_flags)
        np.testing.assert_equal(c.annames[0], 'Tile011')
        np.testing.assert_almost_equal(c.frequency_channels, np.arange(0, 768))
        self.assertEqual(c.frequency_array[0], 167055000.0)
        self.assertEqual(c.frequency_array[-1], 197735000.0)
        inds = np.where(np.array(c.frequency_flags) == 0)
        self.assertEqual(len(inds[0]), 648)
        inds = np.where(np.array(c.frequency_flags) == 1)
        self.assertEqual(len(inds[0]), 768 - 648)
        np.testing.assert_almost_equal(
            c.gain_array[0, 0, 100, :], exp_gains[0, 0, 100, :])
        np.testing.assert_almost_equal(c.amplitudes[0, 0, 100, :], np.array(
            [0.76390112, 0.02917631, 0.0354274, 0.84958042]))
        np.testing.assert_almost_equal(c.phases[0, 0, 100, :], np.array(
            [1.58846751, 1.90056897, 0.87910414, 1.23280019]))
        np.testing.assert_almost_equal(c.convergence, hdu['RESULTS'].data)
        np.testing.assert_almost_equal(
            c.baseline_weights, hdu['BASELINES'].data)
        self.assertFalse(c.norm)
        c = CalFits(calfile, metafits, norm=True)
        self.assertTrue(c.norm)
        self.assertTrue(c.ref_antenna == 127)
        np.testing.assert_almost_equal(
            c.gain_array[0, 0, 100, :], np.array([
                0.71117848-0.70117164j, -0.03601556-0.02672433j,
                0.03182042+0.02749634j, -0.72570327-1.00192601j]))
        np.testing.assert_almost_equal(c.amplitudes[0, 0, 100, :], np.array(
            [0.99870742, 0.04484763, 0.04205458, 1.23713418]))
        np.testing.assert_almost_equal(c.phases[0, 0, 100, :], np.array(
            [-0.77831304, -2.5032171,  0.71262886, -2.19765094]))
        # polynomial testing
        c = CalFits(calfile_poly, metafits)
        self.assertEqual(c.poly_order, 9)
        np.testing.assert_almost_equal(c.poly_mse, 0.2574473278254666)

        def test_iterate_refant(self):
            c = CalFits(calfile)
            ref_antenna = c._iterate_refant()
            self.assertEqual(ref_antenna, 127)

    def test_gains_ind_for(self):
        c = CalFits(calfile)
        ind = c.gains_ind_for(1)
        self.assertTrue(ind == 1)

    def test_check_refant(self):
        c = CalFits(calfile, ref_antenna=77)
        with self.assertRaises(Exception):
            c._check_refant()

    def test_normalized_data(self):
        c = CalFits(calfile, norm=True)
        ngains = c._normalized_data(exp_gains[0, :, :, :])
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(ngains[0, 100, :], expected)

    def test_normalized_gains(self):
        c = CalFits(calfile, norm=True)
        ngains = c.normalized_gains()
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)

    def test_gains_for_antnum(self):
        c = CalFits(calfile, norm=True)
        gains = c.gains_for_antnum(0)
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(gains[0, 100, :], expected)

    def test_gains_for_antpair(self):
        c = CalFits(calfile, norm=True)
        gains_antpair = c.gains_for_antpair((0, 1))
        expected = np.array([3.48835869e-01+0.95405085j,
                             -1.12010126e-03+0.00262549j,
                             2.59994960e-03+0.00177372j,
                             1.22932459e+00+0.74457111j])
        np.testing.assert_almost_equal(gains_antpair[0, 100, :], expected)

    def test_gains_for_receiver(self):
        c = CalFits(calfile)
        with self.assertRaises(Exception):
            c.gains_for_receiver(1)
        c = CalFits(calfile, metafits_path=metafits)
        gains_array = c.gains_for_receiver(1)
        annumbers = c.Metafits.annumbers_for_receiver(1)
        self.assertEqual(gains_array.shape, (1, 8, 768, 4))
        np.testing.assert_almost_equal(
            gains_array[0, 0, 100, :],
            c.gains_for_antnum(annumbers[0])[0, 100, :])

    def test_generate_blackmanharris(self):
        c = CalFits(calfile)
        n = 768
        bm_filter = c.blackmanharris(n)
        self.assertEqual(len(bm_filter), n)
        expected = signal.windows.blackmanharris(n)
        np.testing.assert_almost_equal(bm_filter, expected)

    def test_delays(self):
        c = CalFits(calfile)
        delays = c.delays()
        self.assertEqual(len(delays), 768)
        np.testing.assert_almost_equal(np.max(delays), 12467.447916666668)
        self.assertEqual(delays[0], -12499.999999999998)

    def test_filter_nans(self):
        c = CalFits(calfile)
        gains = c.gain_array
        nonans_inds, nans_inds = c._filter_nans(gains[0, 0, :, 0])
        self.assertEqual(len(nonans_inds), 648)
        self.assertEqual(len(nans_inds), 768 - 648)
        np.testing.assert_almost_equal(nans_inds, np.array(
            [0,   1,  16,  30, 31,  32,  33,  48,  62,  63,  64,  65,  80,
             94,  95,  96,  97, 112, 126, 127, 128, 129, 144, 158, 159, 160,
             161, 176, 190, 191, 192, 193, 208, 222, 223, 224, 225, 240, 254,
             255, 256, 257, 272, 286, 287, 288, 289, 304, 318, 319, 320, 321,
             336, 350, 351, 352, 353, 368, 382, 383, 384, 385, 400, 414, 415,
             416, 417, 432, 446, 447, 448, 449, 464, 478, 479, 480, 481, 496,
             510, 511, 512, 513, 528, 542, 543, 544, 545, 560, 574, 575, 576,
             577, 592, 606, 607, 608, 609, 624, 638, 639, 640, 641, 656, 670,
             671, 672, 673, 688, 702, 703, 704, 705, 720, 734, 735, 736, 737,
             752, 766, 767]))

    def test_gains_fft(self):
        c = CalFits(calfile)
        fft_gains = c.gains_fft()
        self.assertEqual(fft_gains.shape, (1, 128, 768, 4))
        np.testing.assert_almost_equal(fft_gains[0, 0, 100, :],
                                       np.array([0.38595275-0.47685325j,
                                                 0.43570297-0.52856661j,
                                                 -0.15975999-0.36207117j,
                                                 -0.24204551-0.39744718j]))
