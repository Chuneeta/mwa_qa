from mwa_qa.read_calfits import CalFits
from mwa_qa.data import DATA_PATH
from scipy import signal
import unittest
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'hyp_soln_1062784992.fits')
metafits = os.path.join(DATA_PATH, '1062784992.metafits')
hdu = astropy.io.fits.open(calfile)
exp_gains = hdu[1].data[:, :, :, ::2] + hdu[1].data[:, :, :, 1::2] * 1j
_sh = hdu[1].data.shape


class TestCalFits(unittest.TestCase):
    def test__init__(self):
        c = CalFits(calfile)
        self.assertEqual(c.calfits_path, calfile)
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
        np.testing.assert_equal(np.array(c.antenna_flags), expected_flags)
        np.testing.assert_equal(c.annames[0], 'Tile011')
        np.testing.assert_almost_equal(c.frequency_channels, np.arange(0, 768))
        self.assertEqual(c.frequency_array[0], 167055000.0)
        self.assertEqual(c.frequency_array[-1], 197735000.0)
        inds = np.where(np.array(c.frequency_flags) == 0)
        self.assertEqual(len(inds[0]), 768)
        inds = np.where(np.array(c.frequency_flags) == 1)
        self.assertEqual(len(inds[0]), 0)
        np.testing.assert_almost_equal(
            c.gain_array[0, 0, 100, :], exp_gains[0, 0, 100, :])
        np.testing.assert_almost_equal(c.amplitudes[0, 0, 100, :], np.array(
            [0.82267406, 0.03973897, 0.0397497, 0.81913319]))
        np.testing.assert_almost_equal(c.phases[0, 0, 100, :], np.array(
            [1.80144346, -2.03494748,  1.64318736,  1.30800907]))
        np.testing.assert_almost_equal(c.convergence, hdu['RESULTS'].data)
        np.testing.assert_almost_equal(
            c.baseline_weights, hdu['BASELINES'].data)
        self.assertFalse(c.norm)
        c = CalFits(calfile, norm=True)
        self.assertTrue(c.norm)
        self.assertTrue(c.ref_antenna == 127)
        np.testing.assert_almost_equal(
            c.gain_array[0, 0, 100, :], np.array([0.79296893-0.62239205j,  0.01225656+0.05076904j,
                                                  0.02343749-0.05114198j, -0.44148823-1.04256351j]))
        np.testing.assert_almost_equal(c.amplitudes[0, 0, 100, :], np.array(
            [1.00805336, 0.05222757, 0.05625671, 1.13218838]))
        np.testing.assert_almost_equal(c.phases[0, 0, 100, :], np.array(
            [-0.66545834,  1.33391101, -1.14107581, -1.97136534]))
        # polynomial testing
        # c = CalFits(calfile_poly, metafits)
        # self.assertEqual(c.poly_order, 9)
        # np.testing.assert_almost_equal(c.poly_mse, 0.2574473278254666)

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
        expected = np.array([0.79296893-0.62239205j,  0.01225656+0.05076904j,
                             0.02343749-0.05114198j, -0.44148823-1.04256351j])
        np.testing.assert_almost_equal(ngains[0, 100, :], expected)

    def test_normalized_gains(self):
        c = CalFits(calfile, norm=True)
        ngains = c.normalized_gains()
        expected = np.array([0.79296893-0.62239205j,  0.01225656+0.05076904j,
                             0.02343749-0.05114198j, -0.44148823-1.04256351j])
        np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)

    def test_gains_for_antnum(self):
        c = CalFits(calfile, norm=True)
        gains = c.gains_for_antnum(0)
        expected = np.array([0.79296893-0.62239205j,  0.01225656+0.05076904j,
                             0.02343749-0.05114198j, -0.44148823-1.04256351j])
        np.testing.assert_almost_equal(gains[0, 100, :], expected)

    def test_gains_for_antpair(self):
        c = CalFits(calfile, norm=True)
        gains_antpair = c.gains_for_antpair((0, 1))
        expected = np.array([2.94906393e-01+0.89236059j, 4.08641116e-04-0.00110709j,
                             1.35256327e-04+0.00476329j, 9.51426902e-01+0.65670472j])
        np.testing.assert_almost_equal(gains_antpair[0, 100, :], expected)

    def test_gains_for_receiver(self):
        c = CalFits(calfile)
        with self.assertRaises(Exception):
            c.gains_for_receiver(1)
        c = CalFits(calfile)
        gains_array = c.gains_for_receiver(metafits, 1)
        self.assertEqual(gains_array.shape, (1, 8, 768, 4))
        np.testing.assert_almost_equal(
            gains_array[0, 0, 100, :], np.array([0.60166742+0.35781243j,  0.00932383-0.0019097j,
                                                 -0.0134085 - 0.005618j, -0.16567676+0.68909707j]))

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
        self.assertEqual(len(nonans_inds), 647)
        self.assertEqual(len(nans_inds), 768 - 647)
        np.testing.assert_almost_equal(nans_inds, np.array(
            [0,   1,  16,  30,  31,  32,  33,  48,  62,  63,  64,  65,  80,
             94,  95,  96,  97, 112, 126, 127, 128, 129, 144, 158, 159, 160,
             161, 176, 190, 191, 192, 193, 208, 222, 223, 224, 225, 240, 254,
             255, 256, 257, 272, 286, 287, 288, 289, 304, 318, 319, 320, 321,
             324, 336, 350, 351, 352, 353, 368, 382, 383, 384, 385, 400, 414,
             415, 416, 417, 432, 446, 447, 448, 449, 464, 478, 479, 480, 481,
             496, 510, 511, 512, 513, 528, 542, 543, 544, 545, 560, 574, 575,
             576, 577, 592, 606, 607, 608, 609, 624, 638, 639, 640, 641, 656,
             670, 671, 672, 673, 688, 702, 703, 704, 705, 720, 734, 735, 736,
             737, 752, 766, 767]))

    def test_gains_fft(self):
        c = CalFits(calfile)
        fft_gains = c.gains_fft()
        self.assertEqual(fft_gains.shape, (1, 128, 768, 4))
        np.testing.assert_almost_equal(fft_gains[0, 0, 100, :],
                                       np.array([-0.41472297+0.04339564j,
                                                0.00608647-0.33740265j,
                                                 0.10985828-0.18691477j,
                                                 -0.03172375-0.53670078j]))
