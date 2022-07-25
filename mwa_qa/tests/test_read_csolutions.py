from mwa_qa import read_csolutions as rc
from mwa_qa import read_metafits as rm
from mwa_qa.data import DATA_PATH
from scipy import signal
import unittest
import numpy as np
import astropy
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits')
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')
hdu = astropy.io.fits.open(calfile)
exp_gains = hdu[1].data[:, :, :, ::2] + hdu[1].data[:, :, :, 1::2] * 1j
_sh = hdu[1].data.shape


class TestCsoln(unittest.TestCase):
    def test_init__(self):
        c = rc.Csoln(calfile)
        self.assertTrue(c.calfile == calfile)
        c = rc.Csoln(calfile, metafits=metafits)
        self.assertTrue(c.calfile == calfile)
        self.assertTrue(isinstance(c.Metafits, rm.Metafits))
        self.assertFalse(c.norm)
        c = rc.Csoln(calfile, norm=True)
        self.assertTrue(c.norm)
        self.assertTrue(127)

    def test_data_hdu1(self):
        c = rc.Csoln(calfile)
        data = c.data(1)
        self.assertTrue(data.shape == hdu[1].data.shape)
        np.testing.assert_almost_equal(data, hdu[1].data)
        self.assertTrue(data.shape == hdu[1].data.shape)

    def test_data_hdu2(self):
        c = rc.Csoln(calfile)
        data = c.data(2)
        self.assertTrue(data.shape == hdu[2].data.shape)
        self.assertTrue((data == hdu[2].data).all())

    def test_data_hdu3(self):
        c = rc.Csoln(calfile)
        data = c.data(3)
        self.assertTrue(data.shape == hdu[3].data.shape)
        self.assertTrue((data == hdu[3].data).all())

    def test_data_hdu4(self):
        c = rc.Csoln(calfile)
        data = c.data(4)
        self.assertTrue(data.shape == hdu[4].data.shape)
        self.assertTrue((data == hdu[4].data).all())

    def test_data_hdu5(self):
        c = rc.Csoln(calfile)
        data = c.data(5)
        self.assertTrue(data.shape == hdu[5].data.shape)
        np.testing.assert_almost_equal(np.array(data), np.array(hdu[5].data))

    def test_data_hdu6(self):
        c = rc.Csoln(calfile)
        data = c.data(6)
        self.assertTrue(data.shape == hdu[6].data.shape)
        np.testing.assert_almost_equal(np.array(data), np.array(hdu[6].data))

    def header_hdu0(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(0)
        self.assertEqual(cal_hdr, hdu[0].header)

    def header_hdu1(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(1)
        self.assertEqual(cal_hdr, hdu[1].header)

    def header_hdu2(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(2)
        self.assertEqual(cal_hdr, hdu[2].header)

    def header_hdu3(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(3)
        self.assertEqual(cal_hdr, hdu[3].header)

    def header_hdu4(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(4)
        self.assertEqual(cal_hdr, hdu[4].header)

    def header_hdu5(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(5)
        self.assertEqual(cal_hdr, hdu[5].header)

    def header_hdu6(self):
        c = rc.Csoln(calfile)
        cal_hdr = c.header(6)
        self.assertEqual(cal_hdr, hdu[6].header)

    def test_real(self):
        c = rc.Csoln(calfile)
        real_part = c.gains_real()
        self.assertEqual(real_part.shape, (_sh[0], _sh[1], _sh[2], 4))
        np.testing.assert_almost_equal(
            real_part[0, 0, 100, :], hdu[1].data[:, :, :, ::2][0, 0, 100, :])

    def test_imag(self):
        c = rc.Csoln(calfile)
        imag_part = c.gains_imag()
        self.assertEqual(imag_part.shape, (_sh[0], _sh[1], _sh[2], 4))
        np.testing.assert_almost_equal(
            imag_part[0, 0, 100, :], hdu[1].data[:, :, :, 1::2][0, 0, 100, :])

    def test_gains(self):
        c = rc.Csoln(calfile)
        gains = c.gains()
        self.assertEqual(gains.shape, (_sh[0], _sh[1], _sh[2], 4))
        np.testing.assert_almost_equal(
            gains[0, 0, 100, :], exp_gains[0, 0, 100, :])

    def test_gains_shape(self):
        c = rc.Csoln(calfile)
        gains_shape = c.gains_shape()
        self.assertEqual(gains_shape, (_sh[0], _sh[1], _sh[2], 4))

    def test_channel_info(self):
        c = rc.Csoln(calfile, metafits)
        ch_info = c.channel_info()
        np.testing.assert_almost_equal(ch_info['INDEX'], np.arange(0, 768))
        self.assertEqual(ch_info['FREQ'][0], 167055000.0)
        self.assertEqual(ch_info['FREQ'][-1], 197735000.0)
        freq_flags = ch_info['FLAG']
        inds = np.where(np.array(freq_flags) == 0)
        self.assertEqual(len(inds[0]), 648)
        inds = np.where(np.array(freq_flags) == 1)
        self.assertEqual(len(inds[0]), 768 - 648)

    def test_ant_info(self):
        c = rc.Csoln(calfile)
        ant_info = c.ant_info()
        np.testing.assert_almost_equal(ant_info['ANTENNA'], np.arange(0, 128))
        expected_annames = [tl[1] for tl in hdu[3].data]
        self.assertEqual(ant_info['TILENAME'], expected_annames)
        anflags = ant_info['FLAG']
        expected_anflags = np.zeros((128))
        expected_anflags[76] = 1
        np.testing.assert_almost_equal(anflags, expected_anflags)

    def test_gains_ind_for(self):
        c = rc.Csoln(calfile)
        ind = c.gains_ind_for(1)
        self.assertTrue(ind == 1)

    def test_check_refant(self):
        rc.Csoln(calfile, ref_antnum=77)
        with self.assertRaises(Exception):
            self._check_refant()

    def test_normalized_data(self):
        c = rc.Csoln(calfile, norm=True)
        ngains = c._normalized_data(exp_gains[0, :, :, :])
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(ngains[0, 100, :], expected)

    def test_normalized_gains(self):
        c = rc.Csoln(calfile, norm=True)
        ngains = c.normalized_gains()
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)

    def test_select_gains(self):
        c = rc.Csoln(calfile, norm=True)
        ngains = c._select_gains()
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(ngains[0, 0, 100, :], expected)
        c = rc.Csoln(calfile, norm=False)
        ngains = c._select_gains()
        np.testing.assert_almost_equal(
            ngains[0, 0, 100, :], exp_gains[0, 0, 100, :])

    def test_amplitudes(self):
        c = rc.Csoln(calfile, norm=True)
        amps = c.amplitudes()
        np.testing.assert_almost_equal(amps[0, 0, 100, :], np.array(
            [0.99870742, 0.04484763, 0.04205458, 1.23713418]))

    def test_phases(self):
        c = rc.Csoln(calfile, norm=True)
        phases = c.phases()
        np.testing.assert_almost_equal(phases[0, 0, 100, :], np.array(
            [-44.59405224, -143.42377521,   40.83062598, -125.91612395]))

    def test_gains_for_antnum(self):
        c = rc.Csoln(calfile, norm=True)
        gains = c.gains_for_antnum(0)
        expected = np.array([0.71117848-0.70117164j, -0.03601556-0.02672433j,
                             0.03182042+0.02749634j, -0.72570327-1.00192601j])
        np.testing.assert_almost_equal(gains[0, 100, :], expected)

    def test_gains_for_antpair(self):
        c = rc.Csoln(calfile, norm=True)
        gains_antpair = c.gains_for_antpair((0, 1))
        expected = np.array([3.48835869e-01+0.95405085j,
                             -1.12010126e-03+0.00262549j,
                             2.59994960e-03+0.00177372j,
                             1.22932459e+00+0.74457111j])
        np.testing.assert_almost_equal(gains_antpair[0, 100, :], expected)

    def test_gains_for_receiver(self):
        c = rc.Csoln(calfile)
        with self.assertRaises(Exception):
            c.gains_for_receiver(1)
        c = rc.Csoln(calfile, metafits=metafits)
        gains_array = c.gains_for_receiver(1)
        annumbers = c.Metafits.annumbers_for_receiver(1)
        self.assertEqual(gains_array.shape, (1, 8, 768, 4))
        np.testing.assert_almost_equal(
            gains_array[0, 0, 100, :],
            c.gains_for_antnum(annumbers[0])[0, 100, :])

    def test_generate_blackmaneharris(self):
        c = rc.Csoln(calfile)
        n = 768
        bm_filter = c.blackmanharris(n)
        self.assertEqual(len(bm_filter), n)
        expected = signal.windows.blackmanharris(n)
        np.testing.assert_almost_equal(bm_filter, expected)

    def test_delays(self):
        c = rc.Csoln(calfile)
        delays = c.delays()
        self.assertEqual(len(delays), 768)
        np.testing.assert_almost_equal(np.max(delays), 12467.447916666668)
        self.assertEqual(delays[0], 0.0)

    def test_filter_nans(self):
        c = rc.Csoln(calfile)
        gains = c.gains()
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
        c = rc.Csoln(calfile)
        fft_gains = c.gains_fft()
        self.assertEqual(fft_gains.shape, (1, 128, 768, 4))
        np.testing.assert_almost_equal(fft_gains[0, 0, 100, :], np.array(
            [-0.56025554+0.46877677j, -0.13934061+0.03579963j,
             -0.02181193+0.17925411j,  0.04100561-0.41272809j]))
