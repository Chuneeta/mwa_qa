from mwa_qa.read_uvfits import UVfits
from astropy.io import fits
from scipy import signal
import numpy as np
import unittest

uvfits = '../../test_files/1061315688_cal.uvfits'
uvfits_hex = '../../test_files/1320411840_hyp_cal.uvfits'
hdu = fits.open(uvfits)
data0, data1 = hdu[0].data, hdu[1].data
hdr0, hdr1 = hdu[0].header, hdu[1].header
ndata = len(data0)


class TestUVfits(unittest.TestCase):
    def test__init__(self):
        uvf = UVfits(uvfits)
        self.assertEqual(uvf.uvfits_path, uvfits)
        self.assertEqual(uvf.Nfreqs, 768)
        self.assertEqual(uvf.Npols, 4)
        self.assertEqual(uvf.Ntimes, 27)
        self.assertEqual(uvf.Nbls, 8128)
        self.assertEqual(uvf.Nants, 128)
        np.testing.assert_equal(uvf.antenna_numbers, np.arange(0, uvf.Nants))
        self.assertEqual(len(uvf.ant_1_array), ndata)
        # np.testing.assert_equal(uvf.ant_1_array, np.repeat(
        #    np.arange(uvf.Nants), uvf.Nants))
        # np.testing.assert_equal(uvf.ant_2_array, np.array(
        #    np.arange(128).tolist() * uvf.Nants))
        self.assertEqual(len(uvf.ant_2_array), ndata)
        self.assertEqual(len(uvf.baseline_array), ndata)
        self.assertEqual(len(uvf.antpairs), ndata)
        self.assertEqual(len(uvf.antenna_positions), uvf.Nants)
        np.testing.assert_equal(uvf.antpairs[0], np.array([0, 0]))
        np.testing.assert_almost_equal(uvf.antenna_positions[0], np.array([
            4.56250049e+02, -1.49785004e+02,  6.80459899e+01]), decimal=4)
        # self.assertEqual(uvf.baseline_array, np.arange(257, 32897))
        self.assertEqual(uvf.ant_names[0], 'Tile011')
        self.assertEqual(len(uvf.ant_names), uvf.Nants)
        self.assertEqual(len(uvf.freq_array), uvf.Nfreqs)

    def test_auto_pairs(self):
        uvf = UVfits(uvfits)
        auto_antpairs = uvf.auto_antpairs()
        expected = [(ant, ant) for ant in range(uvf.Nants)]
        expected.remove((76, 76))
        np.testing.assert_equal(np.array(auto_antpairs), np.array(expected))

    def test_blt_idxs_for_antpair(self):
        uvf = UVfits(uvfits)
        blt_idx = uvf.blt_idxs_for_antpair((0, 0))
        self.assertEqual(blt_idx[0], 0)

    def test__data_for_antpairs(self):
        uvf = UVfits(uvfits)
        vis_hdu = hdu['PRIMARY']
        data = uvf._data_for_antpairs(vis_hdu, [(0, 0)])
        self.assertEqual(data.shape, (27, 1, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + \
            data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, 0, :, :], expected)

    def test_data_for_antpairs(self):
        uvf = UVfits(uvfits)
        data = uvf.data_for_antpairs([(0, 0)])
        self.assertEqual(data.shape, (27, 1, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + \
            data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, 0, :, :], expected)

    def test_data_for_antpair(self):
        uvf = UVfits(uvfits)
        data = uvf.data_for_antpair((0, 0))
        self.assertEqual(data.shape, (27, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + \
            data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, :, :], expected)

    def test_blackmanharris(self):
        uvf = UVfits(uvfits)
        filter = uvf.blackmanharris(uvf.Nfreqs)
        np.testing.assert_almost_equal(
            filter, signal.windows.blackmanharris(768))

    def test_delays(self):
        uvf = UVfits(uvfits)
        delays = uvf.delays()
        dfreq = uvf.freq_array[1] - uvf.freq_array[0]
        np.testing.assert_almost_equal(delays * 1e-9, np.fft.fftshift(
            np.fft.fftfreq(uvf.Nfreqs, dfreq)), decimal=4)

    def test_fft_array(self):
        uvf = UVfits(uvfits)
        fft_array = uvf.fft_array((0, 0))
        self.assertEqual(fft_array.shape, (27, 768, 4))
        d0 = uvf.data_for_antpair((0, 0))
        # np.testing.assert_almost_equal(
        # fft_array[0, :, 0], np.fft.fftshift(np.fft.fft(d0[0, :, 0])))

    def redundant_antpairs(self):
        uvf = UVfits(uvfits)
        reds = uvf.redundant_antpairs()
        self.assertEqual(len(reds), 25)
        self.assertEqual(list(reds.keys()), [(9, -34, 19),
                                             (14, 32, 27),
                                             (4, -45, 8),
                                             (27, 26, 54),
                                             (21, -85, 40),
                                             (43, -24, 85),
                                             (28, 88, 57),
                                             (14, -112, 28),
                                             (38, 84, 75),
                                             (33, -101, 63),
                                             (19, 122, 39),
                                             (56, 60, 112),
                                             (38, 116, 76),
                                             (56, 75, 111),
                                             (65, 16, 130),
                                             (48, -121, 93),
                                             (78, 13, 154),
                                             (80, 45, 159),
                                             (96, 0, 191),
                                             (96, 108, 190),
                                             (94, 191, 188),
                                             (120, 186, 240),
                                             (13, -399, 29),
                                             (63, -1646, 144),
                                             (663, 1564, 1297)]
                         )
