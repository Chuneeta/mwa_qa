from mwa_qa.data import DATA_PATH
from mwa_qa.read_uvfits import UVfits
import unittest
import os
import numpy as np
from astropy.io import fits
from scipy import signal
import os

from mwa_qa.read_uvfits import UVfits

uvfits = os.path.join(DATA_PATH, '1062784992.uvfits')
hdu = fits.open(uvfits)
data0, data1 = hdu[0].data, hdu[1].data
hdr0, hdr1 = hdu[0].header, hdu[1].header
ndata = len(data0)


class TestUVfits(unittest.TestCase):
    def test__init__(self):
        uvf = UVfits(uvfits)
        self.assertEqual(uvf.uvfits_path, uvfits)
        self.assertEqual(uvf.Nchan, 768)
        self.assertEqual(uvf.Npols, 4)
        self.assertEqual(uvf.Ntimes, 55)
        self.assertEqual(uvf.Nbls, 8256)
        self.assertEqual(uvf.Nants, 128)
        self.assertEqual(uvf.channel_width, 40000.0)
        np.testing.assert_equal(uvf.antenna_numbers, np.arange(0, uvf.Nants))
        self.assertEqual(len(uvf.ant_1_array), 454080)
        self.assertEqual(uvf.ant_1_array[0], 0)
        self.assertEqual(uvf.ant_1_array[-1], 127)
        self.assertEqual(uvf.ant_2_array[0], 0)
        self.assertEqual(uvf.ant_2_array[-1], 127)
        self.assertEqual(len(uvf.ant_2_array), 454080)
        self.assertEqual(len(uvf.baseline_array), 454080)
        self.assertEqual(len(uvf.antpairs), uvf.Nbls)
        self.assertEqual(len(uvf.antenna_positions), uvf.Nants)
        np.testing.assert_equal(uvf.antpairs[0], np.array([0, 0]))
        np.testing.assert_almost_equal(uvf.antenna_positions[0], np.array([
            4.56250049e+02, -1.49785004e+02,  6.80459899e+01]), decimal=4)
        self.assertEqual(len(uvf.unique_baselines), 8256)
        self.assertEqual(uvf.baseline_array[0], 257)
        self.assertEqual(uvf.baseline_array[-1], 32896)
        self.assertEqual(uvf.ant_names[0], 'Tile011')
        self.assertEqual(len(uvf.ant_names), uvf.Nants)
        self.assertEqual(len(uvf.freq_array), uvf.Nchan)
        self.assertEqual(uvf.obsid, 'high_season1_2456545')
        self.assertEqual(len(uvf.antenna_numbers), 128)
        np.testing.assert_equal(uvf.polarization_array,
                                np.array([-4, -5, -6, -7]))

    def test_auto_pairs(self):
        uvf = UVfits(uvfits)
        auto_antpairs = uvf.auto_antpairs()
        expected = [(ant, ant) for ant in range(uvf.Nants)]
        np.testing.assert_equal(np.array(auto_antpairs), np.array(expected))

    def test_blt_idxs_for_antpair(self):
        uvf = UVfits(uvfits)
        blt_idx = uvf.blt_idxs_for_antpair((0, 0))
        self.assertEqual(blt_idx[0], 0)

    def test__data_for_antpairs(self):
        uvf = UVfits(uvfits)
        vis_hdu = hdu['PRIMARY']
        data = uvf._data_for_antpairs(vis_hdu, [(0, 0)])
        self.assertEqual(data.shape, (55, 1, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, 0, :, :], expected)

    def test__flag_for_antpairs(self):
        uvf = UVfits(uvfits)
        vis_hdu = hdu['PRIMARY']
        flag = uvf._flag_for_antpairs(vis_hdu, [(0, 0)])
        self.assertEqual(flag.shape, (55, 1, 768, 4))
        self.assertTrue(np.all(flag[0, 0, 0, :]))

    def test_data_for_antpairs(self):
        uvf = UVfits(uvfits)
        data = uvf.data_for_antpairs([(0, 0)])
        self.assertEqual(data.shape, (55, 1, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, 0, :, :], expected)

    def test_flag_for_antpairs(self):
        uvf = UVfits(uvfits)
        flag = uvf.flag_for_antpairs([(0, 0)])
        self.assertEqual(flag.shape, (55, 1, 768, 4))
        self.assertTrue(np.all(flag[0, 0, 0, :]))

    def test_data_for_antpair(self):
        uvf = UVfits(uvfits)
        data = uvf.data_for_antpair((0, 0))
        self.assertEqual(data.shape, (55, 768, 4))
        expected = data0[0][5][0, 0, :, :, 0] + data0[0][5][0, 0, :, :, 1] * 1j
        np.testing.assert_almost_equal(data[0, :, :], expected)

    def test_flag_for_antpair(self):
        uvf = UVfits(uvfits)
        flag = uvf.flag_for_antpair((0, 0))
        self.assertEqual(flag.shape, (55, 768, 4))
        self.assertTrue(np.all(flag[0, 0, :]))

    def test_amplitude_array(self):
        uvf = UVfits(uvfits)
        amps = uvf.amplitude_array([(0, 0)])
        self.assertEqual(amps.shape, (55, 1, 768))
        np.testing.assert_almost_equal(amps[0, 0, 0:4], np.array(
            [44492.473, 43540.79, 42893.27, 42702.707]), decimal=2)

    def test_phase_array(self):
        uvf = UVfits(uvfits)
        phs = uvf.phase_array([(0, 0)])
        self.assertEqual(phs.shape, (55, 1, 768))
        np.testing.assert_almost_equal(phs[0, 0, 0:4], np.array(
            [7.0759669e-14,  1.3661892e-11, -4.5692186e-11, -1.7285005e-11]))

    def test_blackmanharris(self):
        uvf = UVfits(uvfits)
        filter = uvf.blackmanharris(uvf.Nchan)
        np.testing.assert_almost_equal(
            filter, signal.windows.blackmanharris(768))

    def test_group_antpairs(self):
        uvf = UVfits(uvfits)
        angroups = uvf.group_antpairs(uvf.antenna_positions, bl_tol=1e-2)
        self.assertEqual(len(angroups), 8128)
        self.assertTrue((angroups[(221, 5442, 378)]),  [(1, 0)])

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

    def test_delays(self):
        uvf = UVfits(uvfits)
        delays = uvf.delays()
        dfreq = uvf.freq_array[1] - uvf.freq_array[0]
        np.testing.assert_almost_equal(delays * 1e-9, np.fft.fftshift(
            np.fft.fftfreq(uvf.Nchan, dfreq)), decimal=4)

    def test_fft_data_for_antpair(self):
        uvf = UVfits(uvfits)
        fft_data = uvf.fft_data_for_antpair((0, 1))
        self.assertEqual(fft_data.shape, (55, 768, 4))
        np.testing.assert_almost_equal(fft_data[0, 0, :],
                                       np.array([-572.2727 - 658.0653j, -571.70917-368.39127j,
                                                 1309.6108 - 787.1936j, -250.48773-311.11957j]),
                                       decimal=4)

    def test_fft_data_for_antpairs(self):
        uvf = UVfits(uvfits)
        fft_data = uvf.fft_data_for_antpairs([(0, 0), (0, 1)])
        self.assertEqual(fft_data.shape, (55, 2, 768, 4))
        np.testing.assert_almost_equal(fft_data[0, 1, 0, :], np.array([-572.2727 - 658.0653j,
                                                                       -571.70917-368.39127j,
                                                                       1309.6108 - 787.1936j,
                                                                       -250.48773-311.11957j]),
                                       decimal=4)
