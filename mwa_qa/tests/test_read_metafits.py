
from mwa_qa.read_metafits import Metafits
from mwa_qa.data import DATA_PATH
from collections import OrderedDict
import unittest
import numpy as np
import astropy
import os


metafits = os.path.join(DATA_PATH, '1062784992.metafits')


class Test_Metatits(unittest.TestCase):
    def test_init_(self):
        m = Metafits(metafits)
        self.assertEqual(m.lst, 16.3529561118503)
        self.assertEqual(m.metafits, metafits)
        self.assertEqual(m.pol, 'X')
        self.assertEqual(m.ha, '-25:01:07.44')
        self.assertEqual(m.pointing_centre,
                         (1.129399085412689, -25.96174344493141))
        self.assertEqual(m.phase_centre, (0.0, -27.0))
        self.assertEqual(m.filename, 'high_season1_2456545')
        self.assertEqual(m.eorfield, 'EoR0')
        self.assertEqual(m.exposure, 112)
        self.assertEqual(m.integration, 0.5)
        self.assertEqual(m.obs_time, '2013-09-09T18:02:56')
        self.assertEqual(m.Nants, 128)
        self.assertEqual(m.Nchans, 768)
        np.testing.assert_almost_equal(m.frequency_array[0:4], np.array(
            [167.68, 167.71838331, 167.75676662, 167.79514993]))
        self.assertEqual(m.delays, '6,4,2,0,6,4,2,0,6,4,2,0,6,4,2,0')
        self.assertEqual(m.antenna_positions.shape, (128, 3))
        np.testing.assert_almost_equal(
            m.antenna_positions[0], np.array([-101.53, -585.675,  375.212]), decimal=3)
        self.assertEqual(len(m.antenna_names), 128)
        np.testing.assert_equal(m.antenna_names[0:4], np.array(
            ['Tile104', 'Tile103', 'Tile102', 'Tile101']))
        self.assertEqual(len(m.antenna_numbers), 128)
        np.testing.assert_equal(
            m.antenna_numbers[0:4], np.array([75, 74, 73, 72]))
        self.assertEqual(len(m.tile_ids), 128)
        np.testing.assert_equal(
            m.tile_ids[0:4], np.array([104, 103, 102, 101]))
        self.assertEqual(len(m.receiver_ids), 128)
        np.testing.assert_equal(
            m.receiver_ids[0:4], np.array([10, 10, 10, 10]))
        self.assertEqual(len(m.cable_type), 128)
        np.testing.assert_equal(
            m.cable_type[0:4], np.array(['EL', 'EL', 'EL', 'EL']))
        self.assertEqual(len(m.cable_length), 128)
        np.testing.assert_almost_equal(
            m.cable_length[0:4], np.array([-756.49, -1191.96, -900.98, -904.71]))
        self.assertEqual(len(m.BFTemps), 128)
        np.testing.assert_almost_equal(
            m.BFTemps[0:4], np.array([20.1681, 19.9346, 19.4792, 19.8316]), decimal=4)
        self.assertEqual(len(m.flag_array), 128)
        np.testing.assert_equal(
            m.flag_array[0:4], np.array([0, 0, 0, 0]))
        self.assertEqual(len(m.baseline_array), 8256)
        np.testing.assert_equal(
            m.baseline_array[0:4], np.array([[0, 0], [0, 1], [0, 2], [0, 3]]))
        self.assertEqual(len(m.baseline_lengths), 8256)
        np.testing.assert_almost_equal(
            m.baseline_lengths[0:4], np.array([0., 516.6403, 712.5618, 737.1464]), decimal=4)

    def test_pol_index(self):
        m = Metafits(metafits)
        hdu = astropy.io.fits.open(metafits)
        self.assertEqual(m.pol_index(hdu['TILEDATA'].data), 1)

    def test_check_data(self):
        m = Metafits(metafits)
        hdu = astropy.io.fits.open(metafits)
        m._check_data(hdu['TILEDATA'].data)

    def test_antenna_position_for(self):
        m = Metafits(metafits)
        np.testing.assert_almost_equal(m.antenna_position_for(
            0), np.array([265.814, -149.785,  377.011]), decimal=3)

    def test_baseline_length_for(self):
        m = Metafits(metafits)
        self.assertEqual(m.baseline_length_for((0, 0)), 0.0)
        self.assertAlmostEqual(
            m.baseline_length_for((0, 1)), 516.6403, places=4)

    def test_baselines_greater_than(self):
        m = Metafits(metafits)
        bls = m.baselines_greater_than(500)
        self.assertEqual(len(bls), 5147)
        np.testing.assert_equal(bls[0:4], np.array([[0, 1],
                                                    [0, 2],
                                                    [0, 3],
                                                    [0, 4]]))
        bls = m.baselines_less_than(500)
        self.assertEqual(len(bls), 8256 - 5147)
        np.testing.assert_equal(bls[0:4], np.array([[0, 0],
                                                    [0, 6],
                                                    [0, 7],
                                                    [0, 8]]))

    def test_antenna_numbers_for_receiver(self):
        m = Metafits(metafits)
        antnums = m.antenna_numbers_for_receiver(1)
        np.testing.assert_equal(
            antnums, np.array([3, 2, 1, 0, 7, 6, 5, 4]))
        with self.assertRaises(Exception):
            m.antenna_number_for_receiver(20)

    def test_receiver_for_antenna_number(self):
        m = Metafits(metafits)
        receiver = m.receiver_for_antenna_number(0)
        np.testing.assert_equal(receiver, np.array(1))
        with self.assertRaises(Exception):
            m.receiver_fot_antenna_number(200)

    def test_antpos_dict(self):
        m = Metafits(metafits)
        antpos = m._anpos_dict()
        self.assertEqual(len(list(antpos.keys())), 128)
        self.assertTrue((list(antpos.keys()) == [75, 74, 73, 72, 79, 78, 77, 76, 51, 50, 49,
                                                 48, 55, 54, 53, 52, 123, 122, 121, 120, 127,
                                                 126, 125, 124, 115, 114, 113, 112, 119, 118,
                                                 117, 116, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2,
                                                 1, 0, 7, 6, 5, 4, 67, 66, 65, 64, 71, 70, 69,
                                                 68, 59, 58, 57, 56, 63, 62, 61, 60, 91, 90, 89,
                                                 88, 95, 94, 93, 92, 83, 82, 81, 80, 87, 86, 85,
                                                 84, 107, 106, 105, 104, 111, 110, 109, 108, 99,
                                                 98, 97, 96, 103, 102, 101, 100, 19, 18, 17, 16,
                                                 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28,
                                                 43, 42, 41, 40, 47, 46, 45, 44, 35, 34, 33, 32,
                                                 39, 38, 37, 36]))
        self.assertTrue((antpos[0] == [265.8139953613281, -
                        149.78500366210938, 377.010986328125]))

    def test_group_antpairs(self):
        m = Metafits(metafits)
        ant_groups = m.group_antpairs(bl_tol=1)
        self.assertEqual(len(ant_groups), 7747)
        self.assertTrue((ant_groups[(120, -199, 1)] == [(33, 34)]))

    def test_redundant_antpairs(self):
        m = Metafits(metafits)
        reds = m.redundant_antpairs()
        self.assertEqual(len(list(reds.keys())), 27)
        self.assertTrue((list(reds.keys()) == [(20, -35, 0), (31, 31, 0), (8, -45, 0), (61, 26, 0), (72, -4, 0),
                                               (45, -85, 1), (96, -24, 0), (63,
                                                                            88, 0), (84, 84, 0), (71, -101, 1),
                                               (43, 122, 0), (125, 60,
                                                              0), (124, 75, 0), (146, 17, 0),
                                               (117, -102, -1), (153, -42,
                                                                 0), (105, -121, 1), (48, -161, 1),
                                               (173, 13, 0), (178, 45,
                                                              0), (143, 127, 0), (213, 0, 0),
                                               (212, 108, 0), (210, 191,
                                                               0), (232, 226, 0), (32, -399, 0),
                                               (157, -1647, -9)]))
        self.assertTrue((reds[(20, -35, 0)] == [(26, 12), (2, 1)]))
