from mwa_qa.prepvis_metrics import PrepvisMetrics
from mwa_qa.data import DATA_PATH
from collections import OrderedDict
import numpy as np
import unittest
import os

uvfits = os.path.join(DATA_PATH, '1062784992.uvfits')
metafits = os.path.join(DATA_PATH, '1062784992.metafits')


class TestPrepvisMetrics(unittest.TestCase):
    def test__init__(self):
        vis = PrepvisMetrics(uvfits, metafits)
        self.assertTrue(vis.uvfits_path, uvfits)
        self.assertTrue(vis.metafits_path, metafits)

    def test_autos(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=False)
        np.testing.assert_almost_equal(autos[0, 0, 100, :],
                                       np.array([43051.855 + 4.2532739e-07j, 38573.625 + 4.5283093e-07j,
                                                 -1709.8799+7.0794456e+01j, -1709.8799-7.0794456e+01j]), decimal=3)
        autos = vis.autos(manual_flags=True)
        np.testing.assert_almost_equal(autos[0, 0, 0, :], np.array(
            [np.nan+0.j, np.nan+0.j, np.nan+0.j, np.nan+0.j]))
        np.testing.assert_almost_equal(autos[0, 0, 100, :],
                                       np.array([43051.855 + 4.2532739e-07j, 38573.625 + 4.5283093e-07j,
                                                 -1709.8799+7.0794456e+01j, -1709.8799-7.0794456e+01j]), decimal=3)
        autos = vis.autos(manual_flags=True, ex_annumbers=[3])
        np.testing.assert_almost_equal(autos[0, 3, 100, :], np.array(
            [np.nan+0.j, np.nan+0.j, np.nan+0.j, np.nan+0.j]))
        np.testing.assert_almost_equal(autos[0, 0, 100, :],
                                       np.array([43051.855 + 4.2532739e-07j, 38573.625 + 4.5283093e-07j,
                                                 -1709.8799+7.0794456e+01j, -1709.8799-7.0794456e+01j]), decimal=3)

    def test_flags_from_metafits(self):
        vis = PrepvisMetrics(uvfits, metafits)
        flags = vis.flags_from_metafits()
        self.assertTrue((flags == [75, 98]))

    def test_evaluate_edge_flags(self):
        vis = PrepvisMetrics(uvfits, metafits)
        flags = vis._evaluate_edge_flags()
        self.assertEqual(flags.shape, (768,))
        inds = np.where(flags == True)
        np.testing.assert_almost_equal(inds[0], np.array([0,   1,  16,  30,  31,  32,  33,  48,  62,  63,  64,  65,  80,
                                                          94,  95,  96,  97, 112, 126, 127, 128, 129, 144, 158, 159, 160,
                                                          161, 176, 190, 191, 192, 193, 208, 222, 223, 224, 225, 240, 254,
                                                          255, 256, 257, 272, 286, 287, 288, 289, 304, 318, 319, 320, 321,
                                                          336, 350, 351, 352, 353, 368, 382, 383, 384, 385, 400, 414, 415,
                                                          416, 417, 432, 446, 447, 448, 449, 464, 478, 479, 480, 481, 496,
                                                          510, 511, 512, 513, 528, 542, 543, 544, 545, 560, 574, 575, 576,
                                                          577, 592, 606, 607, 608, 609, 624, 638, 639, 640, 641, 656, 670,
                                                          671, 672, 673, 688, 702, 703, 704, 705, 720, 734, 735, 736, 737,
                                                          752, 766, 767]))

    def test_plot_mode(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=True)
        data = vis._plot_mode(autos[10, 0, 300, :], mode='amp')
        np.testing.assert_almost_equal(data, np.array(
            [36737.727, 34113.54,  1191.0828,  1191.0828]), decimal=3)
        data = vis._plot_mode(autos[10, 0, 300, :], mode='phs')
        np.testing.assert_almost_equal(data, np.array(
            [-1.3968029e-11,  1.4657614e-11,  2.9248583e+00, -2.9248583e+00]), decimal=3)
        data = vis._plot_mode(autos[10, 0, 300, :], mode='real')
        np.testing.assert_almost_equal(data, np.array(
            [36737.727, 34113.54, -1163.2173, -1163.2173]), decimal=3)
        data = vis._plot_mode(autos[10, 0, 300, :], mode='imag')
        np.testing.assert_almost_equal(data, np.array(
            [-5.1315362e-07,  5.0002308e-07,  2.5613214e+02, -2.5613214e+02]), decimal=3)
        data = vis._plot_mode(autos[10, 0, 300, :], mode='log')
        np.testing.assert_almost_equal(data, np.array(
            [4.565112, 4.5329266, 3.075942, 3.075942]), decimal=3)
        with self.assertRaises(Exception):
            data = vis._plot_mode(autos[10, 0, 300, :], mode='abs')

    def test_plot_spectra_across_chan(self):
        pass

    def test_plot_spectra_across_time(self):
        pass

    def test_plot_spectra_2D(self):
        pass

    def test_flag_occupancy(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=True)
        foccupancy, inds = vis.flag_occupancy(autos[10, :, :, 0])
        np.testing.assert_almost_equal(foccupancy, np.array([100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            0., 100., 100., 100.,
                                                            33.33333333, 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.,
                                                            100., 100., 100., 100.]))
        np.testing.assert_equal(inds[0], np.array([76, 80]))

    def test_calculate_rms(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=True)
        rms = vis.calculate_rms(autos[10, :, :, 0])
        np.testing.assert_almost_equal(rms[0:10], np.array([32833.35 + 1.18771766e-08j, 32727.639-4.16548751e-08j,
                                                            36768.977-4.26175504e-08j, 35622.582-3.10633048e-08j,
                                                            24658.205+5.37580824e-08j, 22323.727-1.01629825e-08j,
                                                            23036.38 + 3.13644151e-08j, 21217.994+5.46110925e-08j,
                                                            28355.236+9.70865557e-08j, 29699.135-2.46702836e-08j]), decimal=2)

    def test_calculate_mod_zscore(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=True)
        modz = vis.calculate_mod_zscore(autos[10, :, :, 0])
        np.testing.assert_almost_equal(modz[0:10], np.array([0.49903008-1.9951631e-12j,  0.46797398-1.2897820e-11j,
                                                             1.258051 - 1.1984893e-11j,  1.0161792 - 7.7919494e-12j,
                                                             -1.2551366 + 4.1197675e-12j, -1.6941957 - 3.9573054e-12j,
                                                             -1.5720448 + 2.2617418e-12j, -1.9231831 + 5.4186829e-12j,
                                                             -0.3979257 + 1.2848046e-11j, -0.14511508-1.3222220e-11j]), decimal=4)

    def test_iterative_mod_zscore(self):
        vis = PrepvisMetrics(uvfits, metafits)
        autos = vis.autos(manual_flags=True)
        modz, inds = vis.iterative_mod_zscore(autos[10, :, :, 0], 3, 10)
        np.testing.assert_almost_equal(modz[0][0:10], np.array([0.49903008-1.9951631e-12j,  0.46797398-1.2897820e-11j,
                                                                1.258051 - 1.1984893e-11j,  1.0161792 - 7.7919494e-12j,
                                                                -1.2551366 + 4.1197675e-12j, -1.6941957 - 3.9573054e-12j,
                                                                -1.5720448 + 2.2617418e-12j, -1.9231831 + 5.4186829e-12j,
                                                                -0.3979257 + 1.2848046e-11j, -0.14511508-1.3222220e-11j]))
        np.testing.assert_almost_equal(inds[0], np.array([17, 76, 80]))

    def test_initialize_metrics_dict(self):
        vis = PrepvisMetrics(uvfits, metafits)
        vis._initialize_metrics_dict()
        self.assertTrue(list(vis.metrics.keys()), [
                        'NANTS', 'NTIMES', 'NCHAN', 'NPOLS', 'OBSID', 'ANNUMBERS', 'XX', 'YY'])
        self.assertEqual(vis.metrics['NANTS'], 128)
        self.assertEqual(vis.metrics['NTIMES'], 55)
        self.assertEqual(vis.metrics['NCHAN'], 768)
        self.assertEqual(vis.metrics['NPOLS'], 4)
        self.assertEqual(vis.metrics['OBSID'], 'high_season1_2456545')
        np.testing.assert_equal(vis.metrics['ANNUMBERS'], np.array([0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                                    13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                                    26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                                    39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                                    52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                                    65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                                    78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                                                                    91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
                                                                    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                                                    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]))
        self.assertTrue(isinstance(vis.metrics['XX'], OrderedDict))
        self.assertTrue(isinstance(vis.metrics['YY'], OrderedDict))

    def test_run_metrics(self):
        vis = PrepvisMetrics(uvfits, metafits)
        vis.run_metrics()
        self.assertEqual(list(vis.metrics.keys()), ['NANTS', 'NTIMES', 'NCHAN', 'NPOLS', 'OBSID',
                         'ANNUMBERS', 'XX', 'YY', 'BAD_ANTS', 'BAD_ANTS_PERCENT', 'STATUS', 'THRESHOLD'])
        self.assertEqual(vis.metrics['THRESHOLD'], 3)
        np.testing.assert_equal(
            vis.metrics['BAD_ANTS'], np.array([17, 76, 80]))
        self.assertEqual(vis.metrics['BAD_ANTS_PERCENT'], 2.34375)
        self.assertEqual(vis.metrics['STATUS'], 'GOOD')
        self.assertEqual(list(vis.metrics['XX'].keys()), [
                         'RMS', 'MODZ_SCORE', 'BAD_ANTS'])
        self.assertEqual(list(vis.metrics['YY'].keys()), [
                         'RMS', 'MODZ_SCORE', 'BAD_ANTS'])
        self.assertTrue((vis.metrics['XX']['BAD_ANTS'] == [76, 80, 17]))
        self.assertTrue((vis.metrics['YY']['BAD_ANTS'] == [76, 17]))
        np.testing.assert_almost_equal(vis.metrics['XX']['RMS'][0:10], np.array(
            [0.98502, 0.98553336, 1.0920447, 1.0620629, 0.73594517, 0.6742143,
             0.6892578, 0.6414309, 0.8617734, 0.8987404]))
        np.testing.assert_almost_equal(vis.metrics['XX']['MODZ_SCORE'][0][0:10], np.array(
            [0.44427395,  0.45195827,  1.2281251,  1.0092522, -1.3470997,
             -1.7959414, -1.6857036, -2.033683, -0.44826028, -0.17769146]))
        np.testing.assert_almost_equal(vis.metrics['YY']['RMS'][0:10], np.array(
            [0.8551952, 0.93147486, 1.0507381, 1.0109979, 0.858264,
             0.7061106, 0.70077044, 0.7349987, 0.85218024, 1.0480894]))
        np.testing.assert_almost_equal(vis.metrics['YY']['MODZ_SCORE'][0][0:10], np.array(
            [-0.4171582,  0.05907506,  0.8022736,  0.55377096, -0.38284403,
             -1.3313679, -1.3563856, -1.1527667, -0.43960702,  0.77399105]))

    def test_write_to(self):
        vis = PrepvisMetrics(uvfits, metafits)
        vis._initialize_metrics_dict()
        outfile = uvfits.replace('.uvfits', '_prepvis_metrics.json')
        vis.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        vis.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
