from collections import OrderedDict
from mwa_qa.cal_metrics import CalMetrics
from astropy.io import fits
from mwa_qa.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits')
calfile_poly = os.path.join(DATA_PATH, 'test_1061315688_poly.fits')
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')
hdu = fits.open(calfile)
exp_gains = hdu[1].data[:, :, :, ::2] + hdu[1].data[:, :, :, 1::2] * 1j
_sh = hdu[1].data.shape


class TestCalMetrics(unittest.TestCase):
    def test_init__(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        self.assertEqual(m.calfits_path, calfile)
        self.assertEqual(m.CalFits.calfits_path, calfile)
        self.assertEqual(m.CalFits.Metafits.metafits, metafits)
        self.assertEqual(m.CalFits.Metafits.pol, 'X')
        self.assertEqual(m.Metafits.metafits, metafits)
        self.assertEqual(m.Metafits.pol, 'X')
        self.assertEqual(m.CalFits.gain_array.shape, (1, 128, 768, 4))
        self.assertEqual(m.CalFits.start_time, hdu['TIMEBLOCKS'].data[0][0])
        self.assertEqual(m.CalFits.end_time, hdu['TIMEBLOCKS'].data[0][1])
        self.assertEqual(m.CalFits.average_time, hdu['TIMEBLOCKS'].data[0][2])
        np.testing.assert_equal(np.array(m.CalFits.antenna), np.arange(128))
        expected_flags = np.zeros((128))
        expected_flags[76] = 1
        np.testing.assert_equal(
            np.array(m.CalFits.antenna_flags), expected_flags)
        expected_annames = [tl[1] + ' ' for tl in hdu[3].data]
        np.testing.assert_equal(np.array(m.CalFits.annames), expected_annames)
        np.testing.assert_almost_equal(
            m.CalFits.frequency_channels, np.arange(0, 768))
        self.assertEqual(m.CalFits.frequency_array[0], 167055000.0)
        self.assertEqual(m.CalFits.frequency_array[-1], 197735000.0)
        inds = np.where(np.array(m.CalFits.frequency_flags) == 0)
        self.assertEqual(len(inds[0]), 648)
        inds = np.where(np.array(m.CalFits.frequency_flags) == 1)
        self.assertEqual(len(inds[0]), 768 - 648)
        np.testing.assert_almost_equal(
            m.CalFits.gain_array[0, 0, 100, :], exp_gains[0, 0, 100, :])
        np.testing.assert_almost_equal(m.CalFits.amplitudes[0, 0, 100, :],
                                       np.array(
            [0.76390112, 0.02917631, 0.0354274, 0.84958042]))
        np.testing.assert_almost_equal(m.CalFits.phases[0, 0, 100, :],
                                       np.array(
            [1.58846751, 1.90056897, 0.87910414, 1.23280019]))
        np.testing.assert_almost_equal(
            m.CalFits.convergence, hdu['RESULTS'].data)
        np.testing.assert_almost_equal(
            m.CalFits.baseline_weights, hdu['BASELINES'].data)
        self.assertFalse(m.CalFits.norm)
        m = CalMetrics(calfile, metafits_path=metafits, norm=True)
        self.assertTrue(m.CalFits.norm)
        self.assertTrue(m.CalFits.ref_antenna == 127)
        np.testing.assert_almost_equal(
            m.CalFits.gain_array[0, 0, 100, :], np.array([
                0.71117848-0.70117164j, -0.03601556-0.02672433j,
                0.03182042+0.02749634j, -0.72570327-1.00192601j]))
        np.testing.assert_almost_equal(m.CalFits.amplitudes[0, 0, 100, :],
                                       np.array(
            [0.99870742, 0.04484763, 0.04205458, 1.23713418]))
        np.testing.assert_almost_equal(m.CalFits.phases[0, 0, 100, :],
                                       np.array(
            [-0.77831304, -2.5032171,  0.71262886, -2.19765094]))

    def test_variance_for_antpair(self):
        m = CalMetrics(calfile)
        variance = m.variance_for_antpair((0, 0))
        self.assertEqual(variance.shape, (1, 4))
        np.testing.assert_almost_equal(variance[0, :], np.array(
            [9.39389297e-03, 1.16491587e-06,
             1.94889201e-06, 7.67858028e-03]), decimal=6)

    def test_variance_for_baselines_less_than(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        variances = m.variance_for_baselines_less_than(10)
        self.assertEqual(variances.shape, (1, 23, 4))
        np.testing.assert_almost_equal(variances[0, 0, :], np.array(
            [8.46222442e-03, 8.99869563e-07,
             1.50672493e-06, 8.14717749e-03]), decimal=6)

    def test_skewness_across_uvcut(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        skewness = m.skewness_across_uvcut(10)
        np.testing.assert_almost_equal(skewness, np.array(
            [[0.19086073, 1.24464852, 1.05598102, 0.56893005]]))

    def test_get_receivers(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        receivers = m.get_receivers()
        expected = [10, 10, 10, 10, 10, 10, 10, 10, 7, 7, 7, 7, 7, 7, 7, 7,
                    16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15,
                    15, 15, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                    9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12,
                    12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11,
                    14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13,
                    13, 13, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                    6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5]
        self.assertEqual(receivers, expected)
        m = CalMetrics(calfile)
        receivers = m.get_receivers()
        self.assertEqual(receivers, list(np.arange(1, 17)))

    def test_unused_baselines_percent(self):
        m = CalMetrics(calfile)
        percent = m.unused_baselines_percent()
        self.assertEqual(percent, 13.065944881889763)

    def test_unused_channels_percent(self):
        m = CalMetrics(calfile)
        percent = m.unused_channels_percent()
        self.assertEqual(percent, 15.625)

    def test_unused_antenna_percent(self):
        m = CalMetrics(calfile)
        percent = m.unused_antennas_percent()
        self.assertEqual(percent, 0.78125)

    def test_non_converging_percent(self):
        m = CalMetrics(calfile)
        percent = m.non_converging_percent()
        self.assertEqual(percent, 15.625)

    def test_convergence_variance(self):
        m = CalMetrics(calfile)
        conv_variance = m.convergence_variance()
        np.testing.assert_almost_equal(
            conv_variance, np.array([4.93491705e-14]))

    def test_initialize_metrics_dict(self):
        m = CalMetrics(calfile)
        m._initialize_metrics_dict()
        self.assertTrue(isinstance(m.metrics, OrderedDict))
        self.assertEqual(list(m.metrics.keys()), ['POLS', 'OBSID',
                                                  'UVCUT', 'M_THRESH',
                                                  'NTIME', 'START_FREQ',
                                                  'CH_WIDTH', 'NCHAN',
                                                  'ANTENNA', 'XX', 'YY'])

        m = CalMetrics(calfile_poly)
        m._initialize_metrics_dict()
        self.assertEqual(list(m.metrics.keys()), ['POLS', 'OBSID',
                                                  'UVCUT', 'M_THRESH',
                                                  'NTIME', 'START_FREQ',
                                                  'CH_WIDTH', 'NCHAN',
                                                  'ANTENNA', 'XX', 'YY',
                                                  'POLY_ORDER', 'POLY_MSE'])

    def test_run_metrics(self):
        m = CalMetrics(calfile, metafits)
        m.run_metrics()
        self.assertEqual(list(m.metrics.keys()), ['POLS', 'OBSID',
                                                  'UVCUT', 'M_THRESH',
                                                  'NTIME', 'START_FREQ',
                                                  'CH_WIDTH', 'NCHAN',
                                                  'ANTENNA', 'XX', 'YY',
                                                  'UNUSED_BLS', 'UNUSED_CHS',
                                                  'UNUSED_ANTS',
                                                  'NON_CONVERGED_CHS',
                                                  'CONVERGENCE',
                                                  'CONVERGENCE_VAR',
                                                  'STATUS'])
        self.assertEqual(m.metrics['OBSID'], m.CalFits.obsid)
        self.assertEqual(m.metrics['UVCUT'], m.CalFits.uvcut)
        self.assertEqual(m.metrics['NTIME'], m.CalFits.Ntime)
        self.assertEqual(m.metrics['START_FREQ'], m.CalFits.frequency_array[0])
        np.testing.assert_almost_equal(m.metrics['CH_WIDTH'],
                                       m.CalFits.frequency_array[1]
                                       - m.CalFits.frequency_array[0])
        np.testing.assert_almost_equal(m.metrics['ANTENNA'], m.CalFits.antenna)
        self.assertTrue(isinstance(m.metrics['XX'], OrderedDict))
        self.assertTrue(isinstance(m.metrics['YY'], OrderedDict))
        self.assertEqual(m.metrics['UNUSED_BLS'], 13.065944881889763)
        self.assertEqual(m.metrics['UNUSED_CHS'], 15.625)
        self.assertEqual(m.metrics['UNUSED_ANTS'], 0.78125)
        self.assertEqual(m.metrics['NON_CONVERGED_CHS'], 15.625)
        np.testing.assert_almost_equal(
            m.metrics['CONVERGENCE_VAR'], np.array([4.93491705e-14]))
        self.assertEqual(list(m.metrics['XX'].keys()), ['SKEWNESS_UVCUT',
                                                        'AMPVAR_ANT',
                                                        'AMPRMS_ANT',
                                                        'RMS_AMPVAR_ANT',
                                                        'AMPVAR_FREQ',
                                                        'AMPRMS_FREQ',
                                                        'RMS_AMPVAR_FREQ',
                                                        'DFFT',
                                                        'DFFT_POWER',
                                                        'DFFT_POWER_HIGH_PKPL',
                                                        'DFFT_POWER_HIGH_NKPL',
                                                        'RECEIVER_CHISQVAR'])
        self.assertEqual(list(m.metrics['YY'].keys()), ['SKEWNESS_UVCUT',
                                                        'AMPVAR_ANT',
                                                        'AMPRMS_ANT',
                                                        'RMS_AMPVAR_ANT',
                                                        'AMPVAR_FREQ',
                                                        'AMPRMS_FREQ',
                                                        'RMS_AMPVAR_FREQ',
                                                        'DFFT', 'DFFT_POWER',
                                                        'DFFT_POWER_HIGH_PKPL',
                                                        'DFFT_POWER_HIGH_NKPL',
                                                        'RECEIVER_CHISQVAR'])
        np.testing.assert_almost_equal(
            m.metrics['XX']['SKEWNESS_UVCUT'], 0.43736719426371656)
        np.testing.assert_almost_equal(
            m.metrics['XX']['RMS_AMPVAR_FREQ'], 0.007612282633781024)
        np.testing.assert_almost_equal(
            m.metrics['XX']['RMS_AMPVAR_ANT'], 0.009989448240765735)
        np.testing.assert_almost_equal(
            m.metrics['XX']['DFFT_POWER'], 104927.04311607413)
        np.testing.assert_almost_equal(
            m.metrics['XX']['DFFT_POWER_HIGH_PKPL'], 12011.161546479638)
        np.testing.assert_almost_equal(
            m.metrics['XX']['DFFT_POWER_HIGH_NKPL'], 12260.513518927311)
        np.testing.assert_almost_equal(
            m.metrics['XX']['RECEIVER_CHISQVAR'], 8.06865584510551)
        self.assertTrue(m.metrics['STATUS'])

        m = CalMetrics(calfile, metafits)

    def test_write_metrics(self):
        m = CalMetrics(calfile)
        m._initialize_metrics_dict()
        outfile = calfile.replace('.fits', '_cal_metrics.json')
        print(outfile)
        m.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        m.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
