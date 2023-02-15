from collections import OrderedDict
from mwa_qa.cal_metrics import CalMetrics
from astropy.io import fits
from mwa_qa.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'hyp_soln_1062784992.fits')
metafits = os.path.join(DATA_PATH, '1062784992.metafits')
hdu = fits.open(calfile)


class TestCalMetrics(unittest.TestCase):
    def test_init__(self):
        m = CalMetrics(calfile, metafits)
        self.assertEqual(m.calfits_path, calfile)
        self.assertEqual(m.metafits_path, metafits)
        self.assertEqual(m.CalFits.calfits_path, calfile)
        self.assertEqual(m.MetaFits.metafits, metafits)
        self.assertTrue(m.CalFits.norm)
        self.assertTrue(m.CalFits.reference_antenna == 127)
        self.assertEqual(m.CalFits.gain_array.shape, (1, 128, 768, 4))
        self.assertEqual(m.CalFits.start_time, hdu['TIMEBLOCKS'].data[0][0])
        self.assertEqual(m.CalFits.end_time, hdu['TIMEBLOCKS'].data[0][1])
        self.assertEqual(m.CalFits.average_time, hdu['TIMEBLOCKS'].data[0][2])
        np.testing.assert_equal(np.array(m.CalFits.antenna), np.arange(128))
        np.testing.assert_equal(
            np.array(m.CalFits.antenna_flags), np.zeros(128))
        np.testing.assert_equal(m.CalFits.annames, np.array(['Tile011', 'Tile012', 'Tile013', 'Tile014', 'Tile015',
                                                             'Tile016', 'Tile017', 'Tile018', 'Tile021', 'Tile022',
                                                             'Tile023', 'Tile024', 'Tile025', 'Tile026', 'Tile027',
                                                             'Tile028', 'Tile031', 'Tile032', 'Tile033', 'Tile034',
                                                             'Tile035', 'Tile036', 'Tile037', 'Tile038', 'Tile041',
                                                             'Tile042', 'Tile043', 'Tile044', 'Tile045', 'Tile046',
                                                             'Tile047', 'Tile048', 'Tile051', 'Tile052', 'Tile053',
                                                             'Tile054', 'Tile055', 'Tile056', 'Tile057', 'Tile058',
                                                             'Tile061', 'Tile062', 'Tile063', 'Tile064', 'Tile065',
                                                             'Tile066', 'Tile067', 'Tile068', 'Tile071', 'Tile072',
                                                             'Tile073', 'Tile074', 'Tile075', 'Tile076', 'Tile077',
                                                             'Tile078', 'Tile081', 'Tile082', 'Tile083', 'Tile084',
                                                             'Tile085', 'Tile086', 'Tile087', 'Tile088', 'Tile091',
                                                             'Tile092', 'Tile093', 'Tile094', 'Tile095', 'Tile096',
                                                             'Tile097', 'Tile098', 'Tile101', 'Tile102', 'Tile103',
                                                             'Tile104', 'Tile105', 'Tile106', 'Tile107', 'Tile108',
                                                             'Tile111', 'Tile112', 'Tile113', 'Tile114', 'Tile115',
                                                             'Tile116', 'Tile117', 'Tile118', 'Tile121', 'Tile122',
                                                             'Tile123', 'Tile124', 'Tile125', 'Tile126', 'Tile127',
                                                             'Tile128', 'Tile131', 'Tile132', 'Tile133', 'Tile134',
                                                             'Tile135', 'Tile136', 'Tile137', 'Tile138', 'Tile141',
                                                             'Tile142', 'Tile143', 'Tile144', 'Tile145', 'Tile146',
                                                             'Tile147', 'Tile148', 'Tile151', 'Tile152', 'Tile153',
                                                             'Tile154', 'Tile155', 'Tile156', 'Tile157', 'Tile158',
                                                             'Tile161', 'Tile162', 'Tile163', 'Tile164', 'Tile165',
                                                             'Tile166', 'Tile167', 'Tile168']))
        np.testing.assert_almost_equal(
            m.CalFits.frequency_channels, np.arange(0, 768))
        self.assertEqual(m.CalFits.frequency_array[0], 167055000.0)
        self.assertEqual(m.CalFits.frequency_array[-1], 197735000.0)
        inds = np.where(np.array(m.CalFits.frequency_flags) == 0)
        self.assertEqual(len(inds[0]), 768)
        inds = np.where(np.array(m.CalFits.frequency_flags) == 1)
        self.assertEqual(len(inds[0]), 0)
        np.testing.assert_almost_equal(
            m.CalFits.gain_array[0, 0, 100, :], np.array([0.79296893-0.62239205j,  0.01225656+0.05076904j,
                                                          0.02343749-0.05114198j, -0.44148823-1.04256351j]))
        np.testing.assert_almost_equal(m.CalFits.amplitudes[0, 0, 100, :],
                                       np.array(
            [1.00805336, 0.05222757, 0.05625671, 1.13218838]))
        np.testing.assert_almost_equal(m.CalFits.phases[0, 0, 100, :],
                                       np.array(
            [-0.66545834,  1.33391101, -1.14107581, -1.97136534]))
        np.testing.assert_almost_equal(
            m.CalFits.convergence, hdu['RESULTS'].data)
        np.testing.assert_almost_equal(
            m.CalFits.baseline_weights, hdu['BASELINES'].data)
        m = CalMetrics(calfile, metafits_path=metafits, norm=False)
        self.assertFalse(m.CalFits.norm)
        np.testing.assert_almost_equal(
            m.CalFits.gain_array[0, 0, 100, :], np.array([-0.18806952+0.80088855j, -0.0177897 - 0.03553466j,
                                                          -0.00287501+0.03964559j,  0.21278879+0.79101208j]))
        np.testing.assert_almost_equal(m.CalFits.amplitudes[0, 0, 100, :],
                                       np.array(
            [0.82267406, 0.03973897, 0.0397497, 0.81913319]))
        np.testing.assert_almost_equal(m.CalFits.phases[0, 0, 100, :],
                                       np.array(
            [1.80144346, -2.03494748,  1.64318736,  1.30800907]))

    def test_variance_for_antpair(self):
        m = CalMetrics(calfile, metafits)
        variance = m.variance_for_antpair((0, 0))
        self.assertEqual(variance.shape, (1, 4))
        np.testing.assert_almost_equal(variance[0, :], np.array(
            [5.72259318e-03, 4.77730527e-06, 7.17090402e-06, 1.38760892e-02]), decimal=6)

    def test_variance_for_baselines_less_than(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        variances = m.variance_for_baselines_less_than(10)
        self.assertEqual(variances.shape, (1, 151, 4))
        np.testing.assert_almost_equal(variances[0, 0, :], np.array(
            [5.72259318e-03, 4.77730527e-06, 7.17090402e-06, 1.38760892e-02]), decimal=6)

    def test_skewness_across_uvcut(self):
        m = CalMetrics(calfile, metafits_path=metafits)
        skewness = m.skewness_across_uvcut(10)
        np.testing.assert_almost_equal(skewness, np.array(
            [1.17123073, 0.60040372, 0.5812123, 0.61172205]))

    def test_unused_baselines_percent(self):
        m = CalMetrics(calfile, metafits)
        percent = m.unused_baselines_percent()
        self.assertEqual(percent, 6.619094488188976)

    def test_unused_channels_percent(self):
        m = CalMetrics(calfile, metafits)
        percent = m.unused_channels_percent()
        self.assertEqual(percent, 0.0)

    def test_unused_antenna_percent(self):
        m = CalMetrics(calfile, metafits)
        percent = m.unused_antennas_percent()
        self.assertEqual(percent, 0.0)

    def test_non_converging_percent(self):
        m = CalMetrics(calfile, metafits)
        percent = m.non_converging_percent()
        self.assertEqual(percent, 15.755208333333334)

    def test_convergence_variance(self):
        m = CalMetrics(calfile, metafits)
        conv_variance = m.convergence_variance()
        np.testing.assert_almost_equal(
            conv_variance, 2.0160385085922676e-15)

    def test_initialize_metrics_dict(self):
        m = CalMetrics(calfile, metafits)
        m._initialize_metrics_dict()
        self.assertTrue(isinstance(m.metrics, OrderedDict))
        self.assertEqual(list(m.metrics.keys()), ['POLS', 'OBSID', 'UVCUT', 'M_THRESH',
                         'REF_ANTENNA', 'NTIME', 'START_FREQ', 'CH_WIDTH', 'NCHAN', 'ANTENNA', 'XX', 'YY'])
        self.assertTrue((m.metrics['POLS'] == ['XX', 'YY']))
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

    def test_run_metrics(self):
        m = CalMetrics(calfile, metafits)
        m.run_metrics()
        self.assertEqual(list(m.metrics.keys()), ['POLS',
                                                  'OBSID',
                                                  'UVCUT',
                                                  'M_THRESH',
                                                  'REF_ANTENNA',
                                                  'NTIME',
                                                  'START_FREQ',
                                                  'CH_WIDTH',
                                                  'NCHAN',
                                                  'ANTENNA',
                                                  'XX',
                                                  'YY',
                                                  'PERCENT_UNUSED_BLS',
                                                  'BAD_ANTS',
                                                  'PERCENT_NONCONVERGED_CHS',
                                                  'PERCENT_BAD_ANTS',
                                                  'RMS_CONVERGENCE',
                                                  'SKEWNESS',
                                                  'RECEIVER_VAR',
                                                  'DFFT_POWER'])
        self.assertEqual(m.metrics['PERCENT_UNUSED_BLS'], 6.619094488188976)
        self.assertEqual(
            m.metrics['PERCENT_NONCONVERGED_CHS'], 15.755208333333334)
        self.assertEqual(m.metrics['SKEWNESS'], 1.0281201713254011)
        self.assertEqual(m.metrics['RMS_CONVERGENCE'], 6.30082060084981e-08)
        self.assertEqual(m.metrics['RECEIVER_VAR'], 1193.2543323323973)
        self.assertEqual(list(m.metrics['XX'].keys()), ['RMS',
                                                        'RMS_MODZ',
                                                        'BAD_ANTS',
                                                        'SKEWNESS',
                                                        'DFFT_AMPS',
                                                        'DFFT_POWER']
                         )
        self.assertEqual(m.metrics['XX']['RMS'].shape, (128,))
        np.testing.assert_almost_equal(m.metrics['XX']['RMS'][10:15], np.array(
            [0.03460557, 0.03156614, 0.03036803, 0.03028277, 0.0307147]))
        self.assertEqual(m.metrics['XX']['RMS_MODZ'].shape, (128,))
        np.testing.assert_almost_equal(m.metrics['XX']['RMS_MODZ'][10:15], np.array(
            [-0.59405035, -2.01830524, -2.57972765, -2.6196828, -2.41728384]))
        np.testing.assert_almost_equal(
            m.metrics['XX']['BAD_ANTS'], np.array([17, 76, 80]))
        self.assertEqual(m.metrics['XX']['SKEWNESS'], 0.8703045026090006)
        self.assertEqual(m.metrics['XX']['DFFT_AMPS'].shape, (128, 768))
        np.testing.assert_almost_equal(m.metrics['XX']['DFFT_AMPS'][0, 0:10],
                                       np.array([0.52788904, 0.34286395, 0.29274779, 0.31075246, 0.37095048,
                                                 0.28703637, 0.03902724, 0.13170864, 0.28285714, 0.56093556]))
        self.assertEqual(m.metrics['XX']['DFFT_POWER'], 161026.11413894623)
        self.assertEqual(list(m.metrics['YY'].keys()), ['RMS',
                                                        'RMS_MODZ',
                                                        'BAD_ANTS',
                                                        'SKEWNESS',
                                                        'DFFT_AMPS',
                                                        'DFFT_POWER']
                         )
        self.assertEqual(m.metrics['YY']['RMS'].shape, (128,))
        np.testing.assert_almost_equal(m.metrics['YY']['RMS'][10:15], np.array(
            [0.03965437, 0.03600665, 0.0325334, 0.03592676, 0.03598789]))
        self.assertEqual(m.metrics['YY']['RMS_MODZ'].shape, (128,))
        np.testing.assert_almost_equal(m.metrics['YY']['RMS_MODZ'][10:15], np.array(
            [0.46793586, -1.13925103, -2.6695696, -1.1744511, -1.14751943]))
        np.testing.assert_almost_equal(
            m.metrics['YY']['BAD_ANTS'], np.array([17, 76, 80]))
        self.assertEqual(m.metrics['YY']['SKEWNESS'], 1.0281201713254011)
        self.assertEqual(m.metrics['YY']['DFFT_AMPS'].shape, (128, 768))
        np.testing.assert_almost_equal(m.metrics['YY']['DFFT_AMPS'][0, 0:10],
                                       np.array([0.60251954, 0.35075452, 0.48269033, 0.70336271, 0.52466862,
                                                0.17452712, 0.66135147, 0.67545852, 0.60077982, 0.5961326]))
        self.assertEqual(m.metrics['YY']['DFFT_POWER'], 166956.96124389415)

    def test_write_metrics(self):
        m = CalMetrics(calfile, metafits)
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
