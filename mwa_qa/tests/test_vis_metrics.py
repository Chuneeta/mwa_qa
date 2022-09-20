from mwa_qa.vis_metrics import VisMetrics
from collections import OrderedDict
import numpy as np
import unittest
import os

uvfits_path = '/Users/ridhima/Documents/mwa/calibration/mwa_qa/test_files/'
uvfits = os.path.join(uvfits_path, '1061315688_cal.uvfits')
uvfits_hex = os.path.join(uvfits_path, '1320411840_hyp_cal.uvfits')


class TestVisMetrics(unittest.TestCase):
    def test__init__(self):
        vis = VisMetrics(uvfits)
        self.assertTrue(vis.uvfits_path, uvfits)

    def test_initialize_metrics_dict(self):
        vis = VisMetrics(uvfits)
        vis._initialize_metrics_dict()
        self.assertEqual(list(vis.metrics.keys()), [
            'NANTS', 'NTIMES', 'NFREQS', 'NPOLS', 'OBSID', 'REDUNDANT'])
        self.assertEqual(vis.metrics['NANTS'], 128)
        self.assertEqual(vis.metrics['NFREQS'], 768)
        self.assertEqual(vis.metrics['NPOLS'], 4)
        self.assertEqual(vis.metrics['NTIMES'], 27)
        self.assertEqual(
            list(vis.metrics['REDUNDANT'].keys()), ['XX', 'YY'])
        self.assertTrue(isinstance(
            vis.metrics['REDUNDANT']['XX'], OrderedDict))
        self.assertTrue(isinstance(
            vis.metrics['REDUNDANT']['YY'], OrderedDict))

    def test_run_metrics(self):
        vis = VisMetrics(uvfits)
        vis.run_metrics()
        self.assertEqual(list(vis.metrics.keys()), [
            'NANTS', 'NTIMES', 'NFREQS', 'NPOLS', 'OBSID', 'REDUNDANT'])
        self.assertEqual(list(vis.metrics['REDUNDANT'].keys()), [
            'XX', 'YY', 'RED_PAIRS'])
        self.assertEqual(list(vis.metrics['REDUNDANT']['XX'].keys()), [
            'POOR_BLS', 'AMP_CHISQ', 'NPOOR_BLS'])
        self.assertEqual(vis.metrics['REDUNDANT']['RED_PAIRS'], [])
        np.testing.assert_equal(
            vis.metrics['REDUNDANT']['XX']['AMP_CHISQ'], np.array([]))
        # redundant
        vis = VisMetrics(uvfits_hex)
        vis.run_metrics()
        np.testing.assert_equal(
            vis.metrics['REDUNDANT']['RED_PAIRS'],
            [(27, 35, 54), (0, 70, 0), (28, -35, 54), (27, 105, 54),
             (55, 0, 108), (29, -105, 54), (55, 70, 108),
             (2, -140, -1), (56, -70, 108), (54, 140, 108),
             (56, -140, 107), (82, 35, 162), (29, -175, 53),
             (26, 175, 55), (83, -35, 162), (84, -105, 161),
             (2, -210, -1), (82, 105, 163), (110, 0, 216),
             (53, 210, 109), (57, -210, 107), (84, -175, 161),
             (110, 70, 216), (25, 245, 55), (30, -245, 53),
             (111, -70, 216), (82, 175, 163), (109, 140, 216),
             (3, -280, -1), (111, -140, 216), (137, 35, 270),
             (52, 280, 109), (80, 245, 163), (138, -35, 270),
             (85, -245, 161), (58, -280, 107), (137, 105, 270),
             (108, 210, 217), (112, -210, 215), (25, 315, 55),
             (138, -105, 270), (31, -315, 53), (200, 307, 399),
             (171, 412, 346), (228, 272, 453), (199, 377, 399),
             (227, 342, 453), (256, 237, 507), (171, 482, 346),
             (199, 447, 400), (255, 307, 507), (226, 412, 454),
             (170, 552, 346), (254, 377, 507), (198, 517, 400),
             (283, 272, 561), (226, 482, 454), (282, 342, 561),
             (254, 447, 508), (311, 237, 615), (198, 587, 400),
             (225, 552, 454), (282, 412, 562), (310, 307, 615),
             (253, 517, 508), (309, 377, 616), (281, 482, 562),
             (197, 657, 401), (338, 272, 669), (225, 622, 454),
             (252, 587, 508), (309, 447, 616), (337, 342, 670),
             (281, 552, 562), (337, 412, 670), (224, 692, 455),
             (308, 517, 616), (365, 307, 724), (252, 657, 509),
             (279, 622, 563), (336, 482, 670), (364, 377, 724),
             (308, 587, 616), (251, 727, 509), (364, 447, 724),
             (335, 552, 670), (279, 692, 563), (307, 657, 617),
             (363, 517, 724), (391, 412, 778), (335, 622, 670),
             (391, 482, 778), (363, 587, 725), (306, 727, 617),
             (334, 692, 671), (390, 552, 779), (362, 657, 725)])
        self.assertEqual(list(vis.metrics['REDUNDANT']['XX'].keys()), [
            'POOR_BLS', 'AMP_CHISQ', 'NPOOR_BLS'])
        self.assertEqual(len(vis.metrics['REDUNDANT']['XX']['AMP_CHISQ']), 97)
        self.assertEqual(
            len(vis.metrics['REDUNDANT']['XX']['AMP_CHISQ'][0]), 56)
        self.assertEqual(
            vis.metrics['REDUNDANT']['XX']['AMP_CHISQ'][0][0],
            677.5272216796875)
        self.assertEqual(
            len(vis.metrics['REDUNDANT']['XX']['POOR_BLS']), 81)
        self.assertEqual(vis.metrics['REDUNDANT']
                         ['XX']['POOR_BLS'][0][0], ([60, 65], 4))
        self.assertEqual(vis.metrics['REDUNDANT']['XX']['NPOOR_BLS'], 924)

    def test_write_to(self):
        vis = VisMetrics(uvfits)
        vis._initialize_metrics_dict()
        outfile = uvfits.replace('.uvfits', '_vis_metrics.json')
        vis.write_to()
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
        outfile = 'metrics.json'
        vis.write_to(outfile=outfile)
        self.assertTrue(os.path.exists(outfile))
        os.system('rm -rf {}'.format(outfile))
