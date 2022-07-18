import numpy as np
import unittest
from mwa_qa import coord_utils as ct


class TestCoord(unittest.TestCase):
    def test_deg2hms(self):
        ra_str = ct.deg2hms(0)
        self.assertEqual(ra_str, '0h0m0.0s')
        ra_str = ct.deg2hms(30.01)
        self.assertEqual(ra_str, '2h0m2.4s')

    def test_negative_ra(self):
        with self.assertRaises(Exception):
            ct.deg2hms(-30.)

    def test_deg2dms(self):
        dec_str = ct.deg2dms(0)
        self.assertEqual(dec_str, '0d0m0.0s')
        dec_str = ct.deg2dms(-30.01)
        self.assertEqual(dec_str, '-30d0m36.0s')

    def test_hms2deg(self):
        ra_d = ct.hms2deg('0:0:0.0')
        self.assertEqual(ra_d, 0.0)
        ra_d = ct.hms2deg('2:0:2.4')
        self.assertEqual(ra_d, 30.01)

    def test_negative_hms(self):
        with self.assertRaises(Exception):
            ct.hms2deg('-30:0:0')
        with self.assertRaises(Exception):
            ct.hms2deg('30:-2:0')
        with self.assertRaises(Exception):
            ct.hms2deg('30:0:-3.0')

    def test_dms2deg(self):
        dec_d = ct.dms2deg('0:0:0')
        self.assertEqual(dec_d, 0.0)
        dec_d = ct.dms2deg('-30:0:36.0')
        self.assertEqual(dec_d, -30.01)
