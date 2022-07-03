from collections import OrderedDict
from mwa_qa import read_csolutions as rc
from mwa_qa import read_metafits as rm
from mwa_qa import cal_metrics as cm
from mwa_qa.data import DATA_PATH
import unittest
import numpy as np
import os

calfile = os.path.join(DATA_PATH, 'test_1061315688.fits')
metafits = os.path.join(DATA_PATH, 'test_1061315688.metafits')

class TestCalMetrics(unittest.TestCase):
	def test_init__(self):
		m = cm.CalMetrics(calfile, metafits = metafits)
		self.assertEqual(m.calfile, calfile)
		self.assertEqual(m.Csoln.calfile, calfile)
		self.assertEqual(m.Csoln.Metafits.metafits, metafits)
		self.assertEqual(m. Csoln.Metafits.pol, 'X')

	def test_initialize_metrics_dict(self):
		m = cm.CalMetrics(calfile)
		metrics = m._initialize_metrics_dict()
		self.assertTrue(isinstance(metrics, OrderedDict))
		self.assertEqual(list(metrics.keys()), ['pols', 'freqs', 'tile_ids'])

	def test_run_metrics(self):
		pass
