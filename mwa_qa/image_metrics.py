from collections import OrderedDict
from astropy.io import fits
from mwa_qa import image_utils as iu
from mwa_qa import json_utils as ju
import numpy as np

class ImgMetrics(object):
	def __init__(self, images=[], pols=[]):
		self.images = images
		self.pols = pols

	def _check_object(self):
		assert len(self.image) > 0, "At least one image should be specified"
		assert len(self.images) == len(self.pols), "Number of pols should be the as the number of images"
	
	def _get_index(self, pol):
		return np.where(np.array(self.pols) == pol)[0][0]

	def _initialize_metrics_dict(self, noise_box):
		self.metrics = OrderedDict()
		self.metrics['noise_box'] = noise_box
		for p in self.pols:
			self.metrics[p] = OrderedDict()
		if ('XX' in self.pols) and ('YY' in self.pols):
			self.metrics['XX_YY'] = OrderedDict()
		if ('V' in self.pols) and ('XX' in self.pols):
			self.metrics['V_XX'] = OrderedDict()
		if ('V' in self.pols) and ('YY' in self.pols):
			self.metrics['V_YY'] = OrderedDict()

	def run_metrics(self, noise_box=[100, 100]):
		self._initialize_metrics_dict(noise_box)
		keys = list(self.metrics.keys())
		for p in self.pols:
			ind = self._get_index(p)
			imagename = self.images[ind]
			self.metrics[p]['imagename'] = imagename
			self.metrics[p]['obs-date'] = iu.header(imagename)['DATE-OBS']
			self.metrics[p]['mean_all'] = float(iu.mean(imagename))
			self.metrics[p]['rms_all'] = float(iu.rms(imagename))
			self.metrics[p]['std_all'] = float(iu.std(imagename))
			self.metrics[p]['rms_box'] = float(iu.rms_for(imagename, noise_box[0], noise_box[1]))
			self.metrics[p]['std_box'] = float(iu.std_for(imagename, noise_box[0], noise_box[1]))

		if 'XX_YY' in keys:
			self.metrics['XX_YY']['rms_ratio_all'] = float(self.metrics['XX']['rms_all'] / self.metrics['YY']['rms_all'])
			self.metrics['XX_YY']['std_ratio_all'] = float(self.metrics['XX']['std_all'] / self.metrics['YY']['std_all'])
			self.metrics['XX_YY']['rms_ratio_box'] = float(self.metrics['XX']['rms_box'] / self.metrics['YY']['rms_box'])
			self.metrics['XX_YY']['std_ratio_box'] = float(self.metrics['XX']['std_box'] / self.metrics['YY']['std_box'])

		if 'V_XX' in keys:
			self.metrics['V_XX']['rms_ratio_all'] = float(self.metrics['V']['rms_all'] / self.metrics['XX']['rms_all'])
			self.metrics['V_XX']['std_ratio_all'] = float(self.metrics['V']['std_all'] / self.metrics['XX']['std_all'])
			self.metrics['V_XX']['rms_ratio_box'] = float(self.metrics['V']['rms_box'] / self.metrics['XX']['rms_box'])
			self.metrics['V_XX']['std_ratio_box'] = float(self.metrics['V']['std_box'] / self.metrics['XX']['std_box'])

		if 'V_YY' in keys:
			self.metrics['V_YY']['rms_ratio_all'] = float(self.metrics['V']['rms_all'] / self.metrics['YY']['rms_all'])
			self.metrics['V_YY']['std_ratio_all'] = float(self.metrics['V']['std_all'] / self.metrics['YY']['std_all'])
			self.metrics['V_YY']['rms_ratio_box'] = float(self.metrics['V']['rms_box'] / self.metrics['YY']['rms_box'])
			self.metrics['V_YY']['std_ratio_box'] = float(self.metrics['V']['std_box'] / self.metrics['YY']['std_box'])	

	def write_to(self, outfile=None):
		if outfile is None:
			outfile = self.images[0].replace('.fits', '_metrics.json')
		ju.write_metrics(self.metrics, outfile)
