from collections import OrderedDict
from astropy.io import fits
from mwa_qa import image_utils as iu
from mwa_qa import json_utils as ju
import numpy as np

pol_dict = {-5 : 'XX', -6 : 'YY', -7 : 'XY', 4 : 'V'}

class ImgMetrics(object):
	def __init__(self, images=[]):
		self.images = images

	def _check_object(self):
		assert len(self.image) > 0, "At least one image should be specified"
		#assert len(self.images) == len(self.pols), "Number of pols should be the as the number of images"

	def pols_from_image(self):
		pol_convs = [iu.pol_convention(image) for image in self.images]
		return pol_convs

	def _initialize_metrics_dict(self, noise_box):
		self.metrics = OrderedDict()
		self.metrics['noise_box'] = noise_box
		pol_convs = self.pols_from_image()
		for i, pc in enumerate(pol_convs):
			pol = 'YX' if 'XYi' in self.images[i] else pol_dict[pc]
			self.metrics[pol] = OrderedDict()
		if -5 and -6 in pol_convs:
			self.metrics['{}_{}'.format(pol_dict[-5], pol_dict[-6])] = OrderedDict()
		if 4 and -5 in pol_convs:
			self.metrics['{}_{}'.format(pol_dict[4], pol_dict[-5])] = OrderedDict()
		if 4 and -6 in pol_convs:
			self.metrics['{}_{}'.format(pol_dict[4], pol_dict[-6])] = OrderedDict()

	def run_metrics(self, noise_box=[100, 100]):
		self._initialize_metrics_dict(noise_box)
		keys = list(self.metrics.keys())
		pol_convs = self.pols_from_image()
		for i , pc in enumerate(pol_convs):
			imagename = self.images[i]
			pol = 'YX' if 'XYi' in imagename else pol_dict[pc]
			self.metrics[pol]['imagename'] = imagename
			self.metrics[pol]['obs-date'] = iu.header(imagename)['DATE-OBS']
			self.metrics[pol]['mean_all'] = float(iu.mean(imagename))
			self.metrics[pol]['rms_all'] = float(iu.rms(imagename))
			self.metrics[pol]['std_all'] = float(iu.std(imagename))
			self.metrics[pol]['rms_box'] = float(iu.rms_for(imagename, noise_box[0], noise_box[1]))
			self.metrics[pol]['std_box'] = float(iu.std_for(imagename, noise_box[0], noise_box[1]))

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
