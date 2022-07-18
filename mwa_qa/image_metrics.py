from collections import OrderedDict
from mwa_qa import image_utils as iu
from mwa_qa import json_utils as ju

pol_dict = {-5: 'XX', -6: 'YY', -7: 'XY', 4: 'V'}
# coordinates of PKS0023-26
ra = 6.4549166666666675
dec = -26.04


class ImgMetrics(object):
    def __init__(self, images=[]):
        self.images = images

    def _check_object(self):
        assert len(self.image) > 0, "At least one image should be specified"

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
            self.metrics['{}_{}'.format(
                pol_dict[-5], pol_dict[-6])] = OrderedDict()
        if 4 and -5 in pol_convs:
            self.metrics['{}_{}'.format(
                pol_dict[4], pol_dict[-5])] = OrderedDict()
        if 4 and -6 in pol_convs:
            self.metrics['{}_{}'.format(
                pol_dict[4], pol_dict[-6])] = OrderedDict()

    def run_metrics(self, noise_box=[100, 100], constant=1.):
        self._initialize_metrics_dict(noise_box)
        keys = list(self.metrics.keys())
        pol_convs = self.pols_from_image()
        pols = []
        for i, pc in enumerate(pol_convs):
            imagename = self.images[i]
            pol = pol_dict[pc]
            pols.append(pols)
            self.metrics[pol]['IMAGENAME'] = imagename
            self.metrics[pol]['OBSDATE'] = iu.header(imagename)['DATE-OBS']
            self.metrics[pol]['MEAN_ALL'] = float(iu.mean(imagename))
            self.metrics[pol]['RMS_ALL'] = float(iu.rms(imagename))
            self.metrics[pol]['STD_ALL'] = float(iu.std(imagename))
            self.metrics[pol]['RMS_BOX'] = float(
                iu.rms_for(imagename, noise_box[0], noise_box[1]))
            self.metrics[pol]['STD_BOX'] = float(
                iu.std_for(imagename, noise_box[0], noise_box[1]))
            # flux density of PKS0023-26
            pks_tflux = iu.pix_flux(imagename, ra, dec, constant)[
                'GAUSS_TFLUX']
            self.metrics[pol]['PKS0023_026'] = pks_tflux

        if 'XX_YY' in keys:
            self.metrics['XX_YY']['RMS_RATIO_ALL'] = float(
                self.metrics['XX']['RMS_ALL'] / self.metrics['YY']['RMS_ALL'])
            self.metrics['XX_YY']['STD_RATIO_ALL'] = float(
                self.metrics['XX']['STD_ALL'] / self.metrics['YY']['STD_ALL'])
            self.metrics['XX_YY']['RMS_RATIO_BOX'] = float(
                self.metrics['XX']['RMS_BOX'] / self.metrics['YY']['RMS_BOX'])
            self.metrics['XX_YY']['STD_RATIO_BOX'] = float(
                self.metrics['XX']['STD_BOX'] / self.metrics['YY']['STD_BOX'])

        if 'V_XX' in keys:
            self.metrics['V_XX']['RMS_RATIO_ALL'] = float(
                self.metrics['V']['RMS_ALL'] / self.metrics['XX']['RMS_ALL'])
            self.metrics['V_XX']['STD_RATIO_ALL'] = float(
                self.metrics['V']['STD_ALL'] / self.metrics['XX']['STD_ALL'])
            self.metrics['V_XX']['RMS_RATIO_BOX'] = float(
                self.metrics['V']['RMS_BOX'] / self.metrics['XX']['RMS_BOX'])
            self.metrics['V_XX']['STD_RATIO_BOX'] = float(
                self.metrics['V']['STD_BOX'] / self.metrics['XX']['STD_BOX'])

        if 'V_YY' in keys:
            self.metrics['V_YY']['RMS_RATIO_ALL'] = float(
                self.metrics['V']['RMS_ALL'] / self.metrics['YY']['RMS_ALL'])
            self.metrics['V_YY']['STD_RATIO_ALL'] = float(
                self.metrics['V']['STD_ALL'] / self.metrics['YY']['STD_ALL'])
            self.metrics['V_YY']['RMS_RATIO_BOX'] = float(
                self.metrics['V']['RMS_BOX'] / self.metrics['YY']['RMS_BOX'])
            self.metrics['V_YY']['STD_RATIO_BOX'] = float(
                self.metrics['V']['STD_BOX'] / self.metrics['YY']['STD_BOX'])

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.images[0].replace('.fits', '_metrics.json')
        ju.write_metrics(self.metrics, outfile)
