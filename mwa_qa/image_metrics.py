from collections import OrderedDict
from mwa_qa.read_image import Image
from mwa_qa import json_utils as ju

pol_dict = {-5: 'XX', -6: 'YY', -7: 'XY', 4: 'V'}
# coordinates of PKS0023-26
srcnames = ['PKS0023_026']
srcpos = [(6.4549166666666675, -26.04)]


class ImgMetrics(object):
    def __init__(self, images=[], pix_box=[100, 100]):
        self.images = [Image(img, pix_box=pix_box) for img in images]

    def _check_object(self):
        assert len(self.image) > 0, "At least one image should be specified"

    def pols_from_image(self):
        return [img.polarization for img in self.images]

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['PIX_BOX'] = self.images[0].pix_box
        self.metrics['IMAGE_SIZE'] = self.images[0].image_size
        pol_convs = self.pols_from_image()
        for i, pc in enumerate(pol_convs):
            pol = pol_dict[pc]
            self.metrics[pol] = OrderedDict()
            self.metrics[pol]['IMAGENAME'] = self.images[i].fitspath
            self.metrics[pol]['IMAGE_ID'] = self.images[i].image_ID
            self.metrics[pol]['OBS-DATE'] = self.images[i].obsdate
            for src in srcnames:
                self.metrics[pol][src] = OrderedDict()

    def run_metrics(self, beam_const=1, deconvol=False):
        self._initialize_metrics_dict()
        pol_convs = self.pols_from_image()
        keys = list(self.metrics.keys())
        for i, pc in enumerate(pol_convs):
            pol = pol_dict[pc]
            self.metrics[pol]['MEAN_ALL'] = self.images[i].mean
            self.metrics[pol]['RMS_ALL'] = self.images[i].rms
            self.metrics[pol]['MEAN_BOX'] = self.images[i].mean_across_box
            self.metrics[pol]['RMS_BOX'] = self.images[i].rms_across_box
            for j, src in enumerate(srcnames):
                src_flux = self.images[i].src_flux(
                    srcpos[j], beam_const=beam_const, deconvol=deconvol)
                self.metrics[pol][src]['PEAK_FLUX'] = src_flux[0]
                self.metrics[pol][src]['INT_FLUX'] = src_flux[1]

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.images[0].fitspath.replace('.fits', '_metrics.json')
        ju.write_metrics(self.metrics, outfile)
