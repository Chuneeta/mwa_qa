from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np
import json


class CalMetrics(object):
    def __init__(self, calfile, metafits=None, pol='X'):
        """
        Object that takes in .fits containing the calibration solutions file readable by astropy
        and initializes them as global varaibles
        - calfile : .fits file containing the calibration solutions
        - metafits : Metafits with extension *.metafits or _ppds.fits containing information
                     on an observation done with MWA
        - pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
        with the given pol is provided. Default is X.
        """
        self.calfile = calfile
        self.Csoln = rc.Csoln(calfile, metafits = metafits, pol = pol)
        self.Metafits = rm.Metafits(metafits, pol)

    def variance_for_tilepair(self, tile_pair, norm = True):
        """
        Returns variance across frequency for the given tile pair
         - tile_pair : Tile pair or tuple of tile numbers e.g (102, 103)
         - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
        gains = self.Csoln.gains_for_tilepair(tile_pair, norm = norm)
        return np.nanvar(gains, axis = 2)

    def variance_for_baselines_less_than(self, uv_cut, norm=True):
        """
        Returns bls shorter than the specified cut and the variances calculated across frequency for
        each of the antenna pair
        - baseline_cut : Baseline cut in metres, will use only baselines shorter than the given value
        - norm : boolean, If True returns normalized gains else unormalized gains.
                 Default is set to True.
        """
        baseline_dict = self.Metafits.get_baselines_less_than(uv_cut)
        bls = list(baseline_dict.keys())
        _sh = self.Csoln.gains().shape
        variances = np.zeros((_sh[0], len(bls), _sh[3]))
        for i , bl in enumerate(bls):
            variances[:, i, :] = self.variance_for_tilepair(bl, norm = norm)[:, :, :]
        return bls, variances

    def skewness_across_baselines(self, uv_cut, norm=True):
        """
        Evaluates the Pearson skewness 3 * (mean - median) / std across the variances 
        averaged over baseliness shorter than the given length
        - uv_cut : Baseline cut in metres, will use only baselines shorter than the given value
        - norm : boolean, If True returns normalized gains else unormalized gains.
        Default is set to True.
        """
        _, variances = self.variance_for_baselines_less_than(uv_cut, norm = norm)
        skewness = (3 * ( np.nanmean(variances, axis = 1) - np.nanmedian(variances, axis = 1))) / np.nanstd(variances, axis=1)
        return skewness

    def get_receivers(self, n = 16):
        """
        Returns the receivers connected to the various tiles in the array
        - n : Number of receivers in the array. Optional, enabled if metafits is not provided.
              Default is 16.
        """
        if self.Metafits.metafits is None:
            receivers = list(np.arange(1, n + 1))
        else:
            receivers = self.Metafits.receivers()
        return receivers

    def _initialize_metrics_dict(self):
        """
        Initializes the metric dictionary with some of the default parameters
        """
        metrics = OrderedDict()
        _, freqs, _ = self.Csoln.freqs_info()
        _, tile_ids, tile_flags = self.Csoln.tile_info()
        metrics['pols'] = ['XX', 'XY', 'YX', 'YY']
        metrics['freqs'] = freqs
        metrics['tile_ids'] = tile_ids
        metrics['uvcut'] = 40 # temporarily, will be adjusted
        return metrics

    def run_metrics(self):
        metrics = OrderedDict()
        metrics['mean_freq'] = np.nanmean(self.Csoln.amplitudes(), axis = 2)
        metrics['median_freq'] = np.nanmedian(self.Csoln.amplitudes(), axis = 2)
        metrics['variance_freq'] = np.nanvar(self.Csoln.amplitudes(), axis = 2)
        metrics['rms_freq'] = np.sqrt(np.nanmean(np.abs(gains) ** 2, axis = 2))
        metrics['mean_time'] = np.nanmean(self.Csoln.amplitudes(), axis = 0)
        metrics['median_time'] = np.nanmedian(self.Csoln.amplitudes(), axis = 0)
        metrics['variance_time'] = np.nanvar(self.Csoln.amplitudes(), axis = 0)
        metrics['rms_time'] = np.sqrt(np.nanmean(np.abs(gains) ** 2, axis = 0))
        metrics['skewness_across_baselines'] = self.skewness_across_baselines(metrics['uvcut'])
        return metrics

    def write_metrics(self, metrics, filename = None):
        """
        Writing metrics to output files
        - metrics : dictionary
        - filename : Output file
        """
        if filename is None:
            filename = self.calfile.replace('.fits', '_metrics.json')
        metrics = self.run_metrics()
        utils.write_metrics(metrics, filename)

