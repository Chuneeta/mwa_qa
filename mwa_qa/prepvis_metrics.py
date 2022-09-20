from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


class PrepvisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        autos = self.uvf.data_for_antpairs(auto_antpairs)
        flags = self.uvf.flag_for_antpairs(auto_antpairs)
        autos[flags] = np.nan
        return autos

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nfreqs
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['OBSID'] = self.uvf.obsid
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()

    def run_metrics(self, nbl_limit=10):
        self._initialize_metrics_dict()
        # auto correlations
        autos = self.autos()
        autos_amps = np.abs(autos)  # amplitude
        # rms across antenna
        autos_amps_rms_ant = np.sqrt(np.nanmean(autos_amps, axis=1))
        # rms across frequency
        autos_amps_rms_freq = np.sqrt(np.nanmean(autos_amps, axis=2))

        for p in ['XX', 'YY']:
            # taking the maximum rms out of the timestamps
            rms_amp_ant = np.nanmean(
                autos_amps_rms_ant[:, :, pol_dict[p]], axis=0)
            rms_amp_freq = np.nanmean(
                autos_amps_rms_freq[:, :, pol_dict[p]], axis=0)
            self.metrics[p]['RMS_AMP_ANT'] = rms_amp_ant
            self.metrics[p]['RMS_AMP_FREQ'] = rms_amp_freq
            self.metrics[p]['MXRMS_AMP_ANT'] = np.nanmax(rms_amp_ant)
            self.metrics[p]['MNRMS_AMP_ANT'] = np.nanmin(rms_amp_ant)
            self.metrics[p]['MXRMS_AMP_FREQ'] = np.nanmax(
                rms_amp_freq)
            self.metrics[p]['MNRMS_AMP_FREQ'] = np.nanmin(
                rms_amp_freq)
            # finding outliers in antennas
            threshold_ant_high = np.nanmedian(
                rms_amp_freq) + 3 * np.nanstd(rms_amp_freq)
            threshold_ant_low = np.nanmedian(
                rms_amp_freq) - 3 * np.nanstd(rms_amp_freq)
            inds_rms_ant = np.where((rms_amp_freq < threshold_ant_low) | (
                rms_amp_freq > threshold_ant_high))
            if len(inds_rms_ant) == 0:
                self.metrics[p]['POOR_ANTENNAS'] = []
                self.metrics[p]['NPOOR_ANTENNAS'] = 0
            else:
                self.metrics[p]['POOR_ANTENNAS'] = inds_rms_ant[0]
                self.metrics[p]['NPOOR_ANTENNAS'] = len(
                    inds_rms_ant[0])

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace(
                '.uvfits', '_prepvis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
