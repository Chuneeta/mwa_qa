from mwa_qa import read_uvfits as ru
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np


class VisMetrics(object):
    def __init__(self, uvfits):
        self.uvfits = uvfits
        self.uvf = ru.UVfits(uvfits)

    def autos_for_antnum(self, antnum):
        auto_data = self.uvf.data_for_antpair((antnum, antnum))
        if len(auto_data) == 0:
            print("WARNING: No data found for Antenna Number {}, "
                  "maybe it is flagged".format(antnum))
        return auto_data

    def autos(self):
        annumbers = self.uvf.annumbers()
        auto_array = np.zeros((self.uvf.Ntimes, len(
            annumbers), self.uvf.Nfreqs, self.uvf.Npols), dtype=np.complex64)
        for i, num in enumerate(annumbers):
            auto_data = self.autos_for_antnum(num)
            if len(auto_data) > 0:
                auto_array[:, i, :, :] = auto_data
            else:
                # fill array with Nans
                auto_array[:, i, :, :] = np.nan
        return auto_array

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nfreqs
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['DATA'] = OrderedDict(
            [('AUTOS', OrderedDict()), ('CROSS', OrderedDict())])

    def run_metrics(self):
        self._initialize_metrics_dict()
        self.metrics['NANTS']
        self.metrics['NFREQS']
        self.metrics['NPOLS']
        autos = self.autos()
        # averaged over time
        avg_autos = np.nanmean(autos, axis=0)
        amps_avg_autos = np.abs(avg_autos)
        # metrics across antennas
        mean_amp_ant = np.nanmean(amps_avg_autos, axis=0)
        median_amp_ant = np.nanmedian(amps_avg_autos, axis=0)
        var_amp_ant = np.nanvar(amps_avg_autos, axis=0)
        rms_amp_ant = np.sqrt(np.nanmean(amps_avg_autos ** 2, axis=0))
        # metrics across frequency
        mean_amp_freq = np.nanmean(amps_avg_autos, axis=1)
        median_amp_freq = np.nanmedian(amps_avg_autos, axis=1)
        var_amp_freq = np.nanvar(amps_avg_autos, axis=1)
        rms_amp_freq = np.sqrt(np.nanmean(amps_avg_autos ** 2, axis=1))

        self.metrics['MEAN_AMP_ANT'] = mean_amp_ant
        self.metrics['MEDIAN_AMP_ANT'] = median_amp_ant
        self.metrics['VAR_AMP_ANT'] = var_amp_ant
        self.metrics['RMS_AMP_ANT'] = rms_amp_ant
        self.metrics['MEAN_AMP_FREQ'] = mean_amp_freq
        self.metrics['MEDIAN_AMP_FREQ'] = median_amp_freq
        self.metrics['VAR_AMP_FREQ'] = var_amp_freq
        self.metrics['RMS_AMP_FREQ'] = rms_amp_freq

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
