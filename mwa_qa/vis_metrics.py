from mwa_qa import read_uvfits as ru
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

pol_dict = ru.pol_dict


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
        self.metrics['AUTOS'] = OrderedDict(
            [('XX', OrderedDict()), ('YY', OrderedDict())])

    def run_metrics(self, dev=3):
        self._initialize_metrics_dict()
        self.metrics['NANTS']
        self.metrics['NFREQS']
        self.metrics['NPOLS']
        autos = self.autos()
        # metrics across frequency
        used_pols = ['XX', 'YY']
        for i, p in enumerate(used_pols):
            avg_autos = autos[:, :, :, pol_dict[p]]
            amps_avg_autos = np.abs(avg_autos)
            var_amp_freq = np.nanvar(amps_avg_autos, axis=2)
            rms_amp_freq = np.sqrt(np.nanmean(amps_avg_autos ** 2, axis=2))
            mrms_amp_freq = np.nanmean(rms_amp_freq, axis=0)
            mstd_amp_freq = np.nanstd(rms_amp_freq, axis=0)
            min_lim = mrms_amp_freq - dev * mstd_amp_freq
            max_lim = mrms_amp_freq + dev * mstd_amp_freq
            # picking poor behaving timestamps
            poor_times = []
            for i in range(self.metrics['NANTS']):
                inds_t = np.where((rms_amp_freq[:, i] < min_lim[i]) | (
                    rms_amp_freq[:, i] > max_lim[i]))[0]
            poor_times.append(inds_t.tolist())
            median_amp_time = np.nanmedian(amps_avg_autos, axis=0)
            rms_amp_time = np.sqrt(np.nanmean(amps_avg_autos ** 2, axis=0))
            median_amp_ant = np.nanmedian(amps_avg_autos, axis=1)
            rms_amp_ant = np.sqrt(np.nanmean(amps_avg_autos ** 2, axis=1))
            # writing to json files
            self.metrics['AUTOS'][p]['POOR_TIMES'] = inds_t
            self.metrics['AUTOS'][p]['MEDIAN_AMP_TIME'] = median_amp_time
            self.metrics['AUTOS'][p]['RMS_AMP_TIME'] = rms_amp_time
            self.metrics['AUTOS'][p]['MEDIAN_AMP_ANT'] = median_amp_ant
            self.metrics['AUTOS'][p]['RMS_AMP_ANT'] = rms_amp_ant
            self.metrics['AUTOS'][p]['VAR_AMP_FREQ'] = var_amp_freq
            self.metrics['AUTOS'][p]['RMS_AMP_FREQ'] = rms_amp_freq

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
