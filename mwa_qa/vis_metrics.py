from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


class VisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        autos = self.uvf.data_for_antpairs(auto_antpairs)
        flags = self.uvf.flag_for_antpairs(auto_antpairs)
        autos[flags] = np.nan
        return autos
        # return self.uvf._amps_phs_array(auto_antpairs)

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NBLS'] = self.uvf.Nbls
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nfreqs
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['AUTOS'] = OrderedDict([
            ('XX', OrderedDict()), ('YY', OrderedDict())])
        self.metrics['REDUNDANT'] = OrderedDict([
            ('XX', OrderedDict()), ('YY', OrderedDict())])

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
            self.metrics['AUTOS'][p]['RMS_AMP_ANT'] = rms_amp_ant
            self.metrics['AUTOS'][p]['RMS_AMP_FREQ'] = rms_amp_freq
            self.metrics['AUTOS'][p]['MXRMS_AMP_ANT'] = np.nanmax(rms_amp_ant)
            self.metrics['AUTOS'][p]['MNRMS_AMP_ANT'] = np.nanmin(rms_amp_ant)
            self.metrics['AUTOS'][p]['MXRMS_AMP_FREQ'] = np.nanmax(
                rms_amp_freq)
            self.metrics['AUTOS'][p]['MNRMS_AMP_FREQ'] = np.nanmin(
                rms_amp_freq)
            # finding outliers in frequency channels > mena + 3 sigma
            threshold_freq_high = np.nanmedian(
                rms_amp_ant) + 3 * np.nanstd(rms_amp_ant)
            threshold_freq_low = np.nanmedian(
                rms_amp_ant) - 3 * np.nanstd(rms_amp_ant)
            inds_rms_freq = np.where((rms_amp_ant < threshold_freq_low) | (
                rms_amp_ant > threshold_freq_high))
            if len(inds_rms_freq) == 0:
                self.metrics['AUTOS'][p]['POOR_CHANNELS'] = []
                self.metrics['AUTOS'][p]['NPOOR_CHANNELS'] = 0
            else:
                self.metrics['AUTOS'][p]['POOR_CHANNELS'] = inds_rms_freq[0]
                self.metrics['AUTOS'][p]['NPOOR_CHANNELS'] = len(
                    inds_rms_freq[0])
            # finding outliers in antennas
            threshold_ant_high = np.nanmedian(
                rms_amp_freq) + 3 * np.nanstd(rms_amp_freq)
            threshold_ant_low = np.nanmedian(
                rms_amp_freq) - 3 * np.nanstd(rms_amp_freq)
            inds_rms_ant = np.where((rms_amp_freq < threshold_ant_low) | (
                rms_amp_freq > threshold_ant_high))
            if len(inds_rms_freq) == 0:
                self.metrics['AUTOS'][p]['POOR_ANTENNAS'] = []
                self.metrics['AUTOS'][p]['NPOOR_ANTENNAS'] = 0
            else:
                self.metrics['AUTOS'][p]['POOR_ANTENNAS'] = inds_rms_ant[0]
                self.metrics['AUTOS'][p]['NPOOR_ANTENNAS'] = len(
                    inds_rms_ant[0])
            # need to add FFT of autos here
        # redundant baselines
        red_dict = self.uvf.redundant_antpairs()
        red_keys = list(red_dict.keys())
        self.metrics['REDUNDANT']['RED_PAIRS'] = []
        for p in ['XX', 'YY']:
            self.metrics['REDUNDANT'][p]['POOR_BLS'] = []
            self.metrics['REDUNDANT'][p]['AMP_CHISQ'] = []
        if len(red_keys) > 0:
            red_values = np.array(list(red_dict.values()))
            for i, key in enumerate(red_keys):
                # selecting redundant grouos with at least
                # 10 redundant baselines
                if len(red_values[i]) > nbl_limit:
                    self.metrics['REDUNDANT']['RED_PAIRS'].append(key)
                    d = self.uvf.data_for_antpairs(red_values[i])
                    f = self.uvf.flag_for_antpairs(red_values[i])
                    d[f] = np.nan
                    d_amp = np.nanmean(np.abs(d), axis=0)
                    mean_amp = np.nanmean(d_amp, axis=0)[np.newaxis, :]
                    amp_chisq = np.nansum(
                        (d_amp - mean_amp) ** 2 / mean_amp, axis=1)
                    for p in ['XX', 'YY']:
                        mamp_chisq = np.nanmean(
                            amp_chisq[:, pol_dict[p]])
                        stamp_chisq = np.nanstd(
                            amp_chisq[pol_dict[p]])
                        threshold_high = mamp_chisq + (3 * stamp_chisq)
                        threshold_low = mamp_chisq - (3 * stamp_chisq)
                        inds = np.where((amp_chisq[:, pol_dict[p]]
                                         < threshold_low) |
                                        (amp_chisq[:, pol_dict[p]]
                                         > threshold_high))
                        self.metrics['REDUNDANT'][p]['AMP_CHISQ'].append(
                            amp_chisq[:, pol_dict[p]].tolist())
                        if len(inds[0]) > 0:
                            poor_bls = np.array(red_values[i])[inds[0]]
                            [self.metrics['REDUNDANT'][p]['POOR_BLS'].append(
                                tuple(bl)) for bl in poor_bls]

        self.metrics['REDUNDANT']['XX']['NPOOR_BLS'] = len(
            self.metrics['REDUNDANT']['XX']['POOR_BLS'])
        self.metrics['REDUNDANT']['YY']['NPOOR_BLS'] = len(
            self.metrics['REDUNDANT']['YY']['POOR_BLS'])

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
