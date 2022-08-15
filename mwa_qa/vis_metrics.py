from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YY': 3}


class VisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        return self.uvf.data_for_antpairs(auto_antpairs)
        # return self.uvf._amps_phs_array(auto_antpairs)

    def redundant_metrics(self):
        red_pairs = self.uvf.redundant_antpairs()
        red_keys = list(red_pairs.keys())
        if len(red_keys) > 0:
            red_values = list(red_pairs.values())
            amp_chisqs, phs_chisqs = OrderedDict(), OrderedDict()
            for i in range(len(red_values)):
                d = self.uvf.data_for_antpairs(red_values[i])
                # d = self.uvf._amps_phs_array(red_values[i])
                d_amp = np.nanmean(np.abs(d), axis=0)
                d_phs = np.nanmean(np.angle(d), axis=0)
                mean_amp = np.nanmean(d_amp, axis=0)[np.newaxis, :]
                mean_phs = np.nanmean(d_phs, axis=0)[np.newaxis, :]
                amp_chisq = np.nansum(
                    (d_amp - mean_amp) ** 2 / mean_amp, axis=1)
                phs_chisq = np.nansum(
                    (d_amp - mean_phs) ** 2 / mean_phs, axis=1)
                amp_chisqs[i] = amp_chisq
                phs_chisqs[i] = phs_chisq
                amp_diff = np.diff(d_amp, axis=0)
                phs_diff = np.diff(d_phs, axis=0)
            return red_keys, amp_chisqs, phs_chisqs, amp_diff, phs_diff
        else:
            return None

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

    def run_metrics(self):
        self._initialize_metrics_dict()
        # auto correlations
        autos = self.autos()
        # time difference
        diff_autos = np.diff(autos, axis=0)
        auto_amps = np.abs(autos)
        auto_phs = np.angle(autos)
        avg_amp_rms = np.sqrt(np.nanmean(
            np.nanmean(auto_amps, axis=0), axis=1) ** 2)
        avg_phs_rms = np.sqrt(np.nanmean(
            np.nanmean(auto_phs, axis=0), axis=1) ** 2)
        mavg_amp_rms = np.nanmean(avg_amp_rms, axis=0)
        mavg_phs_rms = np.nanmean(avg_phs_rms, axis=0)
        vavg_amp_rms = np.nanvar(avg_amp_rms, axis=0)
        vavg_phs_rms = np.nanvar(avg_phs_rms, axis=0)
        # writing mwtric to dict
        for p in ['XX', 'YY']:
            self.metrics['AUTOS'][p]['MEAN_RMS_AMPS'] = mavg_amp_rms[pol_dict[p]]
            self.metrics['AUTOS'][p]['VAR_RMS_AMPS'] = vavg_amp_rms[pol_dict[p]]
            self.metrics['AUTOS'][p]['MEAN_RMS_PHS'] = mavg_phs_rms[pol_dict[p]]
            self.metrics['AUTOS'][p]['VAR_RMS_PHS'] = vavg_phs_rms[pol_dict[p]]
        # auto difference:
        if self.uvf.Ntimes > 1:
            vdiff_auto_amps = np.nanvar(np.abs(diff_autos), axis=2)
            vdiff_auto_phs = np.nanvar(np.angle(diff_autos), axis=2)
            self.metrics['AUTOS'][p]['VAR_DIFF_AMPS'] = vdiff_auto_amps[:, pol_dict[p]]
            self.metrics['AUTOS'][p]['VAR_DIFF_PHS'] = vdiff_auto_phs[:, pol_dict[p]]
            for p in ['XX', 'YY']:
                self.metrics['AUTOS'][p]['MX_VAR_DIFF_AMPS'] = np.nanmax(
                    np.max(vdiff_auto_amps[:, pol_dict[p]]))
            self.metrics['AUTOS'][p]['MX_VAR_DIFF_PHS'] = np.nanmax(
                np.max(vdiff_auto_phs))
        # redundancy metrics
        red_metrics = self.redundant_metrics()
        if red_metrics is not None:
            reds, amp_chisqs, phs_chisqs, amp_diff, phs_diff = red_metrics
            var_ampxx_chisqs, var_ampyy_chisqs = [], []
            var_phsxx_chisqs, var_phsyy_chisqs = [], []
            for i in range(len(reds)):
                var_ampxx_chisqs.append(np.nanmean(amp_chisqs[i][:, 0]))
                var_ampyy_chisqs.append(np.nanmean(amp_chisqs[i][:, 1]))
                var_phsxx_chisqs.append(np.nanmean(phs_chisqs[i][:, 0]))
                var_phsyy_chisqs.append(np.nanmean(phs_chisqs[i][:, 1]))

            self.metrics['REDUNDANT']['RED_PAIRS'] = reds
            self.metrics['REDUNDANT']['AMP_CHISQ'] = amp_chisqs
            self.metrics['REDUNDANT']['PHS_CHISQ'] = phs_chisqs
            self.metrics['REDUNDANT']['XX']['VAR_AMP_CHISQ'] = np.nanvar(
                var_ampxx_chisqs)
            self.metrics['REDUNDANT']['XX']['VAR_PHS_CHISQ'] = np.nanvar(
                var_phsxx_chisqs)
            self.metrics['REDUNDANT']['YY']['VAR_AMP_CHISQ'] = np.nanvar(
                var_ampyy_chisqs)
            self.metrics['REDUNDANT']['YY']['VAR_AMP_CHISQ'] = np.nanvar(
                var_phsyy_chisqs)

            for p in ['XX', 'YY']:
                self.metrics['REDUNDANT'][p]['AMP_DIFF'] = amp_diff[:,
                                                                    :, pol_dict[p]]
                self.metrics['REDUNDANT'][p]['PHS_DIFF'] = phs_diff[:,
                                                                    :, pol_dict[p]]
                self.metrics['REDUNDANT'][p]['MVAR_AMP_DIFF'] = np.nanmean(
                    np.nanvar(amp_diff[:, :, pol_dict[p]], axis=1))
                self.metrics['REDUNDANT'][p]['MVAR_PHS_DIFF'] = np.nanmean(
                    np.nanvar(phs_diff[:, :, pol_dict[p]], axis=1))

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
