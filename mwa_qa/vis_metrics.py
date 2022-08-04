from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np


class VisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        return self.uvf._amps_phs_array(auto_antpairs)

    def redundant_metrics(self):
        red_pairs = self.uvf.redundant_antpairs()
        red_keys = list(red_pairs.keys())
        red_values = list(red_pairs.values())
        amp_chisqs, phs_chisqs = [], []
        amp_diff, phs_diff = [], []
        for i in range(len(red_values)):
            d = self.uvf._amps_phs_array(red_values[i])
            d_amp = np.nanmean(d[:, :, :, 0], axis=0)
            d_phs = np.nanmean(d[:, :, :, 1], axis=0)
            mean_amp = np.nanmean(d_amp, axis=0)[np.newaxis, :]
            mean_phs = np.nanmean(d_phs, axis=0)[np.newaxis, :]
            amp_chisqs.append(
                np.nansum((d_amp - mean_amp) ** 2 / mean_amp, axis=1))
            phs_chisqs.append(
                np.nansum((d_amp - mean_phs) ** 2 / mean_phs, axis=1))
            amp_diff = np.diff(d_amp, axis=0)
            phs_diff = np.diff(d_phs, axis=0)
        return red_keys, amp_chisqs, phs_chisqs, amp_diff, phs_diff

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NBLS'] = self.uvf.Nbls
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nfreqs
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['AUTOS'] = OrderedDict()
        self.metrics['REDUNDANT'] = OrderedDict()

    def run_metrics(self):
        self._initialize_metrics_dict()
        # auto correlations
        autos = self.autos()
        # time difference
        diff_autos = np.diff(autos, axis=0)
        auto_amps = autos[:, :, :, 0]
        auto_phs = autos[:, :, :, 1]
        avg_auto_rms = np.sqrt(np.nanmean(
            np.nanmean(auto_amps, axis=0), axis=1) ** 2)
        mavg_auto_rms = np.nanmean(avg_auto_rms)
        vavg_auto_rms = np.nanvar(avg_auto_rms)
        vdiff_auto_amps = np.nanvar(diff_autos[:, :, :, 0], axis=2)
        vdiff_auto_phs = np.nanvar(diff_autos[:, :, :, 1], axis=2)
        # redundant baselines
        reds, amp_chisqs, phs_chisqs, amp_diff, phs_diff \
            = self.redundant_metrics()

        # writing mwtric to dict
        self.metrics['AUTOS']['MEAN_RMS_AMPS'] = mavg_auto_rms
        self.metrics['AUTOS']['VAR_RMS_AMPS'] = vavg_auto_rms
        self.metrics['AUTOS']['VAR_DIFF_AMPS'] = vdiff_auto_amps
        self.metrics['AUTOS']['VAR_DIFF_PHS'] = vdiff_auto_phs
        self.metrics['AUTOS']['MX_VAR_DIFF_AMPS'] = np.nanmax(
            np.max(vdiff_auto_amps))
        self.metrics['AUTOS']['MX_VAR_DIFF_PHS'] = np.nanmax(
            np.max(vdiff_auto_phs))
        self.metrics['REDUNDANT']['RED_PAIRS'] = reds
        self.metrics['REDUNDANT']['AMP_CHISQ'] = amp_chisqs
        self.metrics['REDUNDANT']['PHS_CHISQ'] = phs_chisqs
        self.metrics['REDUNDANT']['VAR_AMP_CHISQ'] = amp_chisqs
        self.metrics['REDUNDANT']['VAR_PHS_CHISQ'] = phs_chisqs
        self.metrics['REDUNDANT']['AMP_DIFF'] = amp_diff
        self.metrics['REDUNDANT']['PHS_DIFF'] = phs_diff
        self.metrics['REDUNDANT']['MVAR_AMP_DIFF'] = np.nanmean(
            np.nanvar(amp_diff, axis=1))
        self.metrics['REDUNDANT']['MVAR_PHS_DIFF'] = np.nanmean(
            np.nanvar(phs_diff, axis=1))

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
