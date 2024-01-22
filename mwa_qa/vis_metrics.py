from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
from scipy import stats
import numpy as np

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


def unique_elm(tup_list):
    unq_list = []
    for tup in tup_list:
        if tup not in unq_list:
            unq_list.append(tup)
    return unq_list


class VisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nchan
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['OBSID'] = self.uvf.obsid
        self.metrics['NPOOR_BLS'] = 0
        self.metrics['POOR_BLS'] = 0
        self.metrics['REDUNDANT'] = OrderedDict([
            ('XX', OrderedDict()), ('YY', OrderedDict())])

    def run_metrics(self, nbl_limit=10, threshold=3.5):
        self._initialize_metrics_dict()
        # redundant baselines
        red_dict = self.uvf.redundant_antpairs()
        red_keys = list(red_dict.keys())
        self.metrics['REDUNDANT']['RED_PAIRS'] = []
        self.metrics['REDUNDANT']['THRESHOLD'] = threshold
        for p in ['XX', 'YY']:
            self.metrics['REDUNDANT'][p]['POOR_BLS'] = []
            self.metrics['REDUNDANT'][p]['AMP_CHISQ'] = []
            self.metrics['REDUNDANT'][p]['MODZ'] = []
            self.metrics['REDUNDANT'][p]['POOR_BLS_INDS'] = []
            self.metrics['REDUNDANT'][p]['NPOOR_BLS'] = 0
        if len(red_keys) > 0:
            red_values = list(red_dict.values())
            for i, key in enumerate(red_keys):
                # selecting redundant grouos with at least
                # 10 redundant baselines
                if len(red_values[i]) > nbl_limit:
                    self.metrics['REDUNDANT']['RED_PAIRS'].append(key)
                    d = self.uvf.data_for_antpairs(red_values[i])
                    f = self.uvf.flag_for_antpairs(red_values[i])
                    d[f] = np.nan
                    d_amp = np.nanmean(np.abs(d), axis=0)
                    mean_amp = np.nanmedian(d_amp, axis=0)[np.newaxis, :]
                    amp_chisq = np.nansum(
                        (d_amp - mean_amp) ** 2 / mean_amp, axis=1)
                    modz = stats.zscore(amp_chisq)
                    for p in ['XX', 'YY']:
                        inds = np.where(
                            (modz[:, pol_dict[p]] < -1 * threshold) | (modz[:, pol_dict[p]] > threshold))
                        self.metrics['REDUNDANT'][p]['AMP_CHISQ'].append(
                            amp_chisq[:, pol_dict[p]].tolist())
                        self.metrics['REDUNDANT'][p]['MODZ'].append(
                            modz[:, pol_dict[p]].tolist())
                        self.metrics['REDUNDANT'][p]['POOR_BLS_INDS'].append(
                            inds[0].tolist())
                        if len(inds[0]) > 0:
                            poor_bls = red_values[i][inds[0][0]]
                            self.metrics['REDUNDANT'][p]['POOR_BLS'].append(
                                poor_bls)

            self.metrics['REDUNDANT']['XX']['NPOOR_BLS'] = len(
                self.metrics['REDUNDANT']['XX']['POOR_BLS'])
            self.metrics['REDUNDANT']['YY']['NPOOR_BLS'] = len(
                self.metrics['REDUNDANT']['YY']['POOR_BLS'])
            poor_bls_all = self.metrics['REDUNDANT']['XX']['POOR_BLS'] + \
                self.metrics['REDUNDANT']['YY']['POOR_BLS']
            self.metrics['POOR_BLS'] = unique_elm(poor_bls_all)
            self.metrics['NPOOR_BLS'] = len(poor_bls_all)

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
