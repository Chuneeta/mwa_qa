from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
from scipy import stats
import numpy as np

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


def unique_elm(input_list):
    unq_list = []
    for elm in input_list:
        if elm not in unq_list:
            unq_list.append(elm)
    return unq_list


def converter(input_list, output_list):
    for elements in input_list:
        if type(elements) == list:
            converter(elements, output_list)
        else:
            output_list.append(elements)
    return output_list


def search_group(red_pairs, antp):
    for i, gp in enumerate(red_pairs):
        if antp in gp:
            return i


class VisMetrics(object):
    def __init__(self, uvfits_path, cutoff_threshold=3.5):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)
        self.cutoff_threshold = cutoff_threshold

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

    def run_metrics(self, nbl_limit=10):
        self._initialize_metrics_dict()
        # redundant baselines
        red_dict = self.uvf.redundant_antpairs()
        red_keys = list(red_dict.keys())
        self.metrics['REDUNDANT']['RED_GROUPS'] = []
        self.metrics['REDUNDANT']['RED_PAIRS'] = []
        self.metrics['REDUNDANT']['THRESHOLD'] = self.cutoff_threshold
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
                    self.metrics['REDUNDANT']['RED_GROUPS'].append(key)
                    self.metrics['REDUNDANT']['RED_PAIRS'].append(
                        red_values[i])
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
                            (modz[:, pol_dict[p]] < -1 * self.cutoff_threshold) | (modz[:, pol_dict[p]] > self.cutoff_threshold))
                        self.metrics['REDUNDANT'][p]['AMP_CHISQ'].append(
                            amp_chisq[:, pol_dict[p]].tolist())
                        self.metrics['REDUNDANT'][p]['MODZ'].append(
                            modz[:, pol_dict[p]].tolist())
                        self.metrics['REDUNDANT'][p]['POOR_BLS_INDS'].append(
                            inds[0].tolist())
                        if len(inds[0]) > 0:
                            poor_bls = [red_values[i][inds[0][k]]
                                        for k in range(len(inds[0]))]
                            self.metrics['REDUNDANT'][p]['POOR_BLS'].append(
                                poor_bls)

        self.metrics['REDUNDANT']['XX']['NPOOR_BLS'] = len(
            self.metrics['REDUNDANT']['XX']['POOR_BLS'])
        self.metrics['REDUNDANT']['YY']['NPOOR_BLS'] = len(
            self.metrics['REDUNDANT']['YY']['POOR_BLS'])
        poor_bls_all = self.metrics['REDUNDANT']['XX']['POOR_BLS'] + \
            self.metrics['REDUNDANT']['YY']['POOR_BLS']
        self.metrics['POOR_BLS'] = converter(unique_elm(poor_bls_all), [])
        self.metrics['NPOOR_BLS'] = len(poor_bls_all)

        # finding the antennas contributing to the poor bls
        modz_gridxx = np.zeros((self.metrics['NANTS'], self.metrics['NANTS']))
        modz_gridyy = np.zeros((self.metrics['NANTS'], self.metrics['NANTS']))
        for i, antp in enumerate(self.metrics['REDUNDANT']['RED_PAIRS']):
            for j, (a1, a2) in enumerate(antp):
                group_number = search_group(
                    self.metrics['REDUNDANT']['RED_PAIRS'], (a1, a2))
                modz_gridxx[a1, a2] = self.metrics['REDUNDANT']['XX']['MODZ'][group_number][j]
                modz_gridyy[a1, a2] = self.metrics['REDUNDANT']['YY']['MODZ'][group_number][j]

        # sigma rule
        modz_xx_sum = np.sum(modz_gridxx, axis=1)
        modz_yy_sum = np.sum(modz_gridyy, axis=1)
        lthreshxx_sum = np.nanmean(
            modz_xx_sum) - self.cutoff_threshold * np.nanstd(modz_xx_sum)
        uthreshxx_sum = np.nanmean(
            modz_xx_sum) + self.cutoff_threshold * np.nanstd(modz_xx_sum)
        lthreshyy_sum = np.nanmean(
            modz_yy_sum) - self.cutoff_threshold * np.nanstd(modz_yy_sum)
        uthreshyy_sum = np.nanmean(
            modz_yy_sum) + self.cutoff_threshold * np.nanstd(modz_yy_sum)
        inds_xx = np.where((modz_xx_sum < lthreshxx_sum) |
                           (modz_xx_sum > uthreshxx_sum))
        inds_yy = np.where((modz_yy_sum < lthreshyy_sum) |
                           (modz_yy_sum > uthreshyy_sum))
        self.metrics['REDUNDANT']['XX']['POOR_ANTS'] = inds_xx[0].tolist()
        self.metrics['REDUNDANT']['YY']['POOR_ANTS'] = inds_yy[0].tolist()
        self.metrics['POOR_ANTS'] = unique_elm(self.metrics['REDUNDANT']['XX']['POOR_ANTS'] +
                                               self.metrics['REDUNDANT']['YY']['POOR_ANTS'])
        self.metrics['NPOOR_ANTS'] = len(self.metrics['POOR_ANTS'])

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace('.uvfits', '_vis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
