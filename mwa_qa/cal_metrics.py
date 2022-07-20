from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
from mwa_qa import json_utils as ju
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import itertools

pol_dict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY':	3}


class CalMetrics(object):
    def __init__(self, calfile, metafits=None, pol='X'):
        """
        Object that takes in .fits containing the calibration solutions
        file readable by astropy and initializes them as global
        varaibles
        - calfile:	.fits file containing the calibration solutions
        - metafits:	Metafits with extension *.metafits or _ppds.fits
                                containing information
        - pol: 	Polarization, can be either 'X' or 'Y'. It should be
                specified so that information associated on an
                observation done with MWA with the given pol is
                provided. Default is X.
        """
        self.calfile = calfile
        self.Csoln = rc.Csoln(calfile, metafits=metafits, pol=pol)
        self.Metafits = rm.Metafits(metafits, pol)

    def variance_for_antpair(self, antpair, norm=True):
        """
        Returns variance across frequency for the given tile pair
        - antpair:	Antenna pair or tuple of antenna numbers
                    e.g (102, 103)
        - norm:		Boolean, If True returns normalized gains
                    else unormalized gains. Default is set to True.
        """
        gain_pairs = self.Csoln.gains_for_antpair(antpair, norm=norm)
        return np.nanvar(gain_pairs, axis=1)

    def variance_for_baselines_less_than(self, uv_cut, norm=True):
        """
        Returns bls shorter than the specified cut and the variances
                calculated across frequency for each of the antenna pair
        - baseline_cut:	Baseline cut in metres, will use only baselines
                        shorter than the given value- norm : boolean,
                        If True returns normalized gains else unormalized
        """
        baseline_dict = self.Metafits.baselines_less_than(uv_cut)
        bls = list(baseline_dict.keys())
        _sh = self.Csoln.gains().shape
        variances = np.zeros((_sh[0], len(bls), _sh[3]))
        for i, bl in enumerate(bls):
            variances[:, i, :] = self.variance_for_antpair(bl, norm=norm)
        return bls, variances

    def skewness_across_uvcut(self, uv_cut, norm=True):
        """
        Evaluates the Pearson skewness 3 * (mean - median) / std across the
        variances averaged over baseliness shorter than the given
        uv length
        - uv_cut:	Baseline cut in metres, will use only baselines shorter
                    than the given value
        - norm:		Boolean, If True returns normalized gains else unormalized
                    gains. Default is set to True.
        """
        _, variances = self.variance_for_baselines_less_than(uv_cut, norm=norm)
        vmean = np.nanmean(variances, axis=1)
        vmedian = np.nanmedian(variances, axis=1)
        vstd = np.nanstd(variances, axis=1)
        skewness = 3 * (vmean - vmedian) / vstd
        return skewness

    def get_receivers(self, n=16):
        """
        Returns the receivers connected to the various tiles in the array
        - n:	Number of receivers in the array. Optional, enabled if
                If metafits is not provided. Default is 16.
        """
        if self.Metafits.metafits is None:
            receivers = list(np.arange(1, n + 1))
        else:
            receivers = self.Metafits.receivers()
        return receivers

    def smooth_calibration_precisions(self, window_length, polyorder):
        cal_precisions = self.Csoln.data(5)
        _sh = cal_precisions.shape
        sm_cal_precisions = np.copy(cal_precisions)
        for t in range(_sh[0]):
            inds_nnans = np.where(~np.isnan(cal_precisions[t, :]))[0]
            inds_nans = np.where(np.isnan(cal_precisions[t, :]))[0]
            sm_cal_precisions[t, inds_nnans] = savgol_filter(
                cal_precisions[t, inds_nnans], window_length, polyorder)
            sm_cal_precisions[t, inds_nans] = np.nan
        return sm_cal_precisions

    def apply_gaussian_filter1D_fft(self, sigma):
        gains_fft = self.Csoln.gains_fft()
        _sh = gains_fft.shape
        # we will be using only xx and yy polarizations
        gains_fft_sm = np.zeros(
            (_sh[0], _sh[1], _sh[2], 2))
        for t in range(_sh[0]):
            for i in range(_sh[1]):
                gains_fft_xx = np.abs(gains_fft[t, i, :, 0])
                gains_fft_yy = np.abs(gains_fft[t, i, :, 3])
                inds_nnans_xx = np.where(~np.isnan(gains_fft_xx))[0]
                inds_nnans_yy = np.where(~np.isnan(gains_fft_yy))[0]
                inds_nans_xx = np.where(np.isnan(gains_fft_xx))[0]
                inds_nans_yy = np.where(np.isnan(gains_fft_yy))[0]
                gains_fft_sm[t, i, inds_nnans_xx, 0] = gaussian_filter1d(
                    gains_fft_xx[inds_nnans_xx], sigma)
                gains_fft_sm[t, i, inds_nnans_yy, 1] = gaussian_filter1d(
                    gains_fft_yy[inds_nnans_yy], sigma)
                gains_fft_sm[t, i, inds_nans_xx, 0] = np.nan
                gains_fft_sm[t, i, inds_nans_yy, 1] = np.nan
        return gains_fft_sm

    def flagged_baselines_percent(self):
        bls_weights = self.Csoln.data(6)
        inds = np.where(np.isnan(bls_weights))[0]
        return len(inds) / len(bls_weights) * 100

    def flagged_channels_percent(self):
        _, _, ch_flags = self.Csoln.freqs_info()
        inds = np.where(np.array(ch_flags) == 1)[0]
        return len(inds) / len(ch_flags) * 100

    def flagged_antennas_percent(self):
        _, _, ant_flags = self.Csoln.ant_info()
        inds = np.where(np.array(ant_flags) == 1)[0]
        return len(inds) / len(ant_flags) * 100

    def non_converging_percent(self):
        convergence = self.Csoln.data(5)
        _sh = convergence.shape
        count = 0
        for t in range(_sh[0]):
            inds = np.where(np.isnan(convergence[t, :]))[0]
            count += len(inds)
        return count / (_sh[0] * _sh[1]) * 100

    def convergence_variance(self):
        return np.nanvar(self.Csoln.data(5), axis=1)

    def _flag_table(self):
        _d = [['ANTENNAS', self.metrics['FLAGGED_ANTS']],
              ['BASELINES', self.metrics['FLAGGED_BLS']],
              ['CHANNELS', self.metrics['FLAGGED_CHS']],
              ['CONVERGENCE', self.metrics['NON_CONVERGED_CHS']]
              ]
        return pd.DataFrame(data=_d, columns=['FIELD', '%FLAG'])

    def _convegence_table(self):
        convergence_var = self.metrics['CONVERGENCE_VAR']
        m_thresh = self.metrics['M_THRESH']
        _d = []
        for t in range(self.metrics['NTIMES']):
            _d.append([convergence_var[t], m_thresh])
            return pd.DataFrame(data=_d, columns=['VAR', 'M_THRESH'])

    def _variance_antenna_stats(self):
        var = self.metrics['VAR_AMP_FREQ']
        mvar = np.nanmean(var, axis=1)
        stdvar = np.nanstd(var, axis=1)
        error = var - mvar
        return error, stdvar

    def variance_antenna_outliers(self, pol, dev=3):
        error, std = self._variance_antenna_stats()
        _sh = error.shape
        pol_ind = pol_dict[pol.upper()]
        d_error = error[:, :, pol_ind]
        d_std = std[:, pol_ind]
        annumbers = self.Metafits.annumbers()
        poor_antennas = []
        for t in range(_sh[0]):
            inds = np.where(d_error[t, :] > dev * d_std[t])[0]
            outliers = np.array(annumbers)[inds]
            poor_antennas.append(outliers.tolist())
            return poor_antennas

    def _variance_table(self, pol):
        var = self.metrics['VAR_AMP_FREQ']
        error, std = self._variance_antenna_stats()
        pol_ind = pol_dict[pol.upper()]
        _sh = var.shape
        _d = []
        for i in range(_sh[1]):
            dvar = [['ANTENNA {}'.format(
                    i), var[t, i, pol_ind], error[t, i, pol_ind],
                std[t, pol_ind]] for t in range(_sh[0])]
            dvar = list(itertools.chain(*dvar))
            _d.append(dvar)
        columns = ['ANTENNA', 'VAR', 'ERROR', 'STD']
        columns *= _sh[0]
        return pd.DataFrame(data=_d, columns=columns)

    def _initialize_metrics_dict(self):
        """
        Initializes the metric dictionary with some of the default
        parameters
        """
        self.metrics = OrderedDict()
        _, freqs, _ = self.Csoln.freqs_info()
        annumbers, _, _ = self.Csoln.ant_info()
        hdr = self.Csoln.header(0)
        # assuming hyperdrive outputs 4 polarizations
        pols = ['XX', 'XY', 'YX', 'YY']
        receivers = np.unique(sorted(self.get_receivers()))
        self.metrics['POLS'] = pols
        self.metrics['OBSID'] = hdr['OBSID']
        self.metrics['UVCUT'] = hdr['UVW_MIN']
        self.metrics['NITER'] = hdr['MAXITER']
        self.metrics['NTIMES'] = self.Csoln.ntimeblocks()
        self.metrics['FREQ_START'] = freqs[0]
        self.metrics['FREQ_WIDTH'] = freqs[1] - freqs[0]
        self.metrics['NFREQ'] = len(freqs)
        self.metrics['ANNUMS'] = annumbers
        self.metrics['RECEIVERS'] = receivers.tolist()
        self.metrics['M_THRESH'] = self.Csoln.header(0)['M_THRESH']
        self.metrics['S_THRESH'] = self.Csoln.header(0)['S_THRESH']
        self.metrics['CONVERGENCE'] = OrderedDict()
        self.metrics['DELAY_SPECTRUM'] = OrderedDict()
        self.metrics['POOR_ANTS'] = OrderedDict()

    def make_clickable(self, val):
        return f'<a target="_blank" href="{val}">{val}</a>'

    def run_metrics(self, window_length=19, polyorder=4, sigma=2,
                    html=None, html_link=None):
        self._initialize_metrics_dict()
        pols = self.metrics['POLS']
        receivers = self.metrics['RECEIVERS']
        ntimes = self.metrics['NTIMES']
        gain_amps = self.Csoln.amplitudes()
        # metrics across antennas
        mean_amp_ant = np.nanmean(gain_amps, axis=1)
        median_amp_ant = np.nanmedian(gain_amps, axis=1)
        var_amp_ant = np.nanvar(gain_amps, axis=1)
        rms_amp_ant = np.sqrt(np.nanmean(gain_amps ** 2, axis=1))
        # metrics amplitude across frequency
        mean_amp_freq = np.nanmean(gain_amps, axis=2)
        median_amp_freq = np.nanmedian(gain_amps, axis=2)
        var_amp_freq = np.nanvar(gain_amps, axis=2)
        rms_amp_freq = np.sqrt(np.nanmean(gain_amps ** 2, axis=2))

        # skewness across ucvut
        # skewness = self.skewness_across_uvcut(self.metrics['UVCUT'])
        # metrics based on antennas connected to receivers
        rcv_chisq = np.zeros((len(receivers), ntimes, 8, len(pols)))
        for i, r in enumerate(receivers):
            rcv_gains = self.Csoln.gains_for_receiver(r)
            rcv_amps = np.abs(rcv_gains)
            rcv_amps_mean = np.nanmean(rcv_amps, axis=1)
            chisq = np.nansum(((rcv_amps - rcv_amps_mean) / rcv_amps), axis=2)
            rcv_chisq[i] = chisq
        # metrics from convergence
        sm_cal_precisions = self.smooth_calibration_precisions(
            window_length=window_length, polyorder=polyorder)
        sm_cal_error = (self.Csoln.data(5) - sm_cal_precisions) ** 2
        # metric from delay spectrum
        fft_spectrum = self.apply_gaussian_filter1D_fft(sigma)
        # writing metrics to json file
        self.metrics['FLAGGED_BLS'] = self.flagged_baselines_percent()
        self.metrics['FLAGGED_CHS'] = self.flagged_channels_percent()
        self.metrics['FLAGGED_ANTS'] = self.flagged_antennas_percent()
        self.metrics['NON_CONVERGED_CHS'] = self.non_converging_percent()
        self.metrics['CONVERGENCE_VAR'] = self.convergence_variance()
        self.metrics['MEAN_AMP_ANT'] = mean_amp_ant
        self.metrics['MEDIAN_AMP_ANT'] = median_amp_ant
        self.metrics['VAR_AMP_ANT'] = var_amp_ant
        self.metrics['RMS_AMP_ANT'] = rms_amp_ant
        self.metrics['MEAN_AMP_FREQ'] = mean_amp_freq
        self.metrics['MEDIAN_AMP_FREQ'] = median_amp_freq
        self.metrics['VAR_AMP_FREQ'] = var_amp_freq
        self.metrics['RMS_AMP_FREQ'] = rms_amp_freq
        self.metrics['RECEIVER_CHISQ'] = rcv_chisq
        # self.metrics['SKEWNESS_UCVUT'] = skewness
        self.metrics['CONVERGENCE']['SM_PRECISIONS'] = sm_cal_precisions
        self.metrics['CONVERGENCE']['SM_PRECISIONS_ERROR'] = sm_cal_error
        self.metrics['DELAY_SPECTRUM']['DFFT'] = fft_spectrum
        self.metrics['POOR_ANTS']['XX'] = self.variance_antenna_outliers('XX')
        self.metrics['POOR_ANTS']['YY'] = self.variance_antenna_outliers(
            'YY')

        if html:
            if html_link is None:
                html_link = 'cal_metrics.html'
            self.metrics['HTML'] = html_link
            # create a HTML table for convergence
            df1 = self._flag_table()
            df2 = self._convegence_table()
            df3 = self._variance_table('XX')
            df4 = self._variance_table('YY')
            with open(html_link, 'w') as f:
                f.write('<center>'
                        + '<h1> {} </h1><br><hr>'.format(self.metrics['OBSID'])
                        + '<h2> FLAGS </h2>' +
                        df1.to_html(index=False, border=2,
                                    justify="center") + '<be><hr>'
                        + '<h2> CALIBRAITON RESULTS </h2>' +
                        df2.to_html(index=False, border=2,
                                    justify="center") + '<br><hr>'
                        + '<h2> GAIN VARAIANCE - XX </h2>' +
                        df3.to_html(index=False, border=2,
                                    justify="center") + '<br><hr>'
                        + '<h2> GAIN VARAIANCE - YY </h2>' +
                        df4.to_html(index=False, border=2,
                                    justify="center") + '<br><hr>'
                        + '</center>')

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.calfile.replace('.fits', '_cal_metrics.json')
        ju.write_metrics(self.metrics, outfile)
