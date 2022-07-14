from mwa_qa import read_metafits as rm
from mwa_qa import read_csolutions as rc
from mwa_qa import json_utils as ju
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np


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
        self.Csoln = rc.Csoln(calfile, metafits=metafits, pol=pol)
        self.Metafits = rm.Metafits(metafits, pol)

    def variance_for_antpair(self, antpair, norm=True):
        """
        Returns variance across frequency for the given tile pair
        - antpair : Antenna pair or tuple of antenna numbers e.g (102, 103)
        - norm : boolean, If True returns normalized gains else unormalized gains.
                                                                                                                                         Default is set to True.
        """
        gains = self.Csoln.gains_for_antpair(antpair, norm=norm)
        return np.nanvar(gains, axis=2)

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
        for i, bl in enumerate(bls):
            variances[:, i, :] = self.variance_for_antpair(bl, norm=norm)[
                :, :, :]
        return bls, variances

    def skewness_across_uvcut(self, uv_cut, norm=True):
        """
        Evaluates the Pearson skewness 3 * (mean - median) / std across the variances
        averaged over baseliness shorter than the given length
        - uv_cut : Baseline cut in metres, will use only baselines shorter than the given value
        - norm : boolean, If True returns normalized gains else unormalized gains.
        Default is set to True.
        """
        _, variances = self.variance_for_baselines_less_than(uv_cut, norm=norm)
        skewness = (3 * (np.nanmean(variances, axis=1) -
                         np.nanmedian(variances, axis=1))) / np.nanstd(variances, axis=1)
        return skewness

    def get_receivers(self, n=16):
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

    def smooth_calibration_precisions(self, window_length, polyorder):
        cal_precisions = self.Csoln.data(5)
        _sh = cal_precisions.shape
        sm_cal_precisions = np.copy(cal_precisions)
        for t in range(_sh[0]):
            inds_nnans = np.where(~np.isnan(cal_precisions[t, :]))[0]
            inds_nans = np.where(np.isnan(cal_precisions[t, :]))[0]
            sm_cal_precisions[t, :] = savgol_filter(
                cal_precisions[t, inds_nnans], window_length, polyorder)
        return sm_cal_precisions

    def apply_gaussian_filter1D_fft(self, data, sigma):
        filtered_data = np.copy(data)
        _sh = filtered_data.shape
        for t in range(_sh[0]):
            inds_nnans = np.where(~np.isnan(data))[0]
            inds_nans = np.where(np.isnan(data))[0]
            filtered_data[t, inds_nnans] = gaussian_filter1d(
                data[t, inds_nnans], sigma)
            filtered_data[t, inds_nans] = np.nan
        return filtered_data

    def _initialize_metrics_dict(self):
        """
        Initializes the metric dictionary with some of the default parameters
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

    def run_metrics(self, window_length=5, polyorder=4, sigma=1):
        self._initialize_metrics_dict()
        pols = self.metrics['POLS']
        receivers = self.metrics['RECEIVERS']
        ntimes = self.metrics['NTIMES']
        nfreqs = self.metrics['NFREQ']
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
        # metrics based on antennas connected to receivers
        mean_chisq = np.zeros((len(receivers), ntimes, 8, len(pols)))
        var_chisq = np.zeros((len(receivers), ntimes, 8, len(pols)))
        for i, r in enumerate(receivers):
            rcv_gains = self.Csoln.gains_for_receiver(r)
            rcv_amps = np.abs(rcv_gains)
            rcv_amps_mean = np.nanmean(rcv_amps, axis=1)
            chisq = (rcv_amps - rcv_amps_mean) / rcv_amps
            mean_chisq[i] = np.nanmean(chisq, axis=2)
            var_chisq[i] = np.nanvar(chisq, axis=2)

        # writing metrics to json file
        self.metrics['MEAN_AMP_ANT'] = mean_amp_ant
        self.metrics['MEDIAN_AMP_ANT'] = median_amp_ant
        self.metrics['VAR_AMP_ANT'] = var_amp_ant
        self.metrics['RMS_AMP_ANT'] = rms_amp_ant
        self.metrics['MEAN_AMP_FREQ'] = mean_amp_freq
        self.metrics['MEDIAN_AMP_FREQ'] = median_amp_freq
        self.metrics['VAR_AMP_FREQ'] = var_amp_freq
        self.metrics['RMS_AMP_FREQ'] = rms_amp_freq
        self.metrics['MEAN_RECEIVER_CHISQ'] = mean_chisq
        self.metrics['VAR_RECEIVER_CHISQ'] = var_chisq

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.calfile.replace('.fits', '_cal_metrics.json')
        ju.write_metrics(self.metrics, outfile)
