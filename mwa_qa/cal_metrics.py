from mwa_qa.read_metafits import Metafits
from mwa_qa.read_calfits import CalFits
from mwa_qa import json_utils as ju
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pol_dict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY':	3}


class CalMetrics(object):
    def __init__(self, calfits_path, metafits_path=None, pol='X',
                 norm=False, ref_antenna=None):
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
        - norm: Boolean, If True, the calibration solutions will be
                normlaized else unnormlaized solutions will be used.
                Default is set to False
        - ref_antenna:   Reference antenna number. If norm is True,
        a reference antenna is require for normalization.
        By default it uses the last antenna in the array.
        If the last antenna is flagged, it will return
        an error.
        """
        self.calfits_path = calfits_path
        self.metafits_path = metafits_path
        self.CalFits = CalFits(calfits_path, metafits_path=metafits_path,
                               pol=pol, norm=norm, ref_antenna=ref_antenna)
        self.Metafits = Metafits(metafits_path, pol)

    def variance_for_antpair(self, antpair):
        """
        Returns variance across frequency for the given tile pair
        - antpair:	Antenna pair or tuple of antenna numbers
                    e.g (102, 103)
        - norm:	Boolean, If True returns normalized gains
                else unormalized gains. Default is set to True.
        """
        gain_pairs = self.CalFits.gains_for_antpair(antpair)
        return np.nanvar(gain_pairs, axis=1)

    def variance_for_baselines_less_than(self, uv_cut):
        """
        Returns bls shorter than the specified cut and the variances
        calculated across frequency for each of the antenna pair
        - uv_cut:	Baseline cut in metres, will use only baselines
        shorter than the given value- norm
        """
        baselines = self.Metafits.baselines_less_than(uv_cut)
        _sh = self.CalFits.gain_array.shape
        variances = np.zeros((_sh[0], len(baselines), 4))
        for i, bl in enumerate(baselines):
            variances[:, i, :] = self.variance_for_antpair(bl)
        return variances

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
        variances = self.variance_for_baselines_less_than(uv_cut)
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
        cal_precisions = self.CalFits.convergence
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
        gains_fft = self.CalFits.gains_fft()
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

    def unused_baselines_percent(self):
        inds_nan = np.where(np.isnan(self.CalFits.baseline_weights))[0]
        inds_wg0 = np.where(self.CalFits.baseline_weights == 0)[0]
        return (len(inds_nan) + len(inds_wg0)) / \
            len(self.CalFits.baseline_weights) * 100

    def unused_channels_percent(self):
        inds = np.where(np.array(self.CalFits.frequency_flags) == 1)[0]
        return len(inds) / len(self.CalFits.frequency_flags) * 100

    def unused_antennas_percent(self):
        inds = np.where(np.array(self.CalFits.antenna_flags) == 1)[0]
        return len(inds) / len(self.CalFits.antenna_flags) * 100

    def non_converging_percent(self):
        sh = self.CalFits.convergence.shape
        count = 0
        for t in range(sh[0]):
            inds = np.where(np.isnan(self.CalFits.convergence[t, :]))[0]
            count += len(inds)
        return count / (sh[0] * sh[1]) * 100

    def convergence_variance(self):
        return np.nanmax(np.nanvar(self.CalFits.convergence, axis=1))

    def _initialize_metrics_dict(self):
        """
        Initializes the metric dictionary with some of the default
        parameters
        """
        self.metrics = OrderedDict()
        # assuming hyperdrive outputs 4 polarizations
        pols = ['XX', 'XY', 'YX', 'YY']
        receivers = np.unique(sorted(self.get_receivers()))
        self.metrics['POLS'] = pols
        self.metrics['OBSID'] = self.CalFits.obsid
        self.metrics['UVCUT'] = self.CalFits.uvcut
        self.metrics['M_THRESH'] = self.CalFits.m_thresh
        if self.CalFits.norm:
            self.metrics['REF_ANTNUM'] = self.CalFits.ref_antenna
        self.metrics['NTIME'] = self.CalFits.Ntime
        self.metrics['START_FREQ'] = self.CalFits.frequency_array[0]
        self.metrics['CH_WIDTH'] = self.CalFits.frequency_array[1] - \
            self.CalFits.frequency_array[0]
        self.metrics['NCHAN'] = self.CalFits.Nchan
        self.metrics['ANTENNA'] = self.CalFits.antenna
        self.metrics['RECEIVERS'] = receivers.tolist()
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()
        # NOTE:polynomial parameters - only the fitted
        # solutions will have poly metrics
        try:
            self.metrics['POLY_ORDER'] = self.CalFits.poly_order
            self.metrics['POLY_MSE'] = self.CalFits.poly_mse
        except AttributeError:
            pass

    def run_metrics(self, dly_cut=2000, sigma=2):
        self._initialize_metrics_dict()
        pols = self.metrics['POLS']
        receivers = self.metrics['RECEIVERS']
        ntimes = self.metrics['NTIME']
        gain_amps = self.CalFits.amplitudes
        # metrics across antennas
        var_amp_ant = np.nanvar(gain_amps, axis=1)
        rms_amp_ant = np.sqrt(np.nanmean(gain_amps ** 2, axis=1))
        rmsvar_amp_ant = np.sqrt(np.nanmean(
            np.nanmean(var_amp_ant, axis=0) ** 2, axis=0))
        # metrics amplitude across frequency
        var_amp_freq = np.nanvar(gain_amps, axis=2)
        rms_amp_freq = np.sqrt(np.nanmean(gain_amps ** 2, axis=2))
        rmsvar_amp_freq = np.sqrt(np.nanmean(
            np.nanmean(var_amp_freq, axis=0) ** 2, axis=0))
        # skewness across ucvut
        skewness = self.skewness_across_uvcut(self.metrics['UVCUT'])
        mskewness = np.nanmean(skewness, axis=0)
        # metrics based on antennas connected to receivers
        rcv_chisq = np.zeros((len(receivers), ntimes, 8, len(pols)))
        for i, r in enumerate(receivers):
            rcv_gains = self.CalFits.gains_for_receiver(r)
            rcv_amps = np.abs(rcv_gains)
        # ignoring zero division
            np.seterr(divide='ignore', invalid='ignore')
            rcv_amps_mean = np.nanmean(rcv_amps, axis=1)
            chisq = np.nansum(((rcv_amps - rcv_amps_mean) / rcv_amps), axis=2)
            rcv_chisq[i] = chisq
        mrcv_chisq = np.nanmean(np.nanmean(rcv_chisq, axis=1), axis=1)
        vmrcv_chisq = np.nanvar(mrcv_chisq, axis=0)
        # metric from delay spectrum
        # fft_spectrum = self.CalFits.gains_fft()
        delays = self.CalFits.delays()
        smfft_spectrum = self.apply_gaussian_filter1D_fft(sigma)
        # convergence metrics
        convergence_var = self.convergence_variance()
        # writing metrics to json file
        self.metrics['UNUSED_BLS'] = self.unused_baselines_percent()
        self.metrics['UNUSED_CHS'] = self.unused_channels_percent()
        self.metrics['UNUSED_ANTS'] = self.unused_antennas_percent()
        self.metrics['NON_CONVERGED_CHS'] = self.non_converging_percent()
        self.metrics['CONVERGENCE'] = self.CalFits.convergence
        self.metrics['CONVERGENCE_VAR'] = convergence_var
        self.metrics['RECEIVER_CHISQ'] = rcv_chisq
        # metrics for each pols
        for i, p in enumerate(['XX', 'YY']):
            self.metrics[p]['SKEWNESS_UVCUT'] = mskewness[pol_dict[p]]
            self.metrics[p]['AMPVAR_ANT'] = var_amp_ant[:, :, pol_dict[p]]
            self.metrics[p]['AMPRMS_ANT'] = rms_amp_ant[:, :, pol_dict[p]]
            self.metrics[p]['RMS_AMPVAR_ANT'] = rmsvar_amp_ant[pol_dict[p]]
            self.metrics[p]['AMPVAR_FREQ'] = var_amp_freq[:, :, pol_dict[p]]
            self.metrics[p]['AMPRMS_FREQ'] = rms_amp_freq[:, :, pol_dict[p]]
            self.metrics[p]['RMS_AMPVAR_FREQ'] = rmsvar_amp_freq[pol_dict[p]]
            # delay spectra
            self.metrics[p]['DFFT'] = np.abs(smfft_spectrum[:, :, :, i])
            self.metrics[p]['DFFT_POWER'] = np.nansum(
                np.abs(smfft_spectrum[:, :, :, i]))
            inds_pos = np.where(delays > dly_cut)[0]
            inds_neg = np.where(delays < -1 * dly_cut)[0]
            self.metrics[p]['DFFT_POWER_HIGH_PKPL'] = np.nansum(
                np.abs(smfft_spectrum[:, :, inds_pos, i]))
            self.metrics[p]['DFFT_POWER_HIGH_NKPL'] = np.nansum(
                np.abs(smfft_spectrum[:, :, inds_neg, i]))
            # receiver variance
            self.metrics[p]['RECEIVER_CHISQVAR'] = vmrcv_chisq[pol_dict[p]]
        # determining if the solutions passes the metrics basics tests
        status = 'FAIL' if np.sqrt(
            convergence_var) > self.CalFits.m_thresh else 'PASS'
        self.metrics['STATUS'] = status
        if status == 'FAIL':
            self.metrics['FAILURE_REASON'] = 'Converging results did not pass the requirement STD(CONV_RESULTS) [{}] < M_THRESH [{}]'.format(
                np.sqrt(convergence_var), self.metrics['M_THRESH'])

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.calfits_path.replace('.fits', '_cal_metrics.json')
        ju.write_metrics(self.metrics, outfile)
