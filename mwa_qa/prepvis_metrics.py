from mwa_qa.read_uvfits import UVfits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np
import pylab
import itertools

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


class PrepvisMetrics(object):
    def __init__(self, uvfits_path):
        self.uvfits_path = uvfits_path
        self.uvf = UVfits(uvfits_path)

    def autos(self, manual_flags, ex_annumbers=[]):
        auto_antpairs = self.uvf.auto_antpairs()
        autos = self.uvf.data_for_antpairs(
            auto_antpairs)  # (only xx and yy pols)
        _sh = autos.shape
        # flags are properly propages for the autocorrelations, therefore
        # we are mually flagging the band edges and centre frequency only
        # for obsids below 13*
        if manual_flags:
            flags = self._evaluate_edge_flags()
            flags = flags[np.newaxis, np.newaxis, :, np.newaxis]
            flags = flags.repeat(_sh[0], axis=0).repeat(
                _sh[1], axis=1).repeat(_sh[3], axis=3)
        else:
            flags = np.zeros((_sh), dtype=bool)
        autos[flags] = np.nan
        if len(ex_annumbers) > 0:
            autos[:, ex_annumbers, :, :] = np.nan
        return autos

    def _evaluate_edge_flags(self):
        edges_ncut = self.uvf.channel_width / 40000.
        if edges_ncut == 1:
            flag_chans = [0, 1, 16, 30, 31]
        elif edges_ncut == 2:
            flag_chans = [0, 16, 31]
        else:
            raise ValueError(
                'Channel width/frequency resolution not supported')
        Ncoarse_chans = 32
        nbands = np.split(np.zeros((self.uvf.Nfreqs), dtype=bool),
                          int(self.uvf.Nfreqs / Ncoarse_chans))
        flags = []
        for band in nbands:
            band[flag_chans] = True
            flags.append(band)
        return np.array(flags).flatten()

    def _plot_mode(self, data, mode):
        if mode == 'amp':
            return np.abs(data)
        elif mode == 'phs':
            return np.angle(data)
        elif mode == 'real':
            return np.real(data)
        elif mode == 'imag':
            return np.imag(data)
        elif mode == 'log':
            return np.log10(np.abs(data))
        else:
            raise ValueError("mode {} is not recognized".format(mode))

    def plot_spectra_across_chan(self, freq_chan, mode='log',
                                 manual_flags=True, ex_annumbers=[], save=None, figname=None):
        autos = self.autos(manual_flags=manual_flags,
                           ex_annumbers=ex_annumbers)
        _sh = autos.shape
        plot_data = self._plot_mode(
            autos[:, :, freq_chan, :], mode=mode)
        fig = pylab.figure(figsize=(9, 7))
        pylab.suptitle('Channel {}'.format(freq_chan))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            for j in range(_sh[1]):
                ax.plot(plot_data[:, j, i], '.-', alpha=0.5)
            ax.set_title(list(pol_dict.keys())[i])
            ax.grid(ls='dotted')
            if i % 2 == 0:
                pylab.ylabel(mode, fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Timestamp', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_f{}_{}_{}.png'.format(freq_chan, mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def plot_spectra_across_time(self, timestamp, mode='log',
                                 manual_flags=True, ex_annumbers=[], save=None, figname=None):
        autos = self.autos(manual_flags=manual_flags,
                           ex_annumbers=ex_annumbers)
        _sh = autos.shape
        plot_data = self._plot_mode(
            autos[timestamp, :, :, :], mode=mode)
        fig = pylab.figure(figsize=(9, 7))
        pylab.suptitle('Timetamp {}'.format(timestamp))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            for j in range(_sh[1]):
                ax.plot(plot_data[j, :, i], '.-', alpha=0.5)
            ax.set_title(list(pol_dict.keys())[i])
            ax.grid(ls='dotted')
            if i % 2 == 0:
                pylab.ylabel(mode, fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Frequency channel', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_t{}_{}_{}.png'.format(timestamp, mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def plot_spectra_2D(self, annumber, manual_flags=True, mode='log',
                        save=None, figname=None):
        autos = self.autos(manual_flags=manual_flags)
        plot_data = self._plot_mode(
            autos[:, annumber, :, :], mode=mode)
        fig = pylab.figure(figsize=(8, 7))
        pylab.suptitle('Ant {} -- {}'.format(annumber, mode))
        for i in range(4):
            ax = pylab.subplot(2, 2, i + 1)
            vmin = np.nanmin(plot_data[:, :, i])
            vmax = np.nanmax(plot_data[:, :, i])
            im = ax.imshow(plot_data[:, :, i],
                           aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(list(pol_dict.keys())[i])
            pylab.colorbar(im)
            if i % 2 == 0:
                pylab.ylabel('Timestamp', fontsize=12)
            if i // 2 > 0:
                pylab.xlabel('Frequency Channel', fontsize=12)
        pylab.subplots_adjust(hspace=0.2, wspace=0.3)
        if save:
            if figname is None:
                outfile = self.uvfits_path.replace(
                    '.uvfits', '_{}_{}.png'.format(mode, pol))
            else:
                outfile = figname
            pylab.savefig(outfile)
        else:
            pylab.show()

    def flag_occupancy(self, data):
        """
        Checking if the flag occupancy of any antenna/tile is greater than 50 percent.
        If so, the tile is discarded and included in the list of bad tiles.
        """
        _sh = data.shape
        count = len(np.where(~np.isnan(data))[0]) / _sh[0]
        percent_gddata = np.count_nonzero(data > 0., axis=1) / count * 100
        # discarding antenna with less than 50% of data
        inds = np.where(percent_gddata < 50.)
        return percent_gddata, inds

    def calculate_rms(self, data):
        """
        Calculating root mean square(rms) across freq 
        """
        # calculating rms across frequency
        _sh = data.shape
        rms = np.sqrt(np.nansum(data ** 2, axis=1) / _sh[1])
        return rms

    def calculate_mod_zscore(self, data):
        median_data = np.nanmedian(data, axis=0)
        diff_data = data - median_data
        mad = np.nanmedian(np.abs(diff_data), axis=0)
        mod_zscore = diff_data / mad / 1.4826
        return np.nanmean(mod_zscore, axis=1)

    def iterative_mod_zscore(self, data, threshold, niter):
        bad_inds = []
        modz_dict = {}
        mod_zscore = self.calculate_mod_zscore(data)
        inds = np.where((mod_zscore < -threshold)
                        | (mod_zscore > threshold))
        count = 1
        modz_dict[count - 1] = mod_zscore
        while (count <= niter and len(inds[0]) > 0):
            bad_inds.append(inds[0])
            data[inds[0]] = np.nan
            mod_zscore = self.calculate_mod_zscore(data)
            inds = np.where(np.abs(mod_zscore) > threshold)
            modz_dict[count] = mod_zscore
            count += 1
        return modz_dict, np.array(bad_inds).flatten()

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NFREQS'] = self.uvf.Nfreqs
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['OBSID'] = self.uvf.obsid
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()

    def run_metrics(self, manual_flags=True, ex_annumbers=[], threshold=3, niter=10):
        self._initialize_metrics_dict()
        # auto correlations
        autos = self.autos(manual_flags=manual_flags,
                           ex_annumbers=ex_annumbers)
        # amplitude averaged over time
        autos_amps = np.abs(np.nanmean(autos, axis=0))
        # normalizing by the median
        autos_amps_norm = autos_amps / np.nanmedian(autos_amps, axis=0)
        # finding misbehaving antennas
        for p in ['XX', 'YY']:
            bad_ants = []
            _, bd_inds = self.flag_occupancy(
                autos_amps_norm[:, :, pol_dict[p]])
            if len(bd_inds) > 0:
                bad_ants.append(self.uvf.antenna_numbers[bd_inds])
            autos_amps_norm[bd_inds, :, pol_dict[p]] = np.nan
            # calculating root mean square
            rms = self.calculate_rms(
                autos_amps_norm[:, :, pol_dict[p]])
            # calculating modifed z-score
            modz_dict, bd_inds = self.iterative_mod_zscore(
                autos_amps_norm[:, :, pol_dict[p]], threshold=threshold, niter=niter)
            if len(bd_inds) > 0:
                bad_ants.append(self.uvf.antenna_numbers[bd_inds])

            # writing stars to metrics instance
            self.metrics[p]['RMS'] = rms
            self.metrics[p]['MODZ_SCORE'] = modz_dict
            self.metrics[p]['BAD_ANTS'] = list(
                itertools.chain.from_iterable(bad_ants))

        # combining bad antennas from both pols to determine if the observation
        # should be considered for processing or not.
        # If %bad_ants > 50, obs is discarded
        self.metrics['BAD_ANTS'] = np.unique(
            self.metrics['XX']['BAD_ANTS'] + self.metrics['YY']['BAD_ANTS'])
        nants = self.metrics['NANTS'] - len(ex_annumbers)
        percent_bdants = len(self.metrics['BAD_ANTS']) / nants * 100
        self.metrics['BAD_ANTS_PERCENT'] = percent_bdants
        self.metrics['STATUS'] = 'GOOD' if percent_bdants < 50 else 'BAD'

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace(
                '.uvfits', '_prepvis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
