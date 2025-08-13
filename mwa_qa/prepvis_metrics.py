from mwa_qa.read_uvfits import UVfits
from mwa_qa.read_metafits import Metafits
from mwa_qa import json_utils as ju
from collections import OrderedDict
import numpy as np
import pylab
import itertools

pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


def converter(input_list, output_list):
    for elements in input_list:
        if type(elements) == list:
            converter(elements, output_list)
        else:
            output_list.append(elements)
    return output_list


class PrepvisMetrics(object):
    def __init__(self, uvfits_path, metafits_path, ex_annumbers=[], edge_flagging=None, antenna_flags=False, cutoff_threshold=3, niter=10):
        self.uvfits_path = uvfits_path
        self.metafits_path = metafits_path
        self.uvf = UVfits(self.uvfits_path)
        self.meta = Metafits(self.metafits_path)
        self.ex_annumbers = ex_annumbers
        self.edge_flagging = edge_flagging
        self.antenna_flags = antenna_flags
        self.cutoff_threshold = cutoff_threshold
        self.niter = niter
        self.antenna_numbers = np.unique(self.uvf.auto_antpairs())

    def autos(self):
        auto_antpairs = self.uvf.auto_antpairs()
        autos = self.uvf.data_for_antpairs(
            auto_antpairs)  # (only xx and yy pols)
        _sh = autos.shape

        # flags are properly propages for the autocorrelations, therefore
        # we are mually flagging the band edges and centre frequency only
        # for obsids below 13*
        if self.edge_flagging:
            flags = self._evaluate_edge_flags()
            flags = flags[np.newaxis, np.newaxis, :, np.newaxis]
            flags = flags.repeat(_sh[0], axis=0).repeat(
                _sh[1], axis=1).repeat(_sh[3], axis=3)
        else:
            flags = np.zeros((_sh), dtype=bool)
        autos[flags] = np.nan
        if len(self.ex_annumbers) > 0:
            autos[:, self.ex_annumbers, :, :] = np.nan
        # applying flags from metafits
        ind = self.flags_from_metafits()
        autos[:, ind, :, :] = np.nan
        return autos

    def flags_from_metafits(self):
        flags = self.meta.flag_array
        inds = np.where(flags == 1)[0]
        flag_inds = []
        if len(inds) > 0:
            [flag_inds.append(
                np.where(self.antenna_numbers == i)[0][0]) for i in inds]
        return flag_inds

    def _evaluate_edge_flags(self):
        cchan_bandwidth = 1_280_000
        edge_bandwidth = 80_000
        freq_res = self.uvf.channel_width
        num_chans = self.uvf.Nchan
        edge_chans = int(edge_bandwidth // freq_res)
        num_fchans = int(cchan_bandwidth // freq_res)
        num_cchans = int(num_chans // num_fchans)
        center_fine_chan = num_fchans // 2
        # start with all flags true
        flags = np.full((num_cchans, num_fchans), True)
        # unflag within edge chans
        flags[:, edge_chans:-edge_chans] = False
        # flag center fine chan
        flags[:, center_fine_chan] = True

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

    def plot_spectra_across_chan(self, freq_chan, mode='log', save=None, figname=None):
        autos = self.autos()
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

    def plot_spectra_across_time(self, timestamp, mode='log', save=None, figname=None):
        autos = self.autos()
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

    def plot_spectra_2D(self, annumber, mode='log',
                        save=None, figname=None):
        autos = self.autos()
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

    def split_annames(self):
        """
        Splitting the antenna indeices as per antenna name conventions ('Tile', 'Hex')
        """
        annames = self.uvf.ant_names[self.antenna_numbers]
        first = [i for i in range(len(annames))
                 if annames[i].startswith('Tile')]
        second = [i for i in range(len(annames))
                  if annames[i].startswith('Hex')]
        # print(first, second)
        return first, second

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
        if len(inds[0]) > 0:
            bad_inds.append(inds[0].tolist())
        count = 1
        modz_dict[count - 1] = mod_zscore
        while (count <= niter and len(inds[0]) > 0):
            data[inds[0]] = np.nan
            mod_zscore = self.calculate_mod_zscore(data)
            inds = np.where(np.abs(mod_zscore) > threshold)
            modz_dict[count] = mod_zscore
            if len(inds[0]) > 0:
                bad_inds.append(inds[0].tolist())
                count += 1
        return modz_dict, converter(bad_inds, [])

    def _initialize_metrics_dict(self):
        self.metrics = OrderedDict()
        self.metrics['NANTS'] = self.uvf.Nants
        self.metrics['NTIMES'] = self.uvf.Ntimes
        self.metrics['NCHAN'] = self.uvf.Nchan
        self.metrics['NPOLS'] = self.uvf.Npols
        self.metrics['OBSID'] = self.uvf.obsid
        if self.antenna_flags:
            self.metrics['ANNUMBERS'] = np.delete(
                self.antenna_numbers, self.flags_from_metafits())
        else:
            self.metrics['ANNUMBERS'] = self.antenna_numbers
        self.metrics['NANTS'] = len(self.metrics['ANNUMBERS'])
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()

    def run_metrics(self, split_autos=False):
        self._initialize_metrics_dict()
        # auto correlations
        autos = self.autos()
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
                bad_ants.append(self.metrics['ANNUMBERS'][bd_inds])
            autos_amps_norm[bd_inds, :, pol_dict[p]] = np.nan
            # calculating root mean square
            rms = self.calculate_rms(
                autos_amps_norm[:, :, pol_dict[p]])
            if split_autos:
                # splitting autos by antenna naming conventions
                first, second = self.split_annames()
                modz_dict_first, bd_inds_first = self.iterative_mod_zscore(
                    autos_amps_norm[first, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                modz_dict_second, bd_inds_second = self.iterative_mod_zscore(
                    autos_amps_norm[second, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.append(np.array(first)[bd_inds_first],
                                    np.array(second)[bd_inds_second])
                # bd_inds_flatten = np.array(bd_inds)
                self.metrics[p]['MODZ_SCORE'] = {}
                self.metrics[p]['MODZ_SCORE']['FIRST'] = modz_dict_first
                self.metrics[p]['MODZ_SCORE']['SECOND'] = modz_dict_second
                self.metrics['ANNUMBERS_FIRST'] = first
                self.metrics['ANNUMBERS_SECOND'] = second
            else:
                # calculating modifed z-score
                modz_dict, bd_inds = self.iterative_mod_zscore(
                    autos_amps_norm[:, :, pol_dict[p]], threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.array(bd_inds)
                self.metrics[p]['MODZ_SCORE'] = modz_dict
            if len(bd_inds) > 0:
                bad_ants.append(self.metrics['ANNUMBERS'][bd_inds])

            # writing stars to metrics instanc
            self.metrics[p]['RMS'] = rms
            self.metrics[p]['BAD_ANTS'] = list(
                itertools.chain.from_iterable(bad_ants))

        # combining bad antennas from both pols to determine if the observation
        # should be considered for processing or not.
        # If %bad_ants > 50, obs is discarded
        self.metrics['BAD_ANTS'] = np.unique(
            self.metrics['XX']['BAD_ANTS'] + self.metrics['YY']['BAD_ANTS'])
        nants = self.metrics['NANTS'] - len(self.ex_annumbers)
        percent_bdants = len(self.metrics['BAD_ANTS']) / nants * 100
        self.metrics['BAD_ANTS_PERCENT'] = percent_bdants
        self.metrics['STATUS'] = 'GOOD' if percent_bdants < 50 else 'BAD'
        self.metrics['THRESHOLD'] = self.cutoff_threshold

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.uvfits_path.replace(
                '.uvfits', '_prepvis_metrics.json')
        ju.write_metrics(self.metrics, outfile)
