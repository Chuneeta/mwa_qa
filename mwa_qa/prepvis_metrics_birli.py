
from mwa_qa.read_metafits import Metafits
from mwa_qa import json_utils as ju
from collections import OrderedDict
from astropy.io import fits
import matplotlib.cm as cm
import numpy as np
import pylab
import itertools
import os


def converter(input_list, output_list):
    for elements in input_list:
        if type(elements) == list:
            converter(elements, output_list)
        else:
            output_list.append(elements)
    return output_list


class PrepvisMetrics(object):
    def __init__(self, data_path, metafits_path, ex_annumbers=[], cutoff_threshold=3, niter=10):
        self.data_path = data_path
        self.metafits_path = metafits_path
        self.meta = Metafits(metafits_path)
        self.ex_annumbers = ex_annumbers
        self.cutoff_threshold = cutoff_threshold
        self.niter = niter

        with fits.open(self.data_path) as hdu:
            autos_xx = hdu['AUTO_POL=XX'].data
            autos_yy = hdu['AUTO_POL=YY'].data
            autos_xy = hdu['AUTO_POL=XY'].data
            autos_yx = hdu['AUTO_POL=YX'].data
            self.autos = [autos_xx, autos_yy, autos_xy, autos_yx]
            self.Nants = hdu['AUTO_POL=XX'].header['N_ANTS']
            self.Nchan = hdu['AUTO_POL=XX'].header['NAXIS1']

    def plot_auto_spectra(self, save=None, figname=None):
        cable_flavors = np.array(self.meta.cable_flavors)
        unq_cable_flavors = np.unique(cable_flavors)
        freqs = self.meta.frequency_array

        # Assign each cable flavor a color
        cmap = cm.get_cmap('viridis', len(unq_cable_flavors))
        flavor_to_color = {cfl: cmap(i)
                           for i, cfl in enumerate(unq_cable_flavors)}
        fig, axes = pylab.subplots(2, 2, figsize=(7, 4.5))

        # Loop over all antennas (or elements in cable_flavors)
        for ant_idx, cfl in enumerate(cable_flavors):
            color = flavor_to_color[cfl]  # pick color based on cable flavor

            # Pol: xx
            axes[0, 0].plot(freqs, np.log10(np.abs(self.autos[0][ant_idx, :])),
                            color=color, alpha=0.5, label=cfl if ant_idx == 0 else "")
            axes[0, 0].set_ylim(0.8, 5.2)

            # Pol: yy
            axes[0, 1].plot(freqs, np.log10(np.abs(self.autos[1][ant_idx, :])),
                            color=color, alpha=0.5)
            axes[0, 1].set_ylim(0.8, 5.2)
            # Pol: xt
            axes[1, 0].plot(freqs, np.log10(np.abs(self.autos[2][ant_idx, :])),
                            color=color, alpha=0.5)
            axes[1, 0].set_ylim(0.8, 5.2)
            # Pol: yx
            axes[1, 1].plot(freqs, np.log10(np.abs(self.autos[3][ant_idx, :])),
                            color=color, alpha=0.7)
            axes[1, 1].set_ylim(0.8, 5.2)
        # Titles and labels
        axes[0, 0].set_title('Pol: xx')
        axes[0, 0].set_ylabel('Amplitude', fontsize=10)

        # One legend entry per cable flavor
        handles = [pylab.Line2D([0], [0], color=flavor_to_color[cfl],
                                lw=2, label=cfl) for cfl in unq_cable_flavors]
        axes[0, 0].legend(handles=handles, ncol=1,
                          fontsize=6, loc='lower left')

        axes[0, 1].set_title('Pol: yy')
        axes[1, 0].set_title('Pol: xy')
        axes[1, 0].set_xlabel('Frequency (MHz)', fontsize=10)
        axes[1, 0].set_ylabel('Amplitude', fontsize=10)
        axes[1, 1].set_title('Pol: yx')
        axes[1, 1].set_xlabel('Frequency (MHz)', fontsize=10)

        pylab.tight_layout()
        pylab.tick_params(labelsize=10)
        if save is None:
            pylab.show()
        else:
            if figname is None:
                figname = self.metafits_path .strip(
                    '.metafits') + '_autos_spectrums.png'
                print('Saving ', figname)
            pylab.savefig(figname, dpi=100)
            pylab.clf()

    def plot_auto_spectra_by_flavor(self, pol="xx", save=None, figname=None):
        # Map pol string to index
        pol_map = {"xx": 0, "yy": 1, "xy": 2, "yx": 3}
        if pol not in pol_map:
            raise ValueError(
                f"Invalid polarization '{pol}'. Must be one of {list(pol_map.keys())}")
        pol_idx = pol_map[pol]

        # Metadata
        cable_flavors = np.array(self.meta.cable_flavors)
        unq_cable_flavors = np.unique(cable_flavors)
        freqs = self.meta.frequency_array

        # Color map for antennas (within each flavor)
        cmap = cm.get_cmap('viridis')

        # Set up figure: one subplot per cable flavor
        n_flavors = len(unq_cable_flavors)
        ncols = 2 if n_flavors > 1 else 1
        nrows = int(np.ceil(n_flavors / ncols))
        fig, axes = pylab.subplots(nrows, ncols, figsize=(
            4 * ncols, 3 * nrows), squeeze=False)

        # Loop over cable flavors
        for ax, cfl in zip(axes.flat, unq_cable_flavors):
            ants_in_flavor = np.where(cable_flavors == cfl)[0]

            # Assign each antenna in this flavor a color shade
            for i, ant_idx in enumerate(ants_in_flavor):
                color = cmap(i / max(1, len(ants_in_flavor) - 1))
                ax.plot(freqs,
                        np.log10(
                            np.abs(self.autos[pol_idx][ant_idx, :]) + 1e-12),
                        color=color, alpha=0.6)

            ax.set_title(f"Cable Flavor: {cfl} ({pol})", fontsize=10)
            ax.set_xlabel("Frequency (MHz)", fontsize=9)
            ax.set_ylabel("Amplitude (log10)", fontsize=9)
            ax.set_ylim(0.8, 5.2)
            ax.tick_params(labelsize=8)

        # Hide unused axes if any
        for ax in axes.flat[n_flavors:]:
            ax.axis("off")

        pylab.tight_layout()

        if save is None:
            pylab.show()
        else:
            if figname is None:
                figname = self.metafits_path .strip(
                    '.metafits') + '_flavor_spectrums_{}.png'.format(pol)
                print('Saving ', figname)
                pylab.savefig(figname, dpi=150, bbox_inches="tight")
                pylab.close(fig)

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
        self.metrics['NANTS'] = self.Nants
        self.metrics['NTIMES'] = ''
        self.metrics['NCHAN'] = self.Nchan
        self.metrics['NPOLS'] = 4
        self.metrics['OBSID'] = str(self.meta.start_gpstime)
        # phase III annumbers are spread, therefore important to get it sorted
        # tilenames also should be arrange according if required
        self.metrics['ANNUMBERS'] = np.sort(self.meta.antenna_numbers)
        self.metrics['NANTS'] = len(self.metrics['ANNUMBERS'])
        self.metrics['XX'] = OrderedDict()
        self.metrics['YY'] = OrderedDict()

    def run_metrics(self, split_autos=False, cable_flavor=False):
        self._initialize_metrics_dict()
        # finding misbehaving antennas
        for i, p in enumerate(['XX', 'YY']):
            if len(self.ex_annumbers) > 0:
                autos[i][self.ex_annumbers, :] = np.nan
            autos_amps_norm = np.abs(
                self.autos[i]) / np.nanmedian(np.abs(self.autos[i]), axis=0)
            bad_ants = []
            _, bd_inds = self.flag_occupancy(
                autos_amps_norm)
            if len(bd_inds) > 0:
                bad_ants.append(self.metrics['ANNUMBERS'][bd_inds])
            autos_amps_norm[bd_inds, :] = np.nan
            bad_ants = np.atleast_1d(bad_ants).astype(
                int)  # this will now be fine
            self.metrics[p]['BAD_ANTS'] = bad_ants.tolist()
            # calculating root mean square
            rms = self.calculate_rms(
                autos_amps_norm)
            if split_autos:
                # splitting autos by antenna naming conventions
                first, second = self.split_annames()
                modz_dict_first, bd_inds_first = self.iterative_mod_zscore(
                    autos_amps_norm[first, :], threshold=self.cutoff_threshold, niter=self.niter)
                modz_dict_second, bd_inds_second = self.iterative_mod_zscore(
                    autos_amps_norm[second, :], threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.append(np.array(first)[bd_inds_first],
                                    np.array(second)[bd_inds_second])
                # bd_inds_flatten = np.array(bd_inds)
                self.metrics[p]['MODZ_SCORE'] = {}
                self.metrics[p]['MODZ_SCORE']['FIRST'] = modz_dict_first
                self.metrics[p]['MODZ_SCORE']['SECOND'] = modz_dict_second
                self.metrics['ANNUMBERS_FIRST'] = first
                self.metrics['ANNUMBERS_SECOND'] = second
            if cable_flavor:
                cable_flavors = np.array(self.meta.cable_flavors)
                unq_cable_flavors = np.unique(cable_flavors)
                # New combined MODZ dict for all antennas
                combined_modz = np.empty(len(self.metrics['ANNUMBERS']))
                combined_modz[:] = np.nan  # fill with NaN initially
                for cfl in unq_cable_flavors:
                    self.metrics[p][cfl] = {}
                    inds = np.where(cable_flavors == cfl)
                    modz_dict, bd_inds_cfl = self.iterative_mod_zscore(
                        autos_amps_norm[inds], threshold=self.cutoff_threshold, niter=self.niter)
                    self.metrics[p][cfl]['MODZ_SCORE'] = modz_dict
                    bd_inds = np.append(
                        bd_inds, inds[0][bd_inds_cfl]).astype(int)
                    combined_modz[inds] = modz_dict[0]
                    if len(bd_inds) > 0:
                        self.metrics[p][cfl]['BAD_ANTS'] = self.metrics['ANNUMBERS'][bd_inds]

                bd_inds = np.unique(bd_inds)
                bd_inds = bd_inds.astype(int)
                self.metrics[p]['MODZ_SCORE'] = {}
                self.metrics[p]['MODZ_SCORE'][0] = combined_modz
            else:
                # calculating modifed z-score
                modz_dict, bd_inds = self.iterative_mod_zscore(
                    autos_amps_norm, threshold=self.cutoff_threshold, niter=self.niter)
                bd_inds = np.array(bd_inds)
                bd_inds = bd_inds.astype(int)
                self.metrics[p]['MODZ_SCORE'] = modz_dict

            if len(bd_inds) > 0:
                bad_ants = np.append(
                    bad_ants, self.metrics['ANNUMBERS'][bd_inds])
            # writing stars to metrics instanc
            self.metrics[p]['RMS'] = rms
            self.metrics[p]['BAD_ANTS'] = bad_ants.tolist()

        # combining bad antennas from both pols to determine if the observation
        self.metrics['BAD_ANTS'] = np.unique(self.metrics['XX']['BAD_ANTS']
                                             + self.metrics['YY']['BAD_ANTS'])
        nants = self.metrics['NANTS'] - len(self.ex_annumbers)
        percent_bdants = len(self.metrics['BAD_ANTS']) / nants * 100
        self.metrics['BAD_ANTS_PERCENT'] = percent_bdants
        self.metrics['STATUS'] = 'GOOD' if percent_bdants < 50 else 'BAD'
        self.metrics['THRESHOLD'] = self.cutoff_threshold

    def write_to(self, outfile=None):
        if outfile is None:
            outfile = self.data_path.replace(
                '.fits', '_prepvis_metrics.json')
            ju.write_metrics(self.metrics, outfile)
