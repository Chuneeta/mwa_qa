from mwa_qa import json_utils as ju
from mpl_toolkits.axes_grid.inset_locator import (
    inset_axes, InsetPosition, mark_inset)
import numpy as np
import pylab
import matplotlib as mpl
import warnings

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

colors = ['springgreen', 'darkorange', 'dodgerblue', 'darkolivegreen', 'lightskyblue', 'sienna',
          'orangered', 'violet', 'maroon', 'pink', 'orange', 'navy', 'crimson']


class CalQA(object):
    def __init__(self, json_path):
        self.json_path = json_path
        self.metrics = ju.load_json(self.json_path)
        self.keys = list(self.metrics.keys())
        if 'XX' in self.keys:
            self.pol_keys = list(self.metrics['XX'].keys())
        elif 'YY' in self.keys:
            self.pol_keys = list(self.metrics['YY'].keys())
        else:
            warnings.warn('No polarization keys found')
        self.nchan = self.metrics['NCHAN']
        self.start_freq = self.metrics['START_FREQ']
        self.ch_width = self.metrics['CH_WIDTH']
        self.frequencies = np.linspace(
            self.start_freq, self.start_freq + self.nchan * self.ch_width, self.nchan)
        self.delays = np.fft.fftshift(
            np.fft.fftfreq(self.nchan, self.ch_width * 1e-9))
        self.antenna = self.metrics['ANTENNA']
        self.obsid = self.metrics['OBSID']

    def _check_key(self, key):
        assert key in self.keys, "Key {} not found".format(key)

    def _check_pol_key(self, key):
        assert key in self.pol_keys, "Key {} not found".format(key)

    def read_key(self, key):
        key = key.upper()
        self._check_key(key)
        return self.metrics[key]

    def read_pol_key(self, pol, key):
        pol = pol.upper()
        key = key.upper()
        self._check_pol_key(key)
        return self.metrics[pol][key]

    def plot_fft(self, timestamp=0, cmap='hot', vmin=None, vmax=None, save=None, figname=None):
        dfft_xx = np.array(self.read_pol_key('XX', 'DFFT'))
        dfft_yy = np.array(self.read_pol_key('YY', 'DFFT'))
        if vmin is None:
            vmin = np.log10(np.abs(np.nanmin(np.array([dfft_xx, dfft_yy]))))
        if vmax is None:
            vmax = np.log10(np.abs(np.nanmax(np.array([dfft_xx, dfft_yy]))))
        fig = pylab.figure(figsize=(10, 7))
        fig.suptitle(self.obsid, size=15)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.log10(np.abs(dfft_xx[timestamp, :, :])), aspect='auto', vmin=vmin, vmax=vmax, extent=(
            self.delays[0], self.delays[-1], self.antenna[-1], self.antenna[0]), cmap=cmap)
        ax1.set_title('XX')
        ax1.set_xlabel('Delay (ns)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.tick_params(labelsize=12, direction='in', length=4, width=2)
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(np.log10(np.abs(dfft_yy[timestamp, :, :])), aspect='auto', vmin=vmin, vmax=vmax, extent=(
            self.delays[0], self.delays[-1], self.antenna[-1], self.antenna[0]), cmap=cmap)
        ax2.set_title('YY')
        ax2.set_xlabel('Delay (ns)', fontsize=12)
        ax2.get_yaxis().set_visible(False)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        ax2.tick_params(labelsize=10, direction='in', length=4, width=2)

        if save:
            if figname is None:
                figname = self.json_path.replace('.json', '_2dfft.png')
            else:
                if figname.split('.')[-1] != 'png':
                    figname += '.png'
            pylab.savefig(figname, dpi=200)
            pylab.close()
        else:
            pylab.show()

    def plot_delay_spectra(self, delays=[], timestamp=0, save=None, figname=None):
        dfft_xx = np.array(self.read_pol_key('XX', 'DFFT'))
        dfft_yy = np.array(self.read_pol_key('YY', 'DFFT'))
        fig = pylab.figure(figsize=(10, 7))
        fig.suptitle('DELAY SPECTRA', size=15)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        for i, dly in enumerate(delays):
            ax1.plot(self.antenna, np.abs(dfft_xx[timestamp, :, dly]),
                     '-', color=colors[i], linewidth=2, label='{} ns'.format(int(self.delays[dly])))
            ax2.plot(self.antenna, np.abs(dfft_yy[timestamp, :, dly]),
                     '-', color=colors[i], linewidth=2, label='{} ns'.format(int(self.delays[dly])))
        ax2.set_xlabel('Antenna Number', fontsize=12)
        ax1.set_ylabel('XX', fontsize=12)
        ax2.set_ylabel('YY', fontsize=12)
        ax1.grid(ls='dotted')
        ax2.grid(ls='dotted')
        ax2.legend(loc='upper right')
        ax1.set_ylim(-0.1, 1)
        ax2.set_ylim(0, 1)
        fig.subplots_adjust(hspace=0)

    def plot_amp_variances(self, timestamp=0, save=None, figname=None):
        varxx_ant = np.array(self.read_pol_key(
            'XX', 'AMPVAR_ANT'))[timestamp, :]
        varyy_ant = np.array(self.read_pol_key(
            'YY', 'AMPVAR_ANT'))[timestamp, :]
        varxx_freq = np.array(self.read_pol_key(
            'XX', 'AMPVAR_FREQ'))[timestamp, :]
        varyy_freq = np.array(self.read_pol_key(
            'YY', 'AMPVAR_FREQ'))[timestamp, :]
        varxx_ant_thresh = np.nanmean(varxx_ant) + 3 * np.nanstd(varxx_ant)
        varyy_ant_thresh = np.nanmean(varxx_ant) + 3 * np.nanstd(varxx_ant)
        varxx_freq_thresh = np.nanmean(varxx_freq) + 3 * np.nanstd(varxx_freq)
        varyy_freq_thresh = np.nanmean(varxx_freq) + 3 * np.nanstd(varxx_freq)
        fig = pylab.figure(figsize=(8, 6))
        fig.suptitle(self.obsid, size=15)
        ax1 = fig.add_subplot(211)
        ax1.semilogy(self.frequencies * 1e-6, varxx_ant,
                     '.', color='indianred', label='XX')
        ax1.semilogy(self.frequencies * 1e-6, varyy_ant,
                     '.', color='dodgerblue', label='YY')
        ax1.set_xlabel('Frequency (MHz)', fontsize=12)
        ax1.set_ylabel('RMS (Ant)', fontsize=12)
        ax1.grid(ls='dotted')
        ax1.legend()
        ax1.tick_params(labelsize=10, direction='in', length=4, width=2)
        ax2 = fig.add_subplot(212)
        ax2.semilogy(self.antenna, varxx_freq, '.',
                     color='indianred', label='XX')
        ax2.semilogy(self.antenna, varyy_freq, '.',
                     color='dodgerblue', label='YY')
        ax2.set_xlabel('Antenna', fontsize=12)
        ax2.set_ylabel('RMS (Freq)', fontsize=12)
        ax2.grid(ls='dotted')
        ax2.legend()
        ax2.tick_params(labelsize=10, direction='in', length=4, width=2)
        pylab.subplots_adjust(hspace=0.3)
        if save:
            if figname is None:
                figname = self.json_path.replace('.json', '_variance.png')
            else:
                if figname.split('.')[-1] != 'png':
                    figname += '.png'
            pylab.savefig(figname, dpi=200)
            pylab.close()
        else:
            pylab.show()
