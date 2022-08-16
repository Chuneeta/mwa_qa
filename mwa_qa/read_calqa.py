from mwa_qa import json_utils as ju
import numpy as np
import pylab
import warnings


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

    def plot_fft(self, timestamp=0, cmap='hot', save=None, figname=None):
        dfft_xx = np.array(self.read_pol_key('XX', 'DFFT'))
        dfft_yy = np.array(self.read_pol_key('YY', 'DFFT'))
        fig = pylab.figure(figsize=(10, 7))
        fig.suptitle('FFT SPECTRUM', size=15)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.log10(np.abs(dfft_xx[timestamp, :, :])), aspect='auto', extent=(
            self.delays[0], self.delays[-1], self.antenna[-1], self.antenna[0]), cmap=cmap)
        ax1.set_title('XX')
        ax1.set_xlabel('Delay (ns)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.tick_params(labelsize=10, direction='in')
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(np.log10(np.abs(dfft_yy[timestamp, :, :])), aspect='auto', extent=(
            self.delays[0], self.delays[-1], self.antenna[-1], self.antenna[0]), cmap=cmap)
        ax2.set_title('YY')
        ax2.set_xlabel('Delay (ns)', fontsize=12)
        ax2.get_yaxis().set_visible(False)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        ax2.tick_params(labelsize=10, direction='in')
        if save:
            if figname is None:
                figname = self.json_path.replace('.json', '_2dfft.png')
            else:
                if figname.split('.')[-1] != 'png':
                    figname += '.png'
            pylab.savefig(figname, dpi=300)
        else:
            pylab.show()
        pylab.close()

    def plot_amp_variances(self, timestamp=0, save=None, figname=None):
        var_ant_xx = np.array(self.read_pol_key('XX', 'AMPVAR_ANT'))
        var_ant_yy = np.array(self.read_pol_key('YY', 'AMPVAR_ANT'))
        var_freq_xx = np.array(self.read_pol_key('XX', 'AMPVAR_FREQ'))
        var_freq_yy = np.array(self.read_pol_key('YY', 'AMPVAR_FREQ'))
        fig = pylab.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(211)
        ax1.semilogy(self.frequencies,
                     var_ant_xx[timestamp, :], '.', color='indianred', label='XX')
        ax1.semilogy(self.frequencies,
                     var_ant_yy[timestamp, :], '.', color='dodgerblue', label='YY')
        ax1.grid(ls='dotted')
        ax1.legend()
        ax1.tick_params(labelsize=10, direction='in')
        ax2 = fig.add_subplot(212)
        ax2.semilogy(self.antenna,
                     var_freq_xx[timestamp, :], '.', color='indianred', label='XX')
        ax2.semilogy(self.antenna,
                     var_freq_yy[timestamp, :], '.', color='dodgerblue', label='YY')
        ax2.grid(ls='dotted')
        ax2.legend()
        ax2.tick_params(labelsize=10, direction='in')
        if save:
            if figname is None:
                figname = self.json_path.replace('.json', '_variance.png')
            else:
                if figname.split('.')[-1] != 'png':
                    figname += '.png'
            pylab.savefig(figname, dpi=300)
        else:
            pylab.show()
        pylab.close()
