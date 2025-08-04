from collections import OrderedDict
from scipy.interpolate import CubicSpline
from scipy import signal
from astropy.io import fits
from astropy import constants as const
import numpy as np
import os
import copy

# speed of light
c = 299_792_458


def make_fits_axis_array(hdu, axis):
    count = hdu.header[f"NAXIS{axis}"]
    crval = hdu.header[f"CRVAL{axis}"]
    cdelt = hdu.header[f"CDELT{axis}"]
    crpix = hdu.header[f"CRPIX{axis}"]
    return cdelt * (np.arange(count) - crpix) + crval


class UVfits(object):
    def __init__(self, uvfits_path, antenna_convention=0):
        self.uvfits_path = uvfits_path
        # numbering of antennas, by default, starting number is 0
        self.antenna_convention = antenna_convention
        with fits.open(self.uvfits_path) as hdus:
            vis_hdu = hdus['PRIMARY']

            # get the data array
            self.data_array = vis_hdu.data['DATA']
            if self.data_array.ndim == 7:  # tweak of ska additional column
                self.data_array = vis_hdu.data['DATA'][:, 0, 0, 0, :,
                                                       :, 0] + 1j * vis_hdu.data['DATA'][:, 0, 0, 0, :, :, 1]
                self.weights_array = vis_hdu.data['DATA'][:, 0, 0, 0, :, :, 2]
            else:
                self.data_array = vis_hdu.data['DATA'][:, 0, 0, :,
                                                       :, 0] + 1j * vis_hdu.data['DATA'][:, 0, 0, :, :, 1]
                self.weights_array = vis_hdu.data['DATA'][:, 0, 0, :, :, 2]
            # the uvfits baseline of each row in the timestep-baseline axis
            try:
                # MWA Observations
                self.baseline_array = np.int64(vis_hdu.data["BASELINE"])
                self.unique_baselines = np.unique(self.baseline_array)
                self.ant_2_array = self.baseline_array % 256 - 1
                self.ant_1_array = (self.baseline_array -
                                    self.ant_2_array) // 256 - 1

            # SKA (OSKAR Simualtions)
            except KeyError:
                self.ant_1_array = vis_hdu.data['ANTENNA1'].astype(int)
                self.ant_2_array = vis_hdu.data['ANTENNA2'].astype(int)

            self.antpairs = np.stack(
                (self.ant_1_array, self.ant_2_array), axis=1)
            self.antpairs = np.sort(
                np.unique(self.antpairs, axis=0))
            self.antpairs = [tuple(antp) for antp in self.antpairs]
            self.Nbls = len(self.antpairs)
            assert len(self.antpairs) == self.Nbls

            self.obsid = vis_hdu.header["OBJECT"]
            self.channel_width = vis_hdu.header['CDELT4']

            # self.time_array = np.float64(vis_hdu.data["DATE"])
            # self.unique_times = np.sort(np.unique(self.time_array))
            self.Ntimes = int(len(self.ant_1_array) / self.Nbls)
            self.freq_array = make_fits_axis_array(vis_hdu, 4)
            self.Nchan = len(self.freq_array)

            self.polarization_array = np.int32(
                make_fits_axis_array(vis_hdu, 3))
            self.Npols = len(self.polarization_array)
            self.uvw_array = -1 * np.array(np.stack((
                vis_hdu.data['UU'],
                vis_hdu.data['VV'],
                vis_hdu.data['WW'],
            )).T) * const.c.to("m/s")

            ant_hdu = hdus['AIPS AN']
            self.ant_names = ant_hdu.data["ANNAME"]
            if self.antenna_convention == 0:
                self.antenna_numbers = ant_hdu.data.field("NOSTA") - 1
            else:
                self.antenna_numbers = ant_hdu.data.field("NOSTA")
            self.antenna_positions = ant_hdu.data.field("STABXYZ")
            self.Nants = len(self.ant_names)

    def debug(self):
        print(
            f"Ntimes={self.Ntimes}, Nbls={self.Nblts}, \
			Nfreqs={self.Nchan}, Npols={self.Npols}")

    def auto_antpairs(self):
        return [(ap[0], ap[1]) for ap in
                self.antpairs if ap[0] == ap[1]]

    def blt_idxs_for_antpair(self, antpair):
        """
        return the indices into the baseline-time axis corresponding
        to the given antpair
        """
        (ant1, ant2) = sorted(antpair)
        return np.where(np.logical_and(
            self.ant_1_array == ant1,
            self.ant_2_array == ant2,
        ))[0]

    def _data_for_antpairs(self, antpairs):
        """
        dimensions: [time, bl, freq, pol]
        """
        # sorted to traverse in the order on disk to minimize seeks
        blt_idxs = np.sort(np.concatenate([
            self.blt_idxs_for_antpair(antpair) for antpair in antpairs]))
        return self.data_array[blt_idxs]

    def data_for_antpair(self, antpair):
        """
        dimensions: [time, freq, pol]
        """
        return self._data_for_antpairs([antpair])

    def data_for_antpairs(self, antpairs):
        """
        dimensions: [time, bl, freq, pol]
        """
        Npairs = len(antpairs)
        result = self._data_for_antpairs(antpairs)
        return result.reshape(
            (self.Ntimes, Npairs, self.Nchan, self.Npols))

    def _flag_for_antpairs(self, antpairs, weight_limit=0):
        """
        dimensions: [time, bl, freq, pol]
        """
        # sorted to traverse in the order on disk to minimize seeks
        blt_idxs = np.sort(np.concatenate([
            self.blt_idxs_for_antpair(antpair) for antpair in antpairs]))
        # weights are clauculate (1/PFB_GAINS * Nf * Nt)
        return self.weights_array[blt_idxs, :, :] <= weight_limit

    def flag_for_antpair(self, antpair):
        """
        dimensions: [time, freq, pol]
        """
        result = self._flag_for_antpairs([antpair])
        return result

    def flag_for_antpairs(self, antpairs):
        """
        dimensions: [time, bl, freq, pol]
        """
        Npairs = len(antpairs)
        result = self._flag_for_antpairs(antpairs)
        return result.reshape(
            (self.Ntimes, Npairs, self.Nchan, self.Npols))

    def _uvw_for_antpairs(self, antpairs):
        """
        dimensions: [uvpoint, 3]
        """
        # sorted to traverse in the order on disk to minimize seeks
        blt_idxs = np.sort(np.concatenate([
            self.blt_idxs_for_antpair(antpair) for antpair in antpairs]))
        # weights are clauculate (1/PFB_GAINS * Nf * Nt)
        return self.uvw_array[blt_idxs, :]

    def uvw_for_antpair(self, antpair):
        """
        dimensions: [time, 3]
        """
        result = self._uvw_for_antpairs([antpair])
        return result

    def uvw_for_antpairs(self, antpairs):
        """
        dimensions: [time, bl, freq, pol]
        """
        Npairs = len(antpairs)
        result = self._uvw_for_antpairs(antpairs)
        return result.reshape(
            (self.Ntimes, Npairs, 3))

    def amplitude_array(self, antpairs):
        return self._amps_phs_array(antpairs)[:, :, :, 0]

    def phase_array(self, antpairs):
        return self._amps_phs_array(antpairs)[:, :, :, 1]

    def blackmanharris(self, n):
        return signal.windows.blackmanharris(n)

    def group_antpairs(self, antenna_positions, bl_tol):
        angroups = OrderedDict()
        delta_z = np.abs(antenna_positions -
                         np.mean(antenna_positions, axis=0))[:, 2]
        is_flat = np.all(delta_z < bl_tol)
        p_m = (-1, 0, 1)
        if is_flat:
            eps = [[dx, dy] for dx in p_m for dy in p_m]
        else:
            eps = [[dx, dy, dz] for dx in p_m for dy in p_m for dz in p_m]

        def _check_neighbours(delta):
            for ep in eps:
                nkey = (delta[0] + ep[0], delta[1] + ep[1], delta[2] + ep[2])
                if nkey in angroups:
                    return nkey

        for i in range(self.Nants):
            for j in range(i+1, self.Nants):
                antpair = (self.antenna_numbers[i], self.antenna_numbers[j])
                delta = tuple(np.round(
                    1.0 * (self.antenna_positions[i] -
                           self.antenna_positions[j])
                    / bl_tol).astype(int))
                nkey = _check_neighbours(delta)
                if nkey is None:
                    nkey = _check_neighbours(tuple([-d for d in delta]))
                    if nkey is not None:
                        antpair = (
                            self.antenna_numbers[j], self.antenna_numbers[i])
                if nkey is not None:
                    if len(self.blt_idxs_for_antpair(antpair)) > 0:
                        angroups[nkey].append(antpair)
                else:
                    # new baseline
                    if delta[0] <= 0 or (delta[0] == 0 and delta[1] <= 0) or \
                            (delta[0] == 0 and delta[1] == 0 and
                             delta[2] <= 0):
                        delta = tuple([-d for d in delta])
                        antpair = (
                            self.antenna_numbers[j], self.antenna_numbers[i])
                    if len(self.blt_idxs_for_antpair(antpair)) > 0:
                        angroups[delta] = [antpair]
        return angroups

    def redundant_antpairs(self, bl_tol=2e-1):
        # keeping only redundant pairs
        angroups = self.group_antpairs(self.antenna_positions, bl_tol=bl_tol)
        ankeys = list(angroups.keys())
        for akey in ankeys:
            if len(angroups[akey]) == 1:
                del angroups[akey]
        # sort keys by shortest baseline length
        sorted_keys = [akey for (length, akey) in sorted(
            zip([np.linalg.norm(akey) for akey in angroups.keys()],
                angroups.keys()))]
        reds = OrderedDict([(key, angroups[key]) for key in sorted_keys])
        return reds

    def delays(self):
        # Evaluates geometric delay (fourier conjugate of frequency)
        dfreq = self.freq_array[1] - self.freq_array[0]
        delays = np.fft.fftfreq(self.Nchan, dfreq)
        return np.fft.fftshift(delays * 1e9)

    def interpolate_data(self, x, x_new, y):
        f = CubicSpline(x, y)
        return f(x_new)

    def fft_data_for_antpair(self, antpair, apply_flag=True):
        data = self.data_for_antpair(antpair)
        flags = self.flag_for_antpair(antpair)
        dflag = data * ~flags
        if apply_flag:
            flag = self.flag_for_antpair(antpair)
            data[flag] = np.nan
        fft_data = copy.deepcopy(data)
        _sh = data.shape
        for t in range(_sh[0]):
            for p in range(_sh[2]):
                try:
                    inds = np.where(dflag[t, :, p] != 0.)[0]
                    window = self.blackmanharris(self.Nchan)
                    d_int = self.interpolate_data(
                        inds, np.arange(self.Nchan), dflag[t, inds, p])
                    d_fft = np.fft.fft(self.interpolate_data(
                        inds, np.arange(self.Nchan), dflag[t, inds, p]) * window)
                    fft_data[t, :, p] = np.fft.fftshift(d_fft)
                except ValueError:
                    fft_data[t, :, p] = np.nan
        return fft_data

    def fft_data_for_antpairs(self, antpairs, apply_flag=True):
        fft_data = np.zeros((self.Ntimes, len(antpairs),
                            self.Nchan, self.Npols), dtype=np.complex64)
        for i, antpair in enumerate(antpairs):
            fft_data[:, i, :, :] = self.fft_data_for_antpair(
                antpair, apply_flag=apply_flag)
        return fft_data

    def _plot_mode(self, data, mode):
        if mode == 'amp':
            return np.abs(data)
        elif mode == 'phs':
            return np.angle(data)
        elif mode == 'real':
            return np.real(data)

    def plot_vis(self, antpair, mode='amp'):
        """
        Plotting visibilities (amp, phases, delay-filtered)
        """
        pass
