from mwa_qa import read_metafits as rm
from scipy import signal
from astropy.io import fits
import numpy as np
import copy


class Csoln(object):
    def __init__(self, calfile, metafits=None, pol='X'):
        """
        Object takes in a calfile in fits format and extracts
        bit and pieces of the required informations
        - calfile:	Fits file readable by astropy containing
                                calibration solutions (support for hyperdrive
                                output only for now) and related information
        - metafits:	Metafits with extension *.metafits containing
                                information corresponding to the observation
                                for which the calibration solutions is derived
        - pol:	Polarization, can be either 'X' or 'Y'. It should be
                        specified so that information associated
                        with the given pol is provided. Default is 'X'
        """
        self.calfile = calfile
        self.Metafits = rm.Metafits(metafits, pol)

    def data(self, hdu):
        """
        Returns the data stored in the specified HDU column of the image
        hdu:	HDU column, ranges from 1 to 6
                                1 - the calibration solution
                                2 - the start time, end time and average time
                                3 - tiles information (antenna, tilename, flag)
                                4 - chanblocks (index, freq, flag)
                5 - calibration results (timeblock, chan, convergence)
                6 - weights used for each baseline
        For more details refer to
        https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
        """
        return fits.open(self.calfile)[hdu].data

    def header(self, hdu):
        """
        Returns the header of the specified HDU column
        hdu:	HDU column, ranges from 0 to 6
                0 - header information on the paramters used for the
                                        calibration process
                1 - header information on the calibration solutions
                2 - header information on the timeblocks
                3 - header information on the tiles (antenna, tilename, flag)
                4 - header information on the chanblocks (index, freq, flag)
                5 - header information on the calibration results
                                        (timeblock, chan, convergence)
                6 - header information on the weights used for each baseline
        For more details refer to
        https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
        """
        return fits.open(self.calfile)[hdu].header

    def gains_real(self):
        """
        Returns the real part of the calibration solutions
        """
        cal_solutions = self.data(1)
        return cal_solutions[:, :, :, ::2]

    def gains_imag(self):
        """
        Returns the imaginary part of the calibration solutions
        """
        cal_solutions = self.data(1)
        return cal_solutions[:, :, :, 1::2]

    def gains(self):
        """
        Combines the real and imaginary parts to form the
                4 polarization (xx, xy, yx and yy)
        """
        return self.gains_real() + self.gains_imag() * 1j

    def gains_shape(self):
        """
        Returns shape of the array containing the gain soultions
        """
        return self.gains().shape

    def ant_info(self):
        """
        Returns the info on the ant induces, tile ID and flags.
        The ant indices/ numbers start from 0 and the ant numbers
                Start from 1 in uvfits
        """
        ant_info = self.data(3)
        # added 1 to the antenna number to match those in uvfits
        annumbers = [ant[0] for ant in ant_info]
        annames = [ant[1] for ant in ant_info]
        anflags = [ant[2] for ant in ant_info]
        return annumbers, annames, anflags

    def ntimeblocks(self):
        """
        Returns the timeblocks of the calibration solutions
        """
        d_header = self.header(1)
        return d_header['NAXIS4']

    def freqs_info(self):
        """
        Returns the frequency index, frequency array and frequency flags
        """
        freqs_info = self.data(4)
        freq_inds = [fq[0] for fq in freqs_info]
        freqs = [fq[1] for fq in freqs_info]
        freq_flags = [fq[2] for fq in freqs_info]
        return freq_inds, freqs, freq_flags

    def gains_ind_for(self, antnum):
        """
        Returns index of the gain solutions fot the given antenna number,
                indices matches the antenna number in this case
        - antnum : Antenna Number
        """
        return antnum

    def _check_refant(self, antnum):
        """
        Checks if the given reference antenna is flagged due to non-convergence
                or any malfunctioning reports
        - antnum : Antenna Number, starts from 0
        """
        annumbers, annames, anflags = self.ant_info()
        ind = self.gains_ind_for(antnum)
        flag = np.array(anflags)[ind]
        assert flag == 0,  "{} seems to be flagged."
        "calibration solutions found, choose a different tile"

    def _normalized_data(self, data, ref_antnum=None):
        """
        Normalizes the gain solutions for each timeblock given a reference tile
        - data:	Input array of shape( tiles, freq, pols) containing the
                                solutions
        - ref_antnum:	Antenna number for the reference antenna (starts from 1).
                        Default is set to the last antenna of the telescope.
                        For example for MWA128T, the reference antenna is
                        Antenna 128
        """
        if ref_antnum is None:
            annumbers, annames, _ = self.ant_info()
            ref_ind = -1
            ref_antnum = np.array(annumbers)[ref_ind]
        else:
            ref_ind = self.gains_ind_for(ref_antnum)
        self._check_refant(ref_antnum)
        refs = []
        for ref in data[ref_ind].reshape((-1, 2, 2)):
            refs.append(np.linalg.inv(ref))
        refs = np.array(refs)
        div_ref = []
        for tile_i in data:
            for (i, ref) in zip(tile_i, refs):
                div_ref.append(i.reshape((2, 2)).dot(ref))
        return np.array(div_ref).reshape(data.shape)

    def normalized_gains(self, ref_antnum=None):
        """
        Returns the normalized gain solutions using the
        given reference Antenna number
        - ref_antnum:	Antenna number for the reference antenna (starts from 1).
                        Default is set to the last antenna of the telescope.
                        For example for MWA128T, the reference antenna is
                        Antenna 128
        """
        gains = self.gains()
        ngains = copy.deepcopy(gains)
        for t in range(len(ngains)):
            ngains[t] = self._normalized_data(gains[t], ref_antnum)
        return ngains

    def _select_gains(self, norm):
        """
        Return normalized if norm is True else unnomalized gains
        - norm:	Boolean, If True returns normalized gains else
                unormalized gains.
        """
        if norm:
            return self.normalized_gains()
        else:
            return self.gains()

    def amplitudes(self, norm=True):
        """
        Returns amplitude of the normalized gain solutions
        - norm:	Boolean, if True returns normalized gains else
                unormalized gains.
                Default is set to True.
        """
        gains = self._select_gains(norm=norm)
        return np.abs(gains)

    def phases(self, norm=True):
        """
        Returns phases in degrees of the normalized gain solutions
        - norm: Boolean, if True returns normalized gains else
                unormalized gains.
                Default is set to True.
        """
        gains = self._select_gains(norm=norm)
        return np.angle(gains) * 180 / np.pi

    def gains_for_antnum(self, antnum, norm=True):
        """
        Returns gain solutions for the given tile ID
        - antnum:	Antenna Number, starts from 1
        - norm:	Boolean, If True returns normalized gains
                else unormalized gains.
                Default is set to True.
        """
        gains = self._select_gains(norm=norm)
        ind = self.gains_ind_for(antnum)
        return gains[:, ind, :, :]

    def gains_for_antpair(self, antpair, norm=True):
        """
        Evaluates conjugation of the gain solutions for antenna pair
                (tile0, tile1)
        - antpair:	Tuple of antenna numbers such as (1, 2)
        """
        gains_t0 = self.gains_for_antnum(antpair[0], norm=norm)
        gains_t1 = self.gains_for_antnum(antpair[1], norm=norm)
        return gains_t0 * np.conj(gains_t1)

    def gains_for_receiver(self, receiver, norm=True):
        """
        Returns the dictionary of gains solutions for all the antennas
                (8 antennas in principles) connected to the given receiver
        """
        assert self.Metafits.metafits is not None, "metafits file associated"
        "with this observation is required to extract the receiver information"
        annumbers = self.Metafits.annumbers_for_receiver(receiver)
        gains0 = self.gains_for_antnum(annumbers[0])
        _sh = gains0.shape
        gains_array = np.zeros(
            (_sh[0], len(annumbers), _sh[1], _sh[2]), dtype=gains0.dtype)
        for i, an in enumerate(annumbers):
            gains_array[:, i, :, :] = self.gains_for_antnum(an)
        return gains_array

    def blackmanharris(self, n):
        return signal.windows.blackmanharris(n)

    def delays(self):
        # Evaluates geometric delay (fourier conjugate of frequency)
        _, freqs, _ = self.freqs_info()
        freqs = np.array(freqs) * 1e-9
        df = freqs[1] - freqs[0]
        delays = np.fft.fftfreq(freqs.size, df)
        return delays

    def _filter_nans(self, data):
        nonans_inds = np.where(~np.isnan(data))[0]
        nans_inds = np.where(np.isnan(data))[0]
        return nonans_inds, nans_inds

    def gains_fft(self):
        gains = self.gains()
        fft_data = np.zeros(gains.shape, dtype=gains.dtype)
        _sh = gains.shape
        _, freqs, _ = self.freqs_info()
        window = self.blackmanharris(len(freqs))
        for t in range(_sh[0]):
            for i in range(_sh[1]):
                for j in range(_sh[3]):
                    try:
                        nonans_inds, nans_inds = self._filter_nans(
                            gains[t, i, :, j])
                        d_fft = np.fft.fft(
                            gains[t, i, nonans_inds, j] * window[nonans_inds])
                        fft_data[t, i, nonans_inds, j] = d_fft
                        fft_data[t, i, nans_inds, j] = np.nan
                    except ValueError:
                        fft_data[t, i, :, j] = np.nan
        return fft_data
