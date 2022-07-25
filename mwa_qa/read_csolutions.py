from mwa_qa import read_metafits as rm
from collections import OrderedDict
from scipy import signal
from astropy.io import fits
import numpy as np
import copy

HDUlist = ['PRIMARY', 'SOLUTIONS', 'TIMEBLOCKS',
           'TILES', 'CHANBLOCKS', 'RESULTS', 'BASELINES']


class Csoln(object):
    def __init__(self, calfile, metafits=None, pol='X',
                 norm=False, ref_antnum=None):
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
        - norm: Boolean, If True, the calibration solutions will be
                normlaized else unnormlaized solutions will be used.
                Default is set to False
        - ref_antnum:   Reference antenna number. If norm is True,
                        a reference antenna is require for normalization.
                        By default it uses the last antenna in the array.
                        If the last antenna is flagged, it will return
                        an error.
        """
        self.calfile = calfile
        self.Metafits = rm.Metafits(metafits, pol)
        self.norm = norm
        if self.norm:
            if ref_antnum is None:
                ref_antnum = self._iterate_refant()
                self.ref_antnum = ref_antnum
            else:
                self.ref_antnum = ref_antnum
                self._check_refant()

    def _iterate_refant(self):
        annumbers = self.ant_info()['ANTENNA']
        anflags = self.ant_info()['FLAG']
        anindex = -1
        while anindex < 0:
            if anflags[anindex] == 0:
                break
            anindex -= 1
        return annumbers[anindex]

    def _check_refant(self):
        """
        Checks if the given reference antenna is flagged due to non-convergence
        or any malfunctioning reports
        """
        anflags = self.ant_info()['FLAG']
        ind = self.gains_ind_for(self.ref_antnum)
        flag = np.array(anflags)[ind]
        assert flag == 0,  "{} seems to be flagged."
        "calibration solutions found, choose a different tile"

    def _hdu(self):
        return fits.open(self.calfile)

    def data(self, hdu_name):
        """
        Returns the data stored in the specified HDU column of the image
        hdu_name:	HDU Column Name. The names should be one from
                    HDULIist.
        For more details refer to
        https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
        """
        hdu = self._hdu()
        index = hdu.index_of(hdu_name)
        return hdu[index].data

    def header(self, hdu_name):
        """
        Returns the header of the specified HDU column
        hdu_name:	HDU Column Name. The names should be one from
                                        HDULIist.
        For more details refer to
        https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyperdrive.html
        """
        hdu = self._hdu()
        index = hdu.index_of(hdu_name)
        return hdu[index].header

    def hdu_shape(self, hdu_name):
        hdr = self.header(hdu_name)
        naxis = hdr['NAXIS']
        shape = []
        for ax in range(1, naxis + 1):
            shape.append(hdr['NAXIS{}'.format(ax)])
        return tuple(shape)

    def hdu_fields(self, hdu_name):
        hdr = self.header(hdu_name)
        fields = []
        try:
            nfield = hdr['TFIELDS']
            for fd in range(1, nfield + 1):
                fields.append(hdr['TTYPE{}'.format(fd)])
        except KeyError:
            print('WARNING: No fields is found for HDU '
                  'Column "{}"'.format(hdu_name))
            pass
        return tuple(fields)

    def gains_real(self):
        """
        Returns the real part of the calibration solutions
        """
        cal_solutions = self.data('SOLUTIONS')
        return cal_solutions[:, :, :, ::2]

    def gains_imag(self):
        """
        Returns the imaginary part of the calibration solutions
        """
        cal_solutions = self.data('SOLUTIONS')
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

    def ntimeblocks(self):
        """
        Returns the timeblocks of the calibration solutions
        """
        d_header = self.header('SOLUTIONS')
        return d_header['NAXIS4']

    def ant_info(self):
        """
        Returns the info on the ant induces, tile ID and flags.
        The ant indices/ numbers start from 0 and the ant numbers.
        """
        ant_data = self.data('TILES')
        fields = self.hdu_fields('TILES')
        ant_info = OrderedDict()
        for i, fd in enumerate(fields):
            ant_info[fd.upper()] = [ant_data[a][i]
                                    for a in range(len(ant_data))]
        return ant_info

    def channel_info(self):
        """
        Returns the channels information as a dictionary
        """
        ch_data = self.data('CHANBLOCKS')
        fields = self.hdu_fields('CHANBLOCKS')
        ch_info = OrderedDict()
        for i, fd in enumerate(fields):
            ch_info[fd.upper()] = [ch_data[ch][i]
                                   for ch in range(len(ch_data))]
        return ch_info

    def gains_ind_for(self, antnum):
        """
        Returns index of the gain solutions fot the given antenna number,
                        indices matches the antenna number in this case
        - antnum : Antenna Number
        """
        return antnum

    def _normalized_data(self, data):
        """
        Normalizes the gain solutions for each timeblock given a reference tile
        - data:	Input array of shape( tiles, freq, pols) containing the
                        solutions
        """
        # if self.ref_antnum is None:
        #    annumbers, annames, _ = self.ant_info()
        #    ref_ind = -1
        #    ref_antnum = np.array(annumbers)[ref_ind]
        # else:
        ref_ind = self.gains_ind_for(self.ref_antnum)
        refs = []
        for ref in data[ref_ind].reshape((-1, 2, 2)):
            refs.append(np.linalg.inv(ref))
        refs = np.array(refs)
        div_ref = []
        for tile_i in data:
            for (i, ref) in zip(tile_i, refs):
                div_ref.append(i.reshape((2, 2)).dot(ref))
        return np.array(div_ref).reshape(data.shape)

    def normalized_gains(self):
        """
        Returns the normalized gain solutions using the
        given reference Antenna number
        """
        gains = self.gains()
        ngains = copy.deepcopy(gains)
        for t in range(len(ngains)):
            ngains[t] = self._normalized_data(gains[t])
        return ngains

    def _select_gains(self):
        """
        Return normalized if norm is True else unnomalized gains
        - norm:	Boolean, If True returns normalized gains else
                        unormalized gains.
        """
        if self.norm:
            return self.normalized_gains()
        else:
            return self.gains()

    def amplitudes(self):
        """
        Returns amplitude of the normalized gain solutions
        - norm:	Boolean, if True returns normalized gains else
                        unormalized gains.
                        Default is set to True.
        """
        gains = self._select_gains()
        return np.abs(gains)

    def phases(self):
        """
        Returns phases in degrees of the normalized gain solutions
        - norm: Boolean, if True returns normalized gains else
                        unormalized gains.
                        Default is set to True.
        """
        gains = self._select_gains()
        return np.angle(gains) * 180 / np.pi

    def gains_for_antnum(self, antnum):
        """
        Returns gain solutions for the given tile ID
        - antnum:	Antenna Number, starts from 1
        - norm:	Boolean, If True returns normalized gains
                        else unormalized gains.
                        Default is set to True.
        """
        gains = self._select_gains()
        ind = self.gains_ind_for(antnum)
        return gains[:, ind, :, :]

    def gains_for_antpair(self, antpair):
        """
        Evaluates conjugation of the gain solutions for antenna pair
                        (tile0, tile1)
        - antpair:	Tuple of antenna numbers such as (1, 2)
        """
        gains_t0 = self.gains_for_antnum(antpair[0])
        gains_t1 = self.gains_for_antnum(antpair[1])
        return gains_t0 * np.conj(gains_t1)

    def gains_for_receiver(self, receiver):
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
        freqs = self.channel_info()['FREQ']
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
        freqs = self.channel_info()['FREQ']
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
