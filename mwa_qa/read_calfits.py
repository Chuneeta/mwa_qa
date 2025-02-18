from mwa_qa.read_metafits import Metafits
from scipy.interpolate import CubicSpline
from scipy import signal
from astropy.io import fits
import numpy as np
import copy
import os


class CalFits(object):
    def __init__(self, calfits_path, pol='X',
                 norm=False, ref_antenna=None):
        """
        Object takes in a calfile in fits format and extracts
        bit and pieces of the required informations
        - calfits_path:	Fits file readable by astropy containing
                    calibration solutions (support for hyperdrive
                    output only for now) and related information
        - pol:	Polarization, can be either 'X' or 'Y'. It should be
                specified so that information associated
                with the given pol is provided. Default is 'X'
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
        with fits.open(calfits_path) as hdus:
            cal_hdu = hdus['SOLUTIONS']
            time_hdu = hdus['TIMEBLOCKS']
            result_hdu = hdus['RESULTS']
            bls_hdu = hdus['BASELINES']

            self.gain_array = cal_hdu.data[:, :, :, ::2] + \
                1j * cal_hdu.data[:, :, :, 1::2]
            self.start_time = time_hdu.data[0][0]
            self.end_time = time_hdu.data[0][1]
            self.average_time = time_hdu.data[0][2]
            self.Ntime = len(self.gain_array)
            self.uvcut = hdus['PRIMARY'].header['UVW_MIN']
            self.obsid = hdus['PRIMARY'].header['OBSID']
            self.s_thresh = hdus['PRIMARY'].header['S_THRESH']
            self.m_thresh = hdus['PRIMARY'].header['M_THRESH']
            self.antenna = hdus['TILES'].data['Antenna']
            self.annames = hdus['TILES'].data['TileName']
            self.antenna_flags = hdus['TILES'].data['Flag']
            self.frequency_array = hdus['CHANBLOCKS'].data['Freq']
            self.frequency_flags = hdus['CHANBLOCKS'].data['Flag']
            self.frequency_channels = hdus['CHANBLOCKS'].data['Index']
            self.Nchan = len(self.frequency_array)
            self.convergence = result_hdu.data
            self.baseline_weights = bls_hdu.data
            self.norm = norm
            if self.norm:
                if ref_antenna is None:
                    ref_antenna = self._iterate_refant()
                    self.reference_antenna = ref_antenna
                else:
                    self.reference_antenna = ref_antenna
                    # self._check_refant()
                self.gain_array = self.normalized_gains()
            self.amplitudes = np.abs(self.gain_array)
            self.phases = np.angle(self.gain_array)
            # self.Metafits = Metafits(metafits_path, pol=pol)
            # NOTE:polynomial parameters - only the fitted solutions will
            # have these parameters
            try:
                self.poly_order = hdus['FIT_COEFFS'].header['ORDER']
                self.poly_mse = hdus['FIT_COEFFS'].header['MSE']
            except KeyError:
                pass

    def _check_refant(self):
        """
        Checks if the given reference antenna is flagged due to non-convergence
        or any malfunctioning reports
        """
        ind = self.gains_ind_for(self.reference_antenna)
        flag = np.array(self.antenna_flags)[ind]
        assert flag == 0,  "{} seems to be flagged."
        "calibration solutions found, choose a different tile"

    def _iterate_refant(self):
        anindex = -1
        while anindex < 0:
            if self.antenna_flags[anindex] == 0:
                break
            anindex -= 1
        return self.antenna[anindex]

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
        ref_ind = self.gains_ind_for(self.reference_antenna)
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
        ngains = copy.deepcopy(self.gain_array)
        for t in range(len(ngains)):
            ngains[t] = self._normalized_data(self.gain_array[t])
        return ngains

    def gains_for_antnum(self, antnum):
        """
        Returns gain solutions for the given tile ID
        - antnum:	Antenna Number, starts from 1
        - norm:	Boolean, If True returns normalized gains
                        else unormalized gains.
                        Default is set to True.
        """
        ind = self.gains_ind_for(antnum)
        return self.gain_array[:, ind, :, :]

    def gains_for_antpair(self, antpair):
        """
        Evaluates conjugation of the gain solutions for antenna pair
                        (tile0, tile1)
        - antpair:	Tuple of antenna numbers such as (1, 2)
        """
        gains_t0 = self.gains_for_antnum(antpair[0])
        gains_t1 = self.gains_for_antnum(antpair[1])
        return gains_t0 * np.conj(gains_t1)

    def gains_for_receiver(self, metafits_path, receiver):
        """
        Returns the dictionary of gains solutions for all the antennas
        (8 antennas in principles) connected to the given receiver
        """
        m = Metafits(metafits_path)
        annumbers = m.antenna_numbers_for_receiver(receiver)
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
        df = (self.frequency_array[1] - self.frequency_array[0]) * 1e-9
        delays = np.fft.fftfreq(len(self.frequency_array), df)
        return np.fft.fftshift(delays)

    def interpolate_gains(self, x, x_new, y):
        f = CubicSpline(x, y)
        return f(x_new)

    def fft_gains(self):
        _sh = self.gain_array.shape
        fft_data = np.zeros(_sh, dtype=self.gain_array.dtype)
        window = self.blackmanharris(self.Nchan)
        for t in range(_sh[0]):
            for i in range(_sh[1]):
                for j in range(_sh[3]):
                    try:
                        inds = np.where(~np.isnan(self.gain_array[t, i, :, j]))
                        d_fft = np.fft.fft(self.interpolate_gains(
                            inds[0], np.arange(self.Nchan), self.gain_array[t, i, inds[0], j]) * window)
                        fft_data[t, i, :, j] = np.fft.fftshift(d_fft)
                    except ValueError:
                        fft_data[t, i, :, j] = np.nan
        return fft_data

    def write_to(self, filename, overwrite=False):
        """
        Overwriting gain solutions
        """
        hdu = fits.open(self.calfits_path)
        if os.path.exists(filename):
            if overwrite:
                hdu['SOLUTIONS'].data[:, :, :, ::2] = self.gain_array.real
                hdu['SOLUTIONS'].data[:, :, :, 1::2] = self.gain_array.imag
                hdu.writeto(filename, overwrite=overwrite)
            else:
                raise FileExistsError('{filename} already exists.')

        # need to add change to haser if required
