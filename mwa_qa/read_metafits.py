from collections import OrderedDict
from astropy.io import fits
import numpy as np


class Metafits(object):
    def __init__(self, metafits, pol):
        """
        Object takes in .metafits or metafits_pps.fits file readable by astropy
        - metafits: Metafits with extension *.metafits or _ppds.fits containing
                    information on an observation done with MWA,
        - pol:  Polarization, can be either 'X' or 'Y'. It should be specified
                so that information associated with the given pol is provided.
        """
        self.metafits = metafits
        self.pol = pol.upper()

    def _read_data(self):
        """
        Reads in data stored in hdu[0] column
        """
        hdu = fits.open(self.metafits)
        data = hdu[1].data
        return data

    def _check_data(self, data):
        """
        Checking if the metafits file has the duplicates tiles containing
        different polarization (East-West and North-South)
        """
        data_length = len(data)
        assert data_length % 2 == 0, "Metafits seems to missing some info, "\
            "the length of objects does not evenly divide"
        pols = [data[i][4] for i in range(data_length)]
        pols_str, pols_ind = np.unique(pols, return_index=True)
        assert len(
            pols_str) == 2, "Two polarizations should be specified, "\
            "found only one or more than two"
        pols_expected = list(
            pols_str[np.array(pols_ind)]) * int(data_length / 2)
        assert pols == pols_expected, "Metafits does not have polarization "\
            "info distribution as per standard, should contain "\
            "consecutive arrangement of the tile duplicates"

    def _pol_index(self, data, pol):
        # checking the first two pols
        pols_str = np.array([])
        for i in range(2):
            pols_str = np.append(pols_str, data[i][4])
        assert len(np.unique(
            pols_str)) == 2, "the different polarization ('X', Y') should be "\
            "alternate"
        return np.where(pols_str == pol.upper())[0]

    def mdata(self):
        """
        Returns data for the specified polarization
        """
        data = self._read_data()
        self._check_data(data)
        ind = self._pol_index(data, self.pol)
        if ind % 2 == 0:
            return data[0::2]
        else:
            return data[1::2]

    def mhdr(self):
        """
        Returns header stored in hdu[1] column
        """
        hdu = fits.open(self.metafits)
        mhdr = hdu[0].header
        return mhdr

    def nchans(self):
        """
        Returns the number of frequency channels
        """
        mhdr = self.mhdr()
        nchans = mhdr['NCHANS']
        return nchans

    def frequencies(self):
        """
        Returns the frequency array of the observation
        """
        mhdr = self.mhdr()
        coarse_ch = mhdr['CHANNELS']
        nchans = self.nchans()
        start = float(coarse_ch.split(',')[0]) * 1.28
        stop = float(coarse_ch.split(',')[-1]) * 1.28
        freqs = np.linspace(start, stop, nchans)
        return freqs

    def obs_time(self):
        """
        Returns UTC time and date of the observation
        """
        mhdr = self.mhdr()
        obsdate = mhdr['DATE-OBS']
        return obsdate

    def int_time(self):
        """
        Returns integration time for the obsrvations
        """
        mhdr = self.mhdr()
        return mhdr['INTTIME']

    def exposure(self):
        """
        Return whole exposure time for the observation
        """
        mhdr = self.mhdr()
        return mhdr['EXPOSURE']

    def start_gpstime(self):
        """
        Returns starting time of the observation
        """
        mhdr = self.mhdr()
        return mhdr['GPSTIME']

    def stop_gpstime(self):
        """
        Return ending time of the observation
        """
        start_time = self.start_gpstime()
        obs_time = self.exposure()
        stop_time = start_time + obs_time
        return stop_time

    def eor_field(self):
        """
        Returns the EoR field for the observation, can be either EoR0 or EoR1
        """
        phase_centre = self.phase_centre()
        if phase_centre == (0.0, -27.0):
            eorfield = 'EoR0'
        elif phase_centre == (60.0, -30.0):
            eorfield = 'EoR1'
        else:
            print(
                'Phase_centre coordinates are not recognised within '
                'the EoR Field')
        return eorfield

    def az_alt(self):
        """
        Returns az alt coordinates of the primary beam centre
        """
        mhdr = self.mhdr()
        az = mhdr['AZIMUTH']
        alt = mhdr['ALTITUDE']
        return (az, alt)

    def ha(self):
        """
        Returns hour angle of the primanry beam centre in degrees
        """
        mhdr = self.mhdr()
        return mhdr['HA']

    def lst(self):
        """
        Returns local sidereal time of the mid point time of the
                        observation in hours
                        """
        mhdr = self.mhdr()
        return mhdr['LST']

    def phase_centre(self):
        """
        Returns coordinates/angle of the ddesired target in degrees
        """
        mhdr = self.mhdr()
        ra = mhdr['RAPHASE']
        dec = mhdr['DECPHASE']
        return (ra, dec)

    def pointing(self):
        """
        Returns coordinates of the primary beam center in degrees
        """
        mhdr = self.mhdr()
        ra = mhdr['RA']
        dec = mhdr['DEC']
        return (ra, dec)

    def delays(self):
        """
        Returns delays values for the primary beam pointing
        """
        mhdr = self.mhdr()
        return mhdr['DELAYS']

    def annumbers(self):
        """
        Returns antenna numbers
        """
        data = self.mdata()
        annumbers = [data[i][1] for i in range(len(data))]
        return annumbers

    def annames(self):
        """
        Returns the antenna names
        """
        data = self.mdata()
        annames = [data[i][3] for i in range(len(data))]
        return annames

    def ind_for_annumber(self, antnum):
        """
        Returns index of the specified antenna number
        - antnum : Antenna Number starts from 0
        """
        annumbers = np.array(self.annumbers())
        return np.where(annumbers == antnum)[0][0]

    def ind_for_anname(self, antname):
        """
        Returns index of the specified antenna number
        - antnum : Antenna Name e.g 'Tile011' or 'HEXS6'
        """
        annames = np.array(self.annames())
        return np.where(annames == antname)[0][0]

    def anpos(self):
        """
        Returns all the antenna position (North, East, Heigth)
        """
        data = self.mdata()
        antpos = np.zeros((len(data), 3))
        for i in range(len(data)):
            antpos[i, 0] = data[i][9]
            antpos[i, 1] = data[i][10]
            antpos[i, 2] = data[i][11]
        return antpos

    def anpos_for(self, antnum):
        """
        Returns tile position (North, East, Heigth) for the given antenna
        number
        - antnum:	Antenna Number, starts from 1
        """
        antpos = self.anpos()
        ind = self.ind_for_annumber(antnum)
        return antpos[ind, :]

    def baseline_length_for(self, antpair):
        """
        Returns length of the given baseline or antpair
        - antpair : Tuple of Antenna numbers
        """
        pos0 = self.anpos_for(antpair[0])
        pos1 = self.anpos_for(antpair[1])
        return np.sqrt((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)

    def baseline_lengths(self):
        """
        Returns dictionary of tile pairs or baseline as keys and
        their corresponding lengths as values
        """
        annumbers = self.annumbers()
        baseline_dict = OrderedDict()
        for i in range(len(annumbers)):
            for j in range(i + 1, len(annumbers)):
                baseline_dict[(annumbers[i], annumbers[j])] = \
                    self.baseline_length_for(
                    (annumbers[i], annumbers[j]))
        return baseline_dict

    def anpos_dict(self):
        anpos = self.anpos()
        annumbers = self.annumbers()
        anpos_dict = OrderedDict()
        for i, ant in enumerate(annumbers):
            anpos_dict[ant] = anpos[i].tolist()
        return anpos_dict

    def group_antpairs(self, bl_tol):
        angroups = OrderedDict()
        anpos_dict = self.anpos_dict()
        ankeys = list(anpos_dict.keys())
        delta_z = np.abs(np.array(list(anpos_dict.values()))[
            :, 2] - np.mean(list(anpos_dict.values()), axis=0)[2])
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

        for i, ant1 in enumerate(ankeys):
            for j, ant2 in enumerate(ankeys[i + 1:]):
                antpair = (ant1, ant2)
                delta = tuple(np.round(
                    1.0 * (np.array(anpos_dict[ant2]) -
                           np.array(anpos_dict[ant1]))
                    / bl_tol).astype(int))
                nkey = _check_neighbours(delta)
                if nkey is None:
                    nkey = _check_neighbours(tuple([-d for d in delta]))
                    if nkey is None:
                        antpair = (ant2, ant1)
                if nkey is not None:
                    angroups[nkey].append(antpair)
                else:
                    # new baseline
                    if delta[0] <= 0 or (delta[0] == 0 and delta[1] <= 0) or \
                        (delta[0] == 0 and delta[1] == 0 and
                         delta[2] <= 0):
                        delta = tuple([-d for d in delta])
                        antpair = (ant2, ant1)
                    angroups[delta] = [antpair]
        return angroups

    def redundant_antpairs(self, bl_tol=2e-1):
        # keeping only redundant pairs
        angroups = self.group_antpairs(bl_tol=bl_tol)
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

    def baselines_greater_than(self, baseline_cut):
        """
        Returns tile pairs/ baselines greater than the given cut
        - baseline_cut : Baseline length cut in metres
        """
        bls_dict = self.baseline_lengths()
        bls = {key: value for key, value in bls_dict.items() if value >
               baseline_cut}
        return bls

    def baselines_greater_than(self, baseline_cut):
        """
        Returns tile pairs/ baselines less than the given cut
        - baseline_cut : Baseline length cut in metres
        """
        angroups = self.group_antpairs(bl_tol=1.)
        keys = list(angroups.keys())
        lengths = [np.linalg.norm(key) for key in keys]
        inds = np.where(np.array(lengths) < baseline_cut)
        chosen_keys = np.array(keys)[inds[0]]
        baselines = []
        for ckey in chosen_keys:
            antpairs = [antp for antp in angroups[tuple(ckey)]]
            for antp in antpairs:
                baselines.append(antp)
        return baselines

    def baselines_less_than(self, baseline_cut):
        """
        Returns tile pairs/ baselines less than the given cut
        - baseline_cut : Baseline length cut in metres
        """
        angroups = self.group_antpairs(bl_tol=1.)
        keys = list(angroups.keys())
        lengths = [np.linalg.norm(key) for key in keys]
        inds = np.where(np.array(lengths) < baseline_cut)
        chosen_keys = np.array(keys)[inds[0]]
        baselines = []
        for ckey in chosen_keys:
            antpairs = [antp for antp in angroups[tuple(ckey)]]
            for antp in antpairs:
                baselines.append(antp)
        return baselines

    def _cable_flavors(self):
        """
        Returns cable flavours for all the tiles
        """
        data = self.mdata()
        ctypes = [data[i][16].split('_')[0] for i in range(0, len(data))]
        clengths = [float(data[i][16].split('_')[1])
                    for i in range(0, len(data))]
        return ctypes, clengths

    def cable_length_for(self, antnum):
        """
        Returns cable length for the given Antenna number
        - antnum : Antenna number
        """
        ind = self.ind_for_annumber(antnum)
        ctype, clength = self._cable_flavors()
        return np.array(clength)[ind]

    def cable_type_for(self, antnum):
        """
        Returns cable type for the given Antenna number
        - antnum : Antenna number
        """
        ind = self.ind_for_annumber(antnum)
        ctype, clength = self._cable_flavors()
        return np.array(ctype)[ind]

    def receivers(self):
        """
        Returns receiver numbers for all tiles
        """
        data = self.mdata()
        receivers = [data[i][5] for i in range(0, len(data))]
        return receivers

    def receiver_for(self, antnum):
        """
        Returns receiver number for the given Antenna number
        - antnum : Antenna Number
        """
        receivers = np.array(self.receivers())
        ind = self.ind_for_annumber(antnum)
        return receivers[ind]

    def annumbers_for_receiver(self, receiver):
        """
        Returns Antenna numbers connected with the given receiver
        - receiver : receiver number 1-16
        """
        annumbers = np.array(self.annumbers())
        receivers = np.array(self.receivers())
        inds = np.where(receivers == receiver)
        return annumbers[inds]

    def btemps(self):
        """
        Returns beamformer temperature in degress
        """
        data = self.mdata()
        btemps = [data[i][13] for i in range(0, len(data))]
        return btemps
