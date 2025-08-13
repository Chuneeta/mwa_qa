from collections import OrderedDict
from astropy.io import fits
import numpy as np


class Metafits(object):
    def __init__(self, metafits, pol='X'):
        """
        Object takes in .metafits or metafits_pps.fits file readable by astropy
        - metafits: Metafits with extension *.metafits or _ppds.fits containing
                    information on an observation done with MWA,
        - pol:  Polarization, can be either 'X' or 'Y'. It should be specified
                so that information associated with the given pol is provided.
                Default is X
        """
        self.metafits = metafits
        self.pol = pol.upper()
        with fits.open(self.metafits) as hdu:
            hdr = hdu['PRIMARY'].header
            self.lst = hdr['LST']
            self.ha = hdr['HA']
            self.az_alt = (hdr['AZIMUTH'], hdr['ALTITUDE'])
            self.pointing_centre = (hdr['RA'], hdr['DEC'])
            self.phase_centre = (hdr['RAPHASE'], hdr['DECPHASE'])
            self.filename = hdr['FILENAME']
            self.start_gpstime = hdr['GPSTIME']
            self.exposure = hdr['EXPOSURE']
            self.integration = hdr['INTTIME']
            self.obs_time = hdr['DATE-OBS']
            self.Nchans = hdr['NCHANS']
            coarse_ch = hdr['CHANNELS']
            start = float(coarse_ch.split(',')[0]) * 1.28
            stop = float(coarse_ch.split(',')[-1]) * 1.28
            self.frequency_array = np.linspace(start, stop, self.Nchans)
            if self.phase_centre == (0.0, -27.0):
                self.eorfield = 'EoR0'
            elif self.phase_centre == (60.0, -30.0):
                self.eorfield = 'EoR1'
            else:
                self.eorfield = 'Unknown'
            tdata = hdu['TILEDATA'].data
            self._check_data(tdata)
            tdata = tdata[self.pol_index(tdata)::2]
            self.delays = tdata['Delays']
            self.dipole_gains = tdata['Calib_Gains']
            self.antenna_positions = np.array(
                [tdata['East'], tdata['North'], tdata['Height']]).T
            self.antenna_numbers = tdata['Antenna']
            self.antenna_names = tdata['TileName']
            self.tile_ids = tdata['Tile']
            self.Nants = len(self.tile_ids)
            self.receiver_ids = tdata['Rx']
            self.cable_flavors = tdata['Flavors']
            self.cable_type = [fl.split('_')[0] for fl in self.cable_flavors]
            self.cable_lengths = []
            for fl in self.cable_flavors:
                try:
                    self.cable_lengths.append(fl.split('_')[1])
                except IndexError:
                    self.cable_lengths.append(' ')
            self.BFTemps = tdata['BFTemps']
            self.flag_array = tdata['Flag']
            self.antpairs = np.sort(np.stack(
                (np.tile(self.antenna_numbers, self.Nants), np.repeat(self.antenna_numbers, self.Nants)), axis=1), axis=1)
            self.antpairs = self.antpairs[np.unique(
                self.antpairs, axis=0, return_index=True)[1]]
            bl_indxs = np.sort(np.stack((np.tile(np.arange(self.Nants), self.Nants), np.repeat(
                np.arange(self.Nants), self.Nants)), axis=1), axis=1)
            bl_indxs = bl_indxs[np.unique(
                bl_indxs, axis=0, return_index=True)[1]]
            self.baseline_lengths = np.linalg.norm(
                self.antenna_positions[bl_indxs[:, 0]] - self.antenna_positions[bl_indxs[:, 1]], axis=1)

    def pol_index(self, data):
        """
        Returns the polarizations index from the fits record
        """
        inds = np.where(data['Pol'] == self.pol)
        return inds[0][0]

    def _check_data(self, data):
        """
        Checking if the metafits file has the duplicates tiles containing
        different polarization (East-West and North-South)
        """
        data_length = len(data)
        assert data_length % 2 == 0, "Metafits seems to missing some info, "\
            "the length of objects does not evenly divide"
        pols = data['Pol']
        pols_str, pols_ind = np.unique(pols, return_index=True)
        assert len(
            pols_str) == 2, "Two polarizations should be specified, "\
            "found only one or more than two"

    def antenna_position_for(self, antnum):
        """
        Returns tile position (North, East, Heigth) for the given antenna
        number
        - antnum:	Antenna Number, starts from 1
        """
        antpos = self.antenna_positions
        ind = np.where(self.antenna_numbers == antnum)
        return antpos[ind[0][0], :]

    def baseline_length_for(self, antpair):
        """
        Returns length of the given baseline or antpair
        - antpair : Tuple of Antenna numbers
        """
        ind = np.where((self.antpairs[:, 0] == antpair[0]) & (
            self.antpairs[:, 1] == antpair[1]))
        if (len(ind[0])) == 0:
            raise ValueError('Given antenna pair does not exist.')
        else:
            return self.baseline_lengths[ind[0][0]]

    def baselines_greater_than(self, baseline_cut):
        """
        Returns tile pairs/ baselines greater than the given cut
        - baseline_cut : Baseline length cut in metres
        """
        return self.antpairs[np.where(self.baseline_lengths > baseline_cut)[0]]

    def baselines_less_than(self, baseline_cut):
        """
        Returns tile pairs/ baselines greater than the given cut
        - baseline_cut : Baseline length cut in metres
        """
        return self.antpairs[np.where(self.baseline_lengths < baseline_cut)[0]]

    def antenna_numbers_for_receiver(self, receiver):
        """
        Returns Antenna numbers connected with the given receiver
        - receiver : receiver number 1-16
        """
        inds = np.where(self.receiver_ids == receiver)
        if len(inds[0]) == 0:
            raise ValueError("Antenna Number does not exist")
        else:
            return self.antenna_numbers[inds[0]]

    def receiver_for_antenna_number(self, antnum):
        """
        Returns receiver number for the given Antenna number
        - antnum : Antenna Number
        """
        ind = np.where(self.antenna_numbers == antnum)
        if len(ind[0]) == 0:
            raise ValueError("Receiver ID does not exist")
        else:
            return self.receiver_ids[ind[0]][0]

    def _anpos_dict(self):
        anpos = self.antenna_positions
        anpos_dict = OrderedDict()
        for i, ant in enumerate(self.antenna_numbers):
            anpos_dict[ant] = anpos[i].tolist()
        return anpos_dict

    def group_antpairs(self, bl_tol):
        angroups = OrderedDict()
        anpos_dict = self._anpos_dict()
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
