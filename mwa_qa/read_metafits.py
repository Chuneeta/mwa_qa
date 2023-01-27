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
            self.pointing_center = (hdr['RA'], hdr['DEC'])
            self.phase_centre = (hdr['RAPHASE'], hdr['DECPHASE'])
            self.filname = hdr['FILENAME']
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
            self.delay_array = hdr['DELAYS']
            tdata = hdu['TILEDATA'].data
            self._check_data(tdata)
            tdata = tdata[self.pol_index(tdata)::2]
            self.antenna_positions = np.array(
                [tdata['North'], tdata['East'], tdata['Height']]).T
            self.annumbers = tdata['Antenna']
            self.annames = tdata['TileName']
            self.tile_ids = tdata['Tile']
            self.Nants = len(self.tile_ids)
            self.receiver_ids = tdata['Rx']
            flavors = tdata['Length']
            self.cable_type = [fl.split('_')[0] for fl in flavors]
            self.cable_length = [float(fl.split('_')[1]) for fl in flavors]
            self.BFTtemps = tdata['BFTemps']
            self.flag_array = tdata['Flag']
            self.baseline_array = np.unique(np.stack(
                (np.tile(self.annumbers, self.Nants), np.repeat(self.annumbers, self.Nants))))
            bl_indxs = np.unique(np.stack(
                (np.tile(np.arange(self.Nants), self.Nants), np.repeat(np.arange(self.Nants), self.Nants))))
            self.baseline_lengths = np.linalg.norm(
                self.antenna_positions[bl_indxs[0]] - self.antenna_positions[bl_indxs[1]], axis=1)

    def pol_index(self, fits_rec):
        """
        Returns the polarizations index from the fits record
        """
        inds = np.where(fits_rec['Pol'] == self.pol)
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
        ind = np.where(self.annumbers == antnum)
        return antpos[ind[0][0], :]

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
