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
            hdr = fits['PRIMARY'].header
            self.lst = hdr['LST']
            self.ha = hdr['HA']
            self.az_alt = (hdr['AZIMUTH'], hdr['ALTITUDE'])
            self.pointing_center = (hdr['RA'], hdr['DEC'])
            self.phase_center = (hdr['RAPHASE'], hdr['DECPHASE'])
            self.filname = hdr['FILENAME']
            self.start_gpstime = hdr['GPSTIME']
            self.exposure = hdr['EXPOSURE']
            self.integration = hdr['INTTIME']
            self.obs_time = hdr['DATA-OBS']
            self.nchans = hdr['NCHAN']
            coarse_ch = hdr['CHANNELS']
            start = float(coarse_ch.split(',')[0]) * 1.28
            stop = float(coarse_ch.split(',')[-1]) * 1.28
            self.frequency_array = np.linspace(start, stop, self.Nchan)
            if self.phase_centre == (0.0, -27.0):
                self.eorfield = 'EoR0'
            elif self.phase_centre == (60.0, -30.0):
                self.eorfield = 'EoR1'
            else:
                self.eorfield = 'Unknown'
            self.delay_array = hdr['DELAYS']
            self.antenna_postions = np.array(
                [tdata['North'], tdata['East'], tdata['Heigth']]).T
            tdata = fits['TILEDATA'].data
            self._check_data(tdata)
            tdata = tdata[self.pol_index::2]
            self.annumbers = tdata['Antenna']
            self.annames = tdata['TileName']
            self.tile_ids = tdata['Tile']
            self.receiver_ids = tdata['Rx']
            flavors = tdata['Length']
            self.cable_type = [fl.split('_')[0] for fl in flavors]
            self.cable_length = [int(fl.split('_')[1]) for fl in flavors]
            self.BFTtemps = tdata['BFTemps']
            self.flag_array = tdata['Flag']

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
        pols_expected = list(
            pols_str[np.array(pols_ind)]) * int(data_length / 2)
        assert pols == pols_expected, "Metafits does not have polarization "\
            "info distribution as per standard, should contain "\
            "consecutive arrangement of the tile duplicates"

    def antenna_position_for(self, antnum):
        """
        Returns tile position (North, East, Heigth) for the given antenna
        number
        - antnum:	Antenna Number, starts from 1
        """
        antpos = self.antenna_positions
        ind = np.where(self.annumbers == antnum)
        return antpos[ind[0][0], :]
