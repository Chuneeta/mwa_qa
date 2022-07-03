from collections import OrderedDict
from astropy.io import fits
import numpy as np

class Metafits(object):
	def __init__(self, metafits, pol):
		""" 
		Object takes in .metafits or metafits_pps.fits file readable by astropy 	
		- metafits : Metafits with extension *.metafits or _ppds.fits containing information 
					 on an observation done with MWA,
		- pol : Polarization, can be either 'X' or 'Y'. It should be specified so that information associated 
				with the given pol is provided. 
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
		Checking if the metafits file has the duplicates tiles containing different polarization (East-West and North-South)
        - data : Array containing the calibration solutions and other related information
		"""
		data_length = len(data)
		assert data_length % 2 == 0, "Metafits seems to missing or extra info, the length of objects does not evenly divide"
		pols = [data[i][4] for i in range(data_length)]
		pols_str, pols_ind = np.unique(pols, return_index = True)
		assert len(pols_str) == 2, "Two polarizations should be specified, found only one or more than two"
		pols_expected = list(pols_str[np.array(pols_ind)]) * int(data_length / 2)
		assert pols == pols_expected, "Metafits does not have polarization info distribution as per standard, should contain consecutive arrangement of the tile duplicates"
	
	def _pol_index(self, data, pol):
		# checking the first two pols
		pols_str = np.array([])
		for i in range(2):
			pols_str = np.append(pols_str, data[i][4])
		assert len(np.unique(pols_str)) == 2, "the different polarization ('X', Y') should be alternate"
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
			print ('Phase_centre coordinates are not recognised within the EoR Field')
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
		Returns local sidereal time of the mid point time of the observation in hours
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

	def tile_ids(self):
		"""
		Returns the tile/antenna ids
		"""
		data = self.mdata()
		tile_ids = [data[i][3] for i in range(len(data))]
		return tile_ids

	def tile_ind_for(self, tile_id):
		"""
		Returns index of the specified tile ids
		- tile : Tile id e.g tile 104 
		"""
		tile_ids = np.array(self.tile_ids())
		return np.where(tile_ids == tile_id)[0]

	def tile_numbers(self):
		"""
		Returns the tile numbers associated with the tile ids
		"""
		tile_ids = self.tile_ids()
		return [int(tl.strip('Tile')) for tl in tile_ids]
	
	def tile_pos(self):
		"""
		Returns all the tile position (North, East, Heigth)
		"""
		data = self.mdata()
		tile_pos = np.zeros((len(data), 3))
		for i in range(len(data)):
			tile_pos[i, 0], tile_pos[i, 1], tile_pos[i, 2] = data[i][9], data[i][10], data[i][11]
		return tile_pos

	def tile_pos_for(self, tile_id):
		"""
        Returns tile position (North, East, Heigth) for the given tile id
        - tile_id : Tile id e.g Tile 103
        """
		tile_pos = self.tile_pos()
		ind = self.tile_ind_for(tile_id)[0]
		return tile_pos[ind, :]

	def baseline_length_for(self, tilepair):
		"""
		Returns length of the given baseline or tilepair
		- tilepair : Tuple of tile numbers
		"""
		pos0 = self.tile_pos_for('Tile{:03g}'.format(tilepair[0]))
		pos1 = self.tile_pos_for('Tile{:03g}'.format(tilepair[1]))
		return np.sqrt((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)

	def baseline_lengths(self):
		"""
		Returns dictionary of tile pairs or baseline as keys and their corresponsing lengths as values
		"""
		tile_numbers = self.tile_numbers()
		tile_pos = self.tile_pos()
		baseline_dict = OrderedDict()
		for i in range(len(tile_numbers)):
			for j in range(i + 1, len(tile_numbers)):
				baseline_dict[(tile_numbers[i], tile_numbers[j])] = self.baseline_length_for((tile_numbers[i], tile_numbers[j]))
		return baseline_dict

	def get_baselines_greater_than(self, baseline_cut):
		"""
		Returns tile pairs/ baselines greater than the given cut
		- baseline_cut : Baseline length cut in metres
		"""
		bls_dict = self.baseline_lengths()
		bls = {key: value for key, value in bls_dict.items() if value > baseline_cut}
		print (bls)
		return bls

	def get_baselines_less_than(self, baseline_cut):
		"""
        Returns tile pairs/ baselines less than the given cut
        - baseline_cut : Baseline length cut in metres
        """
		bls_dict = self.baseline_lengths()
		bls = {key: value for key, value in bls_dict.items() if value < baseline_cut}
		return bls

	def _cable_flavors(self):
		"""
		Returns cable flavours for all the tiles
		"""
		data = self.mdata()
		ctypes = [data[i][16].split('_')[0] for i in range(0, len(data))]
		clengths = [float(data[i][16].split('_')[1]) for i in range(0, len(data))]
		return ctypes, clengths

	def cable_length_for(self, tile_id):
		"""
		Returns cable length for the given tile id
		- tile_id : Tile id e.g Tile 103
		"""
		ind = self.tile_ind_for(tile_id)
		ctype, clength = self._cable_flavors()
		return np.array(clength)[ind]

	def cable_type_for(self, tile_id):
		"""
        Returns cable length for the given tile id
        - tile_id : Tile id e.g Tile 103
        """
		ind = self.tile_ind_for(tile_id)
		ctype, clength = self._cable_flavors()
		return np.array(ctype)[ind]

	def receivers(self):
		"""
		Returns receiver numbers for all tiles
		"""
		data = self.mdata()
		receivers = [data[i][5] for i in range(0, len(data))]
		return receivers

	def receiver_for(self, tile_id):
		"""
        Returns receiver number for the given tile id
        - tile_id : Tile id e.g Tile 103
        """
		receivers = np.array(self.receivers())
		ind = self.tile_ind_for(tile_id)
		return receivers[ind]

	def tiles_for_receiver(self, receiver):
		"""
		Returns tile IDs connected with the given receiver
		- receiver : receiver number 1-16
		"""
		tile_ids = np.array(self.tile_ids())
		receivers = np.array(self.receivers())
		inds = np.where(receivers == receiver)
		return tile_ids[inds]

	def btemps(self):
		"""
		Returns beamformer temperature in degress
		"""
		data = self.mdata()
		btemps = [data[i][13] for i in range(0, len(data))]
		return btemps

