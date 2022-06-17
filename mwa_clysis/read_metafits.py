from astropy.io import fits
from collections import OrderedDict
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

	def _check_data(self, data):
		"""
		Checking if the metafits file has the duplicates tiles containing different polarization (East-West and North-South)
        """
		data_length = len(data)
		assert data_length % 2 == 0, "Metafits seems to missing or extra info, the length of objects does not evenly divide"
		pols = [data[i][4] for i in range(data_length)]
		pols_string = np.unique(pols)
		assert pols_string == 2, "Two plorization should be specified, found only one or more than two"
		pols_expected = pols_string * int(data_length / 2)
		assert pols == pols_expected, "Metafits does not have polarization info distribution as per standard, should contain consecutive arrangement of the tile duplicates"
	
	def _read_mdata(self):
		""" 
		Reads in data stored in hdu[0] column
		"""
		hdu = fits.open(self.metafits)
		data = hdu[1].data
		self._check_data(data)
		return data

	def mdata(self):
		"""
		Returns data for the specified polarization 
		"""
		data = self._read_mdata()
		return [data[i] for i in range(0, len(data), 2)]

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
		hdr = self.mhdr()
		nchans = mhdr['NCHANS']
		return nchans

	def frequecies(self):
		"""
		Returns the frequency array of the observation
		"""
		mhdr = self.mhdr()
		coarse_ch = mhdr['CHANNELS']
		nchans = self.get_nchans()
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

	def stop_gps_time(self):
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

	def phase_center(self):
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
		ra = mhdr['RAPHASE']
		dec = mhdr['DECPHASE']
		return (ra, dec)

	def delays(self):
		"""
		Returns delays values for the primary beam pointing
		"""
		mhdr = self.mhdr()
		return mhdr['DELAYS']

	def actual_delays(self, tile_id):
		"""
		Returns actual delays used for the given tile during the observation
		"""
		
	def _all_tile_ids(self):
		"""
		Returns all the tile ids including the dupplicates
		"""
		mdata = self.mhdr()
		tile_ids = [mdata[i][3] for i in range(len(mdata))]

	def tile_ids(self):
		"""
		Returns the tile/antenna ids
		"""
		tile_ids = self.all_tile_ids()
		return np.unique(tile_ids)

	def get_tile_ind(self, tile):
		"""
		Returns index of the specified tile ids
		"""
		tile_ids = np.array(self.tile_ids())
		return np.where(tile_ids == tile)[0]

	def tile_numbers(self):
		"""
		Returns the tile numbers associated with the tile ids
		"""
		tile_ids = self.tile_ids()
		return [int(tl.strip('Tile')) for tl in tile_ids]

	def btemps(self):
		"""
		Returns beamformer temperature in degress
		"""
		mdata = self.mdata()
		btemps = [mdata[i][13] for i in range(0, len(mdata))]
		return btemps

	def actual_delays(self):
		pass
