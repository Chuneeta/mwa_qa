from mwa_qa import read_metafits as rm
from astropy.io import fits
import numpy as np

# speed of light
c = 299_792_458
pol_dict = {'XX': 0, 'YY': 1, 'XY': 2, 'YX': 3}


class UVfits(object):
    def __init__(self, uvfits, metafits=None, pol='X'):
        self.uvfits = uvfits
        self.Metafits = rm.Metafits(metafits=metafits, pol=pol)
        self._dgroup = self._read_dgroup()
        self.Nants = len(self._ant_info())
        bls = self.baselines()
        self.Nbls = len(np.unique(np.array(bls)))
        self.Ntimes = int(len(bls) / self.Nbls)
        self.Nfreqs = self._dgroup[0][5].shape[2]
        self.Npols = self._dgroup[0][5].shape[3]

    def _read_dgroup(self):
        return fits.open(self.uvfits)[0].data

    def _header(self):
        return fits.open(self.uvfits)[0].header

    def _ant_info(self):
        return fits.open(self.uvfits)[1].data

    def annames(self):
        ant_info = self._ant_info()
        return [ant_info[i][0] for i in range(self.Nants)]

    def annumbers(self):
        ant_info = self._ant_info()
        return [ant_info[i][2] for i in range(self.Nants)]

    def group_count(self):
        hdr = self._header()
        return hdr['GCOUNT']

    def baselines(self):
        gcount = self.group_count()
        baselines = [self._dgroup[i][3] for i in range(gcount)]
        return baselines

    def _encode_baseline(self, ant1_number, ant2_number):
        if ant2_number > 255:
            return ant1_number * 2048 + ant2_number + 65_536
        else:
            return ant1_number * 256 + ant2_number

    def _decode_baseline(self, bl):
        if bl < 65_535:
            ant2_number = bl % 256
            ant1_number = (bl - ant2_number) / 256
        else:
            ant2_number = (bl - 65_536) % 2048
            ant1_number = (bl - ant2_number - 65_536) / 2048
        return (int(ant1_number), int(ant2_number))

    def annumber_to_anname(self, number):
        annumbers = np.array(self.annumbers())
        annames = np.array(self.annames())
        ind = np.where(annumbers == number)[0][0]
        return annames[ind]

    def anname_to_annumber(self, ant_name):
        annumbers = np.array(self.annumbers())
        annames = np.array(self.annames())
        ind = np.where(annames == ant_name)[0][0]
        return annumbers[ind]

    def _indices_for_antpair(self, antpair):
        bls = np.array(self.baselines())
        bl = self._encode_baseline(antpair[0], antpair[1])
        return np.where(bls == bl)[0]

    def antpairs(self):
        baselines = self.baselines()
        antpairs = []
        for bl in baselines:
            antpair = self._decode_baseline(bl)
            antpairs.append((antpair[0], antpair[1]))
        return antpairs

    def uvw(self):
        gcount = self.group_count()
        uvw = np.zeros((3, gcount))
        for i in range(gcount):
            uvw[0, i] = self._dgroup[i][0] * c
            uvw[1, i] = self._dgroup[i][1] * c
            uvw[2, i] = self._dgroup[i][2] * c
        return uvw

    def pols(self):
        # Npols=4 --> ('XX', 'XY', 'YX', 'YY')
        # Npols=2  --> ('XX', 'YY')
        if self.Npols == 2:
            return ['XX', 'YY']
        if self.Npols == 4:
            return ['XX', 'XY', 'YX', 'YY']
        else:
            raise (ValueError, "currently support only 2 and 4 polarizations")

    def data_for_antpair(self, antpair):
        inds = self._indices_for_antpair(antpair)
        pols = self.pols()
        # data shape (times, freqs, pol)
        data = np.zeros((len(inds), self.Nfreqs, self.Npols),
                        dtype=np.complex128)
        for i, ind in enumerate(inds):
            for j, p in enumerate(pols):
                data[i, :, j] = self._dgroup[ind][5][0, 0, :, pol_dict[p], 0]
                + self._dgroup[ind][5][0, 0, :, pol_dict[p], 1] * 1j
        return data

    def data_for_antpairpol(self, antpairpol):
        data_antpair = self.data_for_antpair((antpairpol[0], antpairpol[1]))
        pols = np.array(self.pols())
        return data_antpair[:, :, np.where(pols == antpairpol[2])[0][0]]
