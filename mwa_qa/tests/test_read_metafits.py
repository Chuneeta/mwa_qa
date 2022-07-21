from mwa_qa import read_metafits as rm
import unittest
import numpy as np
import astropy

metafits = '../../test_files/test_1061315688.metafits'
hdu = astropy.io.fits.open(metafits)[1].data
hdr = astropy.io.fits.open(metafits)[0].header
nants = int(len(hdu) / 2)
antnums = [hdu[i][1] for i in range(1, len(hdu), 2)]
ants = [hdu[i][3] for i in range(1, len(hdu), 2)]
expected_btemps = [hdu[i][13] for i in range(1, len(hdu), 2)]
antpairs = []
for i, ant1 in enumerate(antnums):
    for ant2 in antnums[i+1:]:
        antpairs.append((ant1, ant2))


class TestMetafits(unittest.TestCase):
    def test_init__(self):
        m = rm.Metafits(metafits, 'X')
        self.assertEqual(m.metafits, metafits)
        m = rm.Metafits(metafits, 'X')
        self.assertEqual(m.pol, 'X')
        m = rm.Metafits(metafits, 'y')
        self.assertEqual(m.pol, 'Y')

    def test_read_data(self):
        m = rm.Metafits(metafits, 'X')
        data = m._read_data()
        self.assertTrue((data == hdu).all())

    def test_check_data(self):
        m = rm.Metafits(metafits, 'X')
        data = m._read_data()
        with self.assertRaises(Exception):
            m._check_data(data[0])
        with self.assertRaises(Exception):
            tempered_data = data[0:3:2]
            m._check_data(tempered_data)
        with self.assertRaises(Exception):
            m.check_data(data[0:3])

    def test_pol_index(self):
        m = rm.Metafits(metafits, 'X')
        ind = m._pol_index(hdu, 'X')
        self.assertEqual(ind, 1)

    def test_mdata(self):
        m = rm.Metafits(metafits, 'X')
        data = m.mdata()
        self.assertTrue((data == hdu[1::2]).all())

    def test_mhdr(self):
        m = rm.Metafits(metafits, 'X')
        mhdr = m.mhdr()
        self.assertTrue(mhdr == hdr)

    def test_nchans(self):
        m = rm.Metafits(metafits, 'X')
        nchans = m.nchans()
        self.assertEqual(nchans, 768)

    def test_frequencies(self):
        m = rm.Metafits(metafits, 'X')
        frequencies = m.frequencies()
        self.assertEqual(len(frequencies), hdr['NCHANS'])
        expected_frequencies = np.linspace(131 * 1.28, 154 * 1.28, 768)
        self.assertTrue((frequencies == expected_frequencies).all())

    def test_obs_time(self):
        m = rm.Metafits(metafits, 'X')
        obs_time = m.obs_time()
        self.assertEqual(obs_time, hdr['DATE-OBS'])

    def test_int_time(self):
        m = rm.Metafits(metafits, 'X')
        int_time = m.int_time()
        self.assertEqual(int_time, hdr['INTTIME'])

    def test_exposure(self):
        m = rm.Metafits(metafits, 'X')
        exposure = m.exposure()
        self.assertEqual(exposure, hdr['EXPOSURE'])

    def test_start_gpstime(self):
        m = rm.Metafits(metafits, 'X')
        start_gpstime = m.start_gpstime()
        self.assertEqual(start_gpstime, hdr['GPSTIME'])

    def test_stop_gpstime(self):
        m = rm.Metafits(metafits, 'X')
        stop_gpstime = m.stop_gpstime()
        self.assertEqual(stop_gpstime, hdr['GPSTIME'] + hdr['EXPOSURE'])

    def test_eor_field(self):
        m = rm.Metafits(metafits, 'X')
        eor_field = m.eor_field()
        self.assertEqual(eor_field, 'EoR0')

    def test_az_alt(self):
        m = rm.Metafits(metafits, 'X')
        az_alt = m.az_alt()
        self.assertTrue(az_alt, (hdr['AZIMUTH'], hdr['ALTITUDE']))

    def test_ha(self):
        m = rm.Metafits(metafits, 'X')
        ha = m.ha()
        self.assertEqual(ha, hdr['HA'])

    def test_lst(self):
        m = rm.Metafits(metafits, 'X')
        lst = m.lst()
        self.assertEqual(lst, hdr['LST'])

    def test_phase_centre(self):
        m = rm.Metafits(metafits, 'X')
        phase_centre = m.phase_centre()
        self.assertEqual(phase_centre, (hdr['RAPHASE'], hdr['DECPHASE']))

    def test_pointing(self):
        m = rm.Metafits(metafits, 'X')
        pointing = m.pointing()
        self.assertEqual(pointing, (hdr['RA'], hdr['DEC']))

    def test_delays(self):
        m = rm.Metafits(metafits, 'X')
        delays = m.delays()
        self.assertEqual(delays, hdr['DELAYS'])

    def test_annumbers(self):
        m = rm.Metafits(metafits, 'X')
        annumbers = m.annumbers()
        self.assertEqual(annumbers[0:3],  [hdu[0][1], hdu[2][1], hdu[4][1]])

    def test_annames(self):
        m = rm.Metafits(metafits, 'X')
        annames = m.annames()
        self.assertEqual(annames[0:3], [hdu[0][3], hdu[2][3], hdu[4][3]])

    def test_ind_for_annumber(self):
        m = rm.Metafits(metafits, 'X')
        ant_ind = m.ind_for_annumber(hdu[0][1])
        self.assertEqual(ant_ind, 0)
        ant_ind = m.ind_for_annumber(hdu[2][1])
        self.assertEqual(ant_ind, 1)

    def test_ind_for_anname(self):
        m = rm.Metafits(metafits, 'X')
        ant_name = m.ind_for_anname(hdu[0][3])
        self.assertEqual(ant_name, 0)
        ant_name = m.ind_for_anname(hdu[2][3])
        self.assertEqual(ant_name, 1)

    def test_anpos(self):
        m = rm.Metafits(metafits, 'X')
        antpos = m.anpos()
        expected = np.array([[-101.52999878, -585.67498779,  375.21200562],
                             [415.00799561, -575.55700684,  373.37399292],
                             [604.56799316, -489.94299316,  372.90701294]])
        np.testing.assert_almost_equal(antpos[0:3, :], expected)

    def test_anpos_for(self):
        m = rm.Metafits(metafits, 'X')
        antpos = m.anpos_for(74)
        self.assertTrue(len(antpos) == 3)
        np.testing.assert_almost_equal(np.array(antpos), np.array(
            [415.00799561, -575.55700684,  373.37399292]))

    def test_baseline_length_for(self):
        m = rm.Metafits(metafits, 'X')
        baseline_length = m.baseline_length_for((75, 74))
        self.assertEqual(baseline_length, 516.6370807265803)

    # def test_baseline_lengths(self):
     #   m = rm.Metafits(metafits, 'X')
      #  baseline_lengths = m.baseline_lengths()
       # bls = nants * (nants - 1) / 2
        #self.assertEqual(len(list(baseline_lengths.keys())), int(bls))
        #self.assertEqual(list(baseline_lengths.keys()), antpairs)
        #self.assertEqual(baseline_lengths[(75, 74)], 516.6370807265803)

    def test_baselines_greater_than(self):
        m = rm.Metafits(metafits, 'X')
        bls = m.baselines_greater_than(600)
        self.assertEqual(list(bls.keys())[0:1], [(75, 73)])
        self.assertEqual(list(bls.values())[0:1], [712.5580601060333])

    def test_baselines_less_than(self):
        m = rm.Metafits(metafits, 'X')
        bls = m.baselines_less_than(600)
        self.assertEqual(list(bls.keys())[0:1], [(75, 74)])
        self.assertEqual(list(bls.values())[0:1], [516.6370807265803])

    def test_cable_flavors(self):
        m = rm.Metafits(metafits, 'X')
        ctypes, clengths = m._cable_flavors()
        self.assertTrue(len(ctypes) == nants)
        self.assertTrue(len(clengths) == nants)
        self.assertEqual(ctypes[0:3], ['LMR400', 'RG6', 'LMR400'])
        self.assertEqual(clengths[0:3], [524.0, 150.0, 400.0])

    def test_cable_length_for(self):
        m = rm.Metafits(metafits, 'X')
        clength = m.cable_length_for(74)
        self.assertEqual(clength, 150.0)

    def test_cable_type_for(self):
        m = rm.Metafits(metafits, 'X')
        ctype = m.cable_type_for(74)
        self.assertEqual(ctype, 'RG6')

    def test_receivers(self):
        m = rm.Metafits(metafits, 'X')
        receivers = m.receivers()
        self.assertEqual(receivers[0:3], [10, 10, 10])

    def test_receiver_for(self):
        m = rm.Metafits(metafits, 'X')
        receiver = m.receiver_for(74)
        self.assertEqual(receiver, 10)

    def test_annumbers_for_receiver(self):
        m = rm.Metafits(metafits, 'X')
        antnums = m.annumbers_for_receiver(10)
        self.assertTrue((antnums[0:3] == [75, 74, 73]).all())

    def test_btemps(self):
        m = rm.Metafits(metafits, 'X')
        btemps = m.btemps()
        self.assertEqual(len(btemps), nants)
        np.testing.assert_almost_equal(btemps, expected_btemps)


if __name__ == '__main__':
    unittest.main()
