import numpy as np


def deg2hms(ra):
    """
    Converts degrees to hours minutes seconds
    ra : right ascension or any value to convert to hours minutes seconds
    """
    ra_hrs = ra / 15.
    assert ra_hrs >= 0.0, "right ascension or value should be positive"
    ra_h = int(ra_hrs)
    ra_mins = (ra_hrs - ra_h) * 60
    ra_m = int(ra_mins)
    ra_secs = (ra_mins - ra_m) * 60
    ra_s = round(ra_secs, 2)
    return '{}h{}m{:.1f}s'.format(ra_h, ra_m, ra_s)


def deg2dms(dec):
    """
    Converts to degrees to degrees minutes seconds
    dec : declination or any other value to convert to degrees minutes seconds
    """
    dec_a = np.abs(dec)
    dec_deg = int(dec_a)
    dec_mins = (dec_a - dec_deg) * 60
    dec_m = int(dec_mins)
    dec_secs = (dec_mins - dec_m) * 60
    dec_s = round(dec_secs, 2)
    sign = np.sign(dec)
    return '{}d{}m{:.1f}s'.format(int(sign * dec_deg), dec_m, dec_s)


def hms2deg(hms_str):
    """
    Converts hours minutes seconds to degrees
    hms_str : string specifying hours minutes seconds in the format hrs:min:sec
    """
    str_splt = np.array(hms_str.split(':'), dtype=float)
    assert str_splt[0] >= 0, "hours needs to be positive quantity"
    assert str_splt[1] >= 0, "minutes needs to be positive quantity"
    assert str_splt[2] >= 0, "seconds needs to be positive quantity"
    hrs = str_splt[0] + str_splt[1]/60. + str_splt[2]/3600.
    deg = hrs * 15
    return round(deg, 2)


def dms2deg(dms_str):
    """
    Converts degrees minutes seconds to degrees
    dms_str: dtring specifying degrees minutes and seconds in the
    format deg:min:sec
    """
    str_splt = np.array(dms_str.split(':'), dtype=float)
    deg = np.abs(str_splt[0]) + str_splt[1]/60. + str_splt[2]/3600.
    if str_splt[0] < 0:
        multiply = -1
    else:
        multiply = 1
    return round(multiply * deg, 2)
