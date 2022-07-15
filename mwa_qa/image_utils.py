from astropy.io import fits
import numpy as np


def data(image):
    data = fits.open(image)[0].data
    return data.squeeze()


def header(image):
    return fits.open(image)[0].header


def image_size(image):
    hdr = header(image)
    return (hdr['NAXIS1'], hdr['NAXIS2'])


def pol_convention(image):
    hdr = header(image)
    return int(hdr['CRVAL4'])


def mean(image):
    return np.nanmean(data(image))


def rms(image):
    nxpix, nypix = image_size(image)
    return np.sqrt(np.nansum(data(image) ** 2) / (nxpix * nypix))


def std(image):
    return np.nanstd(data(image))


def select_box(image, xpix, ypix):
    _d = data(image)
    return _d[0:xpix, 0:ypix]


def rms_for(image, xpix, ypix):
    pix_box = select_box(image, xpix, ypix)
    return np.sqrt(np.nanmean(pix_box ** 2))


def std_for(image, xpix, ypix):
    pix_box = select_box(image, xpix, ypix)
    return np.nanstd(pix_box)
