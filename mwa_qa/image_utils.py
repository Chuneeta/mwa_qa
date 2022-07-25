from astropy.io import fits
from astropy import wcs
from astropy.modeling import models, fitting
import numpy as np
import warnings
import copy
import pylab


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


def _get_wcs(image):
    """
    Returns the world coordinate system class
        - image:	Input image fitsfile
        """
    return wcs.WCS(image)


def wcs2pix(image, ra, dec):
    """
    Returns pixel numbers corresponding to ra and dec values
    - image: 	Input image fitsfile
    - ra:		Right ascension in degrees
    - dec:		Declination in degrees
    """
    w = _get_wcs(image)
    px1, px2 = w.all_world2pix(ra, dec, 1, 0, 0)[:2]
    px1 = np.around(px1, 0).astype(int)
    px2 = np.around(px2, 0).astype(int)
    return px1, px2


def pix_flux(image, ra, dec, constant, plot=None):
    """
    Returns the statistics obtained from the desired or selected
        ßregion
    image: Input image fitsfile
    ra : Right ascension in degrees
    dec : Declination in degrees
    const : Number/constant by which the radius of the selected area
        is multiplied. If constant is 1, the selected area will be
                confined to the radius of the PSF, if const < 1, then selected
                area is less than the radius and if const > 1 selected area
                ßhas a radius > than the PSF.
        Default is 1.
    """
    imdata = data(image)
    imhdr = header(image)
    nxaxis, nyaxis = image_size(image)
    bmaj = imhdr['BMAJ']
    bmin = imhdr['BMIN']
    bpa = imhdr['BPA']
    # if the image is not deconvolved, the bmin and bmaj would return 0.
    # the hardcoded values are taken from a deconvolved image
    if bmaj == 0.:
        bmaj = 0.0353
    if bmin == 0.:
        bmin = 0.0318
    # computing synthesized beam radius and area in degrees and pixels
    w = _get_wcs(image)
    dx_px = np.abs(w.wcs.cdelt[0])
    dy_px = np.abs(w.wcs.cdelt[1])
    bmaj_px = bmaj / dx_px
    bmin_px = bmin / dy_px
    bm_radius_px = np.sqrt(bmaj_px ** 2 + bmin_px ** 2)
    bm_area = bmaj * bmin * np.pi / 4 / np.log(2)
    px_area = dx_px * dy_px
    bm_npx = bm_area / px_area
    ra_pix, dec_pix = wcs2pix(image, ra, dec)
    if not np.isnan(ra_pix) and not np.isnan(dec_pix):
        ra_pix = int(ra_pix)
        dec_pix = int(dec_pix)
    if (0 <= ra_pix < nxaxis) and (0 <= dec_pix < nyaxis):
        # selecting region with synthesized beam
        l_axis = np.arange(0, nxaxis)
        m_axis = np.arange(0, nyaxis)
        ll, mm = np.meshgrid(l_axis, m_axis)
        R = np.sqrt((ll - ra_pix)**2 + (mm - dec_pix)**2)
        select = R < constant * bm_radius_px
        imdata_select = imdata[select]
        maxval = np.nanmax(imdata_select)
        minval = np.nanmin(imdata_select)
        # allowing to take care of negative components
        peakval = minval if np.abs(minval) > np.abs(maxval) else maxval
        # fitting gaussian to point sources
        gauss_data = copy.deepcopy(imdata)
        gauss_data[~select] = 0
        gauss_data = gauss_data.reshape((nxaxis, nyaxis))
        inds = np.where(gauss_data == peakval)
        mod = models.Gaussian2D(peakval, inds[1], inds[0],
                                bmaj_px/2, bmin_px/2,
                                theta=bpa * np.pi/180)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            try:
                mod = models.Gaussian2D(peakval, inds[1], inds[0],
                                        bmaj_px/2, bmin_px/2,
                                        theta=bpa * np.pi/180)
                gauss_mod = fit_p(mod, ll, mm, gauss_data)
            except ValueError:
                mod = models.Gaussian2D(peakval, inds[0][1], inds[0][0],
                                        bmaj_px/2, bmin_px/2,
                                        theta=bpa * np.pi/180)
                gauss_mod = fit_p(mod, ll, mm, gauss_data)
        gauss_peak = gauss_mod.amplitude.value
        fitted_data = gauss_mod(ll, mm)
        select_err = R < 2 * bm_radius_px
        residual = imdata - fitted_data
        gauss_int = np.nansum(fitted_data) / bm_npx
        gauss_err = np.nanstd(residual[select_err])
    else:
        warnings.warn('WARNING: Right ascension or declination outside \
            image field, therefore values are set to nan', Warning)
        gauss_peak, gauss_int, peakval = np.nan, np.nan, np.nan
        # plotting selected area
    if plot:
        print(ra_pix, dec_pix)
        pylab.figure()
        pylab.imshow(np.flipud(gauss_data))
        pylab.xlim(ra_pix - 200, ra_pix + 200)
        pylab.ylim(dec_pix - 200, dec_pix + 200)
        pylab.colorbar()
        pylab.show()

    return {'FREQ': imhdr['CRVAL3'],
            'GAUSS_PFLUX': gauss_peak,
            'GAUSS_TFLUX': gauss_int,
            'GAUSS_ERROR': gauss_err,
            'PFLUX': peakval}
