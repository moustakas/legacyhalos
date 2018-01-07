"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.

"""
from __future__ import absolute_import, division, print_function

import os
import pickle
import time
import numpy as np
import numpy.ma as ma

from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture

PIXSCALE = 0.262

def _read_multiband(objid, objdir, band=('g', 'r', 'z')):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.

    """
    import fitsio
    from scipy.ndimage.morphology import binary_dilation

    resid = dict()
    for filt in band:

        data = fitsio.read(os.path.join(objdir, '{}-image-{}.fits.fz'.format(objid, filt)))
        model = fitsio.read(os.path.join(objdir, '{}-model-{}.fits.fz'.format(objid, filt)))
        invvar = fitsio.read(os.path.join(objdir, '{}-invvar-{}.fits.fz'.format(objid, filt)))

        # Mask pixels with ivar<=0. Also build an object mask from the model
        # image, to handle systematic residuals.
        sig1 = 1.0 / np.sqrt(np.median(invvar[invvar > 0]))

        mask = (invvar <= 0)*1 # 1=bad, 0=good
        mask = np.logical_or( mask, ( model > (3 * sig1) )*1 )
        mask = binary_dilation(mask, iterations=5) * 1

        _resid = ma.masked_where(mask > 0, data - model)

        ma.set_fill_value(_resid, fill_value=0)
        resid[filt] = _resid

    shape = data.shape
        
    return resid, shape

def _display_multiband(data, band=('g', 'r', 'z'), ellaper=None, isophotfit=None):
    """Display the output of _read_multiband."""

    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval as Interval
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize

    nband = len(data.keys())
    
    fig, ax = plt.subplots(1, 3, figsize=(3 * nband, 3))
    for filt, ax1 in zip(band, ax):
        img = ma.getdata(data[filt]) * (ma.getmask(data[filt]) == 0)
        norm = ImageNormalize(img, interval=Interval(contrast=0.9),
                              stretch=Stretch(a=0.99))
            
        im = ax1.imshow(img, origin='lower', norm=norm, cmap='Blues_r')
        #fig.colorbar(im)
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, fontweight='bold',
                 ha='center', va='center', color='k', fontsize=14)
        
        if ellaper:
            ellaper.plot(color='k', ax=ax1)
            
        if isophotfit:
            nfit = len(isophotfit)
            nplot = np.rint(0.2*nfit).astype('int')
            smas = np.linspace(0, isophotfit.sma.max(), nplot)
            for sma in smas:
                iso = isophotfit.get_closest(sma)
                x, y, = iso.sampled_coordinates()
                ax1.plot(x, y, color='k')
                
        ax1.axis('off')
        ax1.set_adjustable('box-forced')

    fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.01, right=0.99)
    #plt.savefig('junk.png')
    plt.show()

def _initial_ellipse(cat, pixscale=PIXSCALE, shape=(100, 100), verbose=False):
    """Initialize an Ellipse object by converting the Tractor ellipticity
    measurements to eccentricity and position angle.  See
    http://legacysurvey.org/dr5/catalogs and
    http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq for
    more details.

    """
    nx, ny = shape

    galtype = cat.type.strip().upper()
    if galtype == 'DEV':
        sma = cat.shapedev_r / pixscale # [pixels]
        epsilon = np.sqrt(cat.shapedev_e1**2 + cat.shapedev_e2**2)
        pa = 0.5 * np.arctan(cat.shapedev_e2 / cat.shapedev_e1)
    else:
        sma = cat.shapeexp_r / pixscale # [pixels]
        epsilon = np.sqrt(cat.shapeexp_e1**2 + cat.shapeexp_e2**2)
        pa = 0.5 * np.arctan(cat.shapeexp_e2 / cat.shapeexp_e1)

    ba = (1 - np.abs(epsilon)) / (1 + np.abs(epsilon))
    eps = 1 - ba

    if verbose:
        print('Type={}, sma={:.2f}, eps={:.2f}, pa={:.2f} (initial)'.format(
            galtype, sma, eps, np.degrees(pa)))

    geometry = EllipseGeometry(x0=nx/2, y0=ny/2, sma=sma, eps=eps, pa=pa)
    ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                 geometry.sma*(1 - geometry.eps), geometry.pa)
    return geometry, ellaper

def legacyhalos_ellipse(galaxycat, objid=None, objdir=None, ncpu=1,
                        pixscale=0.262, refband='r', band=('g', 'r', 'z'),
                        verbose=False, debug=True):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    """ 
    if objid is None and objdir is None:
        objid, objdir = get_objid(galaxycat)

    # Step 1 - Read the data and then initialize an Ellipse object based on the
    # Tractor fitting results.  Note that the position angle guess is not right.
    data, shape = _read_multiband(objid, objdir, band=band)

    geometry, ellaper = _initial_ellipse(galaxycat, pixscale=pixscale,
                                         shape=shape, verbose=verbose)
    if debug:
        _display_multiband(data, ellaper=ellaper, band=band)

    # Step 2 - Fit in the reference band.
    print('Ellipse-fitting the {}-band image.'.format(refband))
    t0 = time.time()
    ellipse = Ellipse(data[refband], geometry)
    isophotfit = ellipse.fit_image(minsma=0.5, maxsma=shape[0]/2,
                                   integrmode='median')
    print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    isofitfile = os.path.join(objdir, '{}-isophotfit-{}.p'.format(objid, band))
    print('Writing {}'.format(isofitfile))
    with open(isofitfile, 'wb') as out:
        pickle.dump(isophotfit, out)

    if debug:
        _display_multiband(data, isophotfit=isophotfit, band=band)

    import pdb ; pdb.set_trace()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.errorbar(isophotfit.sma, 22.5-2.5*np.log10(isophotfit.intens), isophotfit.int_err/isophotfit.intens/np.log(10), fmt='o')
    plt.xlabel('sma**1/4')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()
    plt.show()

    
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    plt.subplot(2, 2, 1)
    plt.errorbar(isophotfit.sma, isophotfit.eps, yerr=isophotfit.ellip_err,
                 fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')

    plt.subplot(2, 2, 2)
    plt.errorbar(isophotfit.sma, np.degrees(isophotfit.pa),
                 yerr=np.degrees(isophotfit.pa_err), fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')

    plt.subplot(2, 2, 3)
    plt.errorbar(isophotfit.sma, isophotfit.x0, yerr=isophotfit.x0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')

    plt.subplot(2, 2, 4)
    plt.errorbar(isophotfit.sma, isophotfit.y0, yerr=isophotfit.y0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')

    plt.show()

    import pdb ; pdb.set_trace()

    resid = resid[(nx//2-50):(nx//2+50), (ny//2-50):(ny//2+50)]

    return 1 # success!
