"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.

"""
from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io

PIXSCALE = 0.262

def ellipsefit_multiband(objid, objdir, data, mgefit, band=('g', 'r', 'z'), refband='r',
                         nowrite=False, verbose=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    """
    from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                    Isophote, IsophoteList)
    from photutils.isophote.sample import CentralEllipseSample
    from photutils.isophote.fitter import CentralEllipseFitter

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    if verbose:
        print('Initializing an Ellipse object in the reference {}-band image.'.format(refband))
    geometry = EllipseGeometry(x0=mgefit['xpeak'], y0=mgefit['ypeak'], eps=mgefit['eps'],
                               #sma=0.5*mgefit['majoraxis'], 
                               sma=5,
                               pa=np.radians(mgefit['pa']-90))
    
    #print('QA for debugging.')
    #display_multiband(data, geometry=geometry, band=band, mgefit=mgefit)

    ellipsefit = dict()
    ellipsefit['geometry'] = geometry

    # Fit in the reference band...
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))
        
    img = data['{}_masked'.format(refband)]
    ellipse = Ellipse(img, geometry=geometry)

    ellipsefit['{}_sky'.format(refband)] = 22.5 - 2.5 * np.log10( 0.1 * ma.std(img) ) # 10% sky level

    # First fit with the default parameters.
    # https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
    integrmode, sclip, nclip, step, fflag = 'median', 3, 3, 0.1, 0.3
    
    t0 = time.time()
    isophot = ellipse.fit_image(minsma=0.0, maxsma=2*mgefit['majoraxis'],
                                integrmode=integrmode, sclip=sclip, nclip=nclip,
                                step=step, fflag=fflag)
    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    ellipsefit[refband] = isophot

    tall = time.time()
    for filt in band:
        t0 = time.time()
        if verbose:
            print('Ellipse-fitting {}-band image.'.format(filt))

        if filt == refband: # we did it already!
            continue

        img = data['{}_masked'.format(filt)]
        ellipsefit['{}_sky'.format(filt)] = 22.5 - 2.5 * np.log10( 0.1 * ma.std(img) ) # 10% sky level

        # Loop on the reference band isophotes but skip the first isophote,
        # which is a CentralEllipseSample object (see below).
        isobandfit = []
        for iso in isophot[1:]:
            g = iso.sample.geometry # fixed geometry

            # Use the same integration mode and clipping parameters.
            sample = EllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()

            # Create an Isophote instance with the sample.
            isobandfit.append(Isophote(sample, 0, True, 0))

        # Now deal with the central pixel; see
        # https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
        g = EllipseGeometry(geometry.x0, geometry.y0, 0.0, 0.0, 0.0)
        #g.find_center(img)

        # Use the same integration mode and clipping parameters.
        sample = CentralEllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                                      sclip=sclip, nclip=nclip)
        cen = CentralEllipseFitter(sample).fit()
        isobandfit.append(cen)
        isobandfit.sort()

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)
        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    if verbose:
        print('Time for all images = {:.3f} sec'.format( (time.time() - tall) / 1))

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(objid, objdir, ellipsefit, verbose=verbose)

    return ellipsefit

def mgefit_multiband(objid, objdir, data, band=('g', 'r', 'z'), refband='r',
                     pixscale=0.262, debug=False, nowrite=False, nofit=False,
                     verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    #from mge.mge_print_contours import mge_print_contours as print_contours

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    galaxy = find_galaxy(data[refband], nblob=1, binning=3,
                         plot=debug, quiet=not verbose)
    if debug:
        plt.show()
    
    #galaxy.xmed -= 1
    #galaxy.ymed -= 1
    #galaxy.xpeak -= 1
    #galaxy.ypeak -= 1
    
    mgefit = dict()
    for key in ('eps', 'majoraxis', 'pa', 'theta',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        mgefit[key] = getattr(galaxy, key)

    if not nofit:
        t0 = time.time()
        for filt in band:
            if verbose:
                print('Running MGE on the {}-band image.'.format(filt))

            mgephot = sectors_photometry(data[filt], galaxy.eps, galaxy.theta, galaxy.xmed,
                                         galaxy.ymed, n_sectors=11, minlevel=0, plot=debug,
                                         mask=data['{}_mask'.format(filt)])
            if debug:
                plt.show()

            mgefit[filt] = fit_sectors(mgephot.radius, mgephot.angle, mgephot.counts,
                                       galaxy.eps, ngauss=None, negative=False,
                                       sigmaPSF=0, normPSF=1, scale=pixscale,
                                       quiet=not debug, outer_slope=4, bulge_disk=False,
                                       plot=debug)
            if debug:
                plt.show()

            #_ = print_contours(data[refband], galaxy.pa, galaxy.xpeak, galaxy.ypeak, pp.sol, 
            #                   binning=2, normpsf=1, magrange=6, mask=None, 
            #                   scale=pixscale, sigmapsf=0)

        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))
        
    if not nowrite:
        legacyhalos.io.write_mgefit(objid, objdir, mgefit, band=refband, verbose=verbose)

    return mgefit
    
def legacyhalos_ellipse(galaxycat, objid=None, objdir=None, ncpu=1,
                        pixscale=0.262, refband='r', band=('g', 'r', 'z'),
                        verbose=False, debug=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    photutils - do ellipse-fitting using photutils, otherwise use MGE.

    """ 
    if objid is None and objdir is None:
        objid, objdir = get_objid(galaxycat)

    # Read the data.  
    data = legacyhalos.io.read_multiband(objid, objdir, band=band)

    # Find the galaxy and perform MGE fitting
    mgefit = mgefit_multiband(objid, objdir, data, band=band, refband=refband,
                              pixscale=pixscale, verbose=verbose, debug=debug)

    # Do ellipse-fitting
    ellipsefit = ellipsefit_multiband(objid, objdir, data, mgefit, band=band,
                                      refband=refband, verbose=verbose)

    return 1 # success!

