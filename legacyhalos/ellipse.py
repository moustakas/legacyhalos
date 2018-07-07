"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io

def ellipsefit_multiband(objid, objdir, data, sample, mgefit,
                         nowrite=False, verbose=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    """
    from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                    Isophote, IsophoteList)
    from photutils.isophote.sample import CentralEllipseSample
    from photutils.isophote.fitter import CentralEllipseFitter

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']

    ellipsefit = dict()
    ellipsefit['success'] = False
    ellipsefit['redshift'] = sample['z']
    ellipsefit['band'] = band
    ellipsefit['refband'] = refband
    ellipsefit['pixscale'] = pixscale
    for filt in band:
        ellipsefit['psfsigma_{}'.format(filt)] = sample['psfsize_{}'.format(filt)] / 2.355 # [Gaussian sigma, arcsec]

    # Default parameters
    integrmode, sclip, nclip, step, fflag = 'bilinear', 3, 0, 0.1, 0.5
    #integrmode, sclip, nclip, step, fflag = 'median', 3, 0, 0.1, 0.5

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry = EllipseGeometry(x0=mgefit['xpeak'], y0=mgefit['ypeak'],
                               eps=mgefit['eps'],
                               #sma=0.5*mgefit['majoraxis'], 
                               sma=10,
                               pa=np.radians(mgefit['pa']-90))
    ellipsefit['geometry'] = geometry

    def _sky(data, ellipsefit, diameter=2.0):
        """Estimate the sky brightness in each band."""
        #area = diameter**2 # arcsec^2
        for filt in band:
            img = data['{}_masked'.format(filt)]
            #ellipsefit['{}_sky'.format(filt)] = 22.5 - 2.5 * np.log10( ma.std(img) )
            #ellipsefit['mu_{}_sky'.format(filt)] = ellipsefit['{}_sky'.format(filt)] # + 2.5 * np.log10(area)
            ellipsefit['mu_{}_sky'.format(filt)] = 22.5 - 2.5 * np.log10( ma.std(img) )

    _sky(data, ellipsefit)

    # Fit in the reference band...
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))
        
    img = data['{}_masked'.format(refband)]
    ellipse = Ellipse(img, geometry=geometry)

    # First fit with the default parameters.
    try:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for minsma in (10, 1, 5): # try a few different starting minor axes
                print('  Trying minimum sma = {:.1f} pixels.'.format(minsma))
                isophot = ellipse.fit_image(minsma=minsma, maxsma=1.5*mgefit['majoraxis'],
                                            integrmode=integrmode, sclip=sclip, nclip=nclip,
                                            step=step, fflag=fflag)
                if len(isophot) > 0:
                    break
        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

        if len(isophot) == 0:
            print('Ellipse-fitting failed, likely due to complex morphology or poor initial geometry.')
        else:
            ellipsefit['success'] = True
            ellipsefit[refband] = isophot

            tall = time.time()
            for filt in band:
                t0 = time.time()
                if filt == refband: # we did it already!
                    continue

                if verbose:
                    print('Ellipse-fitting {}-band image.'.format(filt))

                img = data['{}_masked'.format(filt)]

                # Loop on the reference band isophotes but skip the first isophote,
                # which is a CentralEllipseSample object (see below).
                isobandfit = []
                for iso in isophot:
                #for iso in isophot[1:]:
                    g = iso.sample.geometry # fixed geometry

                    # Use the same integration mode and clipping parameters.
                    sample = EllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                                           sclip=sclip, nclip=nclip)
                    sample.update()

                    # Create an Isophote instance with the sample.
                    isobandfit.append(Isophote(sample, 0, True, 0))

                # Now deal with the central pixel; see
                # https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
                #import pdb ; pdb.set_trace()
                #g = EllipseGeometry(x0=geometry.x0, y0=geometry.y0, eps=mgefit['eps'], sma=1.0)
                #g.find_center(img)

                ## Use the same integration mode and clipping parameters.
                #sample = CentralEllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                #                              sclip=sclip, nclip=nclip)
                #cen = CentralEllipseFitter(sample).fit()
                #isobandfit.append(cen)
                #isobandfit.sort()

                # Build the IsophoteList instance with the result.
                ellipsefit[filt] = IsophoteList(isobandfit)
                if verbose:
                    print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

            if verbose:
                print('Time for all images = {:.3f} sec'.format( (time.time() - tall) / 1))
                
    except:
        print('Ellipse-fitting failed, likely due to too many masked pixels.')

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(objid, objdir, ellipsefit, verbose=verbose)

    return ellipsefit

def ellipse_sbprofile(ellipsefit, minerr=0.02):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    """
    band, refband = ellipsefit['band'], ellipsefit['refband']
    pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']
    
    indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    sbprofile = dict()
    for filt in band:
        sbprofile['psfsigma_{}'.format(filt)] = ellipsefit['psfsigma_{}'.format(filt)]
    sbprofile['redshift'] = redshift
    
    sbprofile['smaunit'] = 'arcsec'
    sbprofile['sma'] = ellipsefit['r'].sma[indx] * pixscale # [arcsec]

    with np.errstate(invalid='ignore'):
        for filt in band:
            #area = ellipsefit[filt].sarea[indx] * pixscale**2

            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])

            #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
            sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err[indx] / \
              ellipsefit[filt].intens[indx] / np.log(10)
            sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

            #sbprofile['mu_{}'.format(filt)] = sbprofile[filt] + 2.5 * np.log10(area)

            # Just for the plot use a minimum uncertainty
            #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
    sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
    sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)

    # Just for the plot use a minimum uncertainty
    #sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
    #sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

    # # Add the effective wavelength of each bandpass, although this needs to take
    # # into account the DECaLS vs BASS/MzLS filter curves.
    # from speclite import filters
    # filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')
    # for ii, band in enumerate(('g', 'r', 'z', 'W1', 'W2')):
    #     sbprofile.update({'{}_wave_eff'.format(band): filt.effective_wavelengths[ii].value})

    return sbprofile

def mgefit_multiband(objid, objdir, data, debug=False, nowrite=False,
                     nofit=True, verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    #from mge.mge_print_contours import mge_print_contours as print_contours

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    galaxy = find_galaxy(data[refband], nblob=1, binning=3,
                         plot=debug, quiet=not verbose)
    if debug:
        #plt.show()
        pass
    
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
                #plt.show()
                pass

            mgefit[filt] = fit_sectors(mgephot.radius, mgephot.angle, mgephot.counts,
                                       galaxy.eps, ngauss=None, negative=False,
                                       sigmaPSF=0, normPSF=1, scale=pixscale,
                                       quiet=not debug, outer_slope=4, bulge_disk=False,
                                       plot=debug)
            if debug:
                pass
                #plt.show()

            #_ = print_contours(data[refband], galaxy.pa, galaxy.xpeak, galaxy.ypeak, pp.sol, 
            #                   binning=2, normpsf=1, magrange=6, mask=None, 
            #                   scale=pixscale, sigmapsf=0)

        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))
        
    if not nowrite:
        legacyhalos.io.write_mgefit(objid, objdir, mgefit, band=refband, verbose=verbose)

    return mgefit
    
def legacyhalos_ellipse(sample, objid=None, objdir=None, ncpu=1,
                        pixscale=0.262, refband='r', band=('g', 'r', 'z'),
                        verbose=False, debug=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    """ 
    if objid is None and objdir is None:
        objid, objdir = get_objid(sample)

    # Read the data.  
    data = legacyhalos.io.read_multiband(objid, objdir, band=band, refband=refband,
                                         pixscale=pixscale)
    if bool(data):
        # Find the galaxy and perform MGE fitting.
        mgefit = mgefit_multiband(objid, objdir, data, verbose=verbose, debug=debug)

        # Do ellipse-fitting.
        ellipsefit = ellipsefit_multiband(objid, objdir, data, sample, mgefit,
                                          verbose=verbose)
        if ellipsefit['success']:
            return 1
        else:
            return 0
        
    else:
        return 0
