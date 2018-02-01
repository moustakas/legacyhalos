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

import seaborn as sns
sns.set(context='talk', style='ticks')#, palette='Set1')
    
PIXSCALE = 0.262

def _initial_ellipse(cat, pixscale=PIXSCALE, data=None, refband='r',
                     verbose=False, use_tractor=False):
    """Initialize an Ellipse object by converting the Tractor ellipticity
    measurements to eccentricity and position angle.  See
    http://legacysurvey.org/dr5/catalogs and
    http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq for
    more details.

    """
    nx, ny = data[refband].shape

    galtype = cat.type.strip().upper()
    if galtype == 'DEV':
        sma = cat.shapedev_r / pixscale # [pixels]
    else:
        sma = cat.shapeexp_r / pixscale # [pixels]

    if sma == 0:
        sma = 10
        
    if use_tractor:
        if galtype == 'DEV':
            epsilon = np.sqrt(cat.shapedev_e1**2 + cat.shapedev_e2**2)
            pa = 0.5 * np.arctan(cat.shapedev_e2 / cat.shapedev_e1)
        else:
            epsilon = np.sqrt(cat.shapeexp_e1**2 + cat.shapeexp_e2**2)
            pa = 0.5 * np.arctan(cat.shapeexp_e2 / cat.shapeexp_e1)
            
        ba = (1 - np.abs(epsilon)) / (1 + np.abs(epsilon))
        eps = 1 - ba
    else:
        from mge.find_galaxy import find_galaxy
        ff = find_galaxy(data[refband], plot=False, quiet=not verbose)
        eps, pa = ff.eps, np.radians(ff.theta)

    if verbose:
        print('Type={}, sma={:.2f}, eps={:.2f}, pa={:.2f} (initial)'.format(
            galtype, sma, eps, np.degrees(pa)))

    geometry = EllipseGeometry(x0=nx/2, y0=ny/2, sma=sma, eps=eps, pa=pa)
    ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                 geometry.sma*(1 - geometry.eps), geometry.pa)
    return geometry, ellaper

def display_isophotfit(isophotfit, band=('g', 'r', 'z'), redshift=None,
                       pixscale=0.262, indx=None, png=None):
    """Display the isophote fitting results."""

    from matplotlib.ticker import FormatStrFormatter

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    if indx is None:
        indx = np.ones(len(isophotfit[band[0]]))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for filt in band:
        ax1.errorbar(isophotfit[filt].sma[indx] * smascale, isophotfit[filt].eps[indx],
                     isophotfit[filt].ellip_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        #ax1.set_ylim(0, 0.5)
        
        ax2.errorbar(isophotfit[filt].sma[indx] * smascale, np.degrees(isophotfit[filt].pa[indx]),
                     np.degrees(isophotfit[filt].pa_err[indx]), fmt='o',
                     markersize=4)#, color=color[filt])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        #ax2.set_ylim(0, 180)

        ax3.errorbar(isophotfit[filt].sma[indx] * smascale, isophotfit[filt].x0[indx],
                     isophotfit[filt].x0_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        ax4.errorbar(isophotfit[filt].sma[indx] * smascale, isophotfit[filt].y0[indx],
                     isophotfit[filt].y0_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right')
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    ax1.set_ylabel('Ellipticity')
    #ax1.set_xscale('log')
    
    ax2.set_ylabel('Position Angle (deg)')
    #ax2.set_xscale('log')

    ax3.set_xlabel('Semimajor Axis ({})'.format(smaunit))
    ax3.set_ylabel(r'$x_{0}$')
    #ax3.set_xscale('log')

    ax4.set_xlabel('Semimajor Axis ({})'.format(smaunit))
    ax4.set_ylabel(r'$y_{0}$')
    #ax4.set_xscale('log')

    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.15, right=0.85, left=0.15)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_ellipse_sbprofile(isophotfit, band=('g', 'r', 'z'), redshift=None,
                              indx=None, pixscale=0.262, minerr=0.02, png=None):
    """Display the multi-band surface brightness profile."""

    colors = iter(sns.color_palette())

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    if indx is None:
        indx = np.ones(len(isophotfit[band[0]]))
        
    def _sbprofile(isophotfit, indx, smascale):
        """Convert fluxes to magnitudes and colors."""
        sbprofile = dict()
        sbprofile['sma'] = isophotfit['r'].sma[indx] * smascale
        
        with np.errstate(invalid='ignore'):
            for filt in band:
                sbprofile[filt] = 22.5 - 2.5 * np.log10(isophotfit[filt].intens[indx])
                sbprofile['{}_err'.format(filt)] = isophotfit[filt].int_err[indx] / \
                  isophotfit[filt].intens[indx] / np.log(10)

                # Just for the plot use a minimum uncertainty
                sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr
                
        sbprofile['gr'] = sbprofile['g'] - sbprofile['r']
        sbprofile['rz'] = sbprofile['r'] - sbprofile['z']
        sbprofile['gr_err'] = np.sqrt(sbprofile['g_err']**2 + sbprofile['r_err']**2)
        sbprofile['rz_err'] = np.sqrt(sbprofile['r_err']**2 + sbprofile['z_err']**2)

        # Just for the plot use a minimum uncertainty
        sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
        sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

        return sbprofile

    sbprofile = _sbprofile(isophotfit, indx, smascale)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for filt in band:
        ax1.fill_between(sbprofile['sma'], sbprofile[filt]-sbprofile['{}_err'.format(filt)],
                         sbprofile[filt]+sbprofile['{}_err'.format(filt)],
                         label=r'${}$'.format(filt), color=next(colors), alpha=0.75)
    ax1.set_ylabel('AB Magnitude')
    ax1.set_ylim(32.99, 20)
    #ax1.invert_yaxis()
    ax1.legend(loc='upper right')

    ax2.fill_between(sbprofile['sma'], sbprofile['rz']-sbprofile['rz_err'],
                     sbprofile['rz']+sbprofile['rz_err'],
                     label=r'$r - z$', color=next(colors), alpha=0.75)
    
    ax2.fill_between(sbprofile['sma'], sbprofile['gr']-sbprofile['gr_err'],
                     sbprofile['gr']+sbprofile['gr_err'],
                     label=r'$g - r$', color=next(colors), alpha=0.75)

    ax2.set_xlabel('Semimajor Axis ({})'.format(smaunit), alpha=0.75)
    ax2.set_ylabel('Color')
    ax2.set_ylim(-0.5, 2.5)
    ax2.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.0)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_mge_sbprofile(mgefit, band=('g', 'r', 'z'), refband='r', redshift=None,
                          indx=None, pixscale=0.262, png=None):
    """Display the multi-band surface brightness profile."""

    colors = iter(sns.color_palette())

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    #if indx is None:
    #    indx = np.ones_like(mgefit[refband].radius, dtype='bool')
        
    def _sbprofile(mgefit, indx, smascale):
        """Convert fluxes to magnitudes and colors."""
        sbprofile = dict()
        
        with np.errstate(invalid='ignore'):
            for filt in band:
                #indx = np.ones_like(mgefit[filt].radius, dtype='bool')
                indx = np.argsort(mgefit[filt].radius)
                
                sbprofile['{}_sma'.format(filt)] = mgefit[filt].radius[indx] * smascale
                sbprofile[filt] = 22.5 - 2.5 * np.log10(mgefit[filt].counts[indx])
                sbprofile['{}_err'.format(filt)] = 2.5 * mgefit[filt].err[indx] / mgefit[filt].counts[indx] / np.log(10)

                sbprofile['{}_model'.format(filt)] = 22.5 - 2.5 * np.log10(mgefit[filt].yfit[indx])

                #sbprofile['{}_err'.format(filt)] = mgefit[filt].int_err[indx] / \
                #  mgefit[filt].intens[indx] / np.log(10)

                # Just for the plot use a minimum uncertainty
                #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

        #sbprofile['gr'] = sbprofile['g'] - sbprofile['r']
        #sbprofile['rz'] = sbprofile['r'] - sbprofile['z']
        #sbprofile['gr_err'] = np.sqrt(sbprofile['g_err']**2 + sbprofile['r_err']**2)
        #sbprofile['rz_err'] = np.sqrt(sbprofile['r_err']**2 + sbprofile['z_err']**2)
        #
        ## Just for the plot use a minimum uncertainty
        #sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
        #sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

        return sbprofile

    sbprofile = _sbprofile(mgefit, indx, smascale)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for filt in band:
        ax1.scatter(sbprofile['{}_sma'.format(filt)], sbprofile[filt], marker='s',
                     label=r'${}$'.format(filt), color=next(colors), alpha=0.75, s=10)
        #ax1.errorbar(sbprofile['{}_sma'.format(filt)], sbprofile[filt], yerr=sbprofile['{}_err'.format(filt)],
        #             label=r'${}$'.format(filt), color=next(colors), alpha=0.75, fmt='s',
        #             markersize=2)
        #ax1.fill_between(sbprofile['{}_sma'.format(filt)], sbprofile[filt]-sbprofile['{}_err'.format(filt)],
        #                 sbprofile[filt]+sbprofile['{}_err'.format(filt)],
        #                 label=r'${}$'.format(filt), color=next(colors), alpha=0.75)
        #ax1.plot(sbprofile['{}_sma'.format(filt)], sbprofile['{}_model'.format(filt)], lw=2, color='k')
    ax1.set_ylabel('AB Magnitude')
    ax1.set_ylim(32.99, 20)
    #ax1.invert_yaxis()
    ax1.legend(loc='upper right')

    #ax2.fill_between(sbprofile['sma'], sbprofile['rz']-sbprofile['rz_err'],
    #                 sbprofile['rz']+sbprofile['rz_err'],
    #                 label=r'$r - z$', color=next(colors), alpha=0.75)
    #
    #ax2.fill_between(sbprofile['sma'], sbprofile['gr']-sbprofile['gr_err'],
    #                 sbprofile['gr']+sbprofile['gr_err'],
    #                 label=r'$g - r$', color=next(colors), alpha=0.75)

    ax2.set_xlabel('Semimajor Axis ({})'.format(smaunit), alpha=0.75)
    ax2.set_ylabel('Color')
    ax2.set_ylim(-0.5, 2.5)
    #ax2.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.0)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_multiband(data, band=('g', 'r', 'z'), refband='r', geometry=None,
                      mgefit=None, ellipsefit=None, indx=None, magrange=10,
                      contours=False, png=None):
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    """
    from astropy.visualization import ZScaleInterval as Interval
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize
    
    nband = len(band)

    fig, ax = plt.subplots(1, 3, figsize=(2*nband, 3))
    for filt, ax1 in zip(band, ax):

        img = data['{}_masked'.format(filt)]
        #img = data[filt]

        norm = ImageNormalize(img, interval=Interval(contrast=0.95),
                              stretch=Stretch(a=0.95))

        im = ax1.imshow(img, origin='lower', norm=norm, cmap='Blues')
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, #fontweight='bold',
                 ha='center', va='center', color='k', fontsize=14)

        if mgefit:
            from mge.mge_print_contours import _multi_gauss, _gauss2d_mge

            sigmapsf = np.atleast_1d(0)
            normpsf = np.atleast_1d(1)
            _magrange = 10**(-0.4*np.arange(0, magrange, 1)[::-1]) # 0.5 mag/arcsec^2 steps
            #_magrange = 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

            model = _multi_gauss(mgefit[filt].sol, img, sigmapsf, normpsf,
                                 mgefit[filt].xmed, mgefit[filt].ymed,
                                 mgefit[filt].pa)
            
            peak = data[filt][mgefit[filt].xpeak, mgefit[filt].ypeak]
            levels = peak * _magrange
            s = img.shape
            extent = [0, s[1], 0, s[0]]

            ax1.contour(model, levels, colors='k', linestyles='solid', extent=extent,
                        alpha=0.5, lw=1)

        if geometry:
            from photutils import EllipticalAperture
            
            ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                         geometry.sma*(1 - geometry.eps), geometry.pa)
            ellaper.plot(color='k', lw=1, ax=ax1)

        if ellipsefit:

            if len(ellipsefit[filt]) > 0:

                if indx is not None:
                    indx = np.ones(len(ellipsefit[filt]), dtype=bool)

                nfit = len(indx) # len(ellipsefit[filt])
                nplot = np.rint(0.1*nfit).astype('int')
                
                smas = np.linspace(0, ellipsefit[filt].sma[indx].max(), nplot)
                for sma in smas:
                    efit = ellipsefit[filt].get_closest(sma)
                    x, y, = efit.sampled_coordinates()
                    ax1.plot(x, y, color='k', alpha=0.9)
            else:
                from photutils import EllipticalAperture
                geometry = ellipsefit['{}_geometry'.format(refband)]
                ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                             geometry.sma*(1 - geometry.eps), geometry.pa)
                ellaper.plot(color='k', lw=1, ax=ax1)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    if png:
        print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight', pad_inches=0)
        #import pdb ; pdb.set_trace()
        #plt.close(fig)
    else:
        plt.show()

def ellipsefit_multiband(objid, objdir, data, mgefit, band=('g', 'r', 'z'), refband='r',
                         integrmode='bilinear', sclip=5, nclip=3, step=0.1, verbose=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    """
    from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                    Isophote, IsophoteList)

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    if verbose:
        print('Initializing an Ellipse object in the reference {}-band image.'.format(refband))
    geometry = EllipseGeometry(x0=mgefit[refband].xmed, y0=mgefit[refband].ymed,
                               sma=0.5*mgefit[refband].majoraxis, eps=mgefit[refband].eps,
                               pa=np.radians(-mgefit[refband].theta))
    
    #print('QA for debugging.')
    #display_multiband(data, geometry=geometry, band=band, mgefit=mgefit)

    ellipsefit = dict()
    ellipsefit['{}_geometry'.format(refband)] = geometry

    # Fit in the reference band...
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))
    t0 = time.time()

    img = data['{}_masked'.format(refband)]
    ellipse = Ellipse(img, geometry)
    ellipsefit[refband] = ellipse.fit_image(minsma=0.0, maxsma=None, integrmode=integrmode,
                                            sclip=sclip, nclip=nclip, step=step, linear=True)
    import pdb ; pdb.set_trace()

    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    tall = time.time()
    for filt in band:
        t0 = time.time()
        if verbose:
            print('Ellipse-fitting {}-band image.'.format(filt))

        if filt == refband: # we did it already!
            continue

        # Loop on the reference band isophotes.
        isobandfit = []
        for iso in ellipsefit[refband]:

            g = iso.sample.geometry # fixed geometry

            # Use the same integration mode and clipping parameters.
            img = data['{}_masked'.format(filt)]
            sample = EllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()

            # Create an Isophote instance with the sample.
            isobandfit.append(Isophote(sample, 0, True, 0))

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)
        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    if verbose:
        print('Time for all images = {:.3f} sec'.format( (time.time() - tall) / 1))

    # Write out
    legacyhalos.io.write_ellipsefit(objid, objdir, ellipsefit, verbose=verbose)

    return ellipsefit

def mgefit_multiband(objid, objdir, data, band=('g', 'r', 'z'), refband='r',
                     pixscale=0.262, debug=False, verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    from mge.mge_print_contours import mge_print_contours as print_contours

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    galaxy = find_galaxy(data[refband], nblob=1, plot=debug, quiet=not verbose)
    if debug:
        plt.show()

    t0 = time.time()
    
    mgefit = dict()
    for filt in band:

        if verbose:
            print('Running MGE on the {}-band image.'.format(filt))

        mgephot = sectors_photometry(data[filt], galaxy.eps, galaxy.theta, galaxy.xpeak,
                                     galaxy.ypeak, n_sectors=11, minlevel=0, plot=debug,
                                     mask=data['{}_mask'.format(filt)])
        if debug:
            plt.show()

        _mgefit = fit_sectors(mgephot.radius, mgephot.angle, mgephot.counts, galaxy.eps,
                              ngauss=None, negative=False, sigmaPSF=0, normPSF=1,
                              scale=pixscale, quiet=not debug, outer_slope=4,
                              bulge_disk=False, plot=debug)
        _mgefit.eps = galaxy.eps
        _mgefit.majoraxis = galaxy.majoraxis # major axis length in pixels
        _mgefit.pa = galaxy.pa
        _mgefit.theta = galaxy.theta
        _mgefit.xmed = galaxy.xmed
        _mgefit.xpeak = galaxy.xpeak
        _mgefit.ymed = galaxy.ymed
        _mgefit.ypeak = galaxy.ypeak
        
        mgefit[filt] = _mgefit

        if debug:
            plt.show()

        #plt.clf()
        #plt.scatter(phot.radius, 22.5-2.5*np.log10(phot.counts), s=20)
        ##plt.scatter(phot2.radius, 22.5-2.5*np.log10(phot2.counts), s=20)
        #plt.ylim(34, 20)
        #plt.show()        

        #_ = print_contours(data[refband], galaxy.pa, galaxy.xpeak, galaxy.ypeak, pp.sol, 
        #                   binning=2, normpsf=1, magrange=6, mask=None, 
        #                   scale=pixscale, sigmapsf=0)

    legacyhalos.io.write_mgefit(objid, objdir, mgefit, band=refband, verbose=verbose)

    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

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
