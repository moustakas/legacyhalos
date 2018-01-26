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

import seaborn as sns
sns.set(context='talk', style='ticks')#, palette='Set1')
    
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture

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

def read_multiband(objid, objdir, band=('g', 'r', 'z'), photutils=False):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.

    """
    import fitsio
    from scipy.ndimage.morphology import binary_dilation

    data = dict()
    for filt in band:

        image = fitsio.read(os.path.join(objdir, '{}-image-{}.fits.fz'.format(objid, filt)))
        model = fitsio.read(os.path.join(objdir, '{}-model-{}.fits.fz'.format(objid, filt)))
        invvar = fitsio.read(os.path.join(objdir, '{}-invvar-{}.fits.fz'.format(objid, filt)))

        # Mask pixels with ivar<=0. Also build an object mask from the model
        # image, to handle systematic residuals.
        sig1 = 1.0 / np.sqrt(np.median(invvar[invvar > 0]))

        mask = (invvar <= 0)*1 # 1=bad, 0=good
        mask = np.logical_or( mask, ( model > (3 * sig1) )*1 )
        mask = binary_dilation(mask, iterations=5) * 1

        if photutils:
            resid = ma.masked_where(mask > 0, image - model)
            ma.set_fill_value(resid, fill_value=0)
            data[filt] = resid # 0->bad
        else:
            data[filt] = image - model
            data['{}_mask'] = mask > 0 # 0->bad

    return data

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

    #color = dict(g = 'blue', r = 'green', z = 'red')

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
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_sbprofile(isophotfit, band=('g', 'r', 'z'), redshift=None,
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
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_multiband(data, band=('g', 'r', 'z'), ellaper=None, isophotfit=None,
                      indx=None, png=None):
    """Display the output of read_multiband."""

    from astropy.visualization import ZScaleInterval as Interval
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize

    nband = len(data.keys())

    if isophotfit is not None and indx is None:
        indx = np.ones(len(isophotfit[band[0]]))
        
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
            nfit = len(indx)
            #nfit = len(isophotfit[filt])
            nplot = np.rint(0.1*nfit).astype('int')
            smas = np.linspace(0, isophotfit[filt].sma[indx].max(), nplot)
            for sma in smas:
                iso = isophotfit[filt].get_closest(sma)
                x, y, = iso.sampled_coordinates()
                ax1.plot(x, y, color='k')
                
        ax1.axis('off')
        ax1.set_adjustable('box-forced')

    fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.01, right=0.99)
    if png:
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def ellipsefit_multiband(objid, objdir, data, geometry, band=('g', 'r', 'z'), refband='r',
                         integrmode='bilinear', sclip=3, nclip=0, verbose=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    """
    from photutils.isophote import EllipseSample, Isophote, IsophoteList
    from legacyhalos.io import write_isophotfit
    
    nx, ny = data[refband].shape

    # Fit in the reference band to get the geometry.
    if verbose:
        print('Ellipse-fitting the {}-band image.'.format(refband))
    t0 = time.time()
    ellipse = Ellipse(data[refband], geometry)
    #isophotfit = ellipse.fit_image(minsma=0.5, maxsma=nx/2, integrmode=integrmode,
    #                               sclip=sclip, nclip=nclip)
    isophotfit = ellipse.fit_image(minsma=0.5, maxsma=nx/2, integrmode=integrmode,
                                   sclip=sclip, nclip=0)
    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    write_isophotfit(objid, objdir, isophotfit, band=refband, verbose=verbose)

    # Now fit the other two bands.
    isophotfitall = dict()
    isophotfitall[refband] = isophotfit

    for filt in band:
        if filt == refband: # we did it already!
            continue

        if verbose:
            print('Ellipse-fitting the {}-band image.'.format(filt))
        t0 = time.time()
        
        isobandfit = []

        # Loop on the reference band isophotes.
        for iso in isophotfit:

            g = iso.sample.geometry # fixed geometry

            # Use the same integration mode and clipping parameters.
            sample = EllipseSample(data[filt], g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()

            # Create an Isophote instance with the sample.
            isobandfit.append(Isophote(sample, 0, True, 0))

        # Build the IsophoteList instance with the result.
        isophotfitall[filt] = IsophoteList(isobandfit)
        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))
            
        write_isophotfit(objid, objdir, isophotfitall[filt],
                         band=filt, verbose=verbose)

    return isophotfitall
    
def mgfit_multiband(objid, objdir, data, band=('g', 'r', 'z'), refband='r',
                    pixscale=0.262, debug=False, verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    from mge.mge_print_contours import mge_print_contours as print_contours
    from legacyhalos.io import write_mge
    
    nx, ny = data[refband].shape

    mgefitall = dict()
    for filt in band:
        if filt == refband: # we did it already!
            continue

        if verbose:
            print('Ellipse-fitting the {}-band image.'.format(filt))
        t0 = time.time()

        galaxy = find_galaxy(data[filt], nblob=1, plot=debug, quiet=not verbose)
        if debug:
            plt.show()

        phot = sectors_photometry(data[filt], galaxy.eps, galaxy.theta, galaxy.xpeak,
                                  galaxy.ypeak, plot=debug, mask=data['{}_mask'.format(filt)])
        if debug:
            plt.show()

        #plt.scatter(phot.radius, 22.5-2.5*np.log10(phot.counts), s=20)
        #plt.scatter(phot2.radius, 22.5-2.5*np.log10(phot2.counts), s=20)
        #plt.ylim(34, 20)
        #plt.show()        

        mgefit = fit_sectors(phot.radius, phot.angle, phot.counts, galaxy.eps,
                             ngauss=20, negative=False, sigmaPSF=0, normPSF=1,
                             scale=pixscale, quiet=not debug, outer_slope=2,
                             bulge_disk=False, plot=debug)
        if debug:
            plt.show()

        _ = print_contours(data[refband], galaxy.pa, galaxy.xpeak, galaxy.ypeak, pp.sol, 
                           binning=2, normpsf=1, magrange=6, mask=None, 
                           scale=pixscale, sigmapsf=0)

    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    write_isophotfit(objid, objdir, isophotfit, band=refband, verbose=verbose)

    # Now fit the other two bands.
    isophotfitall = dict()
    isophotfitall[refband] = isophotfit

    for filt in band:
        if filt == refband: # we did it already!
            continue

        if verbose:
            print('Ellipse-fitting the {}-band image.'.format(filt))
        t0 = time.time()
        
        isobandfit = []

        # Loop on the reference band isophotes.
        for iso in isophotfit:

            g = iso.sample.geometry # fixed geometry

            # Use the same integration mode and clipping parameters.
            sample = EllipseSample(data[filt], g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()

            # Create an Isophote instance with the sample.
            isobandfit.append(Isophote(sample, 0, True, 0))

        # Build the IsophoteList instance with the result.
        isophotfitall[filt] = IsophoteList(isobandfit)
        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))
            
        write_isophotfit(objid, objdir, isophotfitall[filt],
                         band=filt, verbose=verbose)

    return isophotfitall
    
def legacyhalos_ellipse(galaxycat, objid=None, objdir=None, ncpu=1,
                        pixscale=0.262, refband='r', band=('g', 'r', 'z'),
                        photutils=False, verbose=False, debug=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    photutils - do ellipse-fitting using photutils, otherwise use MGE.

    """ 
    if objid is None and objdir is None:
        objid, objdir = get_objid(galaxycat)

    # Step 1 - Read the data.  
    data = read_multiband(objid, objdir, band=band, photutils=photutils)

    if photutils:
        # Initialize an Ellipse object based on the Tractor fitting results.
        geometry, ellaper = _initial_ellipse(galaxycat, pixscale=pixscale, verbose=verbose,
                                         data=data, refband=refband)
        if debug:
            display_multiband(data, ellaper=ellaper, band=band)

        # Do ellipse-fitting on the multiband data.
        isophotfit = ellipsefit_multiband(objid, objdir, data, geometry, band=band,
                                          refband=refband, verbose=verbose)
        if debug:
            display_multiband(data, isophotfit=isophotfit, band=band)
    else:

        mgefit_multiband(objid, objdir, data, band=band, refband=refband,
                         pixscale=pixscale, verbose=verbose)

        import pdb ; pdb.set_trace()


    return 1 # success!
