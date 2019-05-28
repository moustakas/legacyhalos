"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.
"""
from __future__ import absolute_import, division, print_function

import os, copy, pdb
import time, warnings
import multiprocessing

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io
import legacyhalos.misc
import legacyhalos.hsc

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

def _apphot_one(args):
    """Wrapper function for the multiprocessing."""
    return apphot_one(*args)

def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale):
    """Perform aperture photometry in one elliptical annulus.

    """
    from photutils import EllipticalAperture, aperture_photometry

    aperture = EllipticalAperture((x0, y0), aa, bb, theta)
    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.
    
    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
    apphot = mu_flux * pixscale**2 # [nanomaggies]
    return apphot

def ellipse_apphot(bands, data, refellipsefit, pixscalefactor,
                   pixscale, pool=None):
    """Perform elliptical aperture photometry for the curve-of-growth analysis.

    maxsma in pixels
    pixscalefactor - assumed to be constant for all bandpasses!

    """
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning

    deltaa = 0.5 # pixel spacing 
    maxsma = refellipsefit['maxsma']
    theta = np.radians(refellipsefit['pa']-90)

    results = {}

    for filt in bands:
        img = ma.getdata(data['{}_masked'.format(filt)]) # [nanomaggies/arcsec2]
        #img = ma.getdata(data['{}_masked'.format(filt)]) * pixscale**2 # [nanomaggies/arcsec2-->nanomaggies]
        mask = ma.getmask(data['{}_masked'.format(filt)])

        deltaa_filt = deltaa * pixscalefactor
        sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)
        smb = sma * refellipsefit['eps']

        x0 = refellipsefit['x0'] * pixscalefactor
        y0 = refellipsefit['y0'] * pixscalefactor

        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                apphot = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale)
                                                for aa, bb in zip(sma, smb)])
                apphot = np.hstack(apphot)

        results['apphot_smaunit'] = 'arcsec'
        results['apphot_sma_{}'.format(filt)] = sma * pixscale # [arcsec]
        results['apphot_mag_{}'.format(filt)] = apphot

    print('Modeling the curve of growth.')
    from astropy.modeling import Fittable1DModel, Parameter, fitting
    class CogModel(Fittable1DModel):
        """m(r) = m0 + C1 * exp**(-C2*radius**(-C3))"""
        m0 = Parameter(default=20.0, bounds=(15, 30))
        C1 = Parameter(default=1.0, bounds=(-1, 1))
        C2 = Parameter(default=1.0, bounds=(-1, 1))
        C3 = Parameter(default=1.0, bounds=(-1, 1))
        linear = False
        def __init__(self, m0=m0.default, C1=C1.default, C2=C2.default, C3=C3.default):
            super(CogModel, self).__init__(m0, C1, C2, C3)
        @staticmethod
        def evaluate(radius, m0, C1, C2, C3):
            """Evaluate the COG model."""
            model = m0 + C1 * np.exp(-C2*sma**(-C3))
            return model

    ##custom_model
    #def cogmodel(radius, m0=20.0, C1=1.0, C2=0.1, C3=0.1):
    #    return m0 + C1 * np.exp(-C2*radius**(-C3))

    #class CogFit(object):
    #    def __init__(self):
    #        self.fitter = fitting.LevMarLSWFitter()
    
    for filt in bands:
        radius, cog = results['apphot_sma_{}'.format(filt)], results['apphot_mag_{}'.format(filt)]
    
        params = fitting.LevMarLSQFitter()(CogModel(), radius, cog)
        print(params)

        pdb.set_trace()

    #fig, ax = plt.subplots()
    #for filt in bands:
    #    ax.plot(results['apphot_sma_{}'.format(filt)],
    #            22.5-2.5*np.log10(results['apphot_mag_{}'.format(filt)]), label=filt)
    #ax.set_ylim(30, 5)
    #ax.legend(loc='lower right')
    #plt.show()
    #pdb.set_trace()

    return results

def _unmask_center(img):
    # https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    nn = img.shape[0]
    x0, y0 = geometry.x0, geometry.y0
    rad = geometry.sma # [pixels]
    yy, xx = np.ogrid[-x0:nn-x0, -y0:nn-y0]
    img.mask[xx**2 + yy**2 <= rad**2] = ma.nomask
    return img

def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)

def integrate_isophot_one(iso, img, pixscalefactor, integrmode,
                          sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    """
    #g = iso.sample.geometry # fixed geometry
    g = copy.deepcopy(iso.sample.geometry) # fixed geometry
    
    # Use the same integration mode and clipping parameters.
    # The central pixel is a special case:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if g.sma == 0.0:
            gcen = copy.deepcopy(g)
            gcen.sma = 0.0
            gcen.eps = 0.0
            gcen.pa = 0.0
            censamp = CentralEllipseSample(img, 0.0, geometry=gcen,
                                           integrmode=integrmode, sclip=sclip, nclip=nclip)
            out = CentralEllipseFitter(censamp).fit()
        else:
            g.sma *= pixscalefactor
            g.x0 *= pixscalefactor
            g.y0 *= pixscalefactor

            sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)
        
    return out

def forced_ellipsefit_multiband(galaxy, galaxydir, data, filesuffix='',
                                bands=('g', 'r', 'z'), pixscale=0.262,
                                nproc=1, nowrite=False, verbose=False):
    """Performed "forced" elliptical surface brightness profile fitting (e.g., on
    the SDSS and pipeline-derived image products) using the nominal
    ellipse-fitting results as a reference.

    """
    from legacyhalos.io import read_ellipsefit

    refellipsefit = read_ellipsefit(galaxy, galaxydir, verbose=verbose)
    if not refellipsefit['success']:
        print('No reference ellipsefit file found!')
        return {'success': False}

    refband, refpixscale = refellipsefit['refband'], refellipsefit['refpixscale']
    refisophot = refellipsefit[refband]

    pixscalefactor = refpixscale / pixscale
    maxsma = refellipsefit[refband].sma.max() * pixscalefactor

    integrmode, nclip, sclip = refellipsefit['integrmode'], refellipsefit['nclip'], refellipsefit['sclip']
    step, fflag, linear = refellipsefit['step'], refellipsefit['fflag'], refellipsefit['linear']

    ellipsefit = dict()
    ellipsefit['success'] = False
    ellipsefit['bands'] = bands
    ellipsefit['pixscale'] = pixscale
    ellipsefit['refpixscale'] = refpixscale
    #ellipsefit['refband'] = refband
    #ellipsefit['redshift'] = refellipsefit['redshift']

    #print('Fix me -- what is psfsigma!!')
    #for filt in bands:
    #    ellipsefit['psfsigma_{}'.format(filt)] = 1.3

    newmask = None
    pool = multiprocessing.Pool(nproc)

    # Forced photometry.
    tall = time.time()
    for filt in bands:
        t0 = time.time()
        print('Fitting band {}.'.format(filt))

        img = data['{}_masked'.format(filt)]
        if newmask is not None:
            img.mask = newmask

        # Loop on the reference band isophotes.
        isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)
                                                       for iso in refisophot])

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)
        print('  Time = {:.3f} sec'.format(time.time() - t0))

        #if np.all( np.isnan(ellipsefit['g'].intens) ):
        #    print('ERROR: Ellipse-fitting resulted in all NaN; please check the imaging for band {}'.format(filt))
        #    ellipsefit['success'] = False

    print('Time for all images = {:.3f} sec'.format(time.time()-tall))
    ellipsefit['success'] = True

    # Perform elliptical aperture photometry.
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    apphot = ellipse_apphot(bands, data, refellipsefit, pixscalefactor,
                            pixscale, pool=pool)
    ellipsefit.update(apphot)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        filesuffix=filesuffix, verbose=True)
    pool.close()

    return ellipsefit

def ellipsefit_multiband(galaxy, galaxydir, data, sample, maxsma=None, nproc=1,
                         integrmode='median', nclip=2, sclip=3,
                         step=0.1, fflag=0.7, linear=False, zcolumn='Z',
                         input_ellipse=None, nowrite=False, verbose=False,
                         noellipsefit=True, debug=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    maxsma in (optical) pixels
    zcolumn - name of the redshift column (Z_LAMBDA in redmapper)

    """
    from astropy.stats import sigma_clip
    from legacyhalos.mge import find_galaxy

    pool = multiprocessing.Pool(nproc)
    
    # If noellipsefit=True, use the mean geometry of the galaxy to extract the
    # surface-brightness profile (turn off fitting).
    if noellipsefit:
        maxrit = -1
    else:
        maxrit = None

    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']
    xcen, ycen = data[refband].shape
    xcen /= 2
    ycen /= 2

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    ellipsefit = dict()
    img = data['{}_masked'.format(refband)]

    galprops = find_galaxy(img, nblob=1, fraction=0.05,
                           binning=3, plot=debug, quiet=not verbose)
    galprops.pa = galprops.pa % 180 # put into range [0-180]
    if debug:
        plt.savefig('debug.png')
        
    galprops.centershift = False
    if np.abs(galprops.xpeak-xcen) > 5:
        galprops.xpeak = xcen
        galprops.centershift = True
    if np.abs(galprops.ypeak-ycen) > 5:
        galprops.ypeak = ycen
        galprops.centershift = True

    for key in ('eps', 'majoraxis', 'pa', 'theta', 'centershift',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        ellipsefit['mge_{}'.format(key)] = float(getattr(galprops, key))

    ellipsefit['success'] = False
    ellipsefit['redshift'] = sample[zcolumn]
    ellipsefit['bands'] = bands
    ellipsefit['refband'] = refband
    ellipsefit['refpixscale'] = refpixscale
    for filt in bands: # [Gaussian sigma]
        #if 'PSFSIZE_{}'.format(filt.upper()) in sample.colnames:
        #    psfsize = sample['PSFSIZE_{}'.format(filt.upper())]
        if 'PSFSIZE_{}'.format(filt.upper()) in data.keys():
            psfsize = data['PSFSIZE_{}'.format(filt.upper())]
            #print(filt, psfsize)
        else:
            psfsize = 1.1 # [FWHM, arcsec]
        ellipsefit['psfsigma_{}'.format(filt)] = psfsize / np.sqrt(8 * np.log(2)) # [arcsec]
        ellipsefit['psfsigma_{}'.format(filt)] /= refpixscale # [pixels]

    ##### ##################################################
    #print('MAXSMA HACK!!!')
    #maxsma = 20
    #nclip = 0
    #integrmode = 'bilinear'
    ##### ##################################################
        
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    ellipsefit['step'] = step
    ellipsefit['fflag'] = fflag
    ellipsefit['linear'] = linear

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.
    print('Finding the mean geometry using the reference {}-band image.'.format(refband))

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    t0 = time.time()
    majoraxis = ellipsefit['mge_majoraxis']
    geometry0 = EllipseGeometry(x0=ellipsefit['mge_xpeak'], y0=ellipsefit['mge_ypeak'],
                                eps=ellipsefit['mge_eps'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['mge_pa']-90))
    
    ellipse0 = Ellipse(img, geometry=geometry0)

    smamin, smamax = 3*ellipsefit['psfsigma_{}'.format(refband)], 1.5*majoraxis # inner, outer radius
    #smamin, smamax = 0.05*majoraxis, 5*majoraxis # inner, outer radius
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        factor = (1.1, 1.2, 1.3, 1.4)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin*fac
            iso0 = ellipse0.fit_image(sma0,
                                      #minsma=0, maxsma=smamax,
                                      #minsma=smamin, maxsma=smamax,
                                      #step=0.1, linear=False,
                                      #step=0.2, linear=False, # note bigger step size
                                      integrmode=integrmode, sclip=sclip, nclip=nclip)
            if len(iso0) > 0:
                break

    if len(iso0) == 0:
        print('Initial ellipse-fitting failed!')
        return ellipsefit

    # Mask out outliers and the inner part of the galaxy where seeing dominates.
    #good = ~sigma_clip(iso0.pa, sigma=3).mask
    good = (iso0.sma > smamin) * (iso0.stop_code <= 4) * ~sigma_clip(iso0.pa, sigma=3).mask
    #good = (iso0.sma > 3 * ellipsefit['psfsigma_{}'.format(refband)]) * ~sigma_clip(iso0.pa, sigma=3).mask
    #good = (iso0.stop_code < 4) * ~sigma_clip(iso0.pa, sigma=3).mask
    ngood = np.sum(good)
    if ngood == 0:
        print('Too few good measurements to get ellipse geometry!')
        return ellipsefit

    # Fix the center to be the peak (pixel) values.
    ellipsefit['x0'] = ellipsefit['mge_xpeak']
    ellipsefit['y0'] = ellipsefit['mge_ypeak']
    ellipsefit['x0_median'] = np.median(iso0.x0[good])
    ellipsefit['y0_median'] = np.median(iso0.y0[good])
    ellipsefit['x0_err'] = np.std(iso0.x0[good]) / np.sqrt(ngood)
    ellipsefit['y0_err'] = np.std(iso0.y0[good]) / np.sqrt(ngood)

    ellipsefit['pa'] = (np.degrees(np.median(iso0.pa[good]))+90) % 180
    ellipsefit['pa_err'] = np.degrees(np.std(iso0.pa[good])) / np.sqrt(ngood)
    ellipsefit['eps'] = np.median(iso0.eps[good])
    ellipsefit['eps_err'] = np.std(iso0.eps[good]) / np.sqrt(ngood)

    if verbose:
        print(' x0 = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
            ellipsefit['x0_median'], ellipsefit['x0_err'], ellipsefit['x0']))
        print(' y0 = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
            ellipsefit['y0_median'], ellipsefit['y0_err'], ellipsefit['y0']))
        print(' PA = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
            ellipsefit['pa'], ellipsefit['pa_err'], np.degrees(geometry0.pa)+90))
        print(' eps = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
            ellipsefit['eps'], ellipsefit['eps_err'], geometry0.eps))
    print('  Time = {:.3f} sec'.format(time.time()-t0))

    if False:
        import fitsio
        with fitsio.FITS('junk.fits', 'rw') as ff:
            ff.write(img, overwrite=True)
        fig, ax = plt.subplots()
        ax.imshow(np.log10(img), origin='lower')
        for sma in np.linspace(0, 50, 5):
            iso = iso0.get_closest(sma)
            x, y, = iso.sampled_coordinates()
            ax.plot(x, y, color='green', lw=1, alpha=0.5)
        fig.savefig('junk.png')

    # Re-initialize the EllipseGeometry object, optionally using an external set
    # of ellipticity parameters.
    if input_ellipse:
        ellipsefit['input_ellipse'] = True
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=input_ellipse['eps'], sma=majoraxis, 
                                   pa=np.radians(input_ellipse['pa']-90))
    else:
        # Note: we use the MGE, not fitted geometry here because it's more
        # reliable.
        ellipsefit['input_ellipse'] = False
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=ellipsefit['mge_eps'], sma=majoraxis, 
                                   pa=np.radians(ellipsefit['mge_pa']-90))
    
    geometry_cen = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'], eps=0.0, sma=0.0, pa=0.0)
    ellipsefit['geometry'] = geometry
    ellipse = Ellipse(img, geometry=geometry)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = (data[refband].shape[0]/2) / np.cos(geometry.pa % (np.pi/4))
    ellipsefit['maxsma'] = maxsma

    # Fit the reference bands first then the other bands.
    newmask = None
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))

    # First fit with the default parameters.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print('Fitting the reference band: {}'.format(refband))
        _sma0 = (1, 3, 6, 9, 12)
        for ii, sma0 in enumerate(_sma0): # try a few different starting minor axes
            if ii > 0:
                print('Failed with sma0={:.1f} pixels, trying sma0={:.1f} pixels.'.format(_sma0[ii-1], sma0))
            try:
                isophot = ellipse.fit_image(sma0, maxsma=maxsma, maxrit=maxrit,
                                            integrmode=integrmode, sclip=sclip, nclip=nclip)
            except:
                isophot = []
            if len(isophot) > 0:
                break
    print('  Time = {:.3f} sec'.format(time.time() - t0))

    if len(isophot) == 0:
        print('Ellipse-fitting failed.')
        return ellipsefit
    else:
        ellipsefit['success'] = True
        ellipsefit[refband] = isophot

    # Now do forced photometry at the other bandpasses (or do all the bandpasses
    # if we didn't fit above).
    tall = time.time()
    for filt in bands:
        t0 = time.time()
        if filt == refband: # we did it already!
            continue

        print('Fitting band {}.'.format(filt))

        img = data['{}_masked'.format(filt)]
        if newmask is not None:
            img.mask = newmask

        # Loop on the reference band isophotes.
        isobandfit = pool.map(_integrate_isophot_one, [(iso, img, 1.0, integrmode, sclip, nclip)
                                                       for iso in isophot])

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)
        print('  Time = {:.3f} sec'.format(time.time() - t0))
    print('Time for all images = {:.3f} sec'.format(time.time() - tall))

    # Perform elliptical aperture photometry.
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    apphot = ellipse_apphot(bands, data, ellipsefit, 1.0, pixscale, pool=pool)
    ellipsefit.update(apphot)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        verbose=True)

    pool.close()
    
    return ellipsefit

def ellipse_sbprofile(ellipsefit, minerr=0.0):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    """
    bands, refband = ellipsefit['bands'], ellipsefit['refband']
    refpixscale, redshift = ellipsefit['refpixscale'], ellipsefit['redshift']

    indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    sbprofile = dict()
    for filt in bands:
        sbprofile['psfsigma_{}'.format(filt)] = ellipsefit['psfsigma_{}'.format(filt)]
    sbprofile['redshift'] = redshift
    
    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'arcsec'
    sbprofile['sma'] = ellipsefit['r'].sma[indx] * refpixscale # [arcsec]

    with np.errstate(invalid='ignore'):
        for filt in bands:
            #area = ellipsefit[filt].sarea[indx] * refpixscale**2

            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx]) # [mag/arcsec2]

            #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
            sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err[indx] / \
              ellipsefit[filt].intens[indx] / np.log(10)
            sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

            # Just for the plot use a minimum uncertainty
            #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    if 'g' in bands and 'r' in bands:
        sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
        sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    if 'r' in bands and 'z' in bands:
        sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
        sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)
    # SDSS
    if 'r' in bands and 'i' in bands:
        sbprofile['ri'] = sbprofile['mu_r'] - sbprofile['mu_i']
        sbprofile['ri_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_i_err']**2)
        
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

def legacyhalos_ellipse(onegal, galaxy=None, galaxydir=None, pixscale=0.262,
                        sdss_pixscale=0.396, pipeline=False, nproc=1,
                        refband='r', bands=('g', 'r', 'z'), maxsma=None,
                        integrmode='median', nclip=2, sclip=3, zcolumn='Z',
                        galex_pixscale=1.5, unwise_pixscale=2.75,
                        input_ellipse=None, noellipsefit=False, verbose=False,
                        debug=False, hsc=False, sdss=False, galex=False, unwise=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    noellipsefit - do not fit for the ellipse parameters (use the mean values from MGE).

    pipeline - read the pipeline-built images (default is custom)

    """
    if galaxydir is None or galaxy is None:
        if hsc:
            galaxy, galaxydir = legacyhalos.hsc.get_galaxy_galaxydir(onegal)
        else:
            galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(onegal)

    # Read the data.
    data = legacyhalos.io.read_multiband(galaxy, galaxydir, bands=bands,
                                         refband=refband, pixscale=pixscale,
                                         galex_pixscale=galex_pixscale,
                                         unwise_pixscale=unwise_pixscale,
                                         sdss_pixscale=sdss_pixscale,
                                         sdss=sdss, pipeline=pipeline)

    print('HACK!!!!')
    pipeline = True
    
    # Do ellipse-fitting.
    if bool(data):
        if pipeline or sdss or unwise or galex:
            if pipeline:
                print('Forced ellipse-fitting on the pipeline images.')
                ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data, nproc=nproc,
                                                         filesuffix='pipeline', bands=('g','r','z'),
                                                         pixscale=pixscale, verbose=verbose)
            if sdss:
                print('Forced ellipse-fitting on the SDSS images.')
                ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data, nproc=nproc,
                                                         filesuffix='sdss', bands=('g','r','i'),
                                                         pixscale=sdss_pixscale, verbose=verbose)
            if unwise:
                print('Forced ellipse-fitting on the unWISE images.')
                ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data, nproc=nproc,
                                                         filesuffix='unwise', bands=('W1','W2','W3','W4'),
                                                         pixscale=unwise_pixscale, verbose=verbose)
            if galex:
                print('Forced ellipse-fitting on the GALEX images.')
                ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data, nproc=nproc,
                                                         filesuffix='galex', bands=('NUV','FUV'),
                                                         pixscale=galex_pixscale, verbose=verbose)
        else:
            ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data, onegal,
                                              nproc=nproc, integrmode=integrmode,
                                              nclip=nclip, sclip=sclip, zcolumn=zcolumn,
                                              verbose=verbose, noellipsefit=noellipsefit,
                                              input_ellipse=input_ellipse)
        if ellipsefit['success']:
            return 1
        else:
            return 0
    else:
        return 0
