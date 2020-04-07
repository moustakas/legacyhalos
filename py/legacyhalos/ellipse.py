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

import astropy.modeling

import legacyhalos.io
import legacyhalos.misc

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

class CogModel(astropy.modeling.Fittable1DModel):
    """Class to empirically model the curve of growth.

    radius in arcsec
    r0 - constant scale factor (10)

    m(r) = mtot + mcen * (1-exp**(-alpha1*(radius/r0)**(-alpha2))
    """
    mtot = astropy.modeling.Parameter(default=20.0, bounds=(5, 25)) # integrated magnitude (r-->infty)
    m0 = astropy.modeling.Parameter(default=10.0, bounds=(0, 20)) # central magnitude (r=0)
    alpha1 = astropy.modeling.Parameter(default=0.3, bounds=(0, 5)) # scale factor 1
    alpha2 = astropy.modeling.Parameter(default=0.5, bounds=(0, 5)) # scale factor 2

    def __init__(self, mtot=mtot.default, m0=m0.default,
                 alpha1=alpha1.default, alpha2=alpha2.default):
        super(CogModel, self).__init__(mtot, m0, alpha1, alpha2)

        self.r0 = 10 # scale factor [arcsec]
        
    def evaluate(self, radius, mtot, m0, alpha1, alpha2):
        """Evaluate the COG model."""
        model = mtot + m0 * (1 - np.exp(-alpha1*(radius/self.r0)**(-alpha2)))
        return model
        
def _apphot_one(args):
    """Wrapper function for the multiprocessing."""
    return apphot_one(*args)

def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, variance=False):
    """Perform aperture photometry in a single elliptical annulus.

    """
    from photutils import EllipticalAperture, aperture_photometry

    aperture = EllipticalAperture((x0, y0), aa, bb, theta)
    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.
    
    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
    if variance:
        apphot = np.sqrt(mu_flux) * pixscale**2 # [nanomaggies]
    else:
        apphot = mu_flux * pixscale**2 # [nanomaggies]
    return apphot

def ellipse_cog(bands, data, refellipsefit, pixscalefactor,
                pixscale, centralindx=0, pool=None, seed=1):
    """Measure the curve of growth (CoG) by performing elliptical aperture
    photometry.

    maxsma in pixels
    pixscalefactor - assumed to be constant for all bandpasses!

    """
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from scipy.interpolate import interp1d

    rand = np.random.RandomState(seed)
    
    deltaa = 0.5 # pixel spacing

    #theta, eps = refellipsefit['geometry'].pa, refellipsefit['geometry'].eps
    theta, eps = np.radians(refellipsefit['pa']-90), refellipsefit['eps']
    refband = refellipsefit['refband']
    #maxsma = refellipsefit['maxsma']

    results = {}

    # Build the SB profile and measure the radius (in arcsec) at which mu
    # crosses a few different thresholds like 25 mag/arcsec, etc.
    sbprofile = ellipse_sbprofile(refellipsefit)

    print('We should measure these radii from the extinction-corrected photometry!')
    sberr = sbprofile['muerr_r']
    rr = (sbprofile['sma_r'] * pixscale)**0.25 # [arcsec]
    for sbcut in (24, 25, 25.5, 26):
        if sbprofile['mu_r'].max() < sbcut:
            print('Profile too shallow to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
            results['ellipse_r{:0g}'.format(sbcut)] = -1.0
            continue

        sb = sbprofile['mu_r'] - sbcut
        keep = np.where((sb > -1) * (sb < 1))[0]
        #coeff = np.polyfit(rr[keep], sb[keep], 1, w=1/sberr[keep])
        # Monte Carlo to get the radius
        rcut = []
        for ii in np.arange(20):
            sbfit = rand.normal(sb[keep], sberr[keep])
            coeff = np.polyfit(sbfit, rr[keep], 1)
            rcut.append((np.polyval(coeff, 0))**4)
        meanrcut, sigrcut = np.mean(rcut), np.std(rcut)
        print(rcut, meanrcut, sigrcut)

        #plt.clf() ; plt.plot((rr[keep])**4, sb[keep]) ; plt.axvline(x=meanrcut) ; plt.savefig('junk.png')
        #plt.clf() ; plt.plot(rr, sb+sbcut) ; plt.axvline(x=meanrcut**0.25) ; plt.axhline(y=sbcut) ; plt.xlim(2, 2.6) ; plt.savefig('junk.png')
        #pdb.set_trace()
            
        #try:
        #    rcut = interp1d()(sbcut) # [arcsec]
        #except:
        #    print('Warning: extrapolating r({:0g})!'.format(sbcut))
        #    rcut = interp1d(sbprofile['mu_r'], sbprofile['sma_r'] * pixscale, fill_value='extrapolate')(sbcut) # [arcsec]
        results['ellipse_r{:0g}'.format(sbcut)] = np.float32(meanrcut)
        results['ellipse_r{:0g}_err'.format(sbcut)] = np.float32(sigrcut)

    for filt in bands:
        img = ma.getdata(data['{}_masked'.format(filt)][centralindx]) # [nanomaggies/arcsec2]
        mask = ma.getmask(data['{}_masked'.format(filt)][centralindx])

        deltaa_filt = deltaa * pixscalefactor

        if filt in refellipsefit['bands']:
            maxsma = sbprofile['sma_{}'.format(filt)].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(filt)] # [pixels]
        else:
            maxsma = sbprofile['sma_{}'.format(refband)].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(refband)] # [pixels]
            
        sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)
        smb = sma * eps

        x0 = refellipsefit['x0'] * pixscalefactor
        y0 = refellipsefit['y0'] * pixscalefactor

        #im = np.log10(img) ; im[mask] = 0 ; plt.clf() ; plt.imshow(im, origin='lower') ; plt.scatter(y0, x0, s=50, color='red') ; plt.savefig('junk.png')
        #pdb.set_trace()

        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                cogflux = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale, False)
                                                for aa, bb in zip(sma, smb)])
                cogflux = np.hstack(cogflux)

                if '{}_var'.format(filt) in data.keys():
                    var = data['{}_var'.format(filt)][centralindx] # [nanomaggies**2/arcsec**4]
                    cogferr = pool.map(_apphot_one, [(var, mask, theta, x0, y0, aa, bb, pixscale, True)
                                                    for aa, bb in zip(sma, smb)])
                    cogferr = np.hstack(cogferr)
                else:
                    cogferr = None

        cogmag = 22.5 - 2.5 * np.log10(cogflux) # [mag]
        sma_arcsec = sma * pixscale             # [arcsec]

        results['cog_smaunit'] = 'arcsec'
        results['cog_mag_{}'.format(filt)] = cogmag
        results['cog_sma_{}'.format(filt)] = sma_arcsec

        if cogferr is not None:
            cogmagerr = 2.5 * cogferr / cogflux / np.log(10)
            results['cog_magerr_{}'.format(filt)] = cogmagerr
        else:
            cogmagerr = np.ones(len(cogmag))

        #print('Modeling the curve of growth.')
        def get_chi2(bestfit):
            sbmodel = bestfit(self.radius, self.wave)
            chi2 = np.sum( (self.sb - sbmodel)**2 / self.sberr**2 ) / dof
        
        cogfitter = astropy.modeling.fitting.LevMarLSQFitter()
        cogmodel = CogModel()

        nball = 10

        # perturb the parameter values
        nparams = len(cogmodel.parameters)
        dof = len(cogmag) - nparams

        params = np.repeat(cogmodel.parameters, nball).reshape(nparams, nball)
        for ii, pp in enumerate(cogmodel.param_names):
            pinfo = getattr(cogmodel, pp)
            if pinfo.bounds[0] is not None:
                scale = 0.2 * pinfo.default
                params[ii, :] += rand.normal(scale=scale, size=nball)
                toosmall = np.where( params[ii, :] < pinfo.bounds[0] )[0]
                if len(toosmall) > 0:
                    params[ii, toosmall] = pinfo.default
                toobig = np.where( params[ii, :] > pinfo.bounds[1] )[0]
                if len(toobig) > 0:
                    params[ii, toobig] = pinfo.default
            else:
                params[ii, :] += rand.normal(scale=0.2 * pinfo.default, size=nball)
                
        # perform the fit nball times
        chi2fail = 1e6
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chi2 = np.zeros(nball) + chi2fail
            for jj in range(nball):
                cogmodel.parameters = params[:, jj]
                # Fit up until the curve of growth turns over, but no less than
                # the second moment of the light distribution! Pretty fragile..
                these = np.where(np.diff(cogmag) < 0)[0]
                if sma_arcsec[these[0]] < (refellipsefit['majoraxis'] * pixscale * pixscalefactor):
                    these = np.where(sma_arcsec < refellipsefit['majoraxis'] * pixscale * pixscalefactor)[0]
                
                ballfit = cogfitter(cogmodel, sma_arcsec[these], cogmag[these],
                                    maxiter=100, weights=1/cogmagerr[these])
                bestfit = ballfit(sma_arcsec)

                chi2[jj] = np.sum( (cogmag - bestfit)**2 / cogmagerr**2 ) / dof
                if cogfitter.fit_info['param_cov'] is None: # failed
                    if False:
                        print(jj, cogfitter.fit_info['message'], chi2[jj])
                else:
                    params[:, jj] = ballfit.parameters # update

        # if at least one fit succeeded, re-evaluate the model at the chi2
        # minimum.
        good = chi2 < chi2fail
        if np.sum(good) == 0:
            print('{} CoG modeling failed.'.format(filt))
            #result.update({'fit_message': cogfitter.fit_info['message']})
            #return result
        mindx = np.argmin(chi2)
        minchi2 = chi2[mindx]
        cogmodel.parameters = params[:, mindx]
        P = cogfitter(cogmodel, sma_arcsec, cogmag, weights=1/cogmagerr, maxiter=100)
        print('{} CoG modeling succeeded with a chi^2 minimum of {:.2f}'.format(filt, minchi2))
        
        #P = cogfitter(cogmodel, sma_arcsec, cogmag, weights=1/cogmagerr)
        results['cog_params_{}'.format(filt)] = {'mtot': P.mtot.value, 'm0': P.m0.value,
                                                 'alpha1': P.alpha1.value, 'alpha2': P.alpha2.value,
                                                 'chi2': minchi2}
    #    print(filt, P)
    #    default_model = cogmodel.evaluate(radius, P.mtot.default, P.m0.default, P.alpha1.default, P.alpha2.default)
    #    bestfit_model = cogmodel.evaluate(radius, P.mtot.value, P.m0.value, P.alpha1.value, P.alpha2.value)
    #    plt.scatter(radius, cog, label='Data {}'.format(filt), s=10)
    #    plt.plot(radius, bestfit_model, label='Best Fit {}'.format(filt), color='k', lw=1, alpha=0.5)
    #plt.legend(fontsize=10)
    #plt.ylim(15, 30)
    #plt.savefig('junk.png')
    #pdb.set_trace()

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

def _unpack_isofit(isofit):
    """Unpack the IsophotList objects into a dictionary because the resulting pickle
    files are huge.

    https://photutils.readthedocs.io/en/stable/api/photutils.isophote.IsophoteList.html#photutils.isophote.IsophoteList

    """
    result = {
        'sma': isofit.sma.astype('f4'),
        'eps': isofit.eps.astype('f4'),
        'eps_err': isofit.ellip_err.astype('f4'),
        'pa': isofit.pa.astype('f4'),
        'pa_err': isofit.pa_err.astype('f4'),
        'intens': isofit.intens.astype('f4'),
        'intens_err': isofit.int_err.astype('f4'),
        'x0': isofit.x0.astype('f4'),
        'x0_err': isofit.x0_err.astype('f4'),
        'y0': isofit.y0.astype('f4'),
        'y0_err': isofit.y0_err.astype('f4'),
        'a3': isofit.a3.astype('f4'),
        'a3_err': isofit.a3_err.astype('f4'),
        'a4': isofit.a4.astype('f4'),
        'a4_err': isofit.a4_err.astype('f4'),
        'rms': isofit.rms.astype('f4'),
        'pix_stddev': isofit.pix_stddev.astype('f4'),
        'stop_code': isofit.stop_code.astype(np.int16),
        'ndata': isofit.ndata.astype(np.int16),
        'nflag': isofit.nflag.astype(np.int16),
        'niter': isofit.niter.astype(np.int16)}
        
    return result

def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)

def integrate_isophot_one(img, sma, theta, eps, x0, y0, pixscalefactor,
                          integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    theta in radians

    """
    #g = iso.sample.geometry # fixed geometry
    #g = copy.deepcopy(iso.sample.geometry) # fixed geometry
    g = EllipseGeometry(x0=x0, y0=y0, eps=eps, sma=sma, pa=theta)

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
            sample.update(fixed_parameters=True)
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)
        
    return out

def forced_ellipsefit_multiband(galaxy, galaxydir, data, refellipsefit=None,
                                central_galaxy_id=None, filesuffix='',
                                bands=('g', 'r', 'z'), pixscale=0.262,
                                nproc=1, nowrite=False, verbose=False):
    """Performed "forced" elliptical surface brightness profile fitting using either
    a previous set of ellipse-fitting results or the basic MGE-derived geometry.

    """

    from legacyhalos.io import read_ellipsefit

    refellipsefit = read_ellipsefit(galaxy, galaxydir, verbose=verbose)
    if not bool(refellipsefit) or not refellipsefit['success']:
        print('Reference ellipsefit not found or unsuccessful!')
        return {'success': False}

    refband, refpixscale = refellipsefit['refband'], refellipsefit['refpixscale']
    refisophot = refellipsefit[refband]

    pixscalefactor = refpixscale / pixscale
    maxsma = refellipsefit[refband]['sma'].max() * pixscalefactor

    nclip = refellipsefit['nclip']
    sclip = refellipsefit['sclip']
    integrmode = refellipsefit['integrmode']

    ellipsefit = dict()
    ellipsefit['success'] = False
    ellipsefit['input_ellipse'] = False
    ellipsefit['bands'] = bands
    ellipsefit['pixscale'] = pixscale
    ellipsefit['refpixscale'] = refpixscale
    ellipsefit['refband'] = refellipsefit['geometry']
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
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)
        isobandfit = pool.map(_integrate_isophot_one, [(
            img, _sma, _theta, _eps, _x0, _y0, pixscalefactor, integrmode, sclip, nclip)
            for _sma, _theta, _eps, _x0, _y0 in zip(refisophot['sma'], np.radians(refisophot['pa']-90),
                                                    refisophot['eps'], refisophot['x0'],
                                                    refisophot['y0'])])

        # Build the IsophoteList instance with the result.
        #ellipsefit[filt] = IsophoteList(isobandfit)
        ellipsefit[filt] = _unpack_isofit(IsophoteList(isobandfit))
        print('  Time = {:.3f} sec'.format(time.time() - t0))

        #if np.all( np.isnan(ellipsefit['g'].intens) ):
        #    print('ERROR: Ellipse-fitting resulted in all NaN; please check the imaging for band {}'.format(filt))
        #    ellipsefit['success'] = False

    print('Time for all images = {:.3f} sec'.format(time.time()-tall))
    ellipsefit['success'] = True

    # Perform elliptical aperture photometry.
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, refellipsefit, pixscalefactor,
                      pixscale, pool=pool)
    ellipsefit.update(cog)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        filesuffix=filesuffix, verbose=True)
    pool.close()

    return ellipsefit

def obsolete_ellipsefit_multiband(galaxy, galaxydir, data, redshift=None, maxsma=None, nproc=1,
                                  filesuffix='', integrmode='median', nclip=2, sclip=3, 
                                  input_ellipse=None, nowrite=False, should_be_centered=True,
                                  verbose=False, fitgeometry=False, debug=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    data - dictionary returned by legacyhalos.io.read_multiband, which has the
      initial MGE ellipse geometry in it.

    should_be_centered - set to False if the object is not expected to be at the center

    maxsma in (optical) pixels
    zcolumn - name of the redshift column (Z_LAMBDA in redmapper)

    """
    from astropy.stats import sigma_clip
    from legacyhalos.mge import find_galaxy

    # If fitgeometry=True then fit for the geometry as a function of semimajor
    # axis, otherwise (the default) use the mean geometry of the galaxy to
    # extract the surface-brightness profile.
    if fitgeometry:
        maxrit = None
    else:
        maxrit = -1

    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']

    #xcen, ycen = data['{}_masked'.format(refband)].shape
    #xcen /= 2
    #ycen /= 2
    #
    ## Get the geometry of the galaxy in the reference band.
    #if verbose:
    #    print('Finding the galaxy in the reference {}-band image.'.format(refband))

    # Initialize the output dictionary, starting from the galaxy geometry in the
    # 'data' dictionary.
    ellipsefit = dict()
    ellipsefit['success'] = False
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    if redshift:
        ellipsefit['redshift'] = redshift

    for key in data.keys():
        if not '_masked' in key:
            ellipsefit[key] = data[key]
            
    #img = data['{}'.format(refband)]
    img = data['{}_masked'.format(refband)]

    ##debug=True ; plt.clf()
    #galprops = find_galaxy(img, nblob=1, fraction=0.05, binning=3, quiet=not verbose, plot=debug)
    #galprops.pa = galprops.pa % 180 # put into range [0-180]
    #if debug:
    #    plt.savefig('debug.png')
    #plt.clf()
    #
    #galprops.centershift = False
    #if should_be_centered:
    #    if np.abs(galprops.xpeak-xcen) > 5:
    #        galprops.xpeak = xcen
    #        galprops.centershift = True
    #    if np.abs(galprops.ypeak-ycen) > 5:
    #        galprops.ypeak = ycen
    #        galprops.centershift = True
    #for key in ('eps', 'majoraxis', 'pa', 'theta', 'centershift',
    #            'xmed', 'ymed', 'xpeak', 'ypeak'):
    #    ellipsefit['mge_{}'.format(key)] = np.float32(getattr(galprops, key))

    #ellipsefit['bands'] = bands
    #ellipsefit['refband'] = refband
    #ellipsefit['refpixscale'] = np.float32(refpixscale)
    #for filt in bands: # [Gaussian sigma]
    #    #if 'PSFSIZE_{}'.format(filt.upper()) in sample.colnames:
    #    #    psfsize = sample['PSFSIZE_{}'.format(filt.upper())]
    #    if 'psfsize_{}'.format(filt) in data.keys():
    #        psfsize = data['psfsize_{}'.format(filt)]
    #        #print(filt, psfsize)
    #    else:
    #        psfsize = 1.1 # [FWHM, arcsec]
    #    ellipsefit['psfsigma_{}'.format(filt)] = (psfsize / np.sqrt(8 * np.log(2)) / refpixscale).astype('f4') # [pixels]
    #
    #    ellipsefit['psfsize_{}'.format(filt)] = data['psfsize_{}'.format(filt)] # [FWHM, arcsec]
    #    ellipsefit['psfsize_min_{}'.format(filt)] = data['psfsize_min_{}'.format(filt)]
    #    ellipsefit['psfsize_max_{}'.format(filt)] = data['psfsize_max_{}'.format(filt)]
    #
    #    ellipsefit['psfdepth_{}'.format(filt)] = data['psfdepth_{}'.format(filt)] # [AB mag]
    #    ellipsefit['psfdepth_min_{}'.format(filt)] = data['psfdepth_min_{}'.format(filt)]
    #    ellipsefit['psfdepth_max_{}'.format(filt)] = data['psfdepth_max_{}'.format(filt)]

    ##### ##################################################
    #print('MAXSMA HACK!!!')
    #maxsma = 20
    #nclip = 0
    #integrmode = 'bilinear'
    ##### ##################################################
        
    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.
    print('Finding the mean geometry using the reference {}-band image.'.format(refband))

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    t0 = time.time()
    majoraxis = ellipsefit['mge'][0]['majoraxis']
    geometry0 = EllipseGeometry(x0=ellipsefit['mge'][0]['xpeak'], y0=ellipsefit['mge'][0]['ypeak'],
                                eps=ellipsefit['mge'][0]['eps'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['mge'][0]['pa']-90))
    ellipse0 = Ellipse(img, geometry=geometry0)
    #print('Hack unmask')
    ellipse0 = Ellipse(ma.getdata(img), geometry=geometry0)

    smamin, smamax = ellipsefit['psfsigma_{}'.format(refband)], 1.5*majoraxis # inner, outer radius
    if smamin > majoraxis:
        print('Warning! this galaxy is smaller than three times the seeing FWHM!')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        factor = np.arange(1.0, 6, 0.5) # (1, 2, 3, 3.5, 4, 4.5, 5, 10)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin*fac
            try:
                iso0 = ellipse0.fit_image(sma0, integrmode=integrmode, sclip=sclip, nclip=nclip, minsma=0, maxsma=smamax)
            except:
                iso0 = []
            if len(iso0) > 0:
                break

    if len(iso0) == 0:
        print('Initial ellipse-fitting failed!')
        ellipsefit['x0'] = ellipsefit['mge'][0]['xpeak']
        ellipsefit['y0'] = ellipsefit['mge'][0]['ypeak']

        if input_ellipse is None:
            return ellipsefit

    else:
        # Mask out outliers and the inner part of the galaxy where seeing dominates.
        #good = ~sigma_clip(iso0.pa, sigma=3).mask
        good = (iso0.sma > smamin) * (iso0.stop_code <= 4)
        #good = (iso0.sma > smamin) * (iso0.stop_code <= 4) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.sma > 3 * ellipsefit['psfsigma_{}'.format(refband)]) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.stop_code < 4) * ~sigma_clip(iso0.pa, sigma=3).mask
        ngood = np.sum(good)
        if ngood == 0:
            print('Too few good measurements to get ellipse geometry!')
            return ellipsefit

        ellipsefit['init_smamin'] = iso0.sma[good].min()
        ellipsefit['init_smamax'] = iso0.sma[good].max()

        # Fix the center to be the peak (pixel) values.
        ellipsefit['x0'] = ellipsefit['mge'][0]['xpeak']
        ellipsefit['y0'] = ellipsefit['mge'][0]['ypeak']
        ellipsefit['x0_median'] = np.mean(iso0.x0[good])
        ellipsefit['y0_median'] = np.mean(iso0.y0[good])
        ellipsefit['x0_err'] = np.std(iso0.x0[good]) / np.sqrt(ngood)
        ellipsefit['y0_err'] = np.std(iso0.y0[good]) / np.sqrt(ngood)

        ellipsefit['pa'] = (np.degrees(np.mean(iso0.pa[good]))+90) % 180
        ellipsefit['pa_err'] = np.degrees(np.std(iso0.pa[good])) / np.sqrt(ngood)
        ellipsefit['eps'] = np.mean(iso0.eps[good])
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
        print('Using input ellipse parameters.')
        ellipsefit['input_ellipse'] = True
        input_eps, input_pa = input_ellipse['eps'], input_ellipse['pa'] % 180
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=input_eps, sma=majoraxis, 
                                   pa=np.radians(input_pa-90))
    else:
        # Note: we use the MGE, not fitted geometry here because it's more
        # reliable based on visual inspection.
        ellipsefit['input_ellipse'] = False
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=ellipsefit['mge'][0]['eps'], sma=majoraxis, 
                                   pa=np.radians(ellipsefit['mge'][0]['pa']-90))
    
    geometry_cen = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'], eps=0.0, sma=0.0, pa=0.0)
    ellipsefit['geometry'] = geometry
    ellipse = Ellipse(img, geometry=geometry)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = 0.95 * (data['{}_masked'.format(refband)].shape[0]/2) / np.cos(geometry.pa % (np.pi/4))
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
        smamin = 3 * ellipsefit['psfsigma_{}'.format(refband)]
        factor = (1.0, 1.1, 1.2, 1.3, 1.4)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin * fac
            if ii > 0:
                print('Failed with sma0={:.1f} pixels, trying sma0={:.1f} pixels.'.format(
                    smamin*factor[ii-1], sma0))
            try:
                isophot = ellipse.fit_image(sma0, maxsma=maxsma, maxrit=maxrit,
                                            integrmode=integrmode, sclip=sclip, nclip=nclip)#,
                                            #fix_pa=True, fix_eps=True, fix_center=True)
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
        #ellipsefit[refband] = isophot
        refisophot = _unpack_isofit(isophot)
        ellipsefit[refband] = refisophot

    # Now do forced photometry at the other bandpasses (or do all the bandpasses
    pool = multiprocessing.Pool(nproc)
    
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
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, 1.0, integrmode, sclip, nclip)
        #                                               for iso in isophot])
        isobandfit = pool.map(_integrate_isophot_one, [(
            img, _sma, _pa, _eps, _x0, _y0, 1.0, integrmode, sclip, nclip)
            for _sma, _pa, _eps, _x0, _y0 in zip(refisophot['sma'], refisophot['pa'],
                                                 refisophot['eps'], refisophot['x0'],
                                                 refisophot['y0'])])
        

        # Build the IsophoteList instance with the result.
        #ellipsefit[filt] = IsophoteList(isobandfit)
        ellipsefit[filt] = _unpack_isofit(IsophoteList(isobandfit))
        print('  Time = {:.3f} sec'.format(time.time() - t0))
    print('Time for all images = {:.3f} sec'.format(time.time() - tall))

    # Perform elliptical aperture photometry.
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, 1.0, refpixscale, pool=pool)
    ellipsefit.update(cog)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    # Write out
    if not nowrite:
        legacyhalos.io.write_mge_ellipsefit(galaxy, galaxydir, ellipsefit,
                                            verbose=True, filesuffix=filesuffix)
        #legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
        #                                verbose=True, filesuffix=filesuffix)

    pool.close()
    
    return ellipsefit

def ellipse_sbprofile(ellipsefit, minerr=0.0, snrmin=1.0, sma_not_radius=False,
                      sdss=False, linear=False):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    linear - stay in linear (nanomaggies/arcsec2) units (i.e., don't convert to
      mag/arcsec2) and do not compute colors; used by legacyhalos.integrate

    sma_not_radius - if True, then store the semi-major axis in the 'radius' key
      (converted to arcsec) rather than the circularized radius

    """
    bands = ellipsefit['bands']
    if 'refpixscale' in ellipsefit.keys():
        pixscale = ellipsefit['refpixscale']
    else:
        pixscale = ellipsefit['pixscale']

    if 'geometry' in ellipsefit.keys():
        eps = ellipsefit['geometry'].eps
    else:
        eps = ellipsefit['eps']

    sbprofile = dict()
    for filt in bands:
        psfkey = 'psfsigma_{}'.format(filt)
        if psfkey in ellipsefit.keys():
            sbprofile[psfkey] = ellipsefit[psfkey]

    if 'redshift' in ellipsefit.keys():
        sbprofile['redshift'] = ellipsefit['redshift']
    
    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'pixels'
    sbprofile['radiusunit'] = 'arcsec'

    # semi-major axis and circularized radius
    #sbprofile['sma'] = ellipsefit[bands[0]].sma * pixscale # [arcsec]

    for filt in bands:
        #area = ellipsefit[filt].sarea[indx] * pixscale**2

        sb = ellipsefit[filt]['intens']             # [nanomaggies/arcsec2]
        sberr = np.sqrt(ellipsefit[filt]['intens_err']**2 + (0.4 * np.log(10) * sb * minerr)**2)
        sma = ellipsefit[filt]['sma']                               # semi-major axis [pixels]
        if sma_not_radius:
            radius = sma * pixscale # [arcsec]
        else:
            radius = sma * np.sqrt(1 - eps) * pixscale # circularized radius [arcsec]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            keep = np.isfinite(sb) * ((sb / sberr) > snrmin)

        sbprofile['sma_{}'.format(filt)] = sma[keep]       # [pixels]
        sbprofile['radius_{}'.format(filt)] = radius[keep] # [arcsec]
        if linear:
            sbprofile['mu_{}'.format(filt)] = sb[keep] # [nanomaggies/arcsec2]
            sbprofile['muerr_{}'.format(filt)] = sberr[keep]
            continue
        else:
            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(sb[keep]) # [mag/arcsec2]
            sbprofile['muerr_{}'.format(filt)] = 2.5 * sberr[keep] / sb[keep] / np.log(10)

        #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens)
        #sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err / \
        #  ellipsefit[filt].intens / np.log(10)
        #sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

        # Just for the plot use a minimum uncertainty
        #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    if 'g' in bands and 'r' in bands and 'z' in bands:
        radius_gr, indx_g, indx_r = np.intersect1d(sbprofile['radius_g'], sbprofile['radius_r'], return_indices=True)
        sbprofile['gr'] = sbprofile['mu_g'][indx_g] - sbprofile['mu_r'][indx_r]
        sbprofile['gr_err'] = np.sqrt(sbprofile['muerr_g'][indx_g]**2 + sbprofile['muerr_r'][indx_r]**2)
        sbprofile['radius_gr'] = radius_gr

        radius_rz, indx_r, indx_z = np.intersect1d(sbprofile['radius_r'], sbprofile['radius_z'], return_indices=True)
        sbprofile['rz'] = sbprofile['mu_r'][indx_r] - sbprofile['mu_z'][indx_z]
        sbprofile['rz_err'] = np.sqrt(sbprofile['muerr_r'][indx_r]**2 + sbprofile['muerr_z'][indx_z]**2)
        sbprofile['radius_rz'] = radius_rz
        
    # SDSS
    if sdss and 'g' in bands and 'r' in bands and 'i' in bands:
        radius_gr, indx_g, indx_r = np.intersect1d(sbprofile['radius_g'], sbprofile['radius_r'], return_indices=True)
        sbprofile['gr'] = sbprofile['mu_g'][indx_g] - sbprofile['mu_r'][indx_r]
        sbprofile['gr_err'] = np.sqrt(sbprofile['muerr_g'][indx_g]**2 + sbprofile['muerr_r'][indx_r]**2)
        sbprofile['radius_gr'] = radius_gr

        radius_ri, indx_r, indx_i = np.intersect1d(sbprofile['radius_r'], sbprofile['radius_i'], return_indices=True)
        sbprofile['ri'] = sbprofile['mu_r'][indx_r] - sbprofile['mu_i'][indx_i]
        sbprofile['ri_err'] = np.sqrt(sbprofile['muerr_r'][indx_r]**2 + sbprofile['muerr_i'][indx_i]**2)
        sbprofile['radius_ri'] = radius_ri
        
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

def _fitgeometry_refband(ellipsefit, geometry0, majoraxis, refband='r', verbose=False,
                         integrmode='median', sclip=3, nclip=2):
    """Support routine for ellipsefit_multiband. Optionally use photutils to fit for
    the ellipse geometry as a function of semi-major axis.

    """
    smamax = majoraxis # inner, outer radius
    #smamax = 1.5*majoraxis
    smamin = ellipsefit['psfsigma_{}'.format(refband)]

    if smamin > majoraxis:
        print('Warning! this galaxy is smaller than three times the seeing FWHM!')
        
    t0 = time.time()
    print('Finding the mean geometry using the reference {}-band image...'.format(refband), end='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        factor = np.arange(1.0, 6, 0.5) # (1, 2, 3, 3.5, 4, 4.5, 5, 10)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin*fac
            try:
                iso0 = ellipse0.fit_image(sma0, integrmode=integrmode, sclip=sclip, nclip=nclip)
            except:
                iso0 = []
                sma0 = smamin
            if len(iso0) > 0:
                break
    print('...took {:.3f} sec'.format(time.time()-t0))

    if len(iso0) == 0:
        print('Initial ellipse-fitting failed.')
    else:
        # Try to determine the mean fitted geometry, for diagnostic purposes,
        # masking out outliers and the inner part of the galaxy where seeing
        # dominates.
        good = (iso0.sma > smamin) * (iso0.stop_code <= 4)
        #good = ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.sma > smamin) * (iso0.stop_code <= 4) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.sma > 3 * ellipsefit['psfsigma_{}'.format(refband)]) * ~sigma_clip(iso0.pa, sigma=3).mask
        #good = (iso0.stop_code < 4) * ~sigma_clip(iso0.pa, sigma=3).mask

        ngood = np.sum(good)
        if ngood == 0:
            print('Too few good measurements to get ellipse geometry!')
        else:
            ellipsefit['success'] = True
            ellipsefit['init_smamin'] = iso0.sma[good].min()
            ellipsefit['init_smamax'] = iso0.sma[good].max()

            ellipsefit['x0_median'] = np.mean(iso0.x0[good])
            ellipsefit['y0_median'] = np.mean(iso0.y0[good])
            ellipsefit['x0_err'] = np.std(iso0.x0[good]) / np.sqrt(ngood)
            ellipsefit['y0_err'] = np.std(iso0.y0[good]) / np.sqrt(ngood)

            ellipsefit['pa'] = (np.degrees(np.mean(iso0.pa[good]))+90) % 180
            ellipsefit['pa_err'] = np.degrees(np.std(iso0.pa[good])) / np.sqrt(ngood)
            ellipsefit['eps'] = np.mean(iso0.eps[good])
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

    return ellipsefit

def ellipsefit_multiband(galaxy, galaxydir, data, centralindx=0, galaxyid=None,
                         filesuffix='', refband='r', maxsma=None, nproc=1,
                         integrmode='median', nclip=2, sclip=3, galaxyinfo=None,
                         input_ellipse=None, fitgeometry=False,
                         nowrite=False, verbose=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    Some, but not all hooks for fitgeometry=True are in here, so user beware.

    galaxyinfo - additional dictionary to append to the output file

    galaxyid - add a unique ID number to the output filename (via
      io.write_ellipsefit).

    """
    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']
    
    # If fitgeometry=True then fit for the geometry as a function of semimajor
    # axis, otherwise (the default) use the mean geometry of the galaxy to
    # extract the surface-brightness profile.
    if fitgeometry:
        maxrit = None
    else:
        maxrit = -1
        
    # Initialize the output dictionary, starting from the galaxy geometry in the
    # 'data' dictionary.
    ellipsefit = dict()
    ellipsefit['input_ellipse'] = False
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    ellipsefit['fitgeometry'] = fitgeometry

    # This is fragile, but copy over a specific set of keys from the data dictionary--
    copykeys = ['bands', 'refband', 'refpixscale', 'shape',
                'psfsigma_g', 'psfsigma_r', 'psfsigma_z',
                'psfsize_g', 'psfsize_min_g', 'psfsize_max_g',
                'psfdepth_g', 'psfdepth_min_g', 'psfdepth_max_g', 
                'psfsize_r', 'psfsize_min_r', 'psfsize_max_r',
                'psfdepth_r', 'psfdepth_min_r', 'psfdepth_max_r',
                'psfsize_z', 'psfsize_min_z', 'psfsize_max_z',
                'psfdepth_z', 'psfdepth_min_z', 'psfdepth_max_z']
    for key in copykeys:
        if key in data.keys():
            ellipsefit[key] = data[key]

    img = data['{}_masked'.format(refband)][centralindx]
    mge = data['mge'][centralindx]

    # Fix the center to be the peak (pixel) values. Could also use bx,by here
    # from Tractor.  Also initialize the geometry with the moment-derived
    # values.  Note that (x,y) are switched between MGE and photutils!!
    for key in ['eps', 'pa', 'theta', 'majoraxis', 'ra_med', 'dec_med',
                'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z']:
        ellipsefit[key] = mge[key]
    for mgekey, ellkey in zip(['ymed', 'xmed'], ['x0', 'y0']):
        ellipsefit[ellkey] = mge[mgekey]

    majoraxis = mge['majoraxis'] # [pixel]

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry0 = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                eps=ellipsefit['eps'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['pa']-90))
    ellipse0 = Ellipse(img, geometry=geometry0)
    #import matplotlib.pyplot as plt
    #plt.imshow(img, origin='lower') ; plt.scatter(ellipsefit['y0'], ellipsefit['x0'], s=50, color='red') ; plt.savefig('junk.png')
    #pdb.set_trace()

    if fitgeometry:
        ellipsefit = _fitgeometry_refband(ellipsefit, geometry0, majoraxis, refband,
                                          integrmode=integrmode, sclip=sclip, nclip=nclip,
                                          verbose=verbose)
    
    # Re-initialize the EllipseGeometry object, optionally using an external set
    # of ellipticity parameters.
    if input_ellipse:
        print('Using input ellipse parameters.')
        ellipsefit['input_ellipse'] = True
        input_eps, input_pa = input_ellipse['eps'], input_ellipse['pa'] % 180
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=input_eps, sma=majoraxis, 
                                   pa=np.radians(input_pa-90))
    else:
        # Note: we use the MGE, not fitted geometry here because it's more
        # reliable based on visual inspection.
        geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=ellipsefit['eps'], sma=majoraxis, 
                                   pa=np.radians(ellipsefit['pa']-90))
    
    geometry_cen = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                                   eps=0.0, sma=0.0, pa=0.0)
    #ellipsefit['geometry'] = geometry # can't save an object in an .asdf file
    ellipse = Ellipse(img, geometry=geometry)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = 0.95 * (data['shape'][0]/2) / np.cos(geometry.pa % (np.pi/4))
    ellipsefit['maxsma'] = np.float32(maxsma) # [pixels]

    sma = np.arange(np.ceil(maxsma)).astype('f4')
    #ellipsefit['sma'] = np.arange(np.ceil(maxsma)).astype('f4')

    # Now get the surface brightness profile.  Need some more code for this to
    # work with fitgeometry=True...
    pool = multiprocessing.Pool(nproc)

    tall = time.time()
    for filt in bands:
        print('Fitting {}-band took...'.format(filt), end='')
        img = data['{}_masked'.format(filt)][centralindx]

        # Loop on the reference band isophotes.
        t0 = time.time()
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)
        isobandfit = pool.map(_integrate_isophot_one, [(
            img, _sma, ellipsefit['pa'], ellipsefit['eps'], ellipsefit['x0'],
            ellipsefit['y0'], 1.0, integrmode, sclip, nclip) for _sma in sma])

        # Build the IsophoteList instance with the result.
        #ellipsefit[filt] = IsophoteList(isobandfit)
        ellipsefit[filt] = _unpack_isofit(IsophoteList(isobandfit))
        print('...{:.3f} sec'.format(time.time() - t0))
    print('Time for all images = {:.3f} sec'.format(time.time()-tall))

    ellipsefit['success'] = True
    
    # Perform elliptical aperture photometry--
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, 1.0, refpixscale,
                      centralindx=centralindx, pool=pool)
    ellipsefit.update(cog)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    pool.close()

    # Write out
    if not nowrite:
        if galaxyinfo:
            ellipsefit.update(galaxyinfo)
        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        galaxyid=galaxyid, verbose=True,
                                        filesuffix=filesuffix)

    return ellipsefit

def legacyhalos_ellipse(onegal, galaxy=None, galaxydir=None, pixscale=0.262,
                        sdss_pixscale=0.396, galex_pixscale=1.5, unwise_pixscale=2.75,
                        nproc=1, refband='r', bands=('g','r','z'), sdss_bands=('g','r','i'),
                        integrmode='median', nclip=2, sclip=3, zcolumn=None,
                        fullsample=None, largegalaxy=False, pipeline=False, 
                        input_ellipse=None, fitgeometry=False,
                        verbose=False, debug=False, sdss=False, galex=False, unwise=False):
                        
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    fitgeometry - fit for the ellipse parameters (do not use the mean values
      from MGE).

    pipeline - read the pipeline-built images (default is custom)

    """
    if galaxydir is None or galaxy is None:
        galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(onegal)

    if zcolumn is not None and zcolumn in onegal.columns:
        redshift = onegal[zcolumn]
    else:
        redshift = None

    if largegalaxy:
        filesuffix = 'largegalaxy'
    else:
        filesuffix = ''

    # Read the data and then do ellipse-fitting.
    data = legacyhalos.io.read_multiband(galaxy, galaxydir, bands=bands,
                                         refband=refband, pixscale=pixscale,
                                         galex_pixscale=galex_pixscale,
                                         unwise_pixscale=unwise_pixscale,
                                         verbose=verbose,
                                         largegalaxy=largegalaxy)
    if bool(data):
        for igal in np.arange(len(data['central_galaxy_id'])):
            central_galaxy_id = data['central_galaxy_id'][igal]
            galaxyid = str(central_galaxy_id)
            
            # Try to do an initial round of ellipse-fitting in the reference image.
            if largegalaxy:
                maxsma = 1.5 * data['mge'][igal]['majoraxis'] # [pixels]
                # Supplement the fit results dictionary with some additional info--
                thisgal = fullsample[fullsample['LSLGA_ID'] == central_galaxy_id]
                galaxyinfo = {'lslga_id': central_galaxy_id,
                              'galaxy': str(thisgal['GALAXY'][0])}
                for key, outkey in zip(['ra', 'dec', 'pgc', 'pa', 'ba', 'd25'],
                                       ['ra', 'dec', 'pgc', 'lslga_pa', 'lslga_ba', 'lslga_d25']):
                    galaxyinfo[outkey] = thisgal[key.upper()][0]
            else:
                maxsma, galaxyid = None, None
                galaxyinfo = {'redshift': redshift}

            ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data, centralindx=igal,
                                              galaxyid=galaxyid, filesuffix=filesuffix,
                                              refband='r', nproc=nproc, integrmode=integrmode,
                                              nclip=nclip, sclip=sclip, verbose=verbose,
                                              input_ellipse=input_ellipse, maxsma=maxsma,
                                              fitgeometry=False, galaxyinfo=galaxyinfo)
                
    else:
        return 0

    if pipeline:
        print('Forced ellipse-fitting on the pipeline images.')
        pipeline_data = legacyhalos.io.read_multiband(galaxy, galaxydir, bands=bands,
                                                      refband=refband, pixscale=pixscale,
                                                      pipeline=True, verbose=verbose)
        if bool(pipeline_data):
            pipeline_ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, pipeline_data,
                                                              nproc=nproc, filesuffix='pipeline',
                                                              bands=bands, pixscale=pixscale,
                                                              verbose=verbose)
    if sdss:
        print('Forced ellipse-fitting on the SDSS images.')
        sdss_data = legacyhalos.io.read_multiband(galaxy, galaxydir, bands=sdss_bands,
                                                  refband='r', sdss_pixscale=sdss_pixscale,
                                                  sdss=True)
        
        sdss_ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, sdss_data,
                                                      nproc=nproc, filesuffix='sdss',
                                                      bands=sdss_bands, pixscale=sdss_pixscale,
                                                      verbose=verbose)

    if unwise:
        print('Forced ellipse-fitting on the unWISE images.')
        unwise_ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data,
                                                        nproc=nproc, filesuffix='unwise',
                                                        bands=('W1','W2','W3','W4'),
                                                        pixscale=unwise_pixscale,
                                                        verbose=verbose)
    if galex:
        print('Forced ellipse-fitting on the GALEX images.')
        galex_ellipsefit = forced_ellipsefit_multiband(galaxy, galaxydir, data,
                                                       nproc=nproc, filesuffix='galex',
                                                       bands=('FUV', 'NUV'),
                                                       pixscale=galex_pixscale,
                                                       verbose=verbose)
    if ellipsefit['success']:
        return 1
    else:
        return 0
