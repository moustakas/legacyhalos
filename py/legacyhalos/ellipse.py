"""
legacyhalos.ellipse
===================

Code to do ellipse fitting on the residual coadds.
"""
import os, pdb
import time, warnings
import numpy as np
#import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import astropy.modeling

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

import legacyhalos.io

REF_SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds

def _get_r0():
    r0 = 10.0 # [arcsec]
    return r0

def cog_model(radius, mtot, m0, alpha1, alpha2):
    r0 = _get_r0()
    #return mtot - m0 * np.expm1(-alpha1*((radius / r0)**(-alpha2)))
    return mtot + m0 * np.log1p(alpha1*(radius/10.0)**(-alpha2))

def cog_dofit(sma, mag, mag_err, bounds=None):
    chisq = 1e6
    try:
        popt, _ = curve_fit(cog_model, sma, mag, sigma=mag_err,
                            bounds=bounds, max_nfev=10000)
    except RuntimeError:
        popt = None
    else:
        chisq = (((cog_model(sma, *popt) - mag) / mag_err) ** 2).sum()
        
    return popt, chisq

class CogModel(astropy.modeling.Fittable1DModel):
    """Class to empirically model the curve of growth.

    radius in arcsec
    r0 - constant scale factor (10)

    m(r) = mtot + mcen * (1-exp**(-alpha1*(radius/r0)**(-alpha2))
    """
    mtot = astropy.modeling.Parameter(default=20.0, bounds=(1, 30)) # integrated magnitude (r-->infty)
    m0 = astropy.modeling.Parameter(default=10.0, bounds=(1, 30)) # central magnitude (r=0)
    alpha1 = astropy.modeling.Parameter(default=0.3, bounds=(1e-3, 5)) # scale factor 1
    alpha2 = astropy.modeling.Parameter(default=0.5, bounds=(1e-3, 5)) # scale factor 2

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

def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, variance=False, iscircle=False):
    """Perform aperture photometry in a single elliptical annulus.

    """
    from photutils import EllipticalAperture, CircularAperture, aperture_photometry

    if iscircle:
        aperture = CircularAperture((x0, y0), aa)
    else:
        aperture = EllipticalAperture((x0, y0), aa, bb, theta)

    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.
    
    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]        
    #print(x0, y0, aa, bb, theta, mu_flux, pixscale, img.shape, mask.shape, aperture)
    if variance:
        apphot = np.sqrt(mu_flux) * pixscale**2 # [nanomaggies]
    else:
        apphot = mu_flux * pixscale**2 # [nanomaggies]

    return apphot

def ellipse_cog(bands, data, refellipsefit, igal=0, pool=None,
                seed=1, sbthresh=REF_SBTHRESH):
    """Measure the curve of growth (CoG) by performing elliptical aperture
    photometry.

    maxsma in pixels
    pixscalefactor - assumed to be constant for all bandpasses!

    """
    import numpy.ma as ma
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from scipy import integrate
    from scipy.interpolate import interp1d

    rand = np.random.RandomState(seed)
    
    #deltaa = 1.0 # pixel spacing

    #theta, eps = refellipsefit['geometry'].pa, refellipsefit['geometry'].eps
    theta = np.radians(refellipsefit['pa']-90)
    eps = refellipsefit['eps']
    refband = refellipsefit['refband']
    refpixscale = data['refpixscale']

    #maxsma = refellipsefit['maxsma']

    results = {}

    # Build the SB profile and measure the radius (in arcsec) at which mu
    # crosses a few different thresholds like 25 mag/arcsec, etc.
    sbprofile = ellipse_sbprofile(refellipsefit)

    #print('We should measure these radii from the extinction-corrected photometry!')
    for sbcut in sbthresh:
        if sbprofile['mu_{}'.format(refband)].max() < sbcut or sbprofile['mu_{}'.format(refband)].min() > sbcut:
            print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
            results['sma_sb{:0g}'.format(sbcut)] = np.float32(-1.0)
            results['sma_sb{:0g}_err'.format(sbcut)] = np.float32(-1.0)
            continue

        rr = (sbprofile['sma_{}'.format(refband)] * refpixscale)**0.25 # [arcsec]
        sb = sbprofile['mu_{}'.format(refband)] - sbcut
        sberr = sbprofile['muerr_{}'.format(refband)]
        keep = np.where((sb > -1) * (sb < 1))[0]
        if len(keep) < 5:
            keep = np.where((sb > -2) * (sb < 2))[0]
            if len(keep) < 5:
                print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
                results['sma_sb{:0g}'.format(sbcut)] = np.float32(-1.0)
                results['sma_sb{:0g}_err'.format(sbcut)] = np.float32(-1.0)
                continue

        # Monte Carlo to get the radius
        rcut = []
        for ii in np.arange(20):
            sbfit = rand.normal(sb[keep], sberr[keep])
            coeff = np.polyfit(sbfit, rr[keep], 1)
            rcut.append((np.polyval(coeff, 0))**4)
        meanrcut, sigrcut = np.mean(rcut), np.std(rcut)
        #print(rcut, meanrcut, sigrcut)

        #plt.clf() ; plt.plot((rr[keep])**4, sb[keep]) ; plt.axvline(x=meanrcut) ; plt.savefig('junk.png')
        #plt.clf() ; plt.plot(rr, sb+sbcut) ; plt.axvline(x=meanrcut**0.25) ; plt.axhline(y=sbcut) ; plt.xlim(2, 2.6) ; plt.savefig('junk.png')
        #pdb.set_trace()
            
        #try:
        #    rcut = interp1d()(sbcut) # [arcsec]
        #except:
        #    print('Warning: extrapolating r({:0g})!'.format(sbcut))
        #    rcut = interp1d(sbprofile['mu_{}'.format(refband)], sbprofile['sma_{}'.format(refband)] * pixscale, fill_value='extrapolate')(sbcut) # [arcsec]
        results['sma_sb{:0g}'.format(sbcut)] = np.float32(meanrcut)
        results['sma_sb{:0g}_err'.format(sbcut)] = np.float32(sigrcut)

    chi2fail = 1e6
    nparams = 4
        
    for filt in bands:
        img = ma.getdata(data['{}_masked'.format(filt.lower())][igal]) # [nanomaggies/arcsec2]
        mask = ma.getmask(data['{}_masked'.format(filt.lower())][igal])

        #deltaa_filt = deltaa * pixscalefactor
        if filt in refellipsefit['bands']:
            if len(sbprofile['sma_{}'.format(filt.lower())]) == 0: # can happen with partial coverage in other bands
                maxsma = sbprofile['sma_{}'.format(refband.lower())].max()
            else:
                maxsma = sbprofile['sma_{}'.format(filt.lower())].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(filt.lower())] # [pixels]
        else:
            maxsma = sbprofile['sma_{}'.format(refband)].max()        # [pixels]
            #minsma = 3 * refellipsefit['psfsigma_{}'.format(refband)] # [pixels]
        
        #sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)

        sma = refellipsefit['sma_{}'.format(filt.lower())] * 1.0
        keep = np.where((sma > 0) * (sma <= maxsma))[0]
        #keep = np.where(sma < maxsma)[0]
        if len(keep) > 0:
            sma = sma[keep]
        else:
            print('Too few good semi-major axis pixels!')
            #pdb.set_trace()
            #raise ValueError
        
        smb = sma * eps
        if eps == 0.0:
            iscircle = True
        else:
            iscircle = False

        # handle GALEX and WISE
        if 'filt2pixscale' in data.keys():
            pixscale = data['filt2pixscale'][filt]
            if np.isclose(pixscale, refpixscale): # avoid rounding issues
                pixscale = refpixscale                
                pixscalefactor = 1.0
            else:
                pixscalefactor = refpixscale / pixscale
        else:
            pixscale = refpixscale
            pixscalefactor = 1.0

        x0 = pixscalefactor * refellipsefit['x0']
        y0 = pixscalefactor * refellipsefit['y0']

        #if filt == 'g':
        #    pdb.set_trace()
        #im = np.log10(img) ; im[mask] = 0 ; plt.clf() ; plt.imshow(im, origin='lower') ; plt.scatter(y0, x0, s=50, color='red') ; plt.savefig('junk.png')

        #print(filt, img.shape, pixscale)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                #cogflux = [apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle) for aa, bb in zip(sma, smb)]
                cogflux = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                for aa, bb in zip(sma, smb)])
                if len(cogflux) > 0:
                    cogflux = np.hstack(cogflux)
                else:
                    cogflux = np.array([0.0])

                if '{}_var'.format(filt.lower()) in data.keys():
                    var = data['{}_var'.format(filt.lower())][igal] # [nanomaggies**2/arcsec**4]
                    cogferr = pool.map(_apphot_one, [(var, mask, theta, x0, y0, aa, bb, pixscale, True, iscircle)
                                                    for aa, bb in zip(sma, smb)])
                    if len(cogferr) > 0:
                        cogferr = np.hstack(cogferr)
                    else:
                        cogferr = np.array([0.0])
                else:
                    cogferr = None

        # Aperture fluxes can be negative (or nan?) sometimes--
        with warnings.catch_warnings():
            if cogferr is not None:
                with np.errstate(divide='ignore'):
                    ok = (cogflux > 0) * (cogferr > 0) * np.isfinite(cogflux) * np.isfinite(cogferr) * (cogflux / cogferr > 1)
            else:
                ok = (cogflux > 0) * np.isfinite(cogflux)
                cogmagerr = np.ones(len(cogmag))

        #results['cog_smaunit'] = 'arcsec'

        if np.count_nonzero(ok) == 0:
            results['cog_sma_{}'.format(filt.lower())] = np.float32(-1) # np.array([])
            results['cog_mag_{}'.format(filt.lower())] = np.float32(-1) # np.array([])
            results['cog_magerr_{}'.format(filt.lower())] = np.float32(-1) # np.array([])
            results['cog_mtot_{}'.format(filt.lower())] = np.float32(-1)
            results['cog_m0_{}'.format(filt.lower())] = np.float32(-1)
            results['cog_alpha1_{}'.format(filt.lower())] = np.float32(-1)
            results['cog_alpha2_{}'.format(filt.lower())] = np.float32(-1)
            results['cog_chi2_{}'.format(filt.lower())] = np.float32(1e6)
            results['cog_sma50_{}'.format(filt.lower())] = np.float32(-1)
            for sbcut in sbthresh:
                results['{}_mag_sb{:0g}'.format(filt.lower(), sbcut)] = np.float32(-1)
                results['{}_mag_sb{:0g}_err'.format(filt.lower(), sbcut)] = np.float32(-1)
        else:
            sma_arcsec = sma[ok] * pixscale             # [arcsec]
            cogmag = 22.5 - 2.5 * np.log10(cogflux[ok]) # [mag]
            if cogferr is not None:
                cogmagerr = 2.5 * cogferr[ok] / cogflux[ok] / np.log(10)

            results['cog_sma_{}'.format(filt.lower())] = np.float32(sma_arcsec)
            results['cog_mag_{}'.format(filt.lower())] = np.float32(cogmag)
            results['cog_magerr_{}'.format(filt.lower())] = np.float32(cogmagerr)

            #if filt == 'FUV':
            #    pdb.set_trace()
            
            #print('Modeling the curve of growth.')
            these = np.where((sma_arcsec > 0) * (cogmag > 0) * (cogmagerr > 0))[0]
            if len(these) < nparams:
                print('Warning: Too few {}-band pixels to fit the curve of growth; skipping.'.format(filt))
                results['cog_mtot_{}'.format(filt.lower())] = np.float32(-1)
                results['cog_m0_{}'.format(filt.lower())] = np.float32(-1)
                results['cog_alpha1_{}'.format(filt.lower())] = np.float32(-1)
                results['cog_alpha2_{}'.format(filt.lower())] = np.float32(-1)
                results['cog_chi2_{}'.format(filt.lower())] = np.float32(1e6)
                results['cog_sma50_{}'.format(filt.lower())] = np.float32(-1)
                continue
            else:
                bounds = ([cogmag[these][-1]-1.0, 0, 0, 0], np.inf)
                #bounds = ([cogmag[-1]-0.5, 2.5, 0, 0], np.inf)
                #bounds = (0, np.inf)
                popt, minchi2 = cog_dofit(sma_arcsec[these], cogmag[these], cogmagerr[these], bounds=bounds)
                if minchi2 < chi2fail and popt is not None:
                    mtot, m0, alpha1, alpha2 = popt
                    print('{} CoG modeling succeeded with a chi^2 minimum of {:.2f}'.format(filt, minchi2))
                    results['cog_mtot_{}'.format(filt.lower())] = np.float32(mtot)
                    results['cog_m0_{}'.format(filt.lower())] = np.float32(m0)
                    results['cog_alpha1_{}'.format(filt.lower())] = np.float32(alpha1)
                    results['cog_alpha2_{}'.format(filt.lower())] = np.float32(alpha2)
                    results['cog_chi2_{}'.format(filt.lower())] = np.float32(minchi2)

                    # get the half-light radius (along the major axis)
                    if (m0 != 0) * (alpha1 != 0.0) * (alpha2 != 0.0):
                        #half_light_sma = (- np.log(1.0 - np.log10(2.0) * 2.5 / m0) / alpha1)**(-1.0/alpha2) * _get_r0() # [arcsec]
                        with np.errstate(all='ignore'):                        
                            half_light_sma = ((np.expm1(np.log10(2.0)*2.5/m0)) / alpha1)**(-1.0 / alpha2) * _get_r0() # [arcsec]
                            #if filt == 'W4':
                            #    pdb.set_trace()
                        results['cog_sma50_{}'.format(filt.lower())] = half_light_sma

            #print('Measuring integrated magnitudes to different radii.')
            sb = ellipse_sbprofile(refellipsefit, linear=True)
            radkeys = ['sma_sb{:0g}'.format(sbcut) for sbcut in sbthresh]
            for radkey in radkeys:
                magkey = radkey.replace('sma_', '{}_mag_'.format(filt.lower()))
                magerrkey = '{}_err'.format(magkey)

                smamax = results[radkey] # semi-major axis
                if smamax > 0 and smamax < np.max(sma_arcsec):
                    rmax = smamax * np.sqrt(1 - refellipsefit['eps']) # [circularized radius, arcsec]

                    rr = sb['sma_{}'.format(filt.lower())]    # [circularized radius, arcsec]
                    yy = sb['mu_{}'.format(filt.lower())]        # [surface brightness, nanomaggies/arcsec**2]
                    yyerr = sb['muerr_{}'.format(filt.lower())] # [surface brightness, nanomaggies/arcsec**2]
                    try:
                        #print(filt, rr.max(), rmax)
                        yy_rmax = interp1d(rr, yy)(rmax) # can fail if rmax < np.min(sma_arcsec)
                        yyerr_rmax = interp1d(rr, yyerr)(rmax)

                        # append the maximum radius to the end of the array
                        keep = np.where(rr < rmax)[0]
                        _rr = np.hstack((rr[keep], rmax))
                        _yy = np.hstack((yy[keep], yy_rmax))
                        _yyerr = np.hstack((yyerr[keep], yyerr_rmax))

                        flux = 2 * np.pi * integrate.simps(x=_rr, y=_rr*_yy)
                        fvar = integrate.simps(x=_rr, y=_rr*_yyerr**2)
                        if flux > 0 and fvar > 0:
                            ferr = 2 * np.pi * np.sqrt(fvar)
                            results[magkey] = np.float32(22.5 - 2.5 * np.log10(flux))
                            results[magerrkey] = np.float32(2.5 * ferr / flux / np.log(10))
                        else:
                            results[magkey] = np.float32(-1.0)
                            results[magerrkey] = np.float32(-1.0)
                    except:
                        results[magkey] = np.float32(-1.0)
                        results[magerrkey] = np.float32(-1.0)
                else:
                    results[magkey] = np.float32(-1.0)
                    results[magerrkey] = np.float32(-1.0)
                
    return results

def _unmask_center(img):
    # https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    import numpy.ma as ma
    nn = img.shape[0]
    x0, y0 = geometry.x0, geometry.y0
    rad = geometry.sma # [pixels]
    yy, xx = np.ogrid[-x0:nn-x0, -y0:nn-y0]
    img.mask[xx**2 + yy**2 <= rad**2] = ma.nomask
    return img

def _unpack_isofit(ellipsefit, filt, isofit, failed=False):
    """Unpack the IsophotList objects into a dictionary because the resulting pickle
    files are huge.

    https://photutils.readthedocs.io/en/stable/api/photutils.isophote.IsophoteList.html#photutils.isophote.IsophoteList

    """
    if failed:
        ellipsefit.update({
            'sma_{}'.format(filt.lower()): np.array([-1]).astype(np.int16),
            'intens_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'intens_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'eps_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'eps_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'pa_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'pa_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'x0_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'x0_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'y0_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'y0_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'a3_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'a3_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'a4_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'a4_err_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'rms_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'pix_stddev_{}'.format(filt.lower()): np.array([-1]).astype('f4'),
            'stop_code_{}'.format(filt.lower()): np.array([-1]).astype(np.int16),
            'ndata_{}'.format(filt.lower()): np.array([-1]).astype(np.int16), 
            'nflag_{}'.format(filt.lower()): np.array([-1]).astype(np.int16), 
            'niter_{}'.format(filt.lower()): np.array([-1]).astype(np.int16)})
    else:
        ellipsefit.update({
            'sma_{}'.format(filt.lower()): isofit.sma.astype(np.int16),
            'intens_{}'.format(filt.lower()): isofit.intens.astype('f4'),
            'intens_err_{}'.format(filt.lower()): isofit.int_err.astype('f4'),
            'eps_{}'.format(filt.lower()): isofit.eps.astype('f4'),
            'eps_err_{}'.format(filt.lower()): isofit.ellip_err.astype('f4'),
            'pa_{}'.format(filt.lower()): isofit.pa.astype('f4'),
            'pa_err_{}'.format(filt.lower()): isofit.pa_err.astype('f4'),
            'x0_{}'.format(filt.lower()): isofit.x0.astype('f4'),
            'x0_err_{}'.format(filt.lower()): isofit.x0_err.astype('f4'),
            'y0_{}'.format(filt.lower()): isofit.y0.astype('f4'),
            'y0_err_{}'.format(filt.lower()): isofit.y0_err.astype('f4'),
            'a3_{}'.format(filt.lower()): isofit.a3.astype('f4'),
            'a3_err_{}'.format(filt.lower()): isofit.a3_err.astype('f4'),
            'a4_{}'.format(filt.lower()): isofit.a4.astype('f4'),
            'a4_err_{}'.format(filt.lower()): isofit.a4_err.astype('f4'),
            'rms_{}'.format(filt.lower()): isofit.rms.astype('f4'),
            'pix_stddev_{}'.format(filt.lower()): isofit.pix_stddev.astype('f4'),
            'stop_code_{}'.format(filt.lower()): isofit.stop_code.astype(np.int16),
            'ndata_{}'.format(filt.lower()): isofit.ndata.astype(np.int16),
            'nflag_{}'.format(filt.lower()): isofit.nflag.astype(np.int16),
            'niter_{}'.format(filt.lower()): isofit.niter.astype(np.int16)})
    return ellipsefit

def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)

def integrate_isophot_one(img, sma, theta, eps, x0, y0, 
                          integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    theta in radians

    """
    import copy
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
            #g.sma *= pixscalefactor
            #g.x0 *= pixscalefactor
            #g.y0 *= pixscalefactor

            sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update(fixed_parameters=True)
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)
        
    return out

def ellipse_sbprofile(ellipsefit, minerr=0.0, snrmin=1.0, sma_not_radius=False,
                      cut_on_cog=False, sdss=False, linear=False):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    linear - stay in linear (nanomaggies/arcsec2) units (i.e., don't convert to
      mag/arcsec2) and do not compute colors; used by legacyhalos.integrate

    sma_not_radius - if True, then store the semi-major axis in the 'radius' key
      (converted to arcsec) rather than the circularized radius

    cut_on_cog - if True, limit the sma to where we have successfully measured
      the curve of growth

    """
    sbprofile = dict()
    bands = ellipsefit['bands']
    if 'refpixscale' in ellipsefit.keys():
        pixscale = ellipsefit['refpixscale']
    else:
        pixscale = ellipsefit['pixscale']
    eps = ellipsefit['eps']
    if 'redshift' in ellipsefit.keys():
        sbprofile['redshift'] = ellipsefit['redshift']    
            
    for filt in bands:
        psfkey = 'psfsize_{}'.format(filt.lower())
        if psfkey in ellipsefit.keys():
            sbprofile[psfkey] = ellipsefit[psfkey]

    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'pixels'
    sbprofile['radiusunit'] = 'arcsec'

    # semi-major axis and circularized radius
    #sbprofile['sma'] = ellipsefit[bands[0]].sma * pixscale # [arcsec]

    for filt in bands:
        #area = ellipsefit[filt].sarea[indx] * pixscale**2

        sma = np.atleast_1d(ellipsefit['sma_{}'.format(filt.lower())])   # semi-major axis [pixels]
        sb = np.atleast_1d(ellipsefit['intens_{}'.format(filt.lower())]) # [nanomaggies/arcsec2]
        sberr = np.atleast_1d(np.sqrt(ellipsefit['intens_err_{}'.format(filt.lower())]**2 + (0.4 * np.log(10) * sb * minerr)**2))
            
        if sma_not_radius:
            radius = sma * pixscale # [arcsec]
        else:
            radius = sma * np.sqrt(1 - eps) * pixscale # circularized radius [arcsec]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if linear:
                keep = np.isfinite(sb)
            else:
                keep = np.isfinite(sb) * ((sb / sberr) > snrmin)
                #if filt == 'FUV':
                #    pdb.set_trace()
                
            if cut_on_cog:
                keep *= (ellipsefit['sma_{}'.format(filt.lower())] * pixscale) <= np.max(ellipsefit['cog_sma_{}'.format(filt.lower())])
            keep = np.where(keep)[0]
                
            sbprofile['keep_{}'.format(filt.lower())] = keep

        if len(keep) == 0 or sma[0] == -1:
            sbprofile['sma_{}'.format(filt.lower())] = np.array([-1.0]).astype('f4')    # [pixels]
            sbprofile['radius_{}'.format(filt.lower())] = np.array([-1.0]).astype('f4') # [arcsec]
            sbprofile['mu_{}'.format(filt.lower())] = np.array([-1.0]).astype('f4')     # [nanomaggies/arcsec2]
            sbprofile['muerr_{}'.format(filt.lower())] = np.array([-1.0]).astype('f4')  # [nanomaggies/arcsec2]
        else:
            sbprofile['sma_{}'.format(filt.lower())] = sma[keep]       # [pixels]
            sbprofile['radius_{}'.format(filt.lower())] = radius[keep] # [arcsec]
            if linear:
                sbprofile['mu_{}'.format(filt.lower())] = sb[keep] # [nanomaggies/arcsec2]
                sbprofile['muerr_{}'.format(filt.lower())] = sberr[keep] # [nanomaggies/arcsec2]
                continue
            else:
                sbprofile['mu_{}'.format(filt.lower())] = 22.5 - 2.5 * np.log10(sb[keep]) # [mag/arcsec2]
                sbprofile['muerr_{}'.format(filt.lower())] = 2.5 * sberr[keep] / sb[keep] / np.log(10) # [mag/arcsec2]

        #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens)
        #sbprofile['mu_{}_err'.format(filt.lower())] = 2.5 * ellipsefit[filt].int_err / \
        #  ellipsefit[filt].intens / np.log(10)
        #sbprofile['mu_{}_err'.format(filt.lower())] = np.sqrt(sbprofile['mu_{}_err'.format(filt.lower())]**2 + minerr**2)

        # Just for the plot use a minimum uncertainty
        #sbprofile['{}_err'.format(filt.lower())][sbprofile['{}_err'.format(filt.lower())] < minerr] = minerr

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
    smamin = ellipsefit['psfsize_{}'.format(refband)] / ellipsefit['refpixscale']

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

def ellipsefit_multiband(galaxy, galaxydir, data, igal=0, galaxy_id='',
                         refband='r', nproc=1, 
                         integrmode='median', nclip=3, sclip=3,
                         maxsma=None, logsma=True, delta_logsma=5.0, delta_sma=1.0,
                         sbthresh=REF_SBTHRESH,
                         galaxyinfo=None, input_ellipse=None,
                         fitgeometry=False, nowrite=False, verbose=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    Some, but not all hooks for fitgeometry=True are in here, so user beware.

    galaxyinfo - additional dictionary to append to the output file

    galaxy_id - add a unique ID number to the output filename (via
      io.write_ellipsefit).

    """
    import multiprocessing

    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']
    filesuffix = data['filesuffix']

    if galaxyinfo is not None:
        galaxyinfo = np.atleast_1d(galaxyinfo)
        assert(len(galaxyinfo)==len(data['mge']))
    
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
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = np.int16(sclip)
    ellipsefit['nclip'] = np.int16(nclip)
    ellipsefit['fitgeometry'] = fitgeometry

    if input_ellipse:
        ellipsefit['input_ellipse'] = True
    else:
        ellipsefit['input_ellipse'] = False

    # This is fragile, but copy over a specific set of keys from the data dictionary--
    copykeys = ['bands', 'refband', 'refpixscale',
                'refband_width', 'refband_height',
                #'psfsigma_g', 'psfsigma_r', 'psfsigma_z',
                'psfsize_g', #'psfsize_min_g', 'psfsize_max_g',
                'psfdepth_g', #'psfdepth_min_g', 'psfdepth_max_g', 
                'psfsize_r', #'psfsize_min_r', 'psfsize_max_r',
                'psfdepth_r', #'psfdepth_min_r', 'psfdepth_max_r',
                'psfsize_z', #'psfsize_min_z', 'psfsize_max_z',
                'psfdepth_z'] #'psfdepth_min_z', 'psfdepth_max_z']
    for key in copykeys:
        if key in data.keys():
            ellipsefit[key] = data[key]

    img = data['{}_masked'.format(refband)][igal]
    mge = data['mge'][igal]

    # Fix the center to be the peak (pixel) values. Could also use bx,by here
    # from Tractor.  Also initialize the geometry with the moment-derived
    # values.  Note that (x,y) are switched between MGE and photutils!!
    for key in ['largeshift', 'eps', 'pa', 'theta', 'majoraxis', 'ra_x0y0', 'dec_x0y0']:#,
                #'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z']:
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
        maxsma = 0.95 * (data['refband_width']/2) / np.cos(geometry.pa % (np.pi/4))
    ellipsefit['maxsma'] = np.float32(maxsma) # [pixels]

    if logsma:
        #https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        def _mylogspace(limit, n):
            result = [1]
            if n > 1:  # just a check to avoid ZeroDivisionError
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
            while len(result) < n:
                next_value = result[-1]*ratio
                if next_value - result[-1] >= 1:
                    # safe zone. next_value will be a different integer
                    result.append(next_value)
                else:
                    # problem! same integer. we need to find next_value by artificially incrementing previous value
                    result.append(result[-1]+1)
                    # recalculate the ratio so that the remaining values will scale correctly
                    ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
            # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
            return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.int)

        # this algorithm can fail if there are too few points
        nsma = np.ceil(maxsma / delta_logsma).astype('int')
        sma = _mylogspace(maxsma, nsma).astype('f4')
        assert(len(sma) == len(np.unique(sma)))

        #sma = np.hstack((0, np.logspace(0, np.ceil(np.log10(maxsma)).astype('int'), nsma, dtype=np.int))).astype('f4')
        print('  maxsma={:.2f} pix, delta_logsma={:.1f} log-pix, nsma={}'.format(maxsma, delta_logsma, len(sma)))
    else:
        sma = np.arange(0, np.ceil(maxsma), delta_sma).astype('f4')
        #ellipsefit['sma'] = np.arange(np.ceil(maxsma)).astype('f4')
        print('  maxsma={:.2f} pix, delta_sma={:.1f} pix, nsma={}'.format(maxsma, delta_sma, len(sma)))

    # this assert will fail when integrating the curve of growth using
    # integrate.simps because the x-axis values have to be unique.
    assert(len(np.unique(sma)) == len(sma))

    nbox = 3
    box = np.arange(nbox)-nbox // 2
    
    refpixscale = data['refpixscale']

    # Now get the surface brightness profile.  Need some more code for this to
    # work with fitgeometry=True...
    pool = multiprocessing.Pool(nproc)

    tall = time.time()
    for filt in bands:
        print('Fitting {}-band took...'.format(filt.lower()), end='')
        img = data['{}_masked'.format(filt.lower())][igal]

        # handle GALEX and WISE
        if 'filt2pixscale' in data.keys():
            pixscale = data['filt2pixscale'][filt]            
            if np.isclose(pixscale, refpixscale): # avoid rounding issues
                pixscale = refpixscale                
                pixscalefactor = 1.0
            else:
                pixscalefactor = refpixscale / pixscale
        else:
            pixscalefactor = 1.0

        x0 = pixscalefactor * ellipsefit['x0']
        y0 = pixscalefactor * ellipsefit['y0']
        #if filt == 'W4':
        #    pdb.set_trace()
        filtsma = np.round(sma * pixscalefactor).astype('f4')
        #filtsma = np.round(sma[::int(1/(pixscalefactor))] * pixscalefactor).astype('f4')
        filtsma = np.unique(filtsma)
        assert(len(np.unique(filtsma)) == len(filtsma))
    
        # Loop on the reference band isophotes.
        t0 = time.time()
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)

        # In extreme cases, and despite my best effort in io.read_multiband, the
        # image at the central position of the galaxy can end up masked, which
        # always points to a deeper issue with the data (e.g., bleed trail,
        # extremely bright star, etc.). Capture that corner case here.
        imasked, val = False, []
        for xb in box:
            for yb in box:
                val.append(img.mask[int(xb+x0), int(yb+y0)])
        if np.any(val):
            imasked = True

        if imasked:
        #if img.mask[np.int(ellipsefit['x0']), np.int(ellipsefit['y0'])]:
            print(' Central pixel is masked; resorting to extreme measures!')
            ellipsefit = _unpack_isofit(ellipsefit, filt, None, failed=True)
        else:
            isobandfit = pool.map(_integrate_isophot_one, [(
                img, _sma, ellipsefit['pa'], ellipsefit['eps'], x0,
                y0, integrmode, sclip, nclip) for _sma in filtsma])
            ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))

        #if filt == 'FUV':
        #    pdb.set_trace()
        
        print('...{:.3f} sec'.format(time.time() - t0))
        
    print('Time for all images = {:.3f} min'.format((time.time()-tall)/60))

    ellipsefit['success'] = True

    # Perform elliptical aperture photometry--
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, igal=igal,
                      pool=pool, sbthresh=sbthresh)
    ellipsefit.update(cog)
    del cog
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    pool.close()

    # Write out
    if not nowrite:
        if galaxyinfo is None:
            outgalaxyinfo = None
        else:
            outgalaxyinfo = galaxyinfo[igal]
            ellipsefit.update(galaxyinfo[igal])

        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        galaxy_id=galaxy_id,
                                        galaxyinfo=outgalaxyinfo,
                                        refband=refband,
                                        sbthresh=sbthresh,
                                        bands=ellipsefit['bands'],
                                        verbose=True,
                                        filesuffix=filesuffix)

    return ellipsefit

def legacyhalos_ellipse(galaxy, galaxydir, data, galaxyinfo=None,
                        pixscale=0.262, nproc=1, refband='r',
                        bands=['g', 'r', 'z'], integrmode='median',
                        nclip=3, sclip=3, sbthresh=REF_SBTHRESH,
                        delta_sma=1.0, delta_logsma=5, maxsma=None, logsma=True,
                        input_ellipse=None, fitgeometry=False,
                        verbose=False, debug=False):
                        
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    fitgeometry - fit for the ellipse parameters (do not use the mean values
      from MGE).

    """
    if bool(data):
        if data['missingdata']:
            if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, data['filesuffix']))):
                return 1
            else:
                return 0

        if data['failed']: # all galaxies dropped
            return 1

        if 'galaxy_id' in data.keys():
            galaxy_id = np.atleast_1d(data['galaxy_id'])
        else:
            galaxy_id = ['']
            
        for igal, galid in enumerate(galaxy_id):
            ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data,
                                              galaxyinfo=galaxyinfo,
                                              igal=igal, galaxy_id=str(galid),
                                              delta_logsma=delta_logsma, maxsma=maxsma,
                                              delta_sma=delta_sma, logsma=logsma,
                                              refband=refband, nproc=nproc, sbthresh=sbthresh,
                                              integrmode=integrmode, nclip=nclip, sclip=sclip,
                                              input_ellipse=input_ellipse,
                                              verbose=verbose, fitgeometry=False)
        return 1
    else:
        # An object can get here if it's a "known" failure, e.g., if the object
        # falls off the edge of the footprint (and therefore it will never have
        # coadds).
        if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, 'custom'))):
            return 1
        else:
            return 0
