"""
legacyhalos.sersic
==================

Code to do Sersic on the surface brightness profiles.

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from astropy.modeling import Fittable2DModel

import legacyhalos.io

class SersicSingleWaveModel(Fittable2DModel):
    """
    Define a surface brightness profile model which is three single Sersic
    models connected by a Sersic index and half-light radius which varies
    as a power-law function of wavelength.
    
    See http://docs.astropy.org/en/stable/modeling/new.html#a-step-by-step-definition-of-a-1-d-gaussian-model
    for useful info.

    """
    from astropy.modeling import Parameter
    
    nref = Parameter(default=4, bounds=(0.1, 10))
    r50ref = Parameter(default=10, bounds=(0.1, 100)) # [arcsec]
    alpha = Parameter(default=0.0, bounds=(-1, 1))
    beta = Parameter(default=0.0, bounds=(-1, 1))
    mu50_g = Parameter(default=1.0, bounds=(0, 1e4)) # [nanomaggies at r50] [mag=15-30]
    mu50_r = Parameter(default=1.0, bounds=(0, 1e4))
    mu50_z = Parameter(default=1.0, bounds=(0, 1e4))

    linear = False
    
    def __init__(self, nref=nref.default, r50ref=r50ref.default, 
                 alpha=alpha.default, beta=beta.default, 
                 mu50_g=mu50_g.default, mu50_r=mu50_r.default, mu50_z=mu50_z.default, 
                 psfsigma_g=0.0, psfsigma_r=0.0, psfsigma_z=0.0, 
                 lambda_ref=6470, lambda_g=4890, lambda_r=6470, lambda_z=9196, 
                 pixscale=0.262, seed=None, **kwargs):

        self.band = ('g', 'r', 'z')
        
        #from speclite import filters
        #filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z')
        #print(filt.effective_wavelengths.value)
        
        self.lambda_g = lambda_g
        self.lambda_r = lambda_r
        self.lambda_z = lambda_z
        self.lambda_ref = lambda_ref
        
        self.psfsigma_g = psfsigma_g
        self.psfsigma_r = psfsigma_r
        self.psfsigma_z = psfsigma_z

        self.pixscale = pixscale
        self.seed = seed
        
        super(SersicSingleWaveModel, self).__init__(nref=nref, r50ref=r50ref, alpha=alpha, 
                                                    beta=beta, mu50_g=mu50_g, mu50_r=mu50_r, 
                                                    mu50_z=mu50_z, **kwargs)
        
    def get_sersicn(self, nref, lam, alpha):
        return nref * (lam / self.lambda_ref)**alpha
    
    def get_r50(self, r50ref, lam, beta):
        return r50ref * (lam / self.lambda_ref)**beta
    
    def evaluate(self, r, w, nref, r50ref, alpha, beta, mu50_g, mu50_r, mu50_z):
        """Evaluate the wavelength-dependent single-Sersic model.
        
        Args:
          r : radius [kpc]
          w : wavelength [Angstrom]
          nref : Sersic index at the reference wavelength lambda_ref
          r50ref : half-light radius at lambda_ref
          alpha : power-law slope for the Sersic index
          beta : power-law slope for the half-light radius
          mu50_g : g-band surface brignthess at r=r50_g
          mu50_r : r-band surface brignthess at r=r50_r
          mu50_z : z-band surface brignthess at r=r50_z
        
        """
        from scipy.special import gammaincinv
        from astropy.convolution import Gaussian1DKernel, convolve
        
        mu = np.zeros_like(r)
        
        # Build the surface brightness profile at each wavelength.
        for lam, psfsig, mu50 in zip( (self.lambda_g, self.lambda_r, self.lambda_z), 
                                      (self.psfsigma_g, self.psfsigma_r, self.psfsigma_z), 
                                      (mu50_g, mu50_r, mu50_z) ):
            
            n = self.get_sersicn(nref, lam, alpha)
            r50 = self.get_r50(r50ref, lam, beta)
            
            indx = w == lam
            if np.sum(indx) > 0:
                mu_int = mu50 * np.exp(-gammaincinv(2 * n, 0.5) * ((r[indx] / r50) ** (1 / n) - 1))
            
                # smooth with the PSF
                print(psfsig)
                if psfsig > 0:
                    g = Gaussian1DKernel(stddev=psfsig)#, mode='linear_interp')
                    mu_smooth = convolve(mu_int, g, normalize_kernel=True, boundary='extend')
                    #fix = (r[indx] > 5 * psfsig * self.pixscale)
                    #mu_smooth[fix] = mu_int[fix] # replace with original values
                    mu[indx] = mu_smooth
                else:
                    mu[indx] = mu_int
        
        return mu

class SersicExponentialWaveModel(Fittable2DModel):
    """Define a surface brightness profile model which is three Sersic+exponential
    models connected by two Sersic indices and half-light radiii which vary as a
    power-law function of wavelength.

    """
    from astropy.modeling import Parameter
    
    nref1 = Parameter(default=3, bounds=(0.1, 10))
    nref2 = Parameter(default=1, fixed=True) # fixed exponential

    r50ref1 = Parameter(default=3, bounds=(0.1, 100)) # [arcsec]
    r50ref2 = Parameter(default=10, bounds=(0.1, 100)) # [arcsec]

    alpha1 = Parameter(default=0.0, bounds=(-1, 1))

    beta1 = Parameter(default=0.0, bounds=(-1, 1))
    beta2 = Parameter(default=0.0) # , bounds=(-1, 1))

    mu50_g1 = Parameter(default=1.0, bounds=(0, 1e4))
    mu50_r1 = Parameter(default=1.0, bounds=(0, 1e4))
    mu50_z1 = Parameter(default=1.0, bounds=(0, 1e4))

    mu50_g2 = Parameter(default=0.1, bounds=(0, 1e4))
    mu50_r2 = Parameter(default=0.1, bounds=(0, 1e4))
    mu50_z2 = Parameter(default=0.1, bounds=(0, 1e4))

    linear = False
    
    def __init__(self, nref1=nref1.default, nref2=nref2.default, 
                 r50ref1=r50ref1.default, r50ref2=r50ref2.default, 
                 alpha1=alpha1.default, 
                 beta1=beta1.default, beta2=beta2.default, 
                 mu50_g1=mu50_g1.default, mu50_r1=mu50_r1.default, mu50_z1=mu50_z1.default, 
                 mu50_g2=mu50_g2.default, mu50_r2=mu50_r2.default, mu50_z2=mu50_z2.default, 
                 psfsigma_g=0.0, psfsigma_r=0.0, psfsigma_z=0.0, 
                 lambda_ref=6470, lambda_g=4890, lambda_r=6470, lambda_z=9196,
                 pixscale=0.262, seed=None, **kwargs):

        self.band = ('g', 'r', 'z')
        
        #from speclite import filters
        #filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z')
        #print(filt.effective_wavelengths.value)
        
        self.lambda_g = lambda_g
        self.lambda_r = lambda_r
        self.lambda_z = lambda_z
        self.lambda_ref = lambda_ref
        
        self.psfsigma_g = psfsigma_g
        self.psfsigma_r = psfsigma_r
        self.psfsigma_z = psfsigma_z

        self.pixscale = pixscale
        self.seed = seed
        
        super(SersicExponentialWaveModel, self).__init__(nref1=nref1, nref2=nref2, r50ref1=r50ref1, r50ref2=r50ref2,
                                                         alpha1=alpha1, beta1=beta1, beta2=beta2,
                                                         mu50_g1=mu50_g1, mu50_r1=mu50_r1, mu50_z1=mu50_z1,
                                                         mu50_g2=mu50_g2, mu50_r2=mu50_r2, mu50_z2=mu50_z2,
                                                         **kwargs)

    def get_sersicn(self, nref, lam, alpha):
        return nref * (lam / self.lambda_ref)**alpha
    
    def get_r50(self, r50ref, lam, beta):
        return r50ref * (lam / self.lambda_ref)**beta
    
    def evaluate(self, r, w, nref1, nref2, r50ref1, r50ref2, alpha1, beta1, beta2,
                 mu50_g1, mu50_r1, mu50_z1, mu50_g2, mu50_r2, mu50_z2):
        """Evaluate the wavelength-dependent Sersic-exponential model.
        
        """
        from scipy.special import gammaincinv
        from astropy.convolution import Gaussian1DKernel, convolve
        
        mu = np.zeros_like(r)
        n2 = nref2 # fixed exponential
        
        # Build the surface brightness profile at each wavelength.
        for lam, psfsig, mu50_1, mu50_2 in zip( (self.lambda_g, self.lambda_r, self.lambda_z),
                                                (self.psfsigma_g, self.psfsigma_r, self.psfsigma_z),
                                                (mu50_g1, mu50_r1, mu50_z1), (mu50_g2, mu50_r2, mu50_z2)  ):
            
            n1 = self.get_sersicn(nref1, lam, alpha1)
            r50_1 = self.get_r50(r50ref1, lam, beta1)
            r50_2 = self.get_r50(r50ref2, lam, beta2)
            
            indx = w == lam
            if np.sum(indx) > 0:
                mu_int = ( mu50_1 * np.exp(-gammaincinv(2 * n1, 0.5) * ((r[indx] / r50_1) ** (1 / n1) - 1)) +
                           mu50_2 * np.exp(-gammaincinv(2 * n2, 0.5) * ((r[indx] / r50_2) ** (1 / n2) - 1)) )
            
                # smooth with the PSF
                if psfsig > 0:
                    g = Gaussian1DKernel(stddev=psfsig)#, mode='linear_interp')
                    mu_smooth = convolve(mu_int, g, normalize_kernel=True, boundary='extend')
                    #fix = (r[indx] > 5 * psfsig * self.pixscale)
                    #mu_smooth[fix] = mu_int[fix] # replace with original values
                    mu[indx] = mu_smooth
                else:
                    mu[indx] = mu_int
        
        return mu

class SersicDoubleWaveModel(Fittable2DModel):
    """
    Define a surface brightness profile model which is three double Sersic
    models connected by two Sersic indices and half-light radiii which vary
    as a power-law function of wavelength.

    """
    from astropy.modeling import Parameter
    
    nref1 = Parameter(default=3, bounds=(0.1, 8))
    nref2 = Parameter(default=1, bounds=(0.1, 8))

    r50ref1 = Parameter(default=3, bounds=(0.1, 100)) # [arcsec]
    r50ref2 = Parameter(default=10, bounds=(0.1, 100)) # [arcsec]

    alpha1 = Parameter(default=0.0, bounds=(-1, 1))
    alpha2 = Parameter(default=0.0)#, bounds=(-1, 1))

    beta1 = Parameter(default=0.0, bounds=(-1, 1))
    beta2 = Parameter(default=0.0)#, bounds=(-1, 1))

    mu50_g1 = Parameter(default=1.0, bounds=(0, 1e4))
    mu50_r1 = Parameter(default=1.0, bounds=(0, 1e4))
    mu50_z1 = Parameter(default=1.0, bounds=(0, 1e4))

    mu50_g2 = Parameter(default=0.1, bounds=(0, 1e4))
    mu50_r2 = Parameter(default=0.1, bounds=(0, 1e4))
    mu50_z2 = Parameter(default=0.1, bounds=(0, 1e4))

    linear = False
    
    def __init__(self, nref1=nref1.default, nref2=nref2.default,
                 r50ref1=r50ref1.default, r50ref2=r50ref2.default, 
                 alpha1=alpha1.default, alpha2=alpha2.default,
                 beta1=beta1.default, beta2=beta2.default, 
                 mu50_g1=mu50_g1.default, mu50_r1=mu50_r1.default, mu50_z1=mu50_z1.default, 
                 mu50_g2=mu50_g2.default, mu50_r2=mu50_r2.default, mu50_z2=mu50_z2.default, 
                 psfsigma_g=0.0, psfsigma_r=0.0, psfsigma_z=0.0, 
                 lambda_ref=6470, lambda_g=4890, lambda_r=6470, lambda_z=9196,
                 pixscale=0.262, seed=None, **kwargs):

        self.band = ('g', 'r', 'z')
        
        #from speclite import filters
        #filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z')
        #print(filt.effective_wavelengths.value)
        
        self.lambda_g = lambda_g
        self.lambda_r = lambda_r
        self.lambda_z = lambda_z
        self.lambda_ref = lambda_ref
        
        self.psfsigma_g = psfsigma_g
        self.psfsigma_r = psfsigma_r
        self.psfsigma_z = psfsigma_z

        self.pixscale = pixscale
        self.seed = seed
        
        super(SersicDoubleWaveModel, self).__init__(nref1=nref1, nref2=nref2, r50ref1=r50ref1, r50ref2=r50ref2,
                                                    alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2,
                                                    mu50_g1=mu50_g1, mu50_r1=mu50_r1, mu50_z1=mu50_z1,
                                                    mu50_g2=mu50_g2, mu50_r2=mu50_r2, mu50_z2=mu50_z2,
                                                    **kwargs)
        
    def get_sersicn(self, nref, lam, alpha):
        return nref * (lam / self.lambda_ref)**alpha
    
    def get_r50(self, r50ref, lam, beta):
        return r50ref * (lam / self.lambda_ref)**beta
    
    def evaluate(self, r, w, nref1, nref2, r50ref1, r50ref2, alpha1, alpha2,
                 beta1, beta2, mu50_g1, mu50_r1, mu50_z1, mu50_g2, mu50_r2, mu50_z2):
        """Evaluate the wavelength-dependent double-Sersic model.
        
        """
        from scipy.special import gammaincinv
        from astropy.convolution import Gaussian1DKernel, convolve
        
        mu = np.zeros_like(r)
        
        # Build the surface brightness profile at each wavelength.
        for lam, psfsig, mu50_1, mu50_2 in zip( (self.lambda_g, self.lambda_r, self.lambda_z),
                                                (self.psfsigma_g, self.psfsigma_r, self.psfsigma_z),
                                                (mu50_g1, mu50_r1, mu50_z1), (mu50_g2, mu50_r2, mu50_z2)  ):
            
            n1 = self.get_sersicn(nref1, lam, alpha1)
            n2 = self.get_sersicn(nref2, lam, alpha2)
            r50_1 = self.get_r50(r50ref1, lam, beta1)
            r50_2 = self.get_r50(r50ref2, lam, beta2)
            
            indx = w == lam
            if np.sum(indx) > 0:
                mu_int = ( mu50_1 * np.exp(-gammaincinv(2 * n1, 0.5) * ((r[indx] / r50_1) ** (1 / n1) - 1)) +
                           mu50_2 * np.exp(-gammaincinv(2 * n2, 0.5) * ((r[indx] / r50_2) ** (1 / n2) - 1)) )
            
                # smooth with the PSF
                if psfsig > 0:
                    g = Gaussian1DKernel(stddev=psfsig)#, mode='linear_interp')#, 
                    mu_smooth = convolve(mu_int, g, normalize_kernel=True, boundary='extend')

                    #import matplotlib.pyplot as plt
                    #plt.plot(r[indx], mu_smooth/mu_int) ; plt.show()
                    #plt.plot(r[indx], mu_int) ; plt.plot(r[indx], mu_smooth) ; plt.yscale('log') ; plt.show()
                    #pdb.set_trace()
                    
                    #fix = (r[indx] > 5 * psfsig * self.pixscale)
                    #mu_smooth[fix] = mu_int[fix] # replace with original values
                    mu[indx] = mu_smooth
                else:
                    mu[indx] = mu_int
        
        return mu

class SersicWaveFit(object):
    def __init__(self, ellipsefit, seed=None, minerr=0.01, nradius_uniform=150):

        from astropy.modeling import fitting
        self.rand = np.random.RandomState(seed)
        
        # initialize the fitter
        self.fitter = fitting.LevMarLSQFitter()

        refband, pixscale, redshift = ellipsefit['refband'], ellipsefit['pixscale'], ellipsefit['redshift']
        
        # parse the input sbprofile into the format that SersicSingleWaveModel()
        # expects; also interpolate the surface brightness profile onto a
        # uniform radius grid (in arcsec) so the Gaussian PSF convolution will
        # behave.
        sb, sberr, wave, radius = [], [], [], []
        sb_uniform, sberr_uniform, wave_uniform, radius_uniform = [], [], [], []
        for band, lam in zip( self.initfit.band, (self.initfit.lambda_g, 
                                                  self.initfit.lambda_r, 
                                                  self.initfit.lambda_z) ):
            # any quality cuts on stop_code here?!?
            #rad = ellipsefit[band].sma * pixscale # semi-major axis [arcsec]
            
            _radius = ellipsefit[band].sma * np.sqrt(1 - ellipsefit[band].eps) * pixscale # circularized radius [arcsec]
            _sb = ellipsefit[band].intens
            _sberr = np.sqrt( ellipsefit[band].int_err**2 + (0.4 * np.log(10) * _sb * minerr)**2 ) # minimum uncertainty

            radius.append(_radius)
            sb.append(_sb)
            sberr.append(_sberr)
            wave.append( np.repeat(lam, len(_radius)) )

            # interpolate onto a regular radius grid (in arcsec!)
            _radius_uniform = np.linspace( _radius.min(), _radius.max(), nradius_uniform )
            _sb_uniform = np.interp(_radius_uniform, _radius, _sb)
            _sberr_uniform = np.sqrt(np.interp(_radius_uniform, _radius, _sberr**2))

            radius_uniform.append( _radius_uniform )
            sb_uniform.append( _sb_uniform )
            sberr_uniform.append( _sberr_uniform )
            wave_uniform.append( np.repeat(lam, len(_radius_uniform)) )

            #plt.scatter(_radius, _sb)
            #plt.plot(_radius_uniform, _sb_uniform, color='orange', alpha=0.5)
            #plt.yscale('log')
            #plt.show()
            
        self.sb = np.hstack(sb)
        self.sberr = np.hstack(sberr)
        self.wave = np.hstack(wave)
        self.radius = np.hstack(radius)
        
        self.sb_uniform = np.hstack(sb_uniform)
        self.sberr_uniform = np.hstack(sberr_uniform)
        self.wave_uniform = np.hstack(wave_uniform)
        self.radius_uniform = np.hstack(radius_uniform)

        self.redshift = redshift
        self.minerr = minerr
        self.pixscale = pixscale
        self.seed = seed

    def chi2(self, bestfit):
        dof = len(self.sb_uniform) - len(bestfit.parameters)
        sbmodel = bestfit(self.radius_uniform, self.wave_uniform)
        chi2 = np.sum( (self.sb_uniform - sbmodel)**2 / self.sberr_uniform**2 ) / dof
        return chi2
    
    def integrate(self, bestfit, nrad=50):
        """OBSOLETE -- this functionality is now in legacyhalos-results

        Integrated the data and the model to get the final photometry.
        
        flux_obs_[grz] : observed integrated flux
        flux_int_[grz] : integrated (extrapolated) flux
        deltamag_in_[grz] : flux extrapolated inward
        deltamag_out_[grz] : flux extrapolated outward
        deltamag_[grz] : magnitude correction between flux_obs_[grz] and flux_int_[grz] or
          deltamag_in_[grz] + deltamag_out_[grz]
        
        """
        from scipy import integrate
        from astropy.table import Table, Column

        phot = Table()
        [phot.add_column(Column(name='flux_obs_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]
        [phot.add_column(Column(name='flux_obs_ivar_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]

        [phot.add_column(Column(name='flux_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]
        [phot.add_column(Column(name='flux_ivar_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]
        
        [phot.add_column(Column(name='dm_in_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]
        [phot.add_column(Column(name='dm_out_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]
        [phot.add_column(Column(name='dm_{}'.format(bb), dtype='f4', length=1)) for bb in self.initfit.band]

        for band, lam in zip( self.initfit.band, (self.initfit.lambda_g, 
                                                  self.initfit.lambda_r, 
                                                  self.initfit.lambda_z) ):
            wave = np.repeat(lam, nrad)
            indx = (self.wave == lam) * np.isfinite(self.sb) * (self.sb > 0)
            
            rad = self.radius[indx]
            sb = self.sb[indx]
            sberr = self.sberr[indx]
            
            obsflux = 2 * np.pi * integrate.simps(x=rad, y=rad*sb)
            obsvar = 2 * np.pi * integrate.simps(x=rad, y=rad*sberr**2)

            phot['flux_obs_{}'.format(band)] = obsflux
            if obsvar > 0:
                phot['flux_obs_ivar_{}'.format(band)] = 1/obsvar

            # now integrate inward and outward by evaluating the model
            rad_in = np.linspace(0, rad.min(), nrad)
            sb_in = bestfit(rad_in, wave) # no-convolution!!!
            dm_in = 2 * np.pi * integrate.simps(x=rad_in, y=rad_in*sb_in)
            
            #rad_out = np.logspace(np.log10(rad.max()), 3, nrad)
            rad_out = np.linspace(rad.max()*0.7, 200, 150) # nrad)
            sb_out = bestfit(rad_out, wave)
            dm_out = 2 * np.pi * integrate.simps(x=rad_out, y=rad_out*sb_out)

            #plt.errorbar(rad, 22.5-2.5*np.log10(sb), 2.5*sberr/sb/np.log(10))
            #plt.plot(rad_in, 22.5-2.5*np.log10(sb_in)) ; plt.scatter(rad_out, 22.5-2.5*np.log10(sb_out))
            #plt.ylim(32, 15) ; plt.xlim(0, 30)
            #plt.show()
            #pdb.set_trace()
            
            dm = dm_in + dm_out
            phot['flux_{}'.format(band)] = phot['flux_obs_{}'.format(band)] + dm
            phot['flux_ivar_{}'.format(band)] = phot['flux_obs_ivar_{}'.format(band)] + dm
            
            phot['dm_in_{}'.format(band)] = - 2.5 * np.log10(1 - dm_in / obsflux)
            phot['dm_out_{}'.format(band)] = - 2.5 * np.log10(1 - dm_out / obsflux)
            phot['dm_{}'.format(band)] = - 2.5 * np.log10(1 - dm / obsflux)

        return phot

    def _fit(self, nball=10, chi2fail=1e6, verbose=False, model='single'):
        """Perform the chi2 minimization.
        
        """
        import warnings
        if verbose:
            warnvalue = 'always'
        else:
            warnvalue = 'ignore'

        # initialize the output dictionary
        result = {
            'success': False,
            'converged': False,
            'redshift': self.redshift,
            'radius': self.radius,
            'wave': self.wave,
            'sb': self.sb,
            'sberr': self.sberr,
            'band': self.initfit.band,
            'lambda_ref': self.initfit.lambda_ref,
            'lambda_g': self.initfit.lambda_g,
            'lambda_r': self.initfit.lambda_r,
            'lambda_z': self.initfit.lambda_z,
            'params': self.initfit.param_names,
            'chi2': chi2fail, # initial value
            'dof': len(self.sb) - len(self.initfit.parameters),
            'minerr': self.minerr,
            'pixscale': self.pixscale,
            'seed': self.seed,
            }

        # perturb the parameter values
        nparams = len(self.initfit.parameters)
        params = np.repeat(self.initfit.parameters, nball).reshape(nparams, nball)
        for ii, pp in enumerate(self.initfit.param_names):
            pinfo = getattr(self.initfit, pp)
            if not pinfo.fixed: # don't touch fixed parameters
                if pinfo.bounds[0] is not None:
                    #params[ii, :] = self.rand.uniform(pinfo.bounds[0], pinfo.bounds[1], nball)
                    if pinfo.default == 0:
                        scale = 0.1 * (pinfo.bounds[1] - pinfo.bounds[0])
                    else:
                        scale = 0.2 * pinfo.default
                    params[ii, :] += self.rand.normal(scale=scale, size=nball)
                    toosmall = np.where( params[ii, :] < pinfo.bounds[0] )[0]
                    if len(toosmall) > 0:
                        params[ii, toosmall] = pinfo.default
                    toobig = np.where( params[ii, :] > pinfo.bounds[1] )[0]
                    if len(toobig) > 0:
                        params[ii, toobig] = pinfo.default
                    #if ii == 2:
                    #    print(params[ii, :])
                    #    pdb.set_trace()                              
                else:
                    params[ii, :] += self.rand.normal(scale=0.2 * pinfo.default, size=nball)
        #print(params)
        #pdb.set_trace()
            
        # perform the fit nball times
        with warnings.catch_warnings():
            warnings.simplefilter(warnvalue)

            ## interpolate!
            #radius, wave, sb, sberr = [], [], [], []
            #for lam in (self.initfit.lambda_g, self.initfit.lambda_r, self.initfit.lambda_z):
            #    ww = (self.wave == lam)
            #    if np.sum(ww) > 0:
            #        rmin, rmax = self.radius[ww].min(), self.radius[ww].max()
            #        rr = np.linspace(rmin, rmax, nrr)
            #        radius.append(rr)
            #        wave.append(np.repeat(lam, nrr))
            #        sb.append( np.interp(rr, self.radius[ww], self.sb[ww]) )
            #        sberr.append( np.sqrt(np.interp(rr, self.radius[ww], self.sberr[ww]**2)) )
            #
            #sb = np.hstack(sb)
            #sberr = np.hstack(sberr)
            #wave = np.hstack(wave)
            #radius = np.hstack(radius)

            chi2 = np.zeros(nball) + chi2fail
            for jj in range(nball):
                self.initfit.parameters = params[:, jj]
                ballfit = self.fitter(self.initfit, self.radius_uniform, self.wave_uniform,
                                      self.sb_uniform, weights=1/self.sberr_uniform, maxiter=200)
                #ballfit = self.fitter(self.initfit, self.radius, self.wave, self.sb,
                #                      weights=1/self.sberr, maxiter=200)
                chi2[jj] = self.chi2(ballfit)
                if self.fitter.fit_info['param_cov'] is None: # failed
                    if verbose:
                        print(jj, self.fitter.fit_info['message'], chi2[jj])
                else:
                    params[:, jj] = ballfit.parameters # update

        # did at least one fit succeed?
        good = chi2 < chi2fail
        if np.sum(good) == 0:
            print('{}-Sersic fitting failed.'.format(model.upper()))
            result.update({'fit_message': self.fitter.fit_info['message']})
            return result

        # otherwise, re-evaluate the model at the chi2 minimum
        result['success'] = True
        mindx = np.argmin(chi2)

        self.initfit.parameters = params[:, mindx]
        bestfit = self.fitter(self.initfit, self.radius_uniform, self.wave_uniform,
                              self.sb_uniform, weights=1/self.sberr_uniform)
        #bestfit = self.fitter(self.initfit, self.radius, self.wave, self.sb, weights= 1 / self.sberr)
        minchi2 = chi2[mindx]
        print('{} Sersic fitting succeeded with a chi^2 minimum of {:.2f}'.format(model.upper(), minchi2))

        ## Integrate the data and model over various apertures.
        #phot = self.integrate(bestfit)
        
        # Pack the results in a dictionary and return.
        # https://gist.github.com/eteq/1f3f0cec9e4f27536d52cd59054c55f2
        if self.fitter.fit_info['param_cov'] is not None:
            cov = self.fitter.fit_info['param_cov']
            unc = np.diag(cov)**0.5
            result['converged'] = True
        else:
            cov = np.zeros( (nparams, nparams) )
            unc = np.zeros(nparams)

        count = 0
        for ii, pp in enumerate(bestfit.param_names):
            pinfo = getattr(bestfit, pp)
            result.update({bestfit.param_names[ii]: bestfit.parameters[ii]})
            if pinfo.fixed:
                result.update({bestfit.param_names[ii]+'_err': 0.0})
            elif pinfo.tied:
                pass # hack! see https://github.com/astropy/astropy/issues/7202
            else:
                result.update({bestfit.param_names[ii]+'_err': unc[count]})
                count += 1

        # Fix the uncertainties of tied parameters.  Very hacky here!
        if 'alpha2' in bestfit.param_names and 'alpha1' in bestfit.param_names:
            if bestfit.alpha2.tied is not False:
                result.update({'alpha2_err': result['alpha1_err']})
        if 'beta2' in bestfit.param_names and 'beta1' in bestfit.param_names:
            if bestfit.beta2.tied is not False:
                result.update({'beta2_err': result['beta1_err']})

        result['chi2'] = minchi2
        result.update({
            #'values': bestfit.parameters,
            #'uncertainties': np.diag(cov)**0.5,
            'cov': cov,
            'bestfit': bestfit,
            'fit_message': self.fitter.fit_info['message'],
            #'phot': phot,
        })
        
        return result

class SersicSingleWaveFit(SersicWaveFit):
    """Fit surface brightness profiles with the SersicSingleWaveModel model.""" 
   
    def __init__(self, ellipsefit, minerr=0.01, fix_alpha=False, fix_beta=False, seed=None):
        self.fixed = {'alpha': fix_alpha, 'beta': fix_beta}
        self.initfit = SersicSingleWaveModel(fixed=self.fixed, 
                                             psfsigma_g=ellipsefit['psfsigma_g'],
                                             psfsigma_r=ellipsefit['psfsigma_r'],
                                             psfsigma_z=ellipsefit['psfsigma_z'],
                                             pixscale=ellipsefit['pixscale'],
                                             seed=seed)

        super(SersicSingleWaveFit, self).__init__(ellipsefit, seed=seed)

    def fit(self, nball=10, chi2fail=1e6, verbose=False):

        return self._fit(nball=10, chi2fail=1e6, verbose=verbose, model='single')

class SersicExponentialWaveFit(SersicWaveFit):
    """Fit surface brightness profiles with the SersicExponentialWaveModel model."""
    
    def __init__(self, ellipsefit, minerr=0.01, fix_alpha=False, fix_beta=False, seed=None):

        self.fixed = {'alpha1': fix_alpha, 'beta1': fix_beta, 'beta2': fix_beta}
        #tied = {'r50ref2': self.tie_r50ref2}
        tied = {'beta2': self.tie_beta2}
        
        self.initfit = SersicExponentialWaveModel(fixed=self.fixed, tied=tied,
                                             psfsigma_g=ellipsefit['psfsigma_g'],
                                             psfsigma_r=ellipsefit['psfsigma_r'],
                                             psfsigma_z=ellipsefit['psfsigma_z'],
                                             pixscale=ellipsefit['pixscale'],
                                             seed=seed)

        super(SersicExponentialWaveFit, self).__init__(ellipsefit, seed=seed)

    def tie_beta2(self, model):
        return model.beta1
        
    def tie_r50ref2(self, model):
        if model.r50ref2 < model.r50ref1:
            return model.r50ref1*1.05
        
    def fit(self, nball=10, chi2fail=1e6, verbose=False):
        return self._fit(nball=10, chi2fail=1e6, verbose=verbose, model='exponential')

class SersicDoubleWaveFit(SersicWaveFit):
    """Fit surface brightness profiles with the SersicDoubleWaveModel model."""
    
    def __init__(self, ellipsefit, minerr=0.01, fix_alpha=False, fix_beta=False, seed=None):

        self.fixed = {'alpha1': fix_alpha, 'alpha2': fix_alpha, 'beta1': fix_beta, 'beta2': fix_beta}
        tied = {'alpha2': self.tie_alpha2, 'beta2': self.tie_beta2}

        self.initfit = SersicDoubleWaveModel(fixed=self.fixed, tied=tied,
                                             psfsigma_g=ellipsefit['psfsigma_g'],
                                             psfsigma_r=ellipsefit['psfsigma_r'],
                                             psfsigma_z=ellipsefit['psfsigma_z'],
                                             pixscale=ellipsefit['pixscale'],
                                             seed=seed)

        super(SersicDoubleWaveFit, self).__init__(ellipsefit, seed=seed)

    def tie_alpha2(self, model):
        return model.alpha1
        
    def tie_beta2(self, model):
        return model.beta1
        
    def fit(self, nball=10, chi2fail=1e6, verbose=False):

        return self._fit(nball=10, chi2fail=1e6, verbose=verbose, model='double')

def sersic_single(objid, objdir, ellipsefit, minerr=0.01, seed=None,
                  nowavepower=False, nowrite=False, verbose=False):
    """Wrapper to fit a single Sersic model to an input surface brightness profile.

    nowavepower : no wavelength-dependent variation in the Sersic index or
      half-light radius

    """
    if nowavepower:
        model = 'single-nowavepower'
        sersic = SersicSingleWaveFit(ellipsefit, minerr=minerr, fix_alpha=True,
                                     fix_beta=True, seed=seed)
    else:
        model = 'single'
        sersic = SersicSingleWaveFit(ellipsefit, minerr=minerr, fix_alpha=False,
                                     fix_beta=False, seed=seed)
        
    sersic = sersic.fit(verbose=verbose)

    if not nowrite:
        legacyhalos.io.write_sersic(objid, objdir, sersic, model=model, verbose=verbose)

    return sersic

def sersic_exponential(objid, objdir, ellipsefit, minerr=0.01, seed=None,
                       nowavepower=False, nowrite=False, verbose=False):
    """Wrapper to fit a Sersic+exponential model to an input surface brightness
    profile.

    nowavepower : no wavelength-dependent variation in the Sersic index or
      half-light radius

    """
    if nowavepower:
        model = 'exponential-nowavepower'
        sersic = SersicExponentialWaveFit(ellipsefit, minerr=minerr, fix_alpha=True,
                                          fix_beta=True, seed=seed)
    else:
        model = 'exponential'
        sersic = SersicExponentialWaveFit(ellipsefit, minerr=minerr, fix_alpha=False,
                                          fix_beta=False, seed=seed)
        
    sersic = sersic.fit(verbose=verbose)

    if not nowrite:
        legacyhalos.io.write_sersic(objid, objdir, sersic, model=model, verbose=verbose)

    return sersic

def sersic_double(objid, objdir, ellipsefit, minerr=0.01, seed=None,
                  nowavepower=False, nowrite=False, verbose=False):
    """Wrapper to fit a double Sersic model to an input surface brightness profile. 

    nowavepower : no wavelength-dependent variation in the Sersic index or
      half-light radius

    """
    if nowavepower:
        model = 'double-nowavepower'
        sersic = SersicDoubleWaveFit(ellipsefit, minerr=minerr, fix_alpha=True,
                                     fix_beta=True, seed=None)
    else:
        model = 'double'
        sersic = SersicDoubleWaveFit(ellipsefit, minerr=minerr, fix_alpha=False,
                                     fix_beta=False, seed=None)
        
    sersic = sersic.fit(verbose=verbose)

    if not nowrite:
        legacyhalos.io.write_sersic(objid, objdir, sersic, model=model, verbose=verbose)

    return sersic

def legacyhalos_sersic(sample, objid=None, objdir=None, minerr=0.02, seed=None,
                       verbose=False, debug=False):
    """Top-level wrapper script to model the measured surface-brightness profiles
    with various Sersic models.

    """ 
    #from legacyhalos.ellipse import ellipse_sbprofile

    if objid is None and objdir is None:
        objid, objdir = get_objid(sample)

    # Read the ellipse-fitting results and 
    ellipsefit = legacyhalos.io.read_ellipsefit(objid, objdir)
    if bool(ellipsefit):
        if ellipsefit['success']:

            # Sersic-exponential fit with and without wavelength dependence
            serexp = sersic_exponential(objid, objdir, ellipsefit, minerr=minerr,
                                        verbose=verbose, seed=seed)

            serexp_nowave = sersic_exponential(objid, objdir, ellipsefit, minerr=minerr,
                                               verbose=verbose, nowavepower=True, seed=seed)

            # double Sersic fit with and without wavelength dependence
            double = sersic_double(objid, objdir, ellipsefit, minerr=minerr,
                                   verbose=verbose, seed=seed)
            double_nowave = sersic_double(objid, objdir, ellipsefit, minerr=minerr,
                                          verbose=verbose, nowavepower=True, seed=seed)

            # single Sersic fit with and without wavelength dependence
            single = sersic_single(objid, objdir, ellipsefit, minerr=minerr,
                                   verbose=verbose, seed=seed)
            single_nowave = sersic_single(objid, objdir, ellipsefit, minerr=minerr,
                                          verbose=verbose, nowavepower=True, seed=seed)

            if single['success']:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
