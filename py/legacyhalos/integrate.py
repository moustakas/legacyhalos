"""
legacyhalos.integrate
=====================

Code to integrate the surface brightness profiles, including extrapolation.

"""
import os, warnings, pdb
import multiprocessing
import numpy as np

from astropy.table import Table, Column, vstack
    
import legacyhalos.io
import legacyhalos.misc

def _init_phot(nrad_uniform=50, ngal=1, band=('g', 'r', 'z')):
    """Initialize the output photometry table.

    """
    phot = Table()

    [phot.add_column(Column(name='RMAX_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    
    [phot.add_column(Column(name='FLUX10_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUX30_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUX100_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUXRMAX_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    
    [phot.add_column(Column(name='FLUX10_IVAR_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUX30_IVAR_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUX100_IVAR_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='FLUXRMAX_IVAR_{}'.format(bb.upper()), dtype='f4', length=ngal)) for bb in band]

    phot.add_column(Column(name='RAD', dtype='f4', length=ngal, shape=(nrad_uniform,)))
    phot.add_column(Column(name='RAD_AREA', dtype='f4', length=ngal, shape=(nrad_uniform,)))
    [phot.add_column(Column(name='FLUXRAD_{}'.format(bb.upper()), dtype='f4', length=ngal, shape=(nrad_uniform,))) for bb in band]
    [phot.add_column(Column(name='FLUXRAD_IVAR_{}'.format(bb.upper()), dtype='f4', length=ngal, shape=(nrad_uniform,))) for bb in band]

    return phot

def _dointegrate(radius, sb, sberr, rmin=None, rmax=None, band='r'):
    """Do the actual profile integration.

    """
    from scipy import integrate
    from scipy.interpolate import interp1d

    # Evaluate the profile at r=rmin
    if rmin is None:
        rmin = 0.0
        sberr_rmin = sberr[0]
    else:
        sberr_rmin = interp1d(radius, sberr, kind='linear', fill_value='extrapolate')(rmin)
        
    sb_rmin = interp1d(radius, sb, kind='quadratic', fill_value='extrapolate')(rmin)

    if rmax is None:
        rmax = radius.max() # [kpc]

    if rmax > radius.max():
        return 0.0, 0.0, 0.0 # do not extrapolate outward
    else:
        # Interpolate the last point to the desired rmax
        sb_rmax = interp1d(radius, sb, kind='linear')(rmax)
        sberr_rmax = np.sqrt(interp1d(radius, sberr**2, kind='linear')(rmax))

        keep = np.where((radius > rmin) * (radius < rmax))[0]
        nkeep = len(keep)

        _radius = np.insert(radius[keep], [0, nkeep], [rmin, rmax])
        _sb = np.insert(sb[keep], [0, nkeep], [sb_rmin, sb_rmax])
        _sberr = np.insert(sberr[keep], [0, nkeep], [sberr_rmin, sberr_rmax])

        # Integrate!
        flux = 2 * np.pi * integrate.simps(x=_radius, y=_radius*_sb)    # [nanomaggies]
        ferr = 2 * np.pi * integrate.simps(x=_radius, y=_radius*_sberr) # [nanomaggies]

        if band == 'r':
            area = 2 * np.pi * integrate.simps(x=_radius, y=_radius) # [kpc2]
        else:
            area = 0.0
        
        if flux < 0 or ferr < 0 or np.isnan(flux) or np.isnan(ferr):
            #print('Negative or infinite flux or variance in band {}'.format(band))
            return 0.0, 0.0, 0.0
        else:
            return flux, 1/ferr**2, area

def _integrate_one(args):
    """Wrapper for the multiprocessing."""
    return integrate_one(*args)

def integrate_one(galaxy, galaxydir, phot=None, minerr=0.01, nrad_uniform=50):
    """Integrate over various radial ranges.

    """
    if phot is None:
        phot = _init_phot(ngal=1, nrad_uniform=nrad_uniform)
    phot = Table(phot)

    print(galaxy)
    ellipsefit = legacyhalos.io.read_ellipsefit(galaxy, galaxydir)
    
    if ellipsefit['success'] == False:
        return phot

    allband, pixscale = ellipsefit['band'], ellipsefit['pixscale']
    arcsec2kpc = legacyhalos.misc.arcsec2kpc(ellipsefit['redshift']) # [kpc/arcsec]
    
    def _get_sbprofile(ellipsefit, band, minerr=0.01, snrmin=1):
        radius = ellipsefit[band].sma * np.sqrt(1 - ellipsefit[band].eps) * pixscale * arcsec2kpc # [kpc]
        sb = ellipsefit[band].intens / arcsec2kpc**2 # [nanomaggies/kpc**2]
        sberr = np.sqrt( (ellipsefit[band].int_err/arcsec2kpc**2)**2 + (0.4 * np.log(10) * sb * minerr)**2 )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            good = np.isfinite(sb) * (sb / sberr > snrmin)
            #print('Keeping {} / {} measurements in band {}'.format(np.sum(good), len(radius), band))
            
        return radius[good], sb[good], sberr[good]

    # First integrate to r=10, 30, 100, and max kpc.
    min_r, max_r = [], []
    for band in allband:
        radius, sb, sberr = _get_sbprofile(ellipsefit, band, minerr=minerr)

        min_r.append(radius.min())
        max_r.append(radius.max())

        for rmax in (10, 30, 100, None):
            obsflux, obsivar, _ = _dointegrate(radius, sb, sberr, rmax=rmax, band=band)

            if rmax is not None:
                fkey = 'FLUX{}_{}'.format(rmax, band.upper())
                ikey = 'FLUX{}_IVAR_{}'.format(rmax, band.upper())
            else:
                fkey = 'FLUXRMAX_{}'.format(band.upper())
                ikey = 'FLUXRMAX_IVAR_{}'.format(band.upper())

            phot[fkey] = obsflux
            phot[ikey] = obsivar
        phot['RMAX_{}'.format(band.upper())] = radius.max()

    # Now integrate over fixed apertures to get the differential flux. 
    min_r, max_r = np.min(min_r), np.max(max_r)
    rad_uniform = 10**np.linspace(np.log10(min_r), np.log10(max_r), nrad_uniform+1)
    rmin_uniform, rmax_uniform = rad_uniform[:-1], rad_uniform[1:]
    phot['RAD'][:] = (rmax_uniform - rmin_uniform) / 2 + rmin_uniform
    
    for band in allband:
        radius, sb, sberr = _get_sbprofile(ellipsefit, band, minerr=minerr)
        
        for ii, (rmin, rmax) in enumerate(zip(rmin_uniform, rmax_uniform)):
            obsflux, obsivar, obsarea = _dointegrate(radius, sb, sberr, rmin=rmin, rmax=rmax, band=band)
            #print(band, ii, rmin, rmax, 22.5-2.5*np.log10(obsflux), obsarea)

            if band == 'r':
                phot['RAD_AREA'][0][ii] = obsarea

            phot['FLUXRAD_{}'.format(band.upper())][0][ii] = obsflux
            phot['FLUXRAD_IVAR_{}'.format(band.upper())][0][ii] = obsivar
            
    return phot

def legacyhalos_integrate(sample=None, first=None, last=None, nproc=1,
                          minerr=0.01, nrad_uniform=50, verbose=False,
                          clobber=False):
    """Wrapper script to integrate the profiles for the full sample.

    """
    if sample is None:
        sample = legacyhalos.io.read_paper2_sample(first=first, last=last)
    ngal = len(sample)

    phot = _init_phot(ngal=ngal, nrad_uniform=nrad_uniform)
    galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(sample)
    galaxy, galaxydir = np.atleast_1d(galaxy), np.atleast_1d(galaxydir)

    args = list()
    for ii in range(ngal):
        args.append((galaxy[ii], galaxydir[ii], phot[ii], minerr, nrad_uniform))

    # Divide the sample by cores.
    if nproc > 1:
        pool = multiprocessing.Pool(nproc)
        out = pool.map(_integrate_one, args)
    else:
        out = list()
        for _args in args:
            out.append(_integrate_one(_args))
            
    results = vstack(out)

    return results

