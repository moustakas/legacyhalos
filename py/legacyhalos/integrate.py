"""
legacyhalos.integrate
=====================

Code to integrate the surface brightness profiles, including extrapolation.

"""
import os, warnings, pdb
import multiprocessing
import numpy as np

from scipy.interpolate import interp1d
from astropy.table import Table, Column, vstack, hstack
    
import legacyhalos.io
import legacyhalos.misc
import legacyhalos.hsc
import legacyhalos.ellipse

def _init_phot(nrad_uniform=30, ngal=1, band=('g', 'r', 'z')):
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

    if len(radius) < 10:
        return 0.0, 0.0, 0.0 # need at least 10 points

    # Evaluate the profile at r=rmin
    if rmin is None:
        rmin = 0.0
        sberr_rmin = sberr[0]
    else:
        sberr_rmin = interp1d(radius, sberr, kind='linear', fill_value='extrapolate')(rmin)

    sb_rmin = interp1d(radius, sb, kind='quadratic', fill_value='extrapolate')(rmin)

    if rmax is None:
        rmax = radius.max() # [kpc]

    if rmax > radius.max() or rmax < radius.min():
        return 0.0, 0.0, 0.0 # do not extrapolate outward
    else:
        # Interpolate the last point to the desired rmax
        #if band == 'z':
        #    pdb.set_trace()
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

def integrate_one(galaxy, galaxydir, phot=None, minerr=0.01, snrmin=1,
                  nrad_uniform=30, count=1):
    """Integrate over various radial ranges.

    """
    if phot is None:
        phot = _init_phot(ngal=1, nrad_uniform=nrad_uniform)
    phot = Table(phot)

    print(count, galaxy, nrad_uniform)
    ellipsefit = legacyhalos.io.read_ellipsefit(galaxy, galaxydir)
    if not bool(ellipsefit) or ellipsefit['success'] == False:
        return phot

    sbprofile = legacyhalos.ellipse.ellipse_sbprofile(ellipsefit, minerr=minerr,
                                                      snrmin=snrmin, linear=True)
    allband, refpixscale = ellipsefit['bands'], ellipsefit['refpixscale']
    arcsec2kpc = legacyhalos.misc.arcsec2kpc(ellipsefit['redshift']) # [kpc/arcsec]
    
    def _get_sbprofile(sbprofile, band, minerr=0.01, snrmin=1):
        sb = sbprofile['mu_{}'.format(band)] / arcsec2kpc**2       # [nanomaggies/kpc2]
        sberr = sbprofile['muerr_{}'.format(band)] / arcsec2kpc**2 # [nanomaggies/kpc2]
        radius = sbprofile['radius_{}'.format(band)] * arcsec2kpc  # [kpc]
        return radius, sb, sberr

    # First integrate to r=10, 30, 100, and max kpc.
    min_r, max_r = [], []
    for band in allband:
        radius, sb, sberr = _get_sbprofile(sbprofile, band, minerr=minerr, snrmin=snrmin)
        if len(radius) == 0:
            continue

        min_r.append(radius.min())
        max_r.append(radius.max())

        for rmax in (10, 30, 100, None):
            obsflux, obsivar, _ = _dointegrate(radius, sb, sberr, rmax=rmax, band=band)
            #ff = interp1d(radius, np.cumsum(sb), kind='linear')(rmax)
            
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
    if len(min_r) == 0:
        return phot
    
    min_r, max_r = np.min(min_r), np.max(max_r)
    if False: 
        rad_uniform = 10**np.linspace(np.log10(min_r), np.log10(max_r), nrad_uniform+1) # log-spacing
    else: 
        rad_uniform = np.linspace(min_r**0.25, max_r**0.25, nrad_uniform+1)**4 # r^1/4 spacing
    rmin_uniform, rmax_uniform = rad_uniform[:-1], rad_uniform[1:]
    phot['RAD'][:] = (rmax_uniform - rmin_uniform) / 2 + rmin_uniform
    
    for band in allband:
        radius, sb, sberr = _get_sbprofile(sbprofile, band, minerr=minerr, snrmin=snrmin)
        for ii, (rmin, rmax) in enumerate(zip(rmin_uniform, rmax_uniform)):
            #if band == 'r' and ii == 49:
            #    pdb.set_trace()
            obsflux, obsivar, obsarea = _dointegrate(radius, sb, sberr, rmin=rmin, rmax=rmax, band=band)
            #print(band, ii, rmin, rmax, 22.5-2.5*np.log10(obsflux), obsarea)

            if band == 'r':
                phot['RAD_AREA'][0][ii] = obsarea

            phot['FLUXRAD_{}'.format(band.upper())][0][ii] = obsflux
            phot['FLUXRAD_IVAR_{}'.format(band.upper())][0][ii] = obsivar
            
    return phot

def legacyhalos_integrate(sample, galaxy=None, galaxydir=None, nproc=1,
                          minerr=0.01, snrmin=1, nrad_uniform=30,
                          columns=None, verbose=False, clobber=False):
    """Wrapper script to integrate the profiles for the full sample.

    columns - columns to include in the output table

    """
    ngal = len(sample)

    phot = _init_phot(ngal=ngal, nrad_uniform=nrad_uniform)

    if columns is None:
        columns = ['MEM_MATCH_ID', 'RA', 'DEC', 'Z_LAMBDA', 'LAMBDA_CHISQ', 'ID_CENT',
                   'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z']

    if galaxy is None and galaxydir is None:
        galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(sample)
    
    #if hsc:
    #    galaxy, galaxydir = legacyhalos.hsc.get_galaxy_galaxydir(sample)
    #    columns = ['ID_S16A', 'RA', 'DEC', 'Z_BEST']
    #else:
    #    columns = ['MEM_MATCH_ID', 'RA', 'DEC', 'Z_LAMBDA', 'LAMBDA_CHISQ', 'ID_CENT',
    #               'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z']
            
    integratedfile = legacyhalos.io.get_integrated_filename(hsc=hsc)
    if os.path.exists(integratedfile) and clobber is False:
        print('Output file {} exists; use --clobber.'.format(integratedfile))
        return []
            
    galaxy, galaxydir = np.atleast_1d(galaxy), np.atleast_1d(galaxydir)

    args = list()
    for ii in range(ngal):
        args.append((galaxy[ii], galaxydir[ii], phot[ii], minerr, snrmin, nrad_uniform, ii))

    # Divide the sample by cores.
    if nproc > 1:
        pool = multiprocessing.Pool(nproc)
        out = pool.map(_integrate_one, args)
    else:
        out = list()
        for _args in args:
            out.append(_integrate_one(_args))
    results = vstack(out)
    
    out = hstack((sample[columns], results))
    if verbose:
        print('Writing {}'.format(integratedfile))
    out.write(integratedfile, overwrite=True)

    return out
