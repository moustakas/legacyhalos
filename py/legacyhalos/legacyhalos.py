"""
legacyhalos.legacyhalos
=======================

Code to support the legacyhalos sample and project.

"""
import os, time, shutil, subprocess, pdb
import numpy as np
import astropy
import fitsio

import legacyhalos.io

RACOLUMN = 'RA'
DECCOLUMN = 'DEC'
ZCOLUMN = 'Z_LAMBDA'    
DIAMCOLUMN = 'RADIUS_MOSAIC' # [radius, arcsec]
RICHCOLUMN = 'LAMBDA_CHISQ'
GALAXYCOLUMN = 'CENTRALID'

RADIUS_CLUSTER_KPC = 400.0 # default cluster radius [kpc]

SBTHRESH = [23.0, 24.0, 25.0, 26.0] # surface brightness thresholds

def sample_dir():
    sdir = os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def smf_dir(figures=False, data=False):
    pdir = os.path.join(legacyhalos.io.legacyhalos_dir(), 'science', 'smf')
    if not os.path.isdir(pdir):
        os.makedirs(pdir, exist_ok=True)
    if figures:
        pdir = os.path.join(pdir, 'figures')
        if not os.path.isdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    if data:
        pdir = os.path.join(pdir, 'data')
        if not os.path.isdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    return pdir

def profiles_dir(figures=False, data=False):
    pdir = os.path.join(legacyhalos.io.legacyhalos_dir(), 'science', 'profiles')
    if not os.path.isdir(pdir):
        os.makedirs(pdir, exist_ok=True)
    if figures:
        pdir = os.path.join(pdir, 'figures')
        if not os.path.isdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    if data:
        pdir = os.path.join(pdir, 'data')
        if not os.path.isdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    return pdir

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from legacyhalos.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    #def get_healpix_subdir(nside, pixnum, datadir):
    #    subdir = os.path.join(str(pixnum // 100), str(pixnum))
    #    return os.path.abspath(os.path.join(datadir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        if candidates:
            galaxy = ['{:07d}-{:09d}'.format(cat['MEM_MATCH_ID'], cat['ID'])]
        else:
            galaxy = ['{:07d}-{:09d}'.format(cat['MEM_MATCH_ID'], cat['ID_CENT'][0])]
        pixnum = [radec2pix(nside, cat[RACOLUMN], cat[DECCOLUMN])]
    else:
        ngal = len(cat)
        if candidates:
            galaxy = np.array( ['{:07d}-{:09d}'.format(mid, cid)
                                for mid, cid in zip(cat['MEM_MATCH_ID'], cat['ID'])] )
        else:
            galaxy = np.array( ['{:07d}-{:09d}'.format(mid, cid)
                                for mid, cid in zip(cat['MEM_MATCH_ID'], cat['ID_CENT'][:, 0])] )

        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(datadir, str(pix), gal) for pix, gal in zip(pixnum, galaxy)])
    #galaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, datadir), gal)
    #                      for pix, gal in zip(pixnum, galaxy)])
    if html:
        #htmlgalaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, htmldir), gal)
        #                          for pix, gal in zip(pixnum, galaxy)])
        htmlgalaxydir = np.array([os.path.join(htmldir, str(pix), gal) for pix, gal in zip(pixnum, galaxy)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def missing_files(args, sample, size=1, clobber_overwrite=None):
    import multiprocessing
    from legacyhalos.io import _missing_files_one

    dependson = None
    if args.htmlplots is False and args.htmlindex is False:
        if args.verbose:
            t0 = time.time()
            print('Getting galaxy names and directories...', end='')
        galaxy, galaxydir = get_galaxy_galaxydir(sample)
        if args.verbose:
            print('...took {:.3f} sec'.format(time.time() - t0))
        
    if args.coadds:
        suffix = 'coadds'
        filesuffix = '-custom-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        #filesuffix = '-custom-skytest03-*-ellipse.fits'
        filesuffix = '-custom-ellipse.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.build_catalog:
        suffix = 'build-catalog'
        filesuffix = '-custom-catalog.isdone'
        dependson = '-custom-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-custom-grz-montage.png'
        else:
            filesuffix = '-ccdpos.png'
            #filesuffix = '-custom-maskbits.png'
            #dependson = '-custom-ellipse.isdone'
            dependson = None
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-custom-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        raise ValueError('Need at least one keyword argument.')

    # Make clobber=False for build_catalog and htmlindex because we're not
    # making the files here, we're just looking for them. The argument
    # args.clobber gets used downstream.
    if args.htmlindex:
        clobber = False
    elif args.build_catalog:
        clobber = True
    else:
        clobber = args.clobber

    if clobber_overwrite is not None:
        clobber = clobber_overwrite

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)

    pool = multiprocessing.Pool(args.nproc)
    missargs = []
    for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)):
        #missargs.append([gal, gdir, filesuffix, dependson, clobber])
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if dependson:
            missargs.append([checkfile, os.path.join(gdir, '{}{}'.format(gal, dependson)), clobber])
        else:
            missargs.append([checkfile, None, clobber])

    if args.verbose:
        t0 = time.time()
        print('Finding missing files...', end='')
    todo = np.array(pool.map(_missing_files_one, missargs))
    if args.verbose:
        print('...took {:.3f} min'.format((time.time() - t0)/60))

    itodo = np.where(todo == 'todo')[0]
    idone = np.where(todo == 'done')[0]
    ifail = np.where(todo == 'fail')[0]

    if len(ifail) > 0:
        fail_indices = [indices[ifail]]
    else:
        fail_indices = [np.array([])]

    if len(idone) > 0:
        done_indices = [indices[idone]]
    else:
        done_indices = [np.array([])]

    if len(itodo) > 0:
        _todo_indices = indices[itodo]

        # Assign the sample to ranks to make the diameter distribution per rank ~flat.

        # https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        cumuweight = weight.cumsum() / weight.sum()
        idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        if len(idx) < size: # can happen in corner cases or with 1 rank
            todo_indices = np.array_split(_todo_indices, size) # unweighted
        else:
            todo_indices = np.array_split(_todo_indices, idx) # weighted
        for ii in range(size): # sort by weight
            srt = np.argsort(sample[DIAMCOLUMN][todo_indices[ii]])
            todo_indices[ii] = todo_indices[ii][srt]
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--sdss', action='store_true', help='Analyze the SDSS galaxies.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the coadds with the custom sky.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the coadds with the pipeline sky.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py).')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--sersic', action='store_true', help='Perform Sersic fitting.')
    parser.add_argument('--integrate', action='store_true', help='Integrate the surface brightness profiles.')
    parser.add_argument('--sky', action='store_true', help='Estimate the sky variance.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed (used with --sky and --sersic).')
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--sdss-pixscale', default=0.396, type=float, help='SDSS pixel scale (arcsec/pix).')
    
    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')
    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--sky-tests', action='store_true', help='Test choice of sky apertures in ellipse-fitting.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final photometric catalog.')
    args = parser.parse_args()

    return args

def legacyhalos_cosmology(WMAP=False, Planck=False):
    """Establish the default cosmology for the project."""

    if WMAP:
        from astropy.cosmology import WMAP9 as cosmo
    elif Planck:
        from astropy.cosmology import Planck15 as cosmo
    else:
        from astropy.cosmology import FlatLambdaCDM
        #params = dict(H0=70, Om0=0.3, Ob0=0.0457, Tcmb0=2.7255, Neff=3.046)
        #sigma8 = 0.82
        #ns = 0.96
        #cosmo = FlatLambdaCDM(name='FlatLambdaCDM', **params)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')

    return cosmo

def get_lambdabins(verbose=False):
    """Fixed bins of richness.
    
    nn = 7
    ll = 10**np.linspace(np.log10(5), np.log10(500), nn)
    #ll = np.linspace(5, 500, nn)
    mh = np.log10(lambda2mhalo(ll))
    for ii in range(nn):
        print('{:.3f}, {:.3f}'.format(ll[ii], mh[ii]))    

    """
    # Roughly 13.5, 13.9, 14.2, 14.6, 15, 15.7 Msun
    #lambdabins = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 250.0])

    # Roughly 13.9, 14.2, 14.6, 15, 15.7 Msun
    lambdabins = np.array([20.0, 25.0, 30.0, 40.0, 60.0, 100.0])
    #lambdabins = np.array([10.0, 20.0, 40.0, 80.0, 250.0])
    #lambdabins = np.array([5, 25, 50, 100, 500])
    nlbins = len(lambdabins)
    
    mhalobins = lambda2mhalo(lambdabins, redshift=0.2)

    if verbose:
        for ii in range(nlbins - 1):
            print('Bin {}: lambda={:03d}-{:03d}, Mhalo={:.3f}-{:.3f} Msun'.format(
                ii, lambdabins[ii].astype('int'), lambdabins[ii+1].astype('int'),
                mhalobins[ii], mhalobins[ii+1]))
            
    return lambdabins

def get_zbins(zmin=0.1, zmax=0.3, dt=0.5, verbose=False):
    """Establish redshift bins which are equal in lookback time."""
    import astropy.units as u
    from astropy.cosmology import z_at_value
    
    cosmo = legacyhalos_cosmology()
    tmin, tmax = cosmo.lookback_time([zmin, zmax])
    if verbose:
        print('Cosmic time spanned = {:.3f} Gyr'.format(tmax - tmin))
    
    ntbins = np.round((tmax.value - tmin.value) / dt + 1).astype('int')
    #tbins = np.arange(tmin.value, tmax.value, dt) * u.Gyr
    tbins = np.linspace(tmin.value, tmax.value, ntbins) * u.Gyr
    zbins = np.around([z_at_value(cosmo.lookback_time, tt) for tt in tbins], decimals=3)
    tbins = tbins.value
    
    # Now fix the bins:
    # zbins = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.6])
    #zbins = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    zbins = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    #zbins = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    tbins = cosmo.lookback_time(zbins).value
    
    if verbose:
        for ii in range(ntbins - 1):
            print('Bin {}: z={:.3f}-{:.3f}, t={:.3f}-{:.3f} Gyr, dt={:.3f} Gyr'.format(
                ii, zbins[ii], zbins[ii+1], tbins[ii], tbins[ii+1], tbins[ii+1]-tbins[ii]))
            
    return zbins

def get_mstarbins(deltam=0.1, satellites=False):
    """Fixed bins of stellar mass.
    
    nn = 7
    ll = 10**np.linspace(np.log10(5), np.log10(500), nn)
    #ll = np.linspace(5, 500, nn)
    mh = np.log10(lambda2mhalo(ll))
    for ii in range(nn):
        print('{:.3f}, {:.3f}'.format(ll[ii], mh[ii]))    
    """

    if satellites:
        pass # code me
    else:
        mstarmin, mstarmax = 9.0, 14.0

    nmstarbins = np.round( (mstarmax - mstarmin) / deltam ).astype('int') + 1
    mstarbins = np.linspace(mstarmin, mstarmax, nmstarbins)
    
    return mstarbins

def lambda2mhalo(richness, redshift=0.3, Saro=False):
    """
    Convert cluster richness, lambda, to halo mass, given various 
    calibrations.
    
      * Saro et al. 2015: Equation (7) and Table 2 gives M(500).
      * Melchior et al. 2017: Equation (51) and Table 4 gives M(200).
      * Simet et al. 2017: 
    
    Other SDSS-based calibrations: Li et al. 2016; Miyatake et al. 2016; 
    Farahi et al. 2016; Baxter et al. 2016.

    TODO: Return the variance!

    """
    from colossus.halo import mass_defs
    #from colossus.halo import concentration
    from colossus.cosmology import cosmology

    #cosmo = legacyhalos_cosmology()
    #cosmology.setCosmology(cosmology.fromAstropy(cosmo, ns=0.96, sigma8=0.82))
    params = {'flat': True, 'H0': 70, 'Om0': 0.3, 'Ob0': 0.049, 'sigma8': 0.82, 'ns': 0.95}
    cosmo = cosmology.setCosmology('myCosmo', params)
    
    if Saro:
        pass

    if len(np.atleast_1d(redshift)) == 1:
        zredshift = np.repeat(redshift, len(richness))
    else:
        zredshift = redshift
    
    # Melchior et al. 2017 (default)
    logM0, Flam, Gz, lam0, z0 = 14.371, 1.12, 0.18, 30.0, 0.5
    M200m = 10**logM0 * (richness / lam0)**Flam * ( (1 + zredshift) / (1 + z0) )**Gz

    #np.atleast_1d(M200m)

    # Convert to M200c
    #import pdb ; pdb.set_trace()
    #c200m = concentration.concentration(M200m, '200m', redshift, model='bullock01')
    #M200c, _, _ = mass_defs.changeMassDefinition(M200m, c200m, redshift, '200m', '200c')
    #M200c, _, _ = mass_adv.changeMassDefinitionCModel(M200m, redshift, '200m', '200c')

    # Assume a constant concentration.
    M200c = np.zeros_like(M200m)
    for ii, (mm, zz) in enumerate(zip(M200m, zredshift)):
        mc, _, _ = mass_defs.changeMassDefinition(mm, 3.5, zz, '200m', '200c')
        M200c[ii] = mc
        
    return np.log10(M200c)

def cutout_radius_kpc(redshift, pixscale=None, radius_kpc=RADIUS_CLUSTER_KPC, cosmo=None):
    """Get a cutout radius of RADIUS_KPC [in pixels] at the redshift of the cluster.

    """
    if cosmo is None:
        cosmo = legacyhalos_cosmology()
        
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
    radius = radius_kpc * arcsec_per_kpc # [float arcsec]
    if pixscale:
        radius = np.rint(radius / pixscale).astype(int) # [integer/rounded pixels]
        
    return radius

def cutout_radius_cluster(redshift, cluster_radius, pixscale=0.262, factor=1.0,
                          rmin=50, rmax=500, bound=False, cosmo=None):
    """Get a cutout radius which depends on the richness radius (in h^-1 Mpc)
    R_LAMBDA of each cluster (times an optional fudge factor).

    Optionally bound the radius to (rmin, rmax).

    """
    if cosmo is None:
        cosmo = legacyhalos_cosmology()
        
    radius_kpc = cluster_radius * 1e3 * cosmo.h # cluster radius in kpc
    radius = np.rint(factor * radius_kpc * cosmo.arcsec_per_kpc_proper(redshift).value / pixscale)

    if bound:
        radius[radius < rmin] = rmin
        radius[radius > rmax] = rmax

    return radius # [pixels]

def read_redmapper(rmversion='v6.3.1', sdssdr='dr14', first=None, last=None,
                   galaxylist=None, satellites=False, sdssphot=False, verbose=True):
    """Read the parent redMaPPer cluster catalog and updated photometry.
    
    """
    from astropy.table import Table
    cosmo = legacyhalos_cosmology()
    
    if satellites:
        suffix1, suffix2 = '_members', '-members'
    else:
        suffix1, suffix2 = '', '-centrals'

    lgt = 'lgt5'
        
    cenfile = os.path.join(os.getenv('REDMAPPER_DIR'), rmversion, 
                          'dr8_run_redmapper_{}_{}_catalog.fit'.format(rmversion, lgt))
    satfile = os.path.join(os.getenv('REDMAPPER_DIR'), rmversion, 
                           'dr8_run_redmapper_{}_{}_catalog_members.fit'.format(rmversion, lgt))

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(cenfile)
    nrows = info[ext].get_nrows()

    # select a subset of the full sample for analysis
    # see https://arxiv.org/pdf/1910.01656.pdf
    if True:
        rows = np.arange(nrows)
        cen = fitsio.read(cenfile, columns=[ZCOLUMN, RICHCOLUMN, 'DEC'])#, 'RA'])
        #isouth = np.logical_or((cen['DEC'] < 32.275), (cen['RA'] > 45) * (cen['RA'] < 315))

        if False:
            cut01 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.10) * (cen[ZCOLUMN] < 0.15) * (cen[RICHCOLUMN] >= 20) * (cen[RICHCOLUMN] < 25))[0][:10]
            cut02 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.10) * (cen[ZCOLUMN] < 0.15) * (cen[RICHCOLUMN] >= 25) * (cen[RICHCOLUMN] < 30))[0][:10]
            cut03 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.10) * (cen[ZCOLUMN] < 0.15) * (cen[RICHCOLUMN] >= 30) * (cen[RICHCOLUMN] < 40))[0][:10]
            cut04 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.10) * (cen[ZCOLUMN] < 0.15) * (cen[RICHCOLUMN] >= 40) * (cen[RICHCOLUMN] < 60))[0][:10]
            cut05 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.10) * (cen[ZCOLUMN] < 0.15) * (cen[RICHCOLUMN] >= 60) * (cen[RICHCOLUMN] < 100))[0][:10]

            cut06 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.15) * (cen[ZCOLUMN] < 0.20) * (cen[RICHCOLUMN] >= 20) * (cen[RICHCOLUMN] < 25))[0][:10]
            cut07 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.15) * (cen[ZCOLUMN] < 0.20) * (cen[RICHCOLUMN] >= 25) * (cen[RICHCOLUMN] < 30))[0][:10]
            cut08 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.15) * (cen[ZCOLUMN] < 0.20) * (cen[RICHCOLUMN] >= 30) * (cen[RICHCOLUMN] < 40))[0][:10]
            cut09 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.15) * (cen[ZCOLUMN] < 0.20) * (cen[RICHCOLUMN] >= 40) * (cen[RICHCOLUMN] < 60))[0][:10]
            cut10 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.15) * (cen[ZCOLUMN] < 0.20) * (cen[RICHCOLUMN] >= 60) * (cen[RICHCOLUMN] < 100))[0][:10]

            cut11 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.20) * (cen[ZCOLUMN] < 0.25) * (cen[RICHCOLUMN] >= 20) * (cen[RICHCOLUMN] < 25))[0][:10]
            cut12 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.20) * (cen[ZCOLUMN] < 0.25) * (cen[RICHCOLUMN] >= 25) * (cen[RICHCOLUMN] < 30))[0][:10]
            cut13 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.20) * (cen[ZCOLUMN] < 0.25) * (cen[RICHCOLUMN] >= 30) * (cen[RICHCOLUMN] < 40))[0][:10]
            cut14 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.20) * (cen[ZCOLUMN] < 0.25) * (cen[RICHCOLUMN] >= 40) * (cen[RICHCOLUMN] < 60))[0][:10]
            cut15 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.20) * (cen[ZCOLUMN] < 0.25) * (cen[RICHCOLUMN] >= 60) * (cen[RICHCOLUMN] < 100))[0][:10]

            cut16 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.25) * (cen[ZCOLUMN] < 0.30) * (cen[RICHCOLUMN] >= 20) * (cen[RICHCOLUMN] < 25))[0][:10]
            cut17 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.25) * (cen[ZCOLUMN] < 0.30) * (cen[RICHCOLUMN] >= 25) * (cen[RICHCOLUMN] < 30))[0][:10]
            cut18 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.25) * (cen[ZCOLUMN] < 0.30) * (cen[RICHCOLUMN] >= 30) * (cen[RICHCOLUMN] < 40))[0][:10]
            cut19 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.25) * (cen[ZCOLUMN] < 0.30) * (cen[RICHCOLUMN] >= 40) * (cen[RICHCOLUMN] < 60))[0][:10]
            cut20 = np.where((cen['DEC'] < 32.275) * (cen[ZCOLUMN] >= 0.25) * (cen[ZCOLUMN] < 0.30) * (cen[RICHCOLUMN] >= 60) * (cen[RICHCOLUMN] < 100))[0][:10]

            #samplecut = cut20
            samplecut = np.hstack((
                cut01, cut02, cut03, cut04, cut05,
                cut06, cut07, cut08, cut09, cut10,
                cut11, cut12, cut13, cut14, cut15,
                cut16, cut17, cut18, cut19, cut20))
        else:
            samplecut = np.where(
                (cen[ZCOLUMN] >= 0.1) *
                (cen[ZCOLUMN] < 0.3) *
                (cen[RICHCOLUMN] >= 20) *
                (cen['DEC'] < 32.275))[0]
        rows = rows[samplecut]
        nrows = len(rows)
    else:
        rows = None
    #print('Selecting {} centrals.'.format(nrows))        

    if first is None:
        first = 0
    if last is None:
        last = nrows
        if rows is None:
            rows = np.arange(first, last)
        else:
            rows = rows[np.arange(first, last)]
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        if rows is None:
            rows = np.arange(first, last+1)
        else:
            rows = rows[np.arange(first, last+1)]

    cen = Table(info[ext].read(rows=rows, upper=True))#, columns=columns))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, cenfile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(cen), cenfile))

    # pre-compute the diameter of the mosaic (=2*RADIUS_CLUSTER_KPC kpc) for each cluster
    cen[DIAMCOLUMN] = cutout_radius_kpc(redshift=cen[ZCOLUMN], cosmo=cosmo, # diameter [arcsec]
                                             radius_kpc=2 * RADIUS_CLUSTER_KPC)
    cen[GALAXYCOLUMN] = np.array( ['{:07d}-{:09d}'.format(mid, cid)
                                   for mid, cid in zip(cen['MEM_MATCH_ID'], cen['ID_CENT'][:, 0])] )

    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        these = np.isin(cen[GALAXYCOLUMN], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            cen = cen[these]

    if satellites:
        satid = fitsio.read(satfile, columns='MEM_MATCH_ID')
        satrows = np.where(np.isin(satid, cen['MEM_MATCH_ID']))[0]
        satrows = satrows[np.argsort(satrows)]
        sat = Table(fitsio.read(satfile, rows=satrows, ext=1, upper=True))
        return cen, sat
    else:
        return cen

def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          threshmask=0.001, r50mask=0.1, maxshift=5,
                          verbose=False):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    r50mask - mask satellites whose r50 radius (arcsec) is > r50mask 

    threshmask - mask satellites whose flux ratio is > threshmmask relative to
    the central galaxy.

    """
    import numpy.ma as ma
    from copy import copy
    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import srcs2image, ellipse_mask

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    bands, refband = data['bands'], data['refband']
    #residual_mask = data['residual_mask']

    #nbox = 5
    #box = np.arange(nbox)-nbox // 2
    #box = np.meshgrid(np.arange(nbox), np.arange(nbox))[0]-nbox//2

    xobj, yobj = np.ogrid[0:data['refband_height'], 0:data['refband_width']]

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if 'galaxy_indx' in data.keys():
        galaxy_indx = np.atleast_1d(data['galaxy_indx'])
    else:
        galaxy_indx = np.array([np.argmin((tractor.bx - data['refband_height']/2)**2 +
                                          (tractor.by - data['refband_width']/2)**2)])
        data['galaxy_indx'] = np.atleast_1d(galaxy_indx)
        data['galaxy_id'] = ''

    #print('Import hack!')
    #norm = simple_norm(img, 'log', min_percent=0.05, clip=True)
    #import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm

    # Get the PSF sources.
    psfindx = np.where(tractor.type == 'PSF')[0]
    if len(psfindx) > 0:
        psfsrcs = tractor.copy()
        psfsrcs.cut(psfindx)
    else:
        psfsrcs = None

    def tractor2mge(indx, factor=1.0):
    #def tractor2mge(indx, majoraxis=None):
        # Convert a Tractor catalog entry to an MGE object.
        class MGEgalaxy(object):
            pass

        ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
        ba = (1 - ee) / (1 + ee)
        pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2))
        pa = pa % 180

        mgegalaxy = MGEgalaxy()
        mgegalaxy.xmed = tractor.by[indx]
        mgegalaxy.ymed = tractor.bx[indx]
        mgegalaxy.xpeak = tractor.by[indx]
        mgegalaxy.ypeak = tractor.bx[indx]
        mgegalaxy.eps = 1-ba
        mgegalaxy.pa = pa
        mgegalaxy.theta = (270 - pa) % 180
        mgegalaxy.majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]

        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                               mgegalaxy.majoraxis,
                               mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
                               np.radians(mgegalaxy.theta-90), xobj, yobj)

        return mgegalaxy, objmask

    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        print('Determing the geometry for galaxy {}/{}.'.format(
                ii+1, len(galaxy_indx)))

        # [1] Determine the non-parametricc geometry of the galaxy of interest
        # in the reference band. First, subtract all models except the galaxy
        # and galaxies "near" it. Also restore the original pixels of the
        # central in case there was a poor deblend.
        largeshift = False
        mge, centralmask = tractor2mge(central, factor=5.0)

        iclose = np.where([centralmask[int(by), int(bx)]
                           for by, bx in zip(tractor.by, tractor.bx)])[0]
        
        srcs = tractor.copy()
        srcs.cut(np.delete(np.arange(len(tractor)), iclose))
        model = srcs2image(srcs, data['{}_wcs'.format(refband)],
                           band=refband.lower(),
                           pixelized_psf=data['{}_psf'.format(refband)])

        img = data[refband].data - model
        img[centralmask] = data[refband].data[centralmask]

        mask = np.logical_or(ma.getmask(data[refband]), data['residual_mask'])
        #mask = np.logical_or(data[refband].mask, data['residual_mask'])
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)

        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('debug.png')
        #if True:
        #    import matplotlib.pyplot as plt
        #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('debug.png')
        #    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #    pdb.set_trace()
        #pdb.set_trace()

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            print('Large centroid shift! (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                mgegalaxy.xmed, mgegalaxy.ymed, mge.xmed, mge.ymed))
            largeshift = True
            mgegalaxy = copy(mge)

        radec_med = data['{}_wcs'.format(refband)].pixelToPosition(
            mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        radec_peak = data['{}_wcs'.format(refband)].pixelToPosition(
            mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {
            'largeshift': largeshift,
            'ra': tractor.ra[central], 'dec': tractor.dec[central],
            'bx': tractor.bx[central], 'by': tractor.by[central],
            'mw_transmission_g': tractor.mw_transmission_g[central],
            'mw_transmission_r': tractor.mw_transmission_r[central],
            'mw_transmission_z': tractor.mw_transmission_z[central],
            'ra_x0': radec_med[0], 'dec_y0': radec_med[1],
            #'ra_peak': radec_med[0], 'dec_peak': radec_med[1]
            }
        for key in ('eps', 'majoraxis', 'pa', 'theta', 'xmed', 'ymed', 'xpeak', 'ypeak'):
            mge[key] = np.float32(getattr(mgegalaxy, key))
            if key == 'pa': # put into range [0-180]
                mge[key] = mge[key] % np.float32(180)
        data['mge'].append(mge)

        if False:
            #plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
            plt.clf() ; mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=True, plot=True)
            plt.savefig('/mnt/legacyhalos-data/debug.png')

        # [2] Create the satellite mask in all the bandpasses. Use srcs here,
        # which has had the satellites nearest to the central galaxy trimmed
        # out.
        print('Building the satellite mask.')
        satmask = np.zeros(data[refband].shape, bool)
        for filt in bands:
            cenflux = getattr(tractor, 'flux_{}'.format(filt))[central]
            satflux = getattr(srcs, 'flux_{}'.format(filt))
            if cenflux <= 0.0:
                raise ValueError('Central galaxy flux is negative!')
            
            satindx = np.where(np.logical_or(
                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
                srcs.ref_cat == 'R1'))[0]
            #if np.isin(central, satindx):
            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
            if len(satindx) == 0:
                raise ValueError('All satellites have been dropped!')

            satsrcs = srcs.copy()
            #satsrcs = tractor.copy()
            satsrcs.cut(satindx)
            satimg = srcs2image(satsrcs, data['{}_wcs'.format(filt)],
                                band=filt.lower(),
                                pixelized_psf=data['{}_psf'.format(filt)])
            satmask = np.logical_or(satmask, satimg > 10*data['{}_sigma'.format(filt)])
            #if True:
            #    import matplotlib.pyplot as plt
            #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('debug.png')
            #    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
            #    pdb.set_trace()

        # [3] Build the final image (in each filter) for ellipse-fitting. First,
        # subtract out the PSF sources. Then update the mask (but ignore the
        # residual mask). Finally convert to surface brightness.
        for filt in bands:
            mask = np.logical_or(ma.getmask(data[filt]), satmask)
            mask[centralmask] = False
            #plt.imshow(mask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')

            varkey = '{}_var'.format(filt)
            imagekey = '{}_masked'.format(filt)
            psfimgkey = '{}_psfimg'.format(filt)
            thispixscale = filt2pixscale[filt]
            if imagekey not in data.keys():
                data[imagekey], data[varkey], data[psfimgkey] = [], [], []

            img = ma.getdata(data[filt]).copy()
            if psfsrcs:
                psfimg = srcs2image(psfsrcs, data['{}_wcs'.format(filt)],
                                    band=filt.lower(),
                                    pixelized_psf=data['{}_psf'.format(filt)])
                #data[psfimgkey].append(psfimg)
                img -= psfimg

            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
            var = data['{}_var_'.format(filt)] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)

            data[imagekey].append(img)
            data[varkey].append(var)

        #test = data['r_masked'][0]
        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #pdb.set_trace()

    # Cleanup?
    for filt in bands:
        del data[filt]
        del data['{}_var_'.format(filt)]

    return data            

def read_multiband(galaxy, galaxydir, galaxy_id, filesuffix='custom',
                   refband='r', bands=['g', 'r', 'z'], pixscale=0.262,
                   redshift=None, fill_value=0.0, sky_tests=False, verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    from astrometry.util.fits import fits_table
    from legacypipe.bits import MASKBITS
    from legacyhalos.io import _get_psfsize_and_depth, _read_image_data

    #galaxy_id = np.atleast_1d(galaxy_id)
    #if len(galaxy_id) > 1:
    #    raise ValueError('galaxy_id in read_multiband cannot be a >1-element vector for now!')
    #galaxy_id = galaxy_id[0]
    #assert(np.isscalar(galaxy_id))

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data, filt2imfile, filt2pixscale = {}, {}, {}

    for band in bands:
        filt2imfile.update({band: {'image': '{}-image'.format(filesuffix),
                                   'model': '{}-model'.format(filesuffix),
                                   'invvar': '{}-invvar'.format(filesuffix),
                                   'psf': '{}-psf'.format(filesuffix),
                                   }})
        filt2pixscale.update({band: pixscale})
    filt2imfile.update({'tractor': '{}-tractor'.format(filesuffix),
                        'sample': 'redmapper-sample',
                        'maskbits': '{}-maskbits'.format(filesuffix),
                        })

    # Do all the files exist? If not, bail!
    missing_data = False
    for filt in bands:
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            #if imtype == 'sky': # this is a dictionary entry
            #    continue
            imfile = os.path.join(galaxydir, '{}-{}-{}.fits.fz'.format(galaxy, filt2imfile[filt][imtype], filt))
            #print(imtype, imfile)
            if os.path.isfile(imfile):
                filt2imfile[filt][imtype] = imfile
            else:
                if verbose:
                    print('File {} not found.'.format(imfile))
                missing_data = True
                break
    
    if missing_data:
        return data, None

    # Pack some preliminary info into the output dictionary.
    data['filesuffix'] = filesuffix
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)
    data['failed'] = False # be optimistic!

    # We ~have~ to read the tractor catalog using fits_table because we will
    # turn these catalog entries into Tractor sources later.
    tractorfile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['tractor']))
    if verbose:
        print('Reading {}'.format(tractorfile))
        
    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2',
            'flux_g', 'flux_r', 'flux_z',
            'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
            'nobs_g', 'nobs_r', 'nobs_z',
            'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 
            'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
            'psfsize_g', 'psfsize_r', 'psfsize_z']
    #if galex:
    #    cols = cols+['flux_fuv', 'flux_nuv']
    #if unwise:
    #    cols = cols+['flux_w1', 'flux_w1', 'flux_w1', 'flux_w1']
    tractor = fits_table(tractorfile, columns=cols)
    hdr = fitsio.read_header(tractorfile)
    if verbose:
        print('Read {} sources from {}'.format(len(tractor), tractorfile))
    data.update(_get_psfsize_and_depth(tractor, bands, pixscale, incenter=False))

    # Read the maskbits image and build the starmask.
    maskbitsfile = os.path.join(galaxydir, '{}-{}.fits.fz'.format(galaxy, filt2imfile['maskbits']))
    if verbose:
        print('Reading {}'.format(maskbitsfile))
    maskbits = fitsio.read(maskbitsfile)
    # initialize the mask using the maskbits image
    starmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) | (maskbits & MASKBITS['MEDIUM'] != 0) |
                 (maskbits & MASKBITS['CLUSTER'] != 0) | (maskbits & MASKBITS['ALLMASK_G'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_R'] != 0) | (maskbits & MASKBITS['ALLMASK_Z'] != 0) )

    # Are we doing sky tests? If so, build the dictionary of sky values here.

    # subsky - dictionary of additional scalar value to subtract from the imaging,
    #   per band, e.g., {'g': -0.01, 'r': 0.002, 'z': -0.0001}
    if sky_tests:
        #imfile = os.path.join(galaxydir, '{}-{}-{}.fits.fz'.format(galaxy, filt2imfile[refband]['image'], refband))
        hdr = fitsio.read_header(filt2imfile[refband]['image'], ext=1)
        nskyaps = hdr['NSKYANN'] # number of annuli

        # Add a list of dictionaries to iterate over different sky backgrounds.
        data.update({'sky': []})
        
        for isky in np.arange(nskyaps):
            subsky = {}
            subsky['skysuffix'] = '{}-skytest{:02d}'.format(filesuffix, isky)
            for band in bands:
                refskymed = hdr['{}SKYMD00'.format(band.upper())]
                skymed = hdr['{}SKYMD{:02d}'.format(band.upper(), isky)]
                subsky[band] = refskymed - skymed # *add* the new correction
            print(subsky)
            data['sky'].append(subsky)

    # Read the basic imaging data and masks.
    data = _read_image_data(data, filt2imfile, starmask=starmask,
                            fill_value=fill_value, verbose=verbose)
    
    # Find the central.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    sample = Table(fitsio.read(samplefile))
    print('Read {} sources from {}'.format(len(sample), samplefile))

    data['galaxy_indx'] = []
    data['galaxy_id'] = []
    for galid in np.atleast_1d(galaxy_id):
        galindx = np.where((tractor.ref_cat == 'R1') * (tractor.ref_id == galid))[0]
        if len(galindx) != 1:
            raise ValueError('Problem finding the central galaxy {} in the tractor catalog!'.format(galid))
        data['galaxy_indx'].append(galindx[0])
        data['galaxy_id'].append(galid)

        # Is the flux and/or ivar negative (and therefore perhaps off the
        # footprint?) If so, drop it here.
        for filt in bands:
            cenflux = getattr(tractor, 'flux_{}'.format(filt))[galindx[0]]
            cenivar = getattr(tractor, 'flux_ivar_{}'.format(filt))[galindx[0]]
            if cenflux <= 0.0 or cenivar <= 0.0:
                print('Central galaxy flux is negative. Off footprint or gap in coverage?')
                data['failed'] = True
                return data, []

    # Now build the multiband mask.
    data = _build_multiband_mask(data, tractor, filt2pixscale,
                                 fill_value=fill_value,
                                 verbose=verbose)

    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][0]), origin='lower') ; plt.savefig('junk1.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][1]), origin='lower') ; plt.savefig('junk2.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][2]), origin='lower') ; plt.savefig('junk3.png')
    #pdb.set_trace()

    # Gather some additional info that we want propagated to the output ellipse
    # catalogs.
    if redshift:
        allgalaxyinfo = []
        for galaxy_id, galaxy_indx in zip(data['galaxy_id'], data['galaxy_indx']):
            galaxyinfo = { # (value, units) tuple for the FITS header
                'id_cent': (str(galaxy_id), ''),
                'redshift': (redshift, '')}
            allgalaxyinfo.append(galaxyinfo)
    else:
        allgalaxyinfo = None

    return data, allgalaxyinfo

def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1,
                 filesuffix='custom', bands=['g', 'r', 'z'], refband='r',
                 sky_tests=False, unwise=False, verbose=False,
                 debug=False, logfile=None):
    """Wrapper on legacyhalos.mpi.call_ellipse but with specific preparatory work
    and hooks for the legacyhalos project.

    """
    import astropy.table
    from copy import deepcopy
    from legacyhalos.mpi import call_ellipse as mpi_call_ellipse

    if type(onegal) == astropy.table.Table:
        onegal = onegal[0] # create a Row object
    galaxy_id = onegal['ID_CENT'][0]

    if logfile:
        from contextlib import redirect_stdout, redirect_stderr
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id, bands=bands,
                                                  filesuffix=filesuffix, refband=refband,
                                                  pixscale=pixscale, redshift=onegal[ZCOLUMN],
                                                  sky_tests=sky_tests, verbose=verbose)
    else:
        data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id, bands=bands,
                                          filesuffix=filesuffix, refband=refband,
                                          pixscale=pixscale, redshift=onegal[ZCOLUMN],
                                          sky_tests=sky_tests, verbose=verbose)

    maxsma, delta_logsma = None, 6
    #maxsma, delta_logsma = 200, 10

    if sky_tests:
        from legacyhalos.mpi import _done

        def _wrap_call_ellipse():
            skydata = deepcopy(data) # necessary?
            for isky in np.arange(len(data['sky'])):
                skydata['filesuffix'] = data['sky'][isky]['skysuffix']
                for band in bands:
                    # We want to *add* this delta-sky because it is defined as
                    #   sky_annulus_0 - sky_annulus_isky
                    delta_sky = data['sky'][isky][band] 
                    print('  Adjusting {}-band sky backgroud by {:4g} nanomaggies.'.format(band, delta_sky))
                    for igal in np.arange(len(np.atleast_1d(data['galaxy_indx']))):
                        skydata['{}_masked'.format(band)][igal] = data['{}_masked'.format(band)][igal] + delta_sky

                err = mpi_call_ellipse(galaxy, galaxydir, skydata, galaxyinfo=galaxyinfo,
                                       pixscale=pixscale, nproc=nproc, 
                                       bands=bands, refband=refband, sbthresh=SBTHRESH,
                                       delta_logsma=delta_logsma, maxsma=maxsma,
                                       write_donefile=False,
                                       verbose=verbose, debug=True)#, logfile=logfile)# no logfile and debug=True, otherwise this will crash

                # no need to redo the nominal ellipse-fitting
                if isky == 0:
                    inellipsefile = os.path.join(galaxydir, '{}-{}-{}-ellipse.fits'.format(galaxy, skydata['filesuffix'], galaxy_id))
                    outellipsefile = os.path.join(galaxydir, '{}-{}-{}-ellipse.fits'.format(galaxy, data['filesuffix'], galaxy_id))
                    print('Copying {} --> {}'.format(inellipsefile, outellipsefile))
                    shutil.copy2(inellipsefile, outellipsefile)
            return err

        t0 = time.time()
        if logfile:
            with open(logfile, 'a') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    # Capture corner case of missing data / incomplete coverage (see also
                    # ellipse.legacyhalos_ellipse).
                    if bool(data):
                        if data['failed']:
                            err = 1
                        else:
                            err = _wrap_call_ellipse()
                    else:
                        if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, filesuffix))):
                            err = 1 # successful failure
                        else:
                            err = 0 # failed failure
                    _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix, log=log)
        else:
            log = None
            if bool(data):
                if data['failed']:
                    err = 1
                else:
                    err = _wrap_call_ellipse()
            else:
                if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, filesuffix))):
                    err = 1 # successful failure
                else:
                    err = 0 # failed failure
            _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix, log=log)
    else:
        mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                         pixscale=pixscale, nproc=nproc, 
                         bands=bands, refband=refband, sbthresh=SBTHRESH,
                         delta_logsma=delta_logsma, maxsma=maxsma,
                         verbose=verbose, debug=debug, logfile=logfile)

def get_integrated_filename():
    """Return the name of the file containing the integrated photometry."""
    integratedfile = os.path.join(profiles_dir(data=True), 'integrated-flux.fits')
    return integratedfile

def read_integrated_flux(first=None, last=None, integratedfile=None, verbose=False):
    """Read the output of legacyhalos.integrate.
    
    """
    if integratedfile is None:
        integratedfile = get_integrated_filename()
        
    if not os.path.isfile(integratedfile):
        print('File {} not found.'.format(integratedfile)) # non-catastrophic error is OK
        return None
    
    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(integratedfile)
    nrows = info[ext].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
        rows = np.arange(first, last)
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        rows = np.arange(first, last + 1)
    results = Table(info[ext].read(rows=rows, upper=True))
    
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, integratedfile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(results), integratedfile))
            
    return results

def write_results(lsphot, results=None, sersic_single=None, sersic_double=None,
                  sersic_exponential=None, sersic_single_nowavepower=None,
                  sersic_double_nowavepower=None, sersic_exponential_nowavepower=None,
                  clobber=False, verbose=False):
    """Write out the output of legacyhalos-results

    """
    from astropy.io import fits
    
    lsdir = legacyhalos.io.legacyhalos_dir()
    resultsfile = os.path.join(lsdir, 'legacyhalos-results.fits')
    if not os.path.isfile(resultsfile) or clobber:

        hx = fits.HDUList()

        hdu = fits.table_to_hdu(lsphot)
        hdu.header['EXTNAME'] = 'LHPHOT'
        hx.append(hdu)

        for tt, name in zip( (results, sersic_single, sersic_double, sersic_exponential,
                              sersic_single_nowavepower, sersic_double_nowavepower,
                              sersic_exponential_nowavepower),
                              ('results', 'sersic_single', 'sersic_double', 'sersic_exponential',
                              'sersic_single_nowavepower', 'sersic_double_nowavepower',
                              'sersic_exponential_nowavepower') ):
            hdu = fits.table_to_hdu(tt)
            hdu.header['EXTNAME'] = name.upper()
            hx.append(hdu)

        if verbose:
            print('Writing {}'.format(resultsfile))
        hx.writeto(resultsfile, overwrite=True)
    else:
        print('File {} exists.'.format(resultsfile))

def read_results(first=None, last=None, verbose=False, extname='RESULTS', rows=None):
    """Read the output of io.write_results.

    """
    lsdir = legacyhalos.io.legacyhalos_dir()
    resultsfile = os.path.join(lsdir, 'legacyhalos-results.fits')

    if not os.path.isfile(resultsfile):
        print('File {} not found.'.format(resultsfile))
        return None
    else:
        if rows is not None:
            results = Table(fitsio.read(resultsfile, ext=extname, rows=rows))
        else:
            results = Table(fitsio.read(resultsfile, ext=extname))
        if verbose:
            print('Read {} objects from {} [{}]'.format(len(results), resultsfile, extname))
        return results

def read_jackknife(verbose=False, dr='dr6-dr7'):
    """Read the jackknife table (written by legacyhalos-sample-selection.ipynb).

    """
    jackfile = os.path.join(sample_dir(), 'legacyhalos-jackknife-{}.fits'.format(dr))

    if not os.path.isfile(jackfile):
        print('File {} not found.'.format(jackfile))
        return None, None

    jack, hdr = fitsio.read(jackfile, extname='JACKKNIFE', header=True)
    nside = hdr['NSIDE']
    
    if verbose:
        print('Read {} rows from {}'.format(len(jack), jackfile))
    return Table(jack), nside

def _read_paper_sample(paper='profiles', first=None, last=None, dr='dr8',
                       sfhgrid=1, isedfit_lsphot=False, isedfit_sdssphot=False,
                       isedfit_lhphot=False, candidates=False, kcorr=False,
                       verbose=False):
    """Wrapper to read a sample for a given paper.

    """
    if paper == 'profiles':
        paperdir = profiles_dir(data=True)
    elif paper == 'smf':
        paperdir = smf_dir(data=True)
    else:
        print('Unrecognized paper {}!'.format(paper))
        raise ValueError()
        
    if candidates:
        prefix = 'candidate-centrals'
    else:
        prefix = 'centrals'

    if isedfit_lsphot:
        samplefile = os.path.join(paperdir, '{}-{}-sfhgrid{:02d}-lsphot-{}.fits'.format(paper, prefix, sfhgrid, dr))
    elif isedfit_sdssphot:
        samplefile = os.path.join(paperdir, '{}-{}-sfhgrid{:02d}-sdssphot-dr14.fits'.format(paper, prefix, sfhgrid))
    elif isedfit_lhphot:
        samplefile = os.path.join(paperdir, '{}-{}-sfhgrid{:02d}-lhphot.fits'.format(paper, prefix, sfhgrid))
    else:
        samplefile = os.path.join(paperdir, 'sample-{}-{}-{}.fits'.format(paper, prefix, dr))
        
    if not os.path.isfile(samplefile):
        print('File {} not found.'.format(samplefile))
        return None

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()

    if kcorr:
        ext = 2
    else:
        ext = 1

    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
        rows = np.arange(first, last)
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        rows = np.arange(first, last + 1)

    sample = Table(info[ext].read(rows=rows))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))

    print('Temporary hack to use SDSS coordinates!')
    from astropy.table import Column
    sample.add_column(Column(name='RA', data=sample['RA_SDSS']), index=0)
    sample.add_column(Column(name='DEC', data=sample['DEC_SDSS']), index=1)
    return sample

def read_smf_sample(first=None, last=None, dr='dr8', sfhgrid=1, isedfit_lsphot=False,
                    isedfit_sdssphot=False, isedfit_lhphot=False, candidates=False,
                    kcorr=False, verbose=False):
    """Read the SMF paper sample.

    """
    sample = _read_paper_sample(paper='smf', first=first, last=last, dr=dr,
                                sfhgrid=1, isedfit_lsphot=isedfit_lsphot,
                                isedfit_sdssphot=isedfit_sdssphot,
                                isedfit_lhphot=isedfit_lhphot, kcorr=kcorr,
                                candidates=candidates, verbose=verbose)
    return sample
    
def literature(kravtsov=True, gonzalez=False):
    """Assemble some data from the literature here.

    """
    from colossus.halo import mass_defs

    if kravtsov:
        krav = dict()
        krav['m500c'] = np.log10(np.array([15.6,10.3,7,5.34,2.35,1.86,1.34,0.46,0.47])*1e14)
        krav['mbcg'] = np.array([3.12,4.14,3.06,1.47,0.79,1.26,1.09,0.91,1.38])*1e12
        krav['mbcg'] = krav['mbcg']*0.7**2 # ????
        krav['mbcg_err'] = np.array([0.36,0.3,0.3,0.13,0.05,0.11,0.06,0.05,0.14])*1e12
        krav['mbcg_err'] = krav['mbcg_err'] / krav['mbcg'] / np.log(10)
        krav['mbcg'] = np.log10(krav['mbcg'])

        M200c, _, _ = mass_defs.changeMassDefinition(10**krav['m500c'], 3.5, 0.0, '500c', '200c')
        krav['m200c'] = np.log10(M200c)

        return krav

    if gonzalez:
        gonz = dict()
        gonz['mbcg'] = np.array([0.84,0.87,0.33,0.57,0.85,0.60,0.86,0.93,0.71,0.81,0.70,0.57])*1e12*2.65
        gonz['mbcg'] = gonz['mbcg']*0.7**2 # ????
        gonz['mbcg_err'] = np.array([0.03,0.09,0.01,0.01,0.14,0.03,0.03,0.05,0.07,0.12,0.02,0.01])*1e12*2.65
        gonz['m500c'] = np.array([2.26,5.15,0.95,3.46,3.59,0.99,0.95,3.23,2.26,2.41,2.37,1.45])*1e14
        gonz['m500c_err'] = np.array([0.19,0.42,0.1,0.32,0.28,0.11,0.1,0.19,0.23,0.18,0.24,0.21])*1e14
        gonz['mbcg_err'] = gonz['mbcg_err'] / gonz['mbcg'] / np.log(10)

        M200c, _, _ = mass_defs.changeMassDefinition(gonz['m500c'], 3.5, 0.0, '500c', '200c')
        
        gonz['m200c'] = np.log10(M200c)
        gonz['m500c'] = np.log10(gonz['m500c'])
        gonz['mbcg'] = np.log10(gonz['mbcg'])

        return gonz

def area():
    """Return the area of the DR6+DR7 sample.  See the
    `legacyhalos-sample-selection.ipynb` notebook for this calculation.

    """
    return 6717.906

def obsolete_make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              first=None, last=None, nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
    import subprocess
    import fitsio
    from legacyhalos.misc import RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if sample is None:
        sample = legacyhalos.io.read_sample(first=first, last=last)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    if zcolumn is None:
        zcolumn = ZCOLUMN
        
    galaxy, galaxydir, htmlgalaxydir = legacyhalos.io.get_galaxy_galaxydir(sample, html=True)

    # Write the last-updated date to a webpage.
    js = html_javadate()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * legacyhalos.misc.cutout_radius_kpc(
            redshift=gal[zcolumn], pixscale=0.262,
            radius_kpc=radius_mosaic_kpc) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8'.format(
            baseurl, gal['RA'], gal['DEC'], zoom)
        
        return viewer

    def _skyserver_link(objid):
        return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(objid)

    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
        #shutil.chown(htmldir, group='cosmo')
    homehtmlfile = os.path.join(htmldir, homehtml)

    #if verbose:
    print('Writing {}'.format(homehtmlfile))
    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>LegacyHalos: Central Galaxies</h1>\n')
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Richness</th>\n')
        html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')
        for ii, (gal, galaxy1, htmlgalaxydir1) in enumerate(zip(
            sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir) )):

            htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        if verbose:
            print('Writing {}'.format(trendshtmlfile))
        with open(trendshtmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>LegacyHalos: Sample Trends</h1>\n')
            html.write('<p><a href="https://github.com/moustakas/legacyhalos">Code and documentation</a></p>\n')
            html.write('<a href="trends/ellipticity_vs_sma.png"><img src="trends/ellipticity_vs_sma.png" alt="Missing file ellipticity_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/gr_vs_sma.png"><img src="trends/gr_vs_sma.png" alt="Missing file gr_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/rz_vs_sma.png"><img src="trends/rz_vs_sma.png" alt="Missing file rz_vs_sma.png" height="auto" width="50%"></a>')

            html.write('<br /><br />\n')
            html.write('<b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')
            html.close()

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    # Make a separate HTML page for each object.
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate( zip(
        sample, np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir) ) ):

        radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
            redshift=gal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        diam_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale)

        ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
        
        if not os.path.exists(htmlgalaxydir1):
            os.makedirs(htmlgalaxydir1, mode=0o775)
            #shutil.chown(htmlgalaxydir1, group='cosmo')

        ccdsfile = os.path.join(galaxydir1, '{}-ccds.fits'.format(galaxy1))
        if os.path.isfile(ccdsfile):
            nccds = fitsio.FITS(ccdsfile)[1].get_nrows()
        else:
            nccds = None

        nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
        prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

        htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>Central Galaxy {}</h1>\n'.format(galaxy1))

            html.write('<a href="../../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../../{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../../{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Richness</th>\n')
            html.write('<th>Pcen</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('<th>SkyServer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Image Mosaics</h2>\n')
            html.write('<p>Each mosaic (left to right: data, model of all but the central galaxy, and residual image containing just the central galaxy) is {:.0f} kpc = {:.3f} arcsec = {:.0f} pixels in diameter.</p>\n'.format(2*radius_mosaic_kpc, 2*radius_mosaic_arcsec, diam_mosaic_pixels))
            #html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr><td><a href="{}-grz-montage.png"><img src="{}-grz-montage.png" alt="Missing file {}-grz-montage.png" height="auto" width="100%"></a></td></tr>\n'.format(galaxy1, galaxy1, galaxy1))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            
            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-multiband.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            #html.write('<td><a href="{}-ellipse-ellipsefit.png"><img src="{}-ellipse-ellipsefit.png" alt="Missing file {}-ellipse-ellipsefit.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
            pngfile = '{}-ellipse-sbprofile.png'.format(galaxy1)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-ellipse-cog.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')

            if False:
                html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
                html.write('<table width="90%">\n')

                # single-sersic
                html.write('<tr>\n')
                html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-single-nowavepower.png"><img src="{}-sersic-single-nowavepower.png" alt="Missing file {}-sersic-single-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-single.png"><img src="{}-sersic-single.png" alt="Missing file {}-sersic-single.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                # Sersic+exponential
                html.write('<tr>\n')
                html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-exponential-nowavepower.png"><img src="{}-sersic-exponential-nowavepower.png" alt="Missing file {}-sersic-exponential-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-exponential.png"><img src="{}-sersic-exponential.png" alt="Missing file {}-sersic-exponential.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                # double-sersic
                html.write('<tr>\n')
                html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-double-nowavepower.png"><img src="{}-sersic-double-nowavepower.png" alt="Missing file {}-sersic-double-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-double.png"><img src="{}-sersic-double.png" alt="Missing file {}-sersic-double.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                html.write('</table>\n')

                html.write('<br />\n')

            if nccds and ccdqa:
                html.write('<h2>CCD Diagnostics</h2>\n')
                html.write('<table width="90%">\n')
                html.write('<tr>\n')
                pngfile = '{}-ccdpos.png'.format(galaxy1)
                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                    pngfile))
                html.write('</tr>\n')

                for iccd in range(nccds):
                    html.write('<tr>\n')
                    pngfile = '{}-2d-ccd{:02d}.png'.format(galaxy1, iccd)
                    html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                        pngfile))
                    html.write('</tr>\n')
                html.write('</table>\n')
                html.write('<br />\n')

            html.write('<a href="../../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    # Make the plots.
    if makeplots:
        err = make_plots(sample, datadir=datadir, htmldir=htmldir, refband=refband,
                         bands=bands, pixscale=pixscale, zcolumn=zcolumn, survey=survey,
                         clobber=clobber, verbose=verbose, nproc=nproc, ccdqa=ccdqa,
                         maketrends=maketrends)

    try:
        cmd = '/usr/bin/chgrp -R cosmo {}'.format(htmldir)
        print(cmd)
        err1 = subprocess.call(cmd.split())

        cmd = 'find {} -type d -exec chmod 775 {{}} +'.format(htmldir)
        print(cmd)
        err2 = subprocess.call(cmd.split())

        cmd = 'find {} -type f -exec chmod 664 {{}} +'.format(htmldir)
        print(cmd)
        err3 = subprocess.call(cmd.split())
    except:
        pass

    #if err1 != 0 or err2 != 0 or err3 != 0:
    #    print('Something went wrong updating permissions; please check the logfile.')
    #    return 0
    
    return 1

def _get_mags(cat, rad='10', kpc=False, pipeline=False, cog=False, R24=False, R25=False, R26=False):
    res = []
    for band in ('g', 'r', 'z'):
        mag = None
        if kpc:
            iv = cat['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
            ff = cat['FLUX{}_{}'.format(rad, band.upper())][0]
        elif pipeline:
            iv = cat['flux_ivar_{}'.format(band)]
            ff = cat['flux_{}'.format(band)]
        elif R24:
            mag = cat['{}_mag_sb24'.format(band)]
        elif R25:
            mag = cat['{}_mag_sb25'.format(band)]
        elif R26:
            mag = cat['{}_mag_sb26'.format(band)]
        elif cog:
            mag = cat['{}_cog_params_mtot'.format(band)]
        else:
            print('Thar be rocks ahead!')
        if mag:
            res.append('{:.3f}'.format(mag))
        else:
            if ff > 0:
                mag = 22.5-2.5*np.log10(ff)
                if iv > 0:
                    ee = 1 / np.sqrt(iv)
                    magerr = 2.5 * ee / ff / np.log(10)
                res.append('{:.3f}'.format(mag))
                #res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
            else:
                res.append('...')
    return res

def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn='RA', deccolumn='DEC', diamcolumn='RADIUS_MOSAIC',
                   zcolumn='Z_LAMBDA', maketrends=False, fix_permissions=True,
                   html_raslices=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    import legacyhalos.html
    
    htmlhomefile = os.path.join(htmldir, htmlhome)
    print('Building {}'.format(htmlhomefile))

    js = legacyhalos.html.html_javadate()       

    ## group by RA slices
    #raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    #from legacyhalos.misc import radec2pix
    # = radec2pix(8, cat['RA'], cat['DEC']).data

    with open(htmlhomefile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('p {display: inline-block;;}\n')
        html.write('</style>\n')

        html.write('<h1>LegacyHalos: Central Galaxies</h1>\n')
        
        html.write('<p style="width: 75%">\n')
        html.write("""This project is super neat.</p>\n""")

        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        # The default is to organize the sample by RA slice, but support both options here.
        if False: #html_raslices:
            html.write('<p>The web-page visualizations are organized by one-degree slices of right ascension.</p><br />\n')

            html.write('<table>\n')
            html.write('<tr><th>RA Slice</th><th>Number of Galaxies</th></tr>\n')
            for raslice in sorted(set(raslices)):
                inslice = np.where(raslice == raslices)[0]
                html.write('<tr><td><a href="RA{0}.html"><h3>{0}</h3></a></td><td>{1}</td></tr>\n'.format(raslice, len(inslice)))
            html.write('</table>\n')
        else:
            html.write('<br /><br />\n')
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th> </th>\n')
            html.write('<th>Cluster ID</th>\n')
            html.write('<th>Galaxy ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Richness</th>\n')
            html.write('<th>Pcen</th>\n')
            #html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')
            
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)
            for gal, galaxy1, htmlgalaxydir1 in zip(sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-grz-montage.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-grz-montage.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['MEM_MATCH_ID']))
                html.write('<td>{}</td>\n'.format(gal['ID_CENT'][0]))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                #html.write('<td>{:.4f}</td>\n'.format(diam1))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                html.write('</tr>\n')
            html.write('</table>\n')
            
        # close up shop
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(htmlhomefile, group='cosmo')
        
    # Optionally build the individual pages (one per RA slice).
    if False: # html_raslices:
        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[inslice], html=True)

            slicefile = os.path.join(htmldir, 'RA{}.html'.format(raslice))
            print('Building {}'.format(slicefile))

            with open(slicefile, 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
                html.write('p {width: "75%";}\n')
                html.write('</style>\n')

                html.write('<h3>RA Slice {}</h3>\n'.format(raslice))

                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th> </th>\n')
                html.write('<th>Index</th>\n')
                html.write('<th>ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
                html.write('<th>Viewer</th>\n')

                html.write('</tr>\n')
                for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                    htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                    pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-grz-montage.png'.format(galaxy1))
                    thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-grz-montage.png'.format(galaxy1))

                    ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                    viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                    html.write('<tr>\n')
                    #html.write('<td>{:g}</td>\n'.format(count))
                    #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
                    html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                    html.write('<td>{}</td>\n'.format(gal['INDEX']))
                    html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
                    html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                    html.write('<td>{:.7f}</td>\n'.format(ra1))
                    html.write('<td>{:.7f}</td>\n'.format(dec1))
                    html.write('<td>{:.4f}</td>\n'.format(diam1))
                    #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                    #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                    #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                    html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                    #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                    html.write('</tr>\n')
                html.write('</table>\n')
                #count += 1

                html.write('<br /><br />\n')
                html.write('<b><i>Last updated {}</b></i>\n'.format(js))
                html.write('</html></body>\n')

        if fix_permissions:
            shutil.chown(htmlhomefile, group='cosmo')

    # Optionally build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        print('Building {}'.format(trendshtmlfile))
        with open(trendshtmlfile, 'w') as html:
        #with open(os.open(trendshtmlfile, os.O_CREAT | os.O_WRONLY, 0o664), 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>HSC Massive Galaxies: Sample Trends</h1>\n')
            html.write('<p><a href="https://github.com/moustakas/legacyhalos">Code and documentation</a></p>\n')
            html.write('<a href="trends/ellipticity_vs_sma.png"><img src="trends/ellipticity_vs_sma.png" alt="Missing file ellipticity_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/gr_vs_sma.png"><img src="trends/gr_vs_sma.png" alt="Missing file gr_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/rz_vs_sma.png"><img src="trends/rz_vs_sma.png" alt="Missing file rz_vs_sma.png" height="auto" width="50%"></a>')

            html.write('<br /><br />\n')
            html.write('<b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')

        if fix_permissions:
            shutil.chown(trendshtmlfile, group='cosmo')

def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                       racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy, prevgalaxy,
                       nexthtmlgalaxydir, prevhtmlgalaxydir, verbose, clobber, fix_permissions):
    """Build the web page for a single galaxy.

    """
    import fitsio
    from glob import glob
    import legacyhalos.io
    import legacyhalos.html
    
    if not os.path.exists(htmlgalaxydir1):
        os.makedirs(htmlgalaxydir1)
        if fix_permissions:
            for topdir, dirs, files in os.walk(htmlgalaxydir1):
                for dd in dirs:
                    shutil.chown(os.path.join(topdir, dd), group='cosmo')

    htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
    if os.path.isfile(htmlfile) and not clobber:
        print('File {} exists and clobber=False'.format(htmlfile))
        return
    
    nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
    prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))
    
    js = legacyhalos.html.html_javadate()

    # Support routines--

    def _read_ccds_tractor_sample(prefix):
        nccds, tractor, sample = None, None, None
        
        ccdsfile = glob(os.path.join(galaxydir1, '{}-{}-ccds-*.fits'.format(galaxy1, prefix))) # north or south
        if len(ccdsfile) > 0:
            nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()

        # samplefile can exist without tractorfile when using --just-coadds
        samplefile = os.path.join(galaxydir1, '{}-redmapper-sample.fits'.format(galaxy1))
        #samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))

        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z', 'ra', 'dec']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # just keep the central - this needs its own keyword
            cid = sample[np.argmax(sample['P'])]['ID']
            tractor = tractor[(tractor['ref_cat'] == 'R1') * (tractor['ref_id'] == cid)]
            assert(len(tractor) == 1)

            ## ----------
            #print('Hacking ref_id!')
            #from astropy.table import Table
            #from astrometry.libkd.spherematch import match_radec
            #sample = Table(sample[sample['P'].argmax()])
            #tractor = tractor[tractor['ref_cat'] == 'R1']
            #m1, m2, _ = match_radec(sample['RA'], sample['DEC'], tractor['ra'], tractor['dec'], 1/3600)
            ##srt = np.argsort(m1) ; m1 = m1[srt] ; m2 = m2[srt]
            #sample = sample[m1]
            #tractor = tractor[m2]
            #tractor['ref_id'] = sample['ID']
            ## ----------

            # We just care about the galaxies in our sample
            if prefix == 'custom':
                wt, ws = [], []
                for ii, sid in enumerate(sample['ID']):
                    ww = np.where(tractor['ref_id'] == sid)[0]
                    if len(ww) > 0:
                        #print(ii, ww)
                        wt.append(ww)
                        ws.append(ii)
                if len(wt) == 0:
                    print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxy1))
                    tractor = None
                else:
                    wt = np.hstack(wt)
                    ws = np.hstack(ws)
                    assert(len(wt) == len(ws)) # there are duplicate satellites in tractor!
                    
                    tractor = tractor[wt]
                    sample = sample[ws]
                    srt = np.argsort(tractor['flux_r'])[::-1]

                    tractor = tractor[srt]
                    sample = sample[srt]
                    assert(np.all(tractor['ref_id'] == sample['ID']))
                    
        return nccds, tractor, sample

    def _skyserver_link(objid):
        return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(objid)

    def _html_cluster_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=False)
        #skyserver_link = _skyserver_link(gal['OBJID'])

        html.write('<h2>Cluster Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        #html.write('<th>Index<br />(Primary)</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Richness</th>\n')
        html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
        #html.write('<td>{:g}</td>\n'.format(ii))
        #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
        #html.write('<td>{}</td>\n'.format(gal['INDEX']))
        #html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
        html.write('<td>{}</td>\n'.format(galaxy1))
        html.write('<td>{:.7f}</td>\n'.format(gal[RACOLUMN]))
        html.write('<td>{:.7f}</td>\n'.format(gal[DECCOLUMN]))
        html.write('<td>{:.5f}</td>\n'.format(gal[ZCOLUMN]))
        html.write('<td>{:.4f}</td>\n'.format(gal[RICHCOLUMN]))
        html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(skyserver_link))
        html.write('</tr>\n')
        html.write('</table>\n')

        ## Add the properties of each galaxy.
        #html.write('<h3>Group Members</h3>\n')
        #html.write('<table>\n')
        #html.write('<tr>\n')
        #html.write('<th>ID</th>\n')
        #html.write('<th>Galaxy</th>\n')
        ##html.write('<th>Morphology</th>\n')
        #html.write('<th>RA</th>\n')
        #html.write('<th>Dec</th>\n')
        #html.write('<th>D(25)<br />(arcmin)</th>\n')
        ##html.write('<th>PA<br />(deg)</th>\n')
        ##html.write('<th>e</th>\n')
        #html.write('</tr>\n')
        #for groupgal in sample:
        #    #if '031705' in gal['GALAXY']:
        #    #    print(groupgal['GALAXY'])
        #    html.write('<tr>\n')
        #    html.write('<td>{}</td>\n'.format(groupgal['SGA_ID']))
        #    html.write('<td>{}</td>\n'.format(groupgal['GALAXY']))
        #    #typ = groupgal['MORPHTYPE'].strip()
        #    #if typ == '' or typ == 'nan':
        #    #    typ = '...'
        #    #html.write('<td>{}</td>\n'.format(typ))
        #    html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
        #    html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
        #    html.write('<td>{:.4f}</td>\n'.format(groupgal['DIAM']))
        #    #if np.isnan(groupgal['PA']):
        #    #    pa = 0.0
        #    #else:
        #    #    pa = groupgal['PA']
        #    #html.write('<td>{:.2f}</td>\n'.format(pa))
        #    #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
        #    html.write('</tr>\n')
        #html.write('</table>\n')

    def _html_image_mosaics(html):
        html.write('<h2>Image Mosaics</h2>\n')

        if False:
            html.write('<table>\n')
            html.write('<tr><th colspan="3">Mosaic radius</th><th colspan="3">Point-source depth<br />(5-sigma, mag)</th><th colspan="3">Image quality<br />(FWHM, arcsec)</th></tr>\n')
            html.write('<tr><th>kpc</th><th>arcsec</th><th>grz pixels</th><th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th></tr>\n')
            html.write('<tr><td>{:.0f}</td><td>{:.3f}</td><td>{:.1f}</td>'.format(
                radius_mosaic_kpc, radius_mosaic_arcsec, radius_mosaic_pixels))
            if bool(ellipse):
                html.write('<td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td>'.format(
                    ellipse['psfdepth_g'], ellipse['psfdepth_min_g'], ellipse['psfdepth_max_g'],
                    ellipse['psfdepth_r'], ellipse['psfdepth_min_r'], ellipse['psfdepth_max_r'],
                    ellipse['psfdepth_z'], ellipse['psfdepth_min_z'], ellipse['psfdepth_max_z']))
                html.write('<td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td></tr>\n'.format(
                    ellipse['psfsize_g'], ellipse['psfsize_min_g'], ellipse['psfsize_max_g'],
                    ellipse['psfsize_r'], ellipse['psfsize_min_r'], ellipse['psfsize_max_r'],
                    ellipse['psfsize_z'], ellipse['psfsize_min_z'], ellipse['psfsize_max_z']))
            html.write('</table>\n')
            #html.write('<br />\n')

        pngfile, thumbfile = '{}-custom-grz-montage.png'.format(galaxy1), 'thumb-{}-custom-grz-montage.png'.format(galaxy1)
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
        html.write('<table width="90%">\n')
        for filesuffix in ['custom-grz']:#, 'W1W2'):
            pngfile, thumbfile = '{}-{}-montage.png'.format(galaxy1, filesuffix), 'thumb-{}-{}-montage.png'.format(galaxy1, filesuffix)
            html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile, thumbfile))
        html.write('</table>\n')

        pngfile, thumbfile = '{}-pipeline-grz-montage.png'.format(galaxy1), 'thumb-{}-pipeline-grz-montage.png'.format(galaxy1)
        if os.path.isfile(os.path.join(htmlgalaxydir1, pngfile)):
            html.write('<p>Pipeline (left) data, (middle) model, and (right) residual image mosaic.</p>\n')
            html.write('<table width="90%">\n')
            html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile, thumbfile))
            html.write('</table>\n')

    def _html_ellipsefit_and_photometry(html, tractor, sample):
        html.write('<h2>Elliptical Isophote Analysis</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return
            
        html.write('<h3>Geometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="5">Tractor</th>\n')
        html.write('<th colspan="3">ID</th>\n')
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="5">Ellipse Fitting</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>ID</th>\n')
        html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(25)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(24)<br />(arcsec)</th><th>R(25)<br />(arcsec)</th><th>R(26)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('</tr>\n')

        for tt in tractor:
            ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
            pa = pa % 180

            html.write('<tr><td>{}</td>\n'.format(tt['ref_id']))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['d25_leda']*60/2, ellipse['pa_leda'], 1-ellipse['ba_leda']))
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa'], ellipse['eps']))

                rr = []
                for rad in [ellipse['radius_sb24'], ellipse['radius_sb25'], ellipse['radius_sb26']]:
                    if rad < 0:
                        rr.append('...')
                    else:
                        rr.append('{:.3f}'.format(rad))
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    rr[0], rr[1], rr[2], ellipse['pa'], ellipse['eps']))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th><th></th>\n')
        html.write('<th colspan="3"></th>\n')
        html.write('<th colspan="12">Curve of Growth</th>\n')
        html.write('</tr>\n')
        html.write('<tr><th></th><th></th>\n')
        html.write('<th colspan="3">Tractor</th>\n')
        html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>ID</th><th>Galaxy</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            g, r, z = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(tt['ref_id']))
            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                        galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                g, r, z = _get_mags(ellipse, R24=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R25=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R26=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, cog=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            print('Hacking ref_id!')
            galaxyid = ''
            #galaxyid = '{}-'.format(str(tractor['ref_id'][igal]))
            html.write('<h4>{}</h4>\n'.format(galaxyid))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     verbose=verbose, galaxyid='')
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue
            
            html.write('<table width="90%">\n')
            html.write('<tr>\n')

            pngfile = '{}-custom-{}ellipse-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-{}ellipse-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" width="100%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-custom-{}ellipse-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-custom-{}ellipse-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_maskbits(html):
        html.write('<h2>Masking Geometry</h2>\n')
        pngfile = '{}-custom-maskbits.png'.format(galaxy1)
        html.write('<p>Left panel: color mosaic with the original and final ellipse geometry shown. Middle panel: <i>original</i> maskbits image based on the Hyperleda geometry. Right panel: distribution of all sources and frozen sources (the size of the orange square markers is proportional to the r-band flux).</p>\n')
        html.write('<table width="90%">\n')
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(pngfile))
        html.write('</table>\n')

    def _html_ccd_diagnostics(html):
        html.write('<h2>CCD Diagnostics</h2>\n')

        html.write('<table width="90%">\n')
        pngfile = '{}-ccdpos.png'.format(galaxy1)
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile))
        html.write('</table>\n')
        #html.write('<br />\n')
        
    # Read the catalogs and then build the page--
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='custom')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>{}</h1>\n'.format(galaxy1))
        #raslice = get_raslice(gal[racolumn])
        #html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_cluster_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        #_html_maskbits(html)
        _html_ccd_diagnostics(html)

        html.write('<br /><br />\n')
        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
        html.write('<br />\n')

        html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
        html.write('<br />\n')
        html.write('</html></body>\n')

    if fix_permissions:
        #print('Fixing permissions.')
        shutil.chown(htmlfile, group='cosmo')

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z_LAMBDA', intflux=None,
              racolumn='RA', deccolumn='DEC', diamcolumn='RADIUS_MOSAIC',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=False,
              htmlhome='index.html', html_raslices=False,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None, fix_permissions=True):
    """Make the HTML pages.

    """
    import subprocess
    from astrometry.util.multiproc import multiproc

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width

    datadir = legacyhalos.io.legacyhalos_data_dir()
    htmldir = legacyhalos.io.legacyhalos_html_dir()
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)

    if sample is None:
        sample = read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)
        
    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    _, missing, done, _ = missing_files(args, sample)
    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return
    
    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    trendshtml = 'trends.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   zcolumn=zcolumn, maketrends=maketrends, fix_permissions=fix_permissions,
                   html_raslices=html_raslices)

    # Now build the individual pages in parallel.
    if False: # html_raslices:
        raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
        rasorted = np.argsort(raslices)
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)
    else:
        rasorted = np.arange(len(sample))
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1
