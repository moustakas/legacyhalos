"""
legacyhalos.legacyhalos
=======================

Code to support the legacyhalos sample and project.

"""
import os, time, pdb
import numpy as np
import astropy
import fitsio

import legacyhalos.io

RACOLUMN = 'RA'
DECCOLUMN = 'DEC'
ZCOLUMN = 'Z_LAMBDA'    
DIAMCOLUMN = 'RADIUS_MOSAIC' # [radius, arcsec]
RICHCOLUMN = 'LAMBDA_CHISQ'

RADIUS_CLUSTER_KPC = 250.0 # default cluster radius [kpc]

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

    def get_healpix_subdir(nside, pixnum, datadir):
        subdir = os.path.join(str(pixnum // 100), str(pixnum))
        return os.path.abspath(os.path.join(datadir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        if candidates:
            galaxy = ['{:07d}-{:09d}'.format(cat['MEM_MATCH_ID'], cat['ID'])]
        else:
            galaxy = ['{:07d}-{:09d}'.format(cat['MEM_MATCH_ID'], cat['ID_CENT'][0])]
        pixnum = [radec2pix(nside, cat['RA'], cat['DEC'])]
    else:
        ngal = len(cat)
        if candidates:
            galaxy = np.array( ['{:07d}-{:09d}'.format(mid, cid)
                                for mid, cid in zip(cat['MEM_MATCH_ID'], cat['ID'])] )
        else:
            galaxy = np.array( ['{:07d}-{:09d}'.format(mid, cid)
                                for mid, cid in zip(cat['MEM_MATCH_ID'], cat['ID_CENT'][:, 0])] )

        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, datadir), gal)
                          for pix, gal in zip(pixnum, galaxy)])
    if html:
        htmlgalaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, htmldir), gal)
                                  for pix, gal in zip(pixnum, galaxy)])

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

def obsolete_missing_files(sample, filetype='coadds', size=1, htmldir=None,
                  sdss=False, clobber=False):
    """Find missing data of a given filetype."""    

    if filetype == 'coadds':
        filesuffix = '-pipeline-resid-grz.jpg'
    elif filetype == 'custom-coadds':
        filesuffix = '-custom-resid-grz.jpg'
    elif filetype == 'ellipse':
        filesuffix = '-ellipsefit.p'
    elif filetype == 'sersic':
        filesuffix = '-sersic-single.p'
    elif filetype == 'html':
        filesuffix = '-ccdpos.png'
        #filesuffix = '-sersic-exponential-nowavepower.png'
    elif filetype == 'sdss-coadds':
        filesuffix = '-sdss-image-gri.jpg'
    elif filetype == 'sdss-custom-coadds':
        filesuffix = '-sdss-resid-gri.jpg'
    elif filetype == 'sdss-ellipse':
        filesuffix = '-sdss-ellipsefit.p'
    else:
        print('Unrecognized file type!')
        raise ValueError

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)
    todo = np.ones(ngal, dtype=bool)

    if filetype == 'html':
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    else:
        galaxy, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir)

    for ii, (gal, gdir) in enumerate( zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)) ):
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if os.path.exists(checkfile) and clobber is False:
            todo[ii] = False

    if np.sum(todo) == 0:
        return list()
    else:
        indices = indices[todo]
        
    return np.array_split(indices, size)

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--sdss', action='store_true', help='Analyze the SDSS galaxies.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (used with --sky and --sersic).')

    parser.add_argument('--coadds', action='store_true', help='Build the coadds with the custom sky.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the coadds with the pipeline sky.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py).')
    parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--sersic', action='store_true', help='Perform Sersic fitting.')
    parser.add_argument('--integrate', action='store_true', help='Integrate the surface brightness profiles.')
    parser.add_argument('--sky', action='store_true', help='Estimate the sky variance.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--html-noraslices', dest='html_raslices', action='store_false',
                        help='Do not organize HTML pages by RA slice (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--sdss-pixscale', default=0.396, type=float, help='SDSS pixel scale (arcsec/pix).')
    
    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')
    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final photometric catalog.')
    args = parser.parse_args()

    return args

def cutout_radius_kpc(redshift, pixscale=None, radius_kpc=RADIUS_CLUSTER_KPC, cosmo=None):
    """Get a cutout radius of RADIUS_KPC [in pixels] at the redshift of the cluster.

    """
    if cosmo is None:
        from legacyhalos.misc import cosmology
        cosmo = cosmology()
        
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
        from legacyhalos.misc import cosmology
        cosmo = cosmology()
        
    radius_kpc = cluster_radius * 1e3 * cosmo.h # cluster radius in kpc
    radius = np.rint(factor * radius_kpc * cosmo.arcsec_per_kpc_proper(redshift).value / pixscale)

    if bound:
        radius[radius < rmin] = rmin
        radius[radius > rmax] = rmax

    return radius # [pixels]

def read_redmapper(rmversion='v6.3.1', sdssdr='dr14', first=None, last=None,
                   satellites=False, sdssphot=False, verbose=True):
    """Read the parent redMaPPer cluster catalog and updated photometry.
    
    """
    from astropy.table import Table
    from legacyhalos.misc import cosmology

    cosmo = cosmology()
    
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
        cen = fitsio.read(cenfile, columns=[ZCOLUMN, RICHCOLUMN])
        rows = np.arange(nrows)
        samplecut = np.where((cen[ZCOLUMN] >= 0.1) * (cen[ZCOLUMN] <= 0.3) *
                             (cen[RICHCOLUMN] >= 20))[0]
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

    # pre-compute the radius of the mosaic (=500 kpc) for each cluster
    cen['RADIUS_MOSAIC'] = cutout_radius_kpc(redshift=cen[ZCOLUMN], cosmo=cosmo, # [arcsec]
                                             radius_kpc=RADIUS_CLUSTER_KPC) 

    if satellites:
        satid = fitsio.read(satfile, columns='MEM_MATCH_ID')
        satrows = np.where(np.isin(satid, cen['MEM_MATCH_ID']))[0]
        sat = Table(fitsio.read(satfile, rows=satrows, ext=1, upper=True))
        #sat['RADIUS_MOSAIC'] = cutout_radius_kpc(redshift=sat[ZCOLUMN], cosmo=cosmo, # [arcsec]
        #                                         radius_kpc=RADIUS_CLUSTER_KPC)
        return cen, sat
    else:
        return cen

    ##rm.rename_column('RA', 'RA_REDMAPPER')
    ##rm.rename_column('DEC', 'DEC_REDMAPPER')
    #if sdssphot:
    #    cenphotfile = os.path.join(os.getenv('REDMAPPER_DIR'), rmversion, 
    #                               'redmapper-{}-{}-centrals-sdssWISEphot-{}.fits'.format(rmversion, lgt, sdssdr))
    #    cenphotfile = os.path.join(os.getenv('REDMAPPER_DIR'), rmversion, 
    #                               'redmapper-{}-{}-members-sdssWISEphot-{}.fits'.format(rmversion, lgt, sdssdr))
    #
    #if sdssphot:
    #    print('Need to figure this out.')
    #    pdb.set_trace()
    #    rmphot = Table(fitsio.read(cenphotfile, ext=1, upper=True, rows=rows))
    #    if verbose:
    #        if len(rows) == 1:
    #            print('Read galaxy index {} from {}'.format(first, cenphotfile))
    #        else:
    #            print('Read galaxy indices {} through {} (N={}) from {}'.format(
    #                first, last, len(rmphot), cenphotfile))
    #
    #    rmphot.rename_column('RA', 'RA_SDSS')
    #    rmphot.rename_column('DEC', 'DEC_SDSS')
    #    rmphot.rename_column('OBJID', 'SDSS_OBJID')
    #
    #    assert(np.all(rmphot['MEM_MATCH_ID'] == rm['MEM_MATCH_ID']))
    #    assert(np.all(rmphot['ID'] == rm['ID']))
    #
    #    if satellites:
    #        rm.remove_columns( ('ID', 'MEM_MATCH_ID') )
    #    else:
    #        rm.remove_column('MEM_MATCH_ID')
    #        
    #    rm = hstack((rmphot, rm))
    #    del rmphot
    #
    ## Add a central_id column
    ##rmout.rename_column('MEM_MATCH_ID', 'CENTRAL_ID')
    ##cid = ['{:07d}'.format(cid) for cid in rmout['MEM_MATCH_ID']]
    ##rmout.add_column(Column(name='CENTRAL_ID', data=cid, dtype='U7'), index=0)
    #
    #return rm

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

def obsolete_read_sample(first=None, last=None, dr='dr6-dr7', sfhgrid=1,
                         isedfit_lsphot=False, isedfit_sdssphot=False,
                         isedfit_lhphot=False, candidates=False,
                         kcorr=False, verbose=False):
    """Read the sample.

    """
    if candidates:
        prefix = 'candidate-centrals'
    else:
        prefix = 'centrals'

    if isedfit_lsphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-lsphot-{}.fits'.format(prefix, sfhgrid, dr))
    elif isedfit_sdssphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-sdssphot-dr14.fits'.format(prefix, sfhgrid))
    elif isedfit_lhphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-lhphot.fits'.format(prefix, sfhgrid))
    else:
        samplefile = os.path.join(sample_dir(), 'legacyhalos-{}-{}.fits'.format(prefix, dr))
        
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
            
    return sample

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
    
def obsolete_read_profiles_sample(first=None, last=None, dr='dr8', sfhgrid=1, isedfit_lsphot=False,
                                  isedfit_sdssphot=False, isedfit_lhphot=False, candidates=False,
                                  kcorr=False, verbose=False):
    """Read the profiles paper sample.

    """
    sample = _read_paper_sample(paper='profiles', first=first, last=last, dr=dr,
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

def skyserver_link(gal):
    if 'SDSS_OBJID' in gal.colnames:
        return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['SDSS_OBJID'])
    else:
        return ''

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
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

    def _skyserver_link(gal):
        if 'SDSS_OBJID' in gal.colnames:
            return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['SDSS_OBJID'])
        else:
            return ''

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
