"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
import os, warnings
import pickle, pdb
import numpy as np
import numpy.ma as ma
from glob import glob

import fitsio
from astropy.table import Table, hstack
from astropy.io import fits
from astrometry.util.fits import fits_table, merge_tables

#import legacyhalos.hsc
import legacyhalos.coadds

def missing_files_groups(args, sample, size, htmldir=None):
    """Simple task-specific wrapper on missing_files.

    """
    if args.coadds:
        if args.sdss:
            suffix = 'sdss-coadds'
        else:
            suffix = 'coadds'
    elif args.custom_coadds:
        if args.sdss:
            suffix = 'sdss-custom-coadds'
        else:
            suffix = 'custom-coadds'
    elif args.ellipse:
        if args.sdss:
            suffix = 'sdss-ellipse'
        else:
            suffix = 'ellipse'
    elif args.sersic:
        suffix = 'sersic'
    elif args.sky:
        suffix = 'sky'
    elif args.htmlplots:
        suffix = 'html'
    else:
        suffix = ''        

    if suffix != '':
        groups = missing_files(sample, filetype=suffix, size=size, sdss=args.sdss,
                               clobber=args.clobber, htmldir=htmldir)
    else:
        groups = []        

    return suffix, groups

def missing_files(sample, filetype='coadds', size=1, htmldir=None,
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

def read_all_ccds(dr='dr9'):
    """Read the CCDs files, treating DECaLS and BASS+MzLS separately.

    """
    from astrometry.libkd.spherematch import tree_open
    #survey = LegacySurveyData()

    drdir = os.path.join(sample_dir(), dr)

    kdccds_north = []
    for camera in ('90prime', 'mosaic'):
        ccdsfile = os.path.join(drdir, 'survey-ccds-{}-{}.kd.fits'.format(camera, dr))
        ccds = tree_open(ccdsfile, 'ccds')
        print('Read {} CCDs from {}'.format(ccds.n, ccdsfile))
        kdccds_north.append((ccdsfile, ccds))

    ccdsfile = os.path.join(drdir, 'survey-ccds-decam-{}.kd.fits'.format(dr))
    ccds = tree_open(ccdsfile, 'ccds')
    print('Read {} CCDs from {}'.format(ccds.n, ccdsfile))
    kdccds_south = (ccdsfile, ccds)

    return kdccds_north, kdccds_south

def get_run(onegal):
    """Get the run based on a simple declination cut."""
    if onegal['DEC'] > 32.375:
        if onegal['RA'] < 45 or onegal['RA'] > 315:
            run = 'south'
        else:
            run = 'north'
    else:
        run = 'south'
    return run

def get_run_ccds(onegal, radius_mosaic, pixscale, log=None): # kdccds_north, kdccds_south, log=None):
    """Determine the "run", i.e., determine whether we should use the BASS+MzLS CCDs
    or the DECaLS CCDs file when running the pipeline.

    """
    from astrometry.util.util import Tan
    from astrometry.libkd.spherematch import tree_search_radec
    from legacypipe.survey import ccds_touching_wcs
    
    ra, dec = onegal['RA'], onegal['DEC']
    if dec < 25:
        run = 'decam'
    elif dec > 40:
        run = '90prime-mosaic'
    else:
        width = legacyhalos.coadds._mosaic_width(radius_mosaic, pixscale)
        wcs = Tan(ra, dec, width/2+0.5, width/2+0.5,
                  -pixscale/3600.0, 0.0, 0.0, pixscale/3600.0, 
                  float(width), float(width))

        # BASS+MzLS
        TT = []
        for fn, kd in kdccds_north:
            I = tree_search_radec(kd, ra, dec, 1.0)
            if len(I) == 0:
                continue
            TT.append(fits_table(fn, rows=I))
        if len(TT) == 0:
            inorth = []
        else:
            ccds = merge_tables(TT, columns='fillzero')
            inorth = ccds_touching_wcs(wcs, ccds)
        
        # DECaLS
        fn, kd = kdccds_south
        I = tree_search_radec(kd, ra, dec, 1.0)
        if len(I) > 0:
            ccds = fits_table(fn, rows=I)
            isouth = ccds_touching_wcs(wcs, ccds)
        else:
            isouth = []

        if len(inorth) > len(isouth):
            run = '90prime-mosaic'
        else:
            run = 'decam'
        print('RA, Dec={:.6f}, {:.6f}: run={} ({} north CCDs, {} south CCDs).'.format(
            ra, dec, run, len(inorth), len(isouth)), flush=True, file=log)

    return run

def check_and_read_ccds(galaxy, survey, debug=False, logfile=None):
    """Read the CCDs file generated by the pipeline coadds step.

    """
    ccdsfile_south = os.path.join(survey.output_dir, '{}-ccds-south.fits'.format(galaxy))
    ccdsfile_north = os.path.join(survey.output_dir, '{}-ccds-north.fits'.format(galaxy))
    #ccdsfile_south = os.path.join(survey.output_dir, '{}-ccds-decam.fits'.format(galaxy))
    #ccdsfile_north = os.path.join(survey.output_dir, '{}-ccds-90prime-mosaic.fits'.format(galaxy))
    if os.path.isfile(ccdsfile_south):
        ccdsfile = ccdsfile_south
    elif os.path.isfile(ccdsfile_north):
        ccdsfile = ccdsfile_north
    else:
        if debug:
            print('CCDs file {} not found.'.format(ccdsfile_south), flush=True)
            print('CCDs file {} not found.'.format(ccdsfile_north), flush=True)
            print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True)
        else:
            with open(logfile, 'w') as log:
                print('CCDs file {} not found.'.format(ccdsfile_south), flush=True, file=log)
                print('CCDs file {} not found.'.format(ccdsfile_north), flush=True, file=log)
                print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
        return False
    survey.ccds = survey.cleanup_ccds_table(fits_table(ccdsfile))

    # Check that coadds in all three grz bandpasses were generated in the
    # previous step.
    if ('g' not in survey.ccds.filter) or ('r' not in survey.ccds.filter) or ('z' not in survey.ccds.filter):
        if debug:
            print('Missing grz coadds...skipping.', flush=True)
            print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True)
        else:
            with open(logfile, 'w') as log:
                print('Missing grz coadds...skipping.', flush=True, file=log)
                print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
        return False
    return True

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from legacyhalos.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos_html_dir()

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

def legacyhalos_dir():
    if 'LEGACYHALOS_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def legacyhalos_data_dir():
    if 'LEGACYHALOS_DATA_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def legacyhalos_html_dir():
    if 'LEGACYHALOS_HTML_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYHALOS_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def sample_dir():
    sdir = os.path.join(legacyhalos_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def smf_dir(figures=False, data=False):
    pdir = os.path.join(legacyhalos_dir(), 'science', 'smf')
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
    pdir = os.path.join(legacyhalos_dir(), 'science', 'profiles')
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

def get_integrated_filename():
    """Return the name of the file containing the integrated photometry."""
    if hsc:
        import legacyhalos.hsc
        integratedfile = os.path.join(legacyhalos.hsc.hsc_dir(), 'integrated-flux.fits')
    else:
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

def write_ellipsefit(galaxy, galaxydir, ellipsefit, filesuffix='', verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    if filesuffix.strip() == '':
        ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit.p'.format(galaxy))
    else:
        ellipsefitfile = os.path.join(galaxydir, '{}-{}-ellipsefit.p'.format(galaxy, filesuffix))
        
    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    with open(ellipsefitfile, 'wb') as ell:
        pickle.dump(ellipsefit, ell, protocol=2)

def read_ellipsefit(galaxy, galaxydir, filesuffix='', verbose=True):
    """Read the output of write_ellipsefit.

    """
    if filesuffix.strip() == '':
        ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit.p'.format(galaxy))
    else:
        ellipsefitfile = os.path.join(galaxydir, '{}-{}-ellipsefit.p'.format(galaxy, filesuffix))
        
    try:
        with open(ellipsefitfile, 'rb') as ell:
            ellipsefit = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(ellipsefitfile))
        ellipsefit = dict()

    return ellipsefit

def write_sersic(galaxy, galaxydir, sersic, modeltype='single', verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    sersicfile = os.path.join(galaxydir, '{}-sersic-{}.p'.format(galaxy, modeltype))
    if verbose:
        print('Writing {}'.format(sersicfile))
    with open(sersicfile, 'wb') as ell:
        pickle.dump(sersic, ell, protocol=2)

def read_sersic(galaxy, galaxydir, modeltype='single', verbose=True):
    """Read the output of write_sersic."""

    sersicfile = os.path.join(galaxydir, '{}-sersic-{}.p'.format(galaxy, modeltype))
    try:
        with open(sersicfile, 'rb') as ell:
            sersic = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(sersicfile))
        sersic = dict()

    return sersic

def write_sbprofile(sbprofile, smascale, sbfile):
    """Write a (previously derived) surface brightness profile as a simple ASCII
    file, for use on a webpage.

    """
    data = np.array( [
        sbprofile['sma'],
        sbprofile['sma'] * smascale,
        sbprofile['mu_g'],
        sbprofile['mu_r'],
        sbprofile['mu_z'],
        sbprofile['mu_g_err'],
        sbprofile['mu_r_err'],
        sbprofile['mu_z_err']
        ] ).T

    fixnan = np.isnan(data)
    if np.sum(fixnan) > 0:
        data[fixnan] = -999
        
    np.savetxt(sbfile, data, fmt='%.6f')
    #with open(sbfile, 'wb') as sb:
    #    sb.write('# Yo\n')
    #pdb.set_trace()

    print('Wrote {}'.format(sbfile))

def write_mgefit(galaxy, galaxydir, mgefit, band='r', verbose=False):
    """Pickle an XXXXX object (see, e.g., ellipse.mgefit_multiband).

    """
    mgefitfile = os.path.join(galaxydir, '{}-mgefit.p'.format(galaxy))
    if verbose:
        print('Writing {}'.format(mgefitfile))
    with open(mgefitfile, 'wb') as mge:
        pickle.dump(mgefit, mge, protocol=2)

def read_mgefit(galaxy, galaxydir, verbose=True):
    """Read the output of write_mgefit."""

    mgefitfile = os.path.join(galaxydir, '{}-mgefit.p'.format(galaxy))
    try:
        with open(mgefitfile, 'rb') as mge:
            mgefit = pickle.load(mge)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(mgefitfile))
        mgefit = dict()

    return mgefit

def write_results(lsphot, results=None, sersic_single=None, sersic_double=None,
                  sersic_exponential=None, sersic_single_nowavepower=None,
                  sersic_double_nowavepower=None, sersic_exponential_nowavepower=None,
                  clobber=False, verbose=False):
    """Write out the output of legacyhalos-results

    """
    lsdir = legacyhalos_dir()
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

def _get_psfsize_and_depth(tractor, bands, incenter=False):
    """Helper function for read_multiband. Compute the average PSF size (in arcsec)
    and depth (in 5-sigma AB mags) in each bandpass based on the Tractor
    catalog.

    """
    out = {}
    
    # Get the average PSF size and depth in each bandpass.
    for filt in bands:
        psfsizecol = 'psfsize_{}'.format(filt.lower())
        psfdepthcol = 'psfdepth_{}'.format(filt.lower())
        
        # Optionally choose sources in the center of the field.
        H = np.max(tractor.bx) - np.min(tractor.bx)
        W = np.max(tractor.by) - np.min(tractor.by)
        if incenter:
            dH = 0.1 * H
        else:
            dH = H
        these = ( (tractor.bx > np.int(H / 2 - dH)) * (tractor.bx < np.int(H / 2 + dH)) *
                  (tractor.by > np.int(H / 2 - dH)) * (tractor.by < np.int(H / 2 + dH)) *
                  (tractor.get(psfdepthcol) > 0) )
        if np.sum(these) == 0:
            print('No sources at the center of the field, sonable to get PSF size!')
            continue

        #out['npsfsize_{}'.format(filt)] = np.sum(these).astype(int)
        out['psfsize_{}'.format(filt)] = np.median(tractor.get(psfsizecol)[these]).astype('f4') # [arcsec]
        out['psfsize_min_{}'.format(filt)] = np.min(tractor.get(psfsizecol)[these]).astype('f4')
        out['psfsize_max_{}'.format(filt)] = np.max(tractor.get(psfsizecol)[these]).astype('f4')

        out['psfdepth_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.median(tractor.get(psfdepthcol)[these])))).astype('f4') # [AB mag, 5-sigma]
        out['psfdepth_min_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.min(tractor.get(psfdepthcol)[these])))).astype('f4')
        out['psfdepth_max_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.max(tractor.get(psfdepthcol)[these])))).astype('f4')

    return out

def _read_and_mask(data, bands, refband, filt2imfile, filt2pixscale, tractor,
                   central_galaxy=None, central_galaxy_id=None, fill_value=0.0,
                   starmask=None, verbose=False):
    """Helper function for read_multiband. Read the multi-band imaging and build a
    mask.

    central_galaxy - indices of objects in the tractor catalog to *not* mask

    """
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    from astropy.stats import sigma_clipped_stats

    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    from astrometry.util.util import Tan
    from legacypipe.survey import LegacySurveyWcs

    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import srcs2image, ellipse_mask

    # Loop on each filter and return the masked data.
    residual_mask = None
    for filt in bands:
        # Read the data and initialize the mask with the inverse variance image,
        # if available.
        if verbose:
            print('Reading {}'.format(filt2imfile[filt]['image']))
            print('Reading {}'.format(filt2imfile[filt]['model']))
        image = fitsio.read(filt2imfile[filt]['image'])
        model = fitsio.read(filt2imfile[filt]['model'])

        # Initialize the mask based on the inverse variance
        if 'invvar' in filt2imfile[filt].keys():
            if verbose:
                print('Reading {}'.format(filt2imfile[filt]['invvar']))
            invvar = fitsio.read(filt2imfile[filt]['invvar'])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # Cache the reference image for the next step.
        if filt == refband:
            refimage = image.copy()
            refhdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)

        # Add in the star mask, resizing if necessary for this image/pixel scale.
        if starmask is not None:
            if image.shape != starmask.shape:
                from skimage.transform import resize
                _starmask = resize(starmask, image.shape, mode='reflect')
                mask = np.logical_or(mask, _starmask)
            else:
                mask = np.logical_or(mask, starmask)

        # Flag significant residual pixels after subtracting *all* the models
        # (we will restore the pixels of the galaxies of interest below).
        resid = gaussian_filter(image - model, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        if residual_mask is None:
            residual_mask = np.abs(resid) > 3*sig
        else:
            residual_mask = np.logical_or(residual_mask, np.abs(resid) > 3*sig)

        # Grow the mask slightly and pack into a dictionary.
        mask = binary_dilation(mask, iterations=2)
        
        data[filt] = ma.masked_array(image, mask) # [nanomaggies]

        #if invvar is not None:
        #    var = np.zeros_like(invvar)
        #    var[~mask] = 1 / invvar[~mask]
        #    data['{}_var'.format(filt)] = var / thispixscale**4 # [nanomaggies**2/arcsec**4]

    # Now, build the model image in the reference band using the mean PSF.
    if verbose:
        print('Reading {}'.format(filt2imfile[refband]['psf']))
    psfimg = fitsio.read(filt2imfile[refband]['psf'])
    psf = PixelizedPSF(psfimg)
    H, W = refimage.shape
    xobj, yobj = np.ogrid[0:H, 0:W]

    wcs = Tan(filt2imfile[refband]['image'], 1)
    mjd_tai = refhdr['MJD_MEAN'] # [TAI]

    twcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if central_galaxy is None:
        central_galaxy = np.array([np.argmin((tractor.bx - H/2)**2 + (tractor.by - W/2)**2)])
        central_galaxy_id = None
    data['central_galaxy_id'] = central_galaxy_id

    #print('Import hack!')
    #import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm
    
    # Now, for each 'central_galaxy', "find" it in the reference band and then
    # unmask the pixels belonging to that galaxy in each of the input bands.
    for ii, central in enumerate(central_galaxy):
        print('Building masked image for central {}/{}.'.format(ii+1, len(central_galaxy)))
        
        # Build the model image on-the-fly.
        #model_nocentral = fitsio.read(filt2imfile[filt]['model-nocentral'])
        nocentral = np.delete(np.arange(len(tractor)), central)
        srcs = tractor.copy()
        srcs.cut(nocentral)
        model_nocentral = srcs2image(srcs, twcs, band=refband, pixelized_psf=psf)
        mgegalaxy = find_galaxy(refimage-model_nocentral, nblob=1, binning=3, quiet=True)#, plot=True)

        #for filt in [refband]:
        for filt in bands:
            thispixscale = filt2pixscale[filt]
            
            # Pack the model-subtracted images images corresponding to each
            # (unique) central into a list.
            imagekey = '{}_masked'.format(filt)
            if imagekey not in data.keys():
                data[imagekey] = []

            factor = filt2pixscale[refband] / filt2pixscale[filt]
            majoraxis = 1.3 * mgegalaxy.majoraxis * factor # [pixels]

            central_mask = ellipse_mask(tractor.by[central] * factor, tractor.bx[central] * factor,
                                        majoraxis, majoraxis * (1-mgegalaxy.eps), 
                                        np.radians(mgegalaxy.theta-90), xobj, yobj)

            # "Unmask" the central and pack it away.
            mask = ma.mask_or(residual_mask, ma.getmask(data[filt]))
            mask[central_mask] = ma.nomask

            srcs = tractor.copy()
            srcs.cut(nocentral)
            model_nocentral = srcs2image(srcs, twcs, band=filt, pixelized_psf=psf)
            img = (ma.getdata(data[filt]) - model_nocentral) / thispixscale**2 # [nanomaggies/arcsec**2]
            #img = model_nocentral
            #img[central_mask] = 0
            #img[mask] = 0
            img = ma.masked_array(img, mask)

            # Fill with zeros--
            ma.set_fill_value(img, fill_value)
            img.filled(fill_value)
            data[imagekey].append(img)

            #img = np.log10(tst[0]) ; plt.imshow(img, origin='lower') ; plt.savefig('junk.png')
            #img = np.log10(tst[1]) ; plt.imshow(img, origin='lower') ; plt.savefig('junk4.png')
            #
            #img = np.log10(image) ; plt.imshow(img, origin='lower') ; plt.savefig('junk.png')
            ###
            #plt.imshow(mask, origin='lower') ; plt.savefig('junk3.png')
            #plt.imshow(ma.mask_or(residual_mask, ma.getmask(data[filt])), origin='lower') ; plt.savefig('junk3.png')
            ##
            ##
            ##img = ma.getdata(data[filt])-model_nocentral ; norm = simple_norm(img, 'log') ; plt.imshow(img, origin='lower', norm=norm) ; plt.savefig('junk.png')
            #img = np.log10(image) ; norm = simple_norm(img, 'log') ; plt.imshow(img, origin='lower', norm=norm) ; plt.savefig('junk.png')
            ##
            #plt.imshow(mask, origin='lower') ; plt.savefig('junk{}.png'.format(ii))
            #img = mask ; norm = simple_norm(img, 'log') ; plt.imshow(img, origin='lower', norm=norm) ; plt.savefig('junk{}.png'.format(ii))
            #thisimg = np.log10(img) ; norm = simple_norm(thisimg, 'log') ; plt.imshow(thisimg, origin='lower', norm=norm) ; plt.savefig('junk{}.png'.format(ii+1))
            #thisimg = np.log10(data[imagekey][ii]) ; norm = simple_norm(thisimg, 'log') ; plt.imshow(thisimg, origin='lower', norm=norm) ; plt.savefig('junk{}.png'.format(ii+1))
            #
            #pdb.set_trace()
            #del image, mask, model_nocentral
            
    # Cleanup?
    for filt in bands:
        del data[filt]

    return data

def read_multiband(galaxy, galaxydir, bands=('g', 'r', 'z'), refband='r',
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   sdss_pixscale=0.396, maskfactor=2.0, sdss=False,
                   largegalaxy=False, pipeline=False, verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    filt2imfile, filt2pixscale = {}, {}
    if sdss:
        masksuffix = 'sdss-mask-gri'
        bands = ('g', 'r', 'i')
        tractorprefix = None
        maskbitsprefix = None
        [filt2imfile.update({band: {'image': 'sdss-image',
                                    'model': 'sdss-model',
                                    'model-nocentral': 'sdss-model-nocentral'}}) for band in bands]
        [filt2pixscale.update({band: sdss_pixscale}) for band in bands]
    else:
        if largegalaxy:
            prefix = 'largegalaxy'
        elif pipeline:
            prefix = 'pipeline'
        else:
            prefix = 'custom'
            
        [filt2imfile.update({band: {'image': '{}-image'.format(prefix),
                                    'model': '{}-model'.format(prefix),
                                    'invvar': '{}-invvar'.format(prefix),
                                    'psf': '{}-psf'.format(prefix)}}) for band in bands]
        [filt2pixscale.update({band: pixscale}) for band in bands]
        # Add the tractor and maskbits files.
        filt2imfile.update({'tractor': '{}-tractor'.format(prefix),
                            'maskbits': '{}-maskbits'.format(prefix)})

    # Add GALEX and unWISE - fix me.
    #filt2imfile.update({
    #    'FUV': ['image', 'model-nocentral', 'custom-model', 'invvar'],
    #    'NUV': ['image', 'model-nocentral', 'custom-model', 'invvar'],
    #    'W1':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
    #    'W2':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
    #    'W3':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
    #    'W4':  ['image', 'model-nocentral', 'custom-model', 'invvar']
    #    })
    #filt2pixscale.update({
    #    'FUV': galex_pixscale,
    #    'NUV': galex_pixscale,
    #    'W1':  unwise_pixscale,
    #    'W2':  unwise_pixscale,
    #    'W3':  unwise_pixscale,
    #    'W4':  unwise_pixscale
    #    })

    # Do all the files exist? If not, bail!
    found_data = True
    for filt in bands:
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            for suffix in ('.fz', ''):
                imfile = os.path.join(galaxydir, '{}-{}-{}.fits{}'.format(galaxy, filt2imfile[filt][imtype], filt, suffix))
                #print(imfile)
                if os.path.isfile(imfile):
                    filt2imfile[filt][imtype] = imfile
                    break
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    data = dict()
    if not found_data:
        return data

    # Pack some preliminary info into the dictionary.
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = pixscale

    if 'NUV' in bands:
        data['galex_pixscale'] = galex_pixscale
    if 'W1' in bands:
        data['unwise_pixscale'] = unwise_pixscale

    # Read the tractor and maskbits images (from which we build the starmask).
    tractorfile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['tractor']))
    if os.path.isfile(tractorfile):
        # We ~have~ to read using fits_table because we will turn these catalog
        # entries into Tractor sources later.
        #cols = ['BX', 'BY', 'TYPE', 'REF_CAT', 'REF_ID', 'SERSIC', 'SHAPE_R', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
        #        'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        #tractor = Table(fitsio.read(tractorfile, columns=cols, upper=True))
        cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
                'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                'flux_g', 'flux_r', 'flux_z', 'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                'psfsize_g', 'psfsize_r', 'psfsize_z']
        tractor = fits_table(tractorfile, columns=cols)
        if verbose:
            print('Read {} sources from {}'.format(len(tractor), tractorfile))
        data.update(_get_psfsize_and_depth(tractor, bands, incenter=False))
    else:
        tractor = None

    maskbitsfile = os.path.join(galaxydir, '{}-{}.fits.fz'.format(galaxy, filt2imfile['maskbits']))
    if os.path.isfile(maskbitsfile):
        from legacypipe.bits import MASKBITS
        if verbose:
            print('Reading {}'.format(maskbitsfile))
        maskbits = fitsio.read(maskbitsfile)
        # initialize the mask using the maskbits image
        starmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) | (maskbits & MASKBITS['MEDIUM'] != 0) |
                     (maskbits & MASKBITS['CLUSTER'] != 0) | (maskbits & MASKBITS['ALLMASK_G'] != 0) |
                     (maskbits & MASKBITS['ALLMASK_R'] != 0) | (maskbits & MASKBITS['ALLMASK_Z'] != 0) )
    else:
        starmask = None

    # Read the data. For the large-galaxy project, iterate on LSLGA galaxies in
    # the field, otherwise, take the object closest to the center of the mosaic.
    if largegalaxy:
        # Need to take into account the elliptical mask of each source--
        central_galaxy = np.where(['L' in refcat for refcat in tractor.ref_cat])[0]
        central_galaxy_id = tractor.ref_id[central_galaxy]
    else:
        central_galaxy, central_galaxy_id = None, None

    data = _read_and_mask(data, bands, refband, filt2imfile, filt2pixscale,
                          tractor, central_galaxy=central_galaxy,
                          central_galaxy_id=central_galaxy_id,
                          starmask=starmask, verbose=verbose)
    #pdb.set_trace()
    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][0]), origin='lower') ; plt.savefig('junk1.png')
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][1]), origin='lower') ; plt.savefig('junk2.png')
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][2]), origin='lower') ; plt.savefig('junk3.png')
    #pdb.set_trace()

    return data

def read_results(first=None, last=None, verbose=False, extname='RESULTS', rows=None):
    """Read the output of io.write_results.

    """
    lsdir = legacyhalos_dir()
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

def read_sample(first=None, last=None, dr='dr6-dr7', sfhgrid=1,
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
    
def read_profiles_sample(first=None, last=None, dr='dr8', sfhgrid=1, isedfit_lsphot=False,
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

def read_redmapper(rmversion='v6.3.1', sdssdr='dr14', index=None, satellites=False,
                   get_ngal=False):
    """Read the parent redMaPPer cluster catalog and updated photometry.
    
    """
    if satellites:
        suffix1, suffix2 = '_members', '-members'
    else:
        suffix1, suffix2 = '', '-centrals'
    rmfile = os.path.join( os.getenv('REDMAPPER_DIR'), rmversion, 
                          'dr8_run_redmapper_{}_lgt5_catalog{}.fit'.format(rmversion, suffix1) )
    rmphotfile = os.path.join( os.getenv('REDMAPPER_DIR'), rmversion, 
                          'redmapper-{}-lgt5{}-sdssWISEphot-{}.fits'.format(rmversion, suffix2, sdssdr) )

    if get_ngal:
        ngal = fitsio.FITS(rmfile)[1].get_nrows()
        return ngal
    
    rm = Table(fitsio.read(rmfile, ext=1, upper=True, rows=index))
    rmphot = Table(fitsio.read(rmphotfile, ext=1, upper=True, rows=index))

    print('Read {} galaxies from {}'.format(len(rm), rmfile))
    print('Read {} galaxies from {}'.format(len(rmphot), rmphotfile))
    
    rm.rename_column('RA', 'RA_REDMAPPER')
    rm.rename_column('DEC', 'DEC_REDMAPPER')
    rmphot.rename_column('RA', 'RA_SDSS')
    rmphot.rename_column('DEC', 'DEC_SDSS')
    rmphot.rename_column('OBJID', 'SDSS_OBJID')

    assert(np.sum(rmphot['MEM_MATCH_ID'] - rm['MEM_MATCH_ID']) == 0)
    if satellites:
        assert(np.sum(rmphot['ID'] - rm['ID']) == 0)
        rm.remove_columns( ('ID', 'MEM_MATCH_ID') )
    else:
        rm.remove_column('MEM_MATCH_ID')
    rmout = hstack( (rmphot, rm) )
    del rmphot, rm

    # Add a central_id column
    #rmout.rename_column('MEM_MATCH_ID', 'CENTRAL_ID')
    #cid = ['{:07d}'.format(cid) for cid in rmout['MEM_MATCH_ID']]
    #rmout.add_column(Column(name='CENTRAL_ID', data=cid, dtype='U7'), index=0)
    
    return rmout

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

# For the HSC analysis---

def hsc_dir():
    ddir = os.path.join(legacyhalos_dir(), 'hsc')
    if not os.path.isdir(ddir):
        os.makedirs(ddir, exist_ok=True)
    return ddir

def hsc_data_dir():
    ddir = os.path.join(legacyhalos_data_dir(), 'hsc')
    if not os.path.isdir(ddir):
        os.makedirs(ddir, exist_ok=True)
    return ddir

def hsc_data_dir():
    ddir = os.path.join(legacyhalos_data_dir(), 'hsc')
    if not os.path.isdir(ddir):
        os.makedirs(ddir, exist_ok=True)
    return ddir

def read_hsc_vs_decals(verbose=False):
    """Read the parent sample."""
    ddir = hsc_vs_decals_dir()
    catfile = os.path.join(ddir, 'hsc-vs-decals.fits')
    cat = Table(fitsio.read(catfile, upper=True))
    if verbose:
        print('Read {} objects from {}'.format(len(cat), catfile))
    return cat
