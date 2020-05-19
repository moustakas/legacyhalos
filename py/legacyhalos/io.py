"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
import os, pdb
import numpy as np
import numpy.ma as ma

import fitsio
import astropy.units as u
from astropy.table import Table, Column
from astrometry.util.fits import fits_table

# build out the FITS header
def legacyhalos_header(hdr=None):
    """Build a header with code versions, etc.

    """
    import subprocess
    import pydl
    import legacyhalos

    if hdr is None:
        hdr = fitsio.FITSHDR()

    cmd = 'cd {} && git describe --tags'.format(os.path.dirname(legacyhalos.__file__))
    ver = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
    hdr.add_record(dict(name='LEGHALOV', value=ver, comment='legacyhalos git version'))

    depvers, headers = [], []
    for name, pkg in [('pydl', pydl)]:
        hdr.add_record(dict(name=name, value=pkg.__version__, comment='{} version'.format(name)))

    return hdr
    
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
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.util.util import Tan
    from astrometry.libkd.spherematch import tree_search_radec
    from legacypipe.survey import ccds_touching_wcs
    import legacyhalos.coadds
    
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

def _write_ellipsefit(galaxy, galaxydir, ellipsefit, filesuffix='', galaxyid='',
                     verbose=False, use_pickle=False):
    """Write out an ASDF file based on the output of
    legacyhalos.ellipse.ellipse_multiband..

    use_pickle - write an old-style pickle file

    OBSOLETE - we now use FITS

    """
    import pickle
    from astropy.io import fits
    from asdf import fits_embed
    #import asdf
    
    if use_pickle:
        suff = '.p'
    else:
        suff = '.fits'
        #suff = '.asdf'

    if galaxyid.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxyid)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)
        
    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse{}'.format(galaxy, fsuff, galid, suff))
        
    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    if use_pickle:
        with open(ellipsefitfile, 'wb') as ell:
            pickle.dump(ellipsefit, ell, protocol=2)
    else:
        pdb.set_trace()
        hdu = fits.HDUList()
        af = fits_embed.AsdfInFits(hdu, ellipsefit)
        af.write_to(ellipsefitfile)
        #af = asdf.AsdfFile(ellipsefit)
        #af.write_to(ellipsefitfile)

def _read_ellipsefit(galaxy, galaxydir, filesuffix='', galaxyid='', verbose=True, use_pickle=False):
    """Read the output of write_ellipsefit.

    OBSOLETE - we now use FITS

    """
    import pickle
    import asdf
    
    if use_pickle:
        suff = '.p'
    else:
        suff = '.asdf'
    
    if galaxyid.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxyid)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)

    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse{}'.format(galaxy, fsuff, galid, suff))
        
    try:
        if use_pickle:
            with open(ellipsefitfile, 'rb') as ell:
                ellipsefit = pickle.load(ell)
        else:
            #with asdf.open(ellipsefitfile) as af:
            #    ellipsefit = af.tree
            ellipsefit = asdf.open(ellipsefitfile)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(ellipsefitfile))
        ellipsefit = dict()

    return ellipsefit

# ellipsefit data model
def _get_ellipse_datamodel(refband='r'):
    cols = [
        ('bands', ''),
        ('refband', ''),
        ('refpixscale', u.arcsec / u.pixel),
        ('success', ''),
        ('fitgeometry', ''),
        ('input_ellipse', ''),
        ('largeshift', ''),

        ('ra_x0', u.degree),
        ('dec_y0', u.degree),
        ('x0', u.pixel),
        ('y0', u.pixel),
        ('eps', ''),
        ('pa', u.degree),
        ('theta', u.degree),
        ('majoraxis', u.pixel),
        ('maxsma', u.pixel),

        ('integrmode', ''),
        ('sclip', ''),
        ('nclip', ''),

        #('psfsigma_g', u.pixel),
        #('psfsigma_r', u.pixel),
        #('psfsigma_z', u.pixel),

        ('psfsize_g', u.arcsec),
        #('psfsize_min_g', u.arcsec),
        #('psfsize_max_g', u.arcsec),
        ('psfsize_r', u.arcsec),
        #('psfsize_min_r', u.arcsec),
        #('psfsize_max_r', u.arcsec),
        ('psfsize_z', u.arcsec),
        #('psfsize_min_z', u.arcsec),
        #('psfsize_max_z', u.arcsec),

        ('psfdepth_g', u.mag),
        #('psfdepth_min_g', u.mag),
        #('psfdepth_max_g', u.mag),
        ('psfdepth_r', u.mag),
        #('psfdepth_min_r', u.mag),
        #('psfdepth_max_r', u.mag),
        ('psfdepth_z', u.mag),
        #('psfdepth_min_z', u.mag),
        #('psfdepth_max_z', u.mag),

        ('mw_transmission_g', ''),
        ('mw_transmission_r', ''),
        ('mw_transmission_z', ''),

        ('{}_width'.format(refband), u.pixel),
        ('{}_height'.format(refband), u.pixel),

        ('g_sma', u.pixel),
        ('g_eps', ''),
        ('g_eps_err', ''),
        ('g_pa', u.degree),
        ('g_pa_err', u.degree),
        ('g_intens', u.maggy/u.arcsec**2),
        ('g_intens_err', u.maggy/u.arcsec**2),
        ('g_x0', u.pixel),
        ('g_x0_err', u.pixel),
        ('g_y0', u.pixel),
        ('g_y0_err', u.pixel),
        ('g_a3', ''), # units?
        ('g_a3_err', ''),
        ('g_a4', ''),
        ('g_a4_err', ''),
        ('g_rms', u.maggy/u.arcsec**2),
        ('g_pix_stddev', u.maggy/u.arcsec**2),
        ('g_stop_code', ''),
        ('g_ndata', ''),
        ('g_nflag', ''),
        ('g_niter', ''),

        ('r_sma', u.pixel),
        ('r_eps', ''),
        ('r_eps_err', ''),
        ('r_pa', u.degree),
        ('r_pa_err', u.degree),
        ('r_intens', u.maggy/u.arcsec**2),
        ('r_intens_err', u.maggy/u.arcsec**2),
        ('r_x0', u.pixel),
        ('r_x0_err', u.pixel),
        ('r_y0', u.pixel),
        ('r_y0_err', u.pixel),
        ('r_a3', ''),
        ('r_a3_err', ''),
        ('r_a4', ''),
        ('r_a4_err', ''),
        ('r_rms', u.maggy/u.arcsec**2),
        ('r_pix_stddev', u.maggy/u.arcsec**2),
        ('r_stop_code', ''),
        ('r_ndata', ''),
        ('r_nflag', ''),
        ('r_niter', ''),

        ('z_sma', u.pixel),
        ('z_eps', ''),
        ('z_eps_err', ''),
        ('z_pa', u.degree),
        ('z_pa_err', u.degree),
        ('z_intens', u.maggy/u.arcsec**2),
        ('z_intens_err', u.maggy/u.arcsec**2),
        ('z_x0', u.pixel),
        ('z_x0_err', u.pixel),
        ('z_y0', u.pixel),
        ('z_y0_err', u.pixel),
        ('z_a3', ''),
        ('z_a3_err', ''),
        ('z_a4', ''),
        ('z_a4_err', ''),
        ('z_rms', u.maggy/u.arcsec**2),
        ('z_pix_stddev', u.maggy/u.arcsec**2),
        ('z_stop_code', ''),
        ('z_ndata', ''),
        ('z_nflag', ''),
        ('z_niter', ''),

        #('cog_smaunit', ''),

        ('g_cog_sma', u.arcsec),
        ('g_cog_mag', u.mag),
        ('g_cog_magerr', u.mag),
        ('g_cog_params_mtot', u.mag),
        ('g_cog_params_m0', u.mag),
        ('g_cog_params_alpha1', ''),
        ('g_cog_params_alpha2', ''),
        ('g_cog_params_chi2', ''),

        ('r_cog_sma', u.arcsec),
        ('r_cog_mag', u.mag),
        ('r_cog_magerr', u.mag),
        ('r_cog_params_mtot', u.mag),
        ('r_cog_params_m0', u.mag),
        ('r_cog_params_alpha1', ''),
        ('r_cog_params_alpha2', ''),
        ('r_cog_params_chi2', ''),

        ('z_cog_sma', u.arcsec),
        ('z_cog_mag', u.mag),
        ('z_cog_magerr', u.mag),
        ('z_cog_params_mtot', u.mag),
        ('z_cog_params_m0', u.mag),
        ('z_cog_params_alpha1', ''),
        ('z_cog_params_alpha2', ''),
        ('z_cog_params_chi2', ''),

        ('radius_sb23', u.arcsec),
        ('radius_sb23_err', u.arcsec),
        ('radius_sb24', u.arcsec),
        ('radius_sb24_err', u.arcsec),
        ('radius_sb25', u.arcsec),
        ('radius_sb25_err', u.arcsec),
        ('radius_sb25.5', u.arcsec),
        ('radius_sb25.5_err', u.arcsec),
        ('radius_sb26', u.arcsec),
        ('radius_sb26_err', u.arcsec),

        ('g_mag_sb23', u.mag),
        ('g_mag_sb23_err', u.mag),
        ('g_mag_sb24', u.mag),
        ('g_mag_sb24_err', u.mag),
        ('g_mag_sb25', u.mag),
        ('g_mag_sb25_err', u.mag),
        ('g_mag_sb25.5', u.mag),
        ('g_mag_sb25.5_err', u.mag),
        ('g_mag_sb26', u.mag),
        ('g_mag_sb26_err', u.mag),

        ('r_mag_sb23', u.mag),
        ('r_mag_sb23_err', u.mag),
        ('r_mag_sb24', u.mag),
        ('r_mag_sb24_err', u.mag),
        ('r_mag_sb25', u.mag),
        ('r_mag_sb25_err', u.mag),
        ('r_mag_sb25.5', u.mag),
        ('r_mag_sb25.5_err', u.mag),
        ('r_mag_sb26', u.mag),
        ('r_mag_sb26_err', u.mag),

        ('z_mag_sb23', u.mag),
        ('z_mag_sb23_err', u.mag),
        ('z_mag_sb24', u.mag),
        ('z_mag_sb24_err', u.mag),
        ('z_mag_sb25', u.mag),
        ('z_mag_sb25_err', u.mag),
        ('z_mag_sb25.5', u.mag),
        ('z_mag_sb25.5_err', u.mag),
        ('z_mag_sb26', u.mag),
        ('z_mag_sb26_err', u.mag),
        ]
    return cols

def write_ellipsefit(galaxy, galaxydir, ellipsefit, filesuffix='', galaxyid='',
                     galaxyinfo=None, refband='r', verbose=False):
    """Write out a FITS file based on the output of
    legacyhalos.ellipse.ellipse_multiband..

    ellipsefit - input dictionary

    """
    from astropy.table import QTable
    #from astropy.io import fits
    
    if galaxyid.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxyid)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)
        
    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse.fits'.format(galaxy, fsuff, galid))

    # Turn the ellipsefit dictionary into a FITS table, starting with the
    # galaxyinfo dictionary (if provided).
    out = QTable()
    if galaxyinfo:
        for key in galaxyinfo.keys():
            data = galaxyinfo[key][0]
            if np.isscalar(data):
                data = np.atleast_1d(data)
            else:
                data = np.atleast_2d(data)
            unit = galaxyinfo[key][1] # add units
            if type(unit) is not str:
                data *= unit
            col = Column(name=key, data=data)
            out.add_column(col)

    # First, unpack the nested dictionaries.
    datadict = {}
    for key in ellipsefit.keys():
        #if type(ellipsefit[key]) is dict: # obsolete
        #    for key2 in ellipsefit[key].keys():
        #        datadict['{}_{}'.format(key, key2)] = ellipsefit[key][key2]
        #else:
        #    datadict[key] = ellipsefit[key]
        datadict[key] = ellipsefit[key]
    del ellipsefit

    # Add to the data table
    datakeys = datadict.keys()
    for key, unit in _get_ellipse_datamodel(refband=refband):
        if key not in datakeys:
            raise ValueError('Data model change -- no column {} for galaxy {}!'.format(key, galaxy))
        data = datadict[key]
        if np.isscalar(data):# or len(np.array(data)) > 1:
            data = np.atleast_1d(data)
        else:
            data = np.atleast_2d(data)
        if type(unit) is not str:
            data *= unit
        col = Column(name=key, data=data)
        out.add_column(col)

    if np.logical_not(np.all(np.isin([*datakeys], out.colnames))):
        raise ValueError('Data model change -- non-documented columns have been added to ellipsefit dictionary!')

    hdr = legacyhalos_header()

    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    #out.write(ellipsefitfile, overwrite=True)
    fitsio.write(ellipsefitfile, out.as_array(), extname='ELLIPSE', header=hdr, clobber=True)

def read_ellipsefit(galaxy, galaxydir, filesuffix='', galaxyid='', verbose=True):
    """Read the output of write_ellipsefit. Convert the astropy Table into a
    dictionary so we can use a bunch of legacy code.

    """
    if galaxyid.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxyid)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)

    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse.fits'.format(galaxy, fsuff, galid))
        
    if os.path.isfile(ellipsefitfile):
        data = Table.read(ellipsefitfile)

        # Convert (back!) into a dictionary.
        ellipsefit = {}
        for key in data.colnames:
            val = data[key].tolist()[0]
            if np.logical_not(np.isscalar(val)) and len(val) > 0:
                val = np.array(val)
            ellipsefit[key] = val
    else:
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
    from astropy.io import fits
    
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

def _get_psfsize_and_depth(tractor, bands, pixscale, incenter=False):
    """Support function for read_multiband. Compute the average PSF size (in arcsec)
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
            these = np.where((tractor.bx >= np.int(H / 2 - dH)) * (tractor.bx <= np.int(H / 2 + dH)) *
                             (tractor.by >= np.int(H / 2 - dH)) * (tractor.by <= np.int(H / 2 + dH)) *
                             (tractor.get(psfdepthcol) > 0))[0]
        else:
            these = np.where(tractor.get(psfdepthcol) > 0)[0]
            
        if len(these) == 0:
            print('No sources at the center of the field, unable to get PSF size!')
            continue

        # Get the PSF size and image depth.
        psfsize = tractor.get(psfsizecol)[these]   # [FWHM, arcsec]
        psfdepth = tractor.get(psfdepthcol)[these] # [AB mag, 5-sigma]
        psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

        out['psfsigma_{}'.format(filt)] = np.median(psfsigma).astype('f4') 
        out['psfsize_{}'.format(filt)] = np.median(psfsize).astype('f4') 
        #out['psfsize_min_{}'.format(filt)] = np.min(psfsize).astype('f4')
        #out['psfsize_max_{}'.format(filt)] = np.max(psfsize).astype('f4')

        out['psfdepth_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.median(psfdepth)))).astype('f4') 
        #out['psfdepth_min_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.min(psfdepth)))).astype('f4')
        #out['psfdepth_max_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.max(psfdepth)))).astype('f4')
        
    return out

def _read_and_mask(data, bands, refband, filt2imfile, filt2pixscale, tractor,
                   central_galaxy=None, central_galaxy_id=None, fill_value=0.0,
                   starmask=None, verbose=False, largegalaxy=False):
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
        sz = image.shape

        # Initialize the mask based on the inverse variance
        if 'invvar' in filt2imfile[filt].keys():
            if verbose:
                print('Reading {}'.format(filt2imfile[filt]['invvar']))
            invvar = fitsio.read(filt2imfile[filt]['invvar'])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # Cache the reference image header for the next step.
        if filt == refband:
            HH, WW = sz
            data['{}_width'.format(refband)] = np.float32(WW)
            data['{}_height'.format(refband)] = np.float32(HH)
            refhdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)

        # Add in the star mask, resizing if necessary for this image/pixel scale.
        if starmask is not None:
            if sz != starmask.shape:
                from skimage.transform import resize
                _starmask = resize(starmask, sz, mode='reflect')
                mask = np.logical_or(mask, _starmask)
            else:
                mask = np.logical_or(mask, starmask)

        # Flag significant residual pixels after subtracting *all* the models
        # (we will restore the pixels of the galaxies of interest below).
        resid = gaussian_filter(image - model, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        if residual_mask is None:
            residual_mask = np.abs(resid) > 5*sig
        else:
            residual_mask = np.logical_or(residual_mask, np.abs(resid) > 5*sig)

        # Dilate the mask, mask out a 10% border, and pack into a dictionary.
        mask = binary_dilation(mask, iterations=2)
        edge = np.int(0.02*sz[0])
        mask[:edge, :] = True
        mask[:, :edge] = True
        mask[:, sz[0]-edge:] = True
        mask[sz[0]-edge:, :] = True

        data[filt] = ma.masked_array(image, mask) # [nanomaggies]
        ma.set_fill_value(data[filt], fill_value)

        if invvar is not None:
            var = np.zeros_like(invvar)
            ok = invvar > 0
            var[ok] = 1 / invvar[ok]
            data['{}_var_'.format(filt)] = var # [nanomaggies**2]
            #data['{}_var'.format(filt)] = var / thispixscale**4 # [nanomaggies**2/arcsec**4]
            if np.any(invvar < 0):
                print('Warning! Negative pixels in the {}-band inverse variance map!'.format(filt))
                #pdb.set_trace()

    # Now, build the model image in the reference band using the mean PSF.
    if verbose:
        print('Reading {}'.format(filt2imfile[refband]['psf']))
    psfimg = fitsio.read(filt2imfile[refband]['psf'])
    psf = PixelizedPSF(psfimg)
    xobj, yobj = np.ogrid[0:HH, 0:WW]

    nbox = 5
    box = np.arange(nbox)-nbox // 2
    #box = np.meshgrid(np.arange(nbox), np.arange(nbox))[0]-nbox//2

    wcs = Tan(filt2imfile[refband]['image'], 1)
    mjd_tai = refhdr['MJD_MEAN'] # [TAI]

    twcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
    data['wcs'] = twcs

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if central_galaxy is None:
        central_galaxy = np.array([np.argmin((tractor.bx - HH/2)**2 + (tractor.by - WW/2)**2)])
        central_galaxy_id = None
    data['central_galaxy_id'] = central_galaxy_id

    #print('Import hack!')
    #import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm
    
    # Now, loop through each 'central_galaxy' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(central_galaxy):
        if verbose:
            print('Building masked image for central {}/{}.'.format(ii+1, len(central_galaxy)))

        # Build the model image (of every object except the central)
        # on-the-fly. Need to be smarter about Tractor sources of resolved
        # structure (i.e., sources that "belong" to the central).
        nocentral = np.delete(np.arange(len(tractor)), central)
        srcs = tractor.copy()
        srcs.cut(nocentral)
        model_nocentral = srcs2image(srcs, twcs, band=refband, pixelized_psf=psf)

        # Mask all previous (brighter) central galaxies, if any.
        img, newmask = ma.getdata(data[refband]) - model_nocentral, ma.getmask(data[refband])
        for jj in np.arange(ii):
            geo = data['mge'][jj] # the previous galaxy

            # Do this step iteratively to capture the possibility where the
            # previous galaxy has masked the central pixels of the *current*
            # galaxy, in each iteration reducing the size of the mask.
            for shrink in np.arange(0.1, 1.05, 0.05)[::-1]:
                maxis = shrink * geo['majoraxis']
                _mask = ellipse_mask(geo['xmed'], geo['ymed'], maxis, maxis * (1-geo['eps']),
                                     np.radians(geo['theta']-90), xobj, yobj)
                notok = False
                for xb in box:
                    for yb in box:
                        if _mask[int(yb+tractor.by[central]), int(xb+tractor.bx[central])]:
                            notok = True
                            break
                if notok:
                #if _mask[int(tractor.by[central]), int(tractor.bx[central])]:
                    print('The previous central has masked the current central with shrink factor {:.2f}'.format(shrink))
                else:
                    break
            newmask = ma.mask_or(_mask, newmask)

        # Next, get the basic galaxy geometry and pack it into a dictionary. If
        # the object of interest has been masked by, e.g., an adjacent star
        # (see, e.g., IC4041), temporarily unmask those pixels using the Tractor
        # geometry.
        
        minsb = 10**(-0.4*(27.5-22.5)) / filt2pixscale[refband]**2
        #import matplotlib.pyplot as plt ; plt.clf()
        #mgegalaxy = find_galaxy(img / filt2pixscale[refband]**2, nblob=1, binning=3, quiet=not verbose, plot=True, level=minsb)
        #mgegalaxy = find_galaxy(img / filt2pixscale[refband]**2, nblob=1, fraction=0.1, binning=3, quiet=not verbose, plot=True)
        notok, val = False, []
        for xb in box:
            for yb in box:
                #print(xb, yb, val)
                val.append(newmask[int(yb+tractor.by[central]), int(xb+tractor.bx[central])])
                
        # Use np.any() here to capture the case where a handful of the central
        # pixels are masked due to, e.g., saturation, which if we don't do, will
        # cause issues in the ellipse-fitting (specifically with
        # CentralEllipseFitter(censamp).fit() if the very central pixel is
        # masked).  For a source masked by a star, np.all() would have worked
        # fine.
        if np.any(val):
            notok = True
            
        if notok:
            print('Central position has been masked, possibly by a star (or saturated core).')
            xmed, ymed = tractor.by[central], tractor.bx[central]
            #if largegalaxy:
            #    ba = tractor.ba_leda[central]
            #    pa = tractor.pa_leda[central]
            #    maxis = tractor.d25_leda[central] * 60 / 2 / filt2pixscale[refband] # [pixels]
            ee = np.hypot(tractor.shape_e1[central], tractor.shape_e2[central])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[central], tractor.shape_e1[central]) / 2))
            pa = pa % 180
            maxis = 1.5 * tractor.shape_r[central] / filt2pixscale[refband] # [pixels]
            theta = (270 - pa) % 180
                
            fixmask = ellipse_mask(xmed, ymed, maxis, maxis*ba, np.radians(theta-90), xobj, yobj)
            newmask[fixmask] = ma.nomask
        
        #import matplotlib.pyplot as plt ; plt.clf()
        mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), 
                                nblob=1, binning=3, level=minsb)#, plot=True)#, quiet=not verbose
        #plt.savefig('junk.png') ; pdb.set_trace()

        # Above, we used the Tractor positions, so check one more time here with
        # the light-weighted positions, which may have shifted into a masked
        # region (e.g., check out the interacting pair PGC052639 & PGC3098317).
        val = []
        for xb in box:
            for yb in box:
                val.append(newmask[int(xb+mgegalaxy.xmed), int(yb+mgegalaxy.ymed)])
        if np.any(val):
            notok = True

        # If we fit the geometry by unmasking pixels using the Tractor fit then
        # we're probably sitting inside the mask of a bright star, so call
        # find_galaxy a couple more times to try to grow the "unmasking".
        if notok:
            print('Iteratively unmasking pixels:')
            print('  r={:.2f} pixels'.format(maxis))
            maxis = 1.0 * mgegalaxy.majoraxis # [pixels]
            prevmaxis, iiter, maxiter = 0.0, 0, 4
            while (maxis > prevmaxis) and (iiter < maxiter):
                #print(prevmaxis, maxis, iiter, maxiter)
                print('  r={:.2f} pixels'.format(maxis))
                fixmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed,
                                       maxis, maxis * (1-mgegalaxy.eps), 
                                       np.radians(mgegalaxy.theta-90), xobj, yobj)
                newmask[fixmask] = ma.nomask
                mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), 
                                        nblob=1, binning=3, quiet=True, plot=False, level=minsb)
                prevmaxis = maxis.copy()
                maxis = 1.2 * mgegalaxy.majoraxis # [pixels]
                iiter += 1

        #plt.savefig('junk.png') ; pdb.set_trace()
        print(mgegalaxy.xmed, tractor.by[central], mgegalaxy.ymed, tractor.bx[central])
        maxshift = 10
        if (np.abs(mgegalaxy.xmed-tractor.by[central]) > maxshift or # note [xpeak,ypeak]-->[by,bx]
            np.abs(mgegalaxy.ymed-tractor.bx[central]) > maxshift):
            print('Peak position has moved by more than {} pixels---falling back on Tractor geometry!'.format(maxshift))
            #import matplotlib.pyplot as plt ; plt.clf()
            #mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), nblob=1, binning=3, quiet=False, plot=True, level=minsb)
            #plt.savefig('junk.png') ; pdb.set_trace()
            #pdb.set_trace()
            largeshift = True
            
            ee = np.hypot(tractor.shape_e1[central], tractor.shape_e2[central])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[central], tractor.shape_e1[central]) / 2))
            mgegalaxy.xmed = tractor.by[central]
            mgegalaxy.ymed = tractor.bx[central]
            mgegalaxy.xpeak = tractor.by[central]
            mgegalaxy.ypeak = tractor.bx[central]
            mgegalaxy.eps = 1 - ba
            mgegalaxy.pa = pa % 180
            mgegalaxy.theta = (270 - pa) % 180
            mgegalaxy.majoraxis = 2 * tractor.shape_r[central] / filt2pixscale[refband] # [pixels]
            print('  r={:.2f} pixels'.format(mgegalaxy.majoraxis))
        else:
            largeshift = False

        #if tractor.ref_id[central] == 474614:
        #    import matplotlib.pyplot as plt
        #    plt.imshow(mask, origin='lower')
        #    plt.savefig('junk.png')
        #    pdb.set_trace()
            
        radec_med = data['wcs'].pixelToPosition(mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        radec_peak = data['wcs'].pixelToPosition(mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {'largeshift': largeshift,
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

        # Now, loop on each filter and build a custom image and mask for each
        # central. Specifically, pack the model-subtracted images images
        # corresponding to each (unique) central into a list. Note that there's
        # a little bit of code to deal with different pixel scales but this case
        # requires more work.
        
        #for filt in [refband]:
        for filt in bands:
            thispixscale = filt2pixscale[filt]
            
            imagekey, varkey = '{}_masked'.format(filt), '{}_var'.format(filt)
            if imagekey not in data.keys():
                data[imagekey], data[varkey] = [], []

            factor = filt2pixscale[refband] / filt2pixscale[filt]
            majoraxis = 1.5 * factor * mgegalaxy.majoraxis # [pixels]

            # Grab the pixels belonging to this galaxy so we can unmask them below.
            central_mask = ellipse_mask(mge['xmed'] * factor, mge['ymed'] * factor, 
                                        majoraxis, majoraxis * (1-mgegalaxy.eps), 
                                        np.radians(mgegalaxy.theta-90), xobj, yobj)
            if np.sum(central_mask) == 0:
                print('No pixels belong to the central galaxy---this is bad!')
                pdb.set_trace()

            # Build the mask from the (cumulative) residual-image mask and the
            # inverse variance mask for this galaxy, but then "unmask" the
            # pixels belonging to the central.
            _residual_mask = residual_mask.copy()
            _residual_mask[central_mask] = ma.nomask
            mask = ma.mask_or(_residual_mask, newmask, shrink=False)

            # Need to be smarter about the srcs list...
            srcs = tractor.copy()
            srcs.cut(nocentral)
            model_nocentral = srcs2image(srcs, twcs, band=filt, pixelized_psf=psf)

            # Convert to surface brightness and 32-bit precision.
            img = (ma.getdata(data[filt]) - model_nocentral) / thispixscale**2 # [nanomaggies/arcsec**2]
            img = ma.masked_array(img.astype('f4'), mask)
            var = data['{}_var_'.format(filt)] / thispixscale**4 # [nanomaggies/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #img.filled(fill_value)
            data[imagekey].append(img)
            data[varkey].append(var)

            #if tractor.ref_id[central] == 474614:
            #    import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm ; plt.clf()
            #    thisimg = np.log10(data[imagekey][ii]) ; norm = simple_norm(thisimg, 'log') ; plt.imshow(thisimg, origin='lower', norm=norm) ; plt.savefig('junk{}.png'.format(ii+1))
            #    pdb.set_trace()
            
    # Cleanup?
    for filt in bands:
        del data[filt]
        del data['{}_var_'.format(filt)]

    return data

def read_multiband(galaxy, galaxydir, bands=('g', 'r', 'z'), refband='r',
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   sdss_pixscale=0.396, return_sample=False,
                   #central_galaxy_id=None,
                   sdss=False, largegalaxy=False, pipeline=False, verbose=False):
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
                            'sample': '{}-sample'.format(prefix),
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
                if verbose:
                    print('File {} not found.'.format(imfile))
                found_data = False

    data = dict()
    if not found_data:
        if return_sample:
            return data, Table()
        else:
            return data

    # Pack some preliminary info into the dictionary.
    data['failed'] = False # be optimistic!
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)

    if 'NUV' in bands:
        data['galex_pixscale'] = galex_pixscale
    if 'W1' in bands:
        data['unwise_pixscale'] = unwise_pixscale

    # Read the tractor and full-sample catalogs.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    if os.path.isfile(samplefile):
        sample = Table(fitsio.read(samplefile, upper=True))
        if verbose:
            print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))

    tractorfile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['tractor']))
    if os.path.isfile(tractorfile):
        # We ~have~ to read using fits_table because we will turn these catalog
        # entries into Tractor sources later.
        #cols = ['BX', 'BY', 'TYPE', 'REF_CAT', 'REF_ID', 'SERSIC', 'SHAPE_R', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
        #        'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        #tractor = Table(fitsio.read(tractorfile, columns=cols, upper=True))
        cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
                'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                'flux_g', 'flux_r', 'flux_z',
                'nobs_g', 'nobs_r', 'nobs_z',
                'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 
                'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                'psfsize_g', 'psfsize_r', 'psfsize_z']
        tractor = fits_table(tractorfile, columns=cols)
        hdr = fitsio.read_header(tractorfile)
        if verbose:
            print('Read {} sources from {}'.format(len(tractor), tractorfile))
        data.update(_get_psfsize_and_depth(tractor, bands, pixscale, incenter=False))

    # Read the maskbits image and build the starmask.
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
    # the field, otherwise, take the object closest to the center of the mosaic
    # (which we figure out in _read_and_mask, after we know the size of the
    # mosaic).
    if largegalaxy:
        # I'm going to be pedantic here to be sure I get it right (np.isin
        # doens't preserve order)--
        msg = []
        islslga = ['L' in refcat for refcat in tractor.ref_cat] # e.g., L6
        minsize = 2.0     # [arcsec]
        minsize_rex = 5.0 # minimum size for REX [arcsec]
        central_galaxy, reject_galaxy, keep_galaxy = [], [], []
        data['tractor_flags'] = {}
        for ii, sid in enumerate(sample['ID']):
            I = np.where((sid == tractor.ref_id) * islslga)[0]
            if len(I) == 0: # dropped by Tractor
                reject_galaxy.append(ii)
                data['tractor_flags'].update({str(sid): 'dropped'})
                msg.append('Dropped by Tractor (spurious?)')
            else:
                r50 = tractor.shape_r[I][0]
                refflux = tractor.get('flux_{}'.format(refband))[I][0]
                ng, nr, nz = tractor.nobs_g[I][0], tractor.nobs_z[I][0], tractor.nobs_z[I][0]
                if ng < 1 or nr < 1 or nz < 1:
                    reject_galaxy.append(ii)
                    data['tractor_flags'].update({str(sid): 'nogrz'})
                    msg.append('Missing 3-band coverage')
                elif tractor.type[I] == 'PSF': # always reject
                    reject_galaxy.append(ii)
                    data['tractor_flags'].update({str(sid): 'psf'})
                    msg.append('Tractor type=PSF')
                elif refflux <= 0:
                    reject_galaxy.append(ii)
                    data['tractor_flags'].update({str(sid): 'negflux'})
                    msg.append('{}-band flux={:.3g} (<=0)'.format(refband, refflux))
                elif r50 < minsize:
                    reject_galaxy.append(ii)
                    data['tractor_flags'].update({str(sid): 'anytype_toosmall'})
                    msg.append('type={}, r50={:.3f} (<{:.1f}) arcsec'.format(tractor.type[I], r50, minsize))
                elif tractor.type[I] == 'REX':
                    if r50 > minsize_rex: # REX must have a minimum size
                        keep_galaxy.append(ii)
                        central_galaxy.append(I)
                    else:
                        reject_galaxy.append(ii)
                        data['tractor_flags'].update({str(sid): 'rex_toosmall'})
                        msg.append('Tractor type=REX & r50={:.3f} (<{:.1f}) arcsec'.format(r50, minsize_rex))
                else:
                    keep_galaxy.append(ii)
                    central_galaxy.append(I)

        if len(reject_galaxy) > 0:
            reject_galaxy = np.hstack(reject_galaxy)
            for jj, rej in enumerate(reject_galaxy):
                print('  Dropping {} (ID={}, RA, Dec = {:.7f} {:.7f}): {}'.format(
                    sample[rej]['GALAXY'], sample[rej]['ID'], sample[rej]['RA'], sample[rej]['DEC'], msg[jj]))

        if len(central_galaxy) > 0:
            keep_galaxy = np.hstack(keep_galaxy)
            central_galaxy = np.hstack(central_galaxy)
            sample = sample[keep_galaxy]
        else:
            data['failed'] = True
            if return_sample:
                return data, Table()
            else:
                return data

        #sample = sample[np.searchsorted(sample['ID'], tractor.ref_id[central_galaxy])]
        assert(np.all(sample['ID'] == tractor.ref_id[central_galaxy]))
        
        tractor.d25_leda = np.zeros(len(tractor), dtype='f4')
        tractor.pa_leda = np.zeros(len(tractor), dtype='f4')
        tractor.ba_leda = np.zeros(len(tractor), dtype='f4')
        tractor.d25_leda[central_galaxy] = sample['D25_LEDA']
        tractor.pa_leda[central_galaxy] = sample['PA_LEDA']
        tractor.ba_leda[central_galaxy] = sample['BA_LEDA']
        
        # Do we need to take into account the elliptical mask of each source??
        srt = np.argsort(tractor.flux_r[central_galaxy])[::-1]
        central_galaxy = central_galaxy[srt]
        print('Sort by flux! ', tractor.flux_r[central_galaxy])
        central_galaxy_id = tractor.ref_id[central_galaxy]
    else:
        central_galaxy, central_galaxy_id = None, None

    data = _read_and_mask(data, bands, refband, filt2imfile, filt2pixscale,
                          tractor, central_galaxy=central_galaxy,
                          central_galaxy_id=central_galaxy_id,
                          starmask=starmask, verbose=verbose,
                          largegalaxy=largegalaxy)
    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.imshow(np.log10(data['r_masked'][0]), origin='lower') ; plt.savefig('junk1.png')
    #plt.clf() ; plt.imshow(np.log10(data['r_masked'][1]), origin='lower') ; plt.savefig('junk2.png')
    #plt.clf() ; plt.imshow(np.log10(data['r_masked'][2]), origin='lower') ; plt.savefig('junk3.png')
    #pdb.set_trace()

    if return_sample:
        return data, sample
    else:
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
    from astropy.table import hstack
    
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
