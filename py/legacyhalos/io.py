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

# build out the FITS header
def legacyhalos_header(hdr=None):
    """Build a header with code versions, etc.

    """
    import subprocess
    from astropy.io import fits
    import pydl
    import legacyhalos

    if False:
        if hdr is None:
            hdr = fitsio.FITSHDR()

        cmd = 'cd {} && git describe --tags'.format(os.path.dirname(legacyhalos.__file__))
        ver = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
        hdr.add_record(dict(name='LEGHALOV', value=ver, comment='legacyhalos git version'))

        depvers, headers = [], []
        for name, pkg in [('pydl', pydl)]:
            hdr.add_record(dict(name=name, value=pkg.__version__, comment='{} version'.format(name)))
    else:
        if hdr is None:
            hdr = fits.header.Header()

        cmd = 'cd {} && git describe --tags'.format(os.path.dirname(legacyhalos.__file__))
        ver = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
        hdr['LEGHALOV'] = (ver, 'legacyhalos git version')

        depvers, headers = [], []
        for name, pkg in [('pydl', pydl)]:
            hdr[name] = (pkg.__version__, '{} version'.format(name))

    return hdr
    
def _missing_files_one(args):
    """Wrapper for the multiprocessing."""
    return missing_files_one(*args)

def missing_files_one(checkfile, dependsfile, clobber):
#def missing_files_one(galaxy, galaxydir, filesuffix, dependson, clobber):
    #checkfile = os.path.join(galaxydir, '{}{}'.format(galaxy, filesuffix))
    #print('missing_files_one: ', checkfile)
    #print(checkfile, dependsfile, clobber)
    from pathlib import Path
    #from glob import glob
    #if os.path.isfile(checkfile) and clobber is False:
    #checkfile = glob(checkfile)
    #if len(checkfile) > 0:
    #    checkfile = checkfile[0]
    #else:
    #    checkfile = '_'
    if Path(checkfile).exists() and clobber is False:
        # Is the stage that this stage depends on done, too?
        #print(checkfile, dependsfile, clobber)
        if dependsfile is None:
            return 'done'
        else:
            if Path(dependsfile).exists():
            #if os.path.isfile(dependsfile):
                return 'done'
            else:
                return 'todo'
    else:
        #print('missing_files_one: ', checkfile)
        # Did this object fail?
        if checkfile[-6:] == 'isdone':
            failfile = checkfile[:-6]+'isfail'
            if Path(failfile).exists():
            #if os.path.isfile(failfile):
                if clobber is False:
                    return 'fail'
                else:
                    os.remove(failfile)
                    return 'todo'
            else:
                return 'todo'
            #if dependsfile is None:
            #    return 'todo'
            #else:
            #    if os.path.isfile(dependsfile):
            #        return 'todo'
            #    else:
            #        return 'todo'
        else:
            if dependsfile is not None:
                if os.path.isfile(dependsfile):
                    return 'todo'
                else:
                    print('Missing depends file {}'.format(dependsfile))
                    return 'fail'
            else:
                return 'todo'

        return 'todo'
            
def get_run(onegal, racolumn='RA', deccolumn='DEC'):
    """Get the run based on a simple declination cut."""
    if onegal[deccolumn] > 32.375:
        if onegal[racolumn] < 45 or onegal[racolumn] > 315:
            run = 'south'
        else:
            run = 'north'
    else:
        run = 'south'
    return run

# ellipsefit data model
def _get_ellipse_datamodel(sbthresh, bands=['g', 'r', 'z']):
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

        #('mw_transmission_g', ''),
        #('mw_transmission_r', ''),
        #('mw_transmission_z', ''),

        ('refband_width', u.pixel),
        ('refband_height', u.pixel)]

    for band in bands:
        cols.append(('{}_sma'.format(band), u.pixel))
        cols.append(('{}_intens'.format(band), u.maggy/u.arcsec**2))
        cols.append(('{}_intens_err'.format(band), u.maggy/u.arcsec**2))
        cols.append(('{}_eps'.format(band), ''))
        cols.append(('{}_eps_err'.format(band), ''))
        cols.append(('{}_pa'.format(band), u.degree))
        cols.append(('{}_pa_err'.format(band), u.degree))
        cols.append(('{}_x0'.format(band), u.pixel))
        cols.append(('{}_x0_err'.format(band), u.pixel))
        cols.append(('{}_y0'.format(band), u.pixel))
        cols.append(('{}_y0_err'.format(band), u.pixel))
        cols.append(('{}_a3'.format(band), '')) # units?
        cols.append(('{}_a3_err'.format(band), ''))
        cols.append(('{}_a4'.format(band), ''))
        cols.append(('{}_a4_err'.format(band), ''))
        cols.append(('{}_rms'.format(band), u.maggy/u.arcsec**2))
        cols.append(('{}_pix_stddev'.format(band), u.maggy/u.arcsec**2))
        cols.append(('{}_stop_code'.format(band), ''))
        cols.append(('{}_ndata'.format(band), ''))
        cols.append(('{}_nflag'.format(band), ''))
        cols.append(('{}_niter'.format(band), ''))
        cols.append(('{}_cog_sma'.format(band), u.arcsec))
        cols.append(('{}_cog_mag'.format(band), u.mag))
        cols.append(('{}_cog_magerr'.format(band), u.mag))
        cols.append(('{}_cog_params_mtot'.format(band), u.mag))
        cols.append(('{}_cog_params_m0'.format(band), u.mag))
        cols.append(('{}_cog_params_alpha1'.format(band), ''))
        cols.append(('{}_cog_params_alpha2'.format(band), ''))
        cols.append(('{}_cog_params_chi2'.format(band), ''))
        cols.append(('{}_cog_sma50'.format(band), u.arcsec))

    for thresh in sbthresh:
        cols.append(('sma_sb{:0g}'.format(thresh), u.arcsec))
        cols.append(('sma_sb{:0g}_err'.format(thresh), u.arcsec))
        
    for band in bands:
        for thresh in sbthresh:
            cols.append(('{}_mag_sb{:0g}'.format(band, thresh), u.mag))
            cols.append(('{}_mag_sb{:0g}_err'.format(band, thresh), u.mag))

    return cols

def write_ellipsefit(galaxy, galaxydir, ellipsefit, filesuffix='', galaxy_id='',
                     galaxyinfo=None, refband='r', sbthresh=None, verbose=False):
    """Write out a FITS file based on the output of
    legacyhalos.ellipse.ellipse_multiband..

    ellipsefit - input dictionary

    """
    from astropy.io import fits
    from astropy.table import QTable

    if type(galaxy_id) is not str:
        galaxy_id = str(galaxy_id)

    if galaxy_id.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxy_id)
    if filesuffix.strip() == '':
        fsuff = ''
    else:
        fsuff = '-{}'.format(filesuffix)
        
    ellipsefitfile = os.path.join(galaxydir, '{}{}{}-ellipse.fits'.format(galaxy, fsuff, galid))

    if sbthresh is None:
        from legacyhalos.ellipse import REF_SBTHRESH as sbthresh
    
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
                #data *= unit
                data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
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
    for key, unit in _get_ellipse_datamodel(sbthresh):
        if key not in datakeys:
            raise ValueError('Data model change -- no column {} for galaxy {}!'.format(key, galaxy))
        data = datadict[key]
        if np.isscalar(data):# or len(np.array(data)) > 1:
            data = np.atleast_1d(data)
        #elif len(data) == 0:
        #    data = np.atleast_1d(data)
        else:
            data = np.atleast_2d(data)
        if type(unit) is not str:
            data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
        col = Column(name=key, data=data)
        #if 'z_cog' in key:
        #    print(key)
        #    pdb.set_trace()
        out.add_column(col)
        
    if np.logical_not(np.all(np.isin([*datakeys], out.colnames))):
        raise ValueError('Data model change -- non-documented columns have been added to ellipsefit dictionary!')

    # uppercase!
    for col in out.colnames:
        out.rename_column(col, col.upper())

    hdr = legacyhalos_header()

    #for col in out.colnames:
    #    print(col, out[col])
    hdu = fits.convenience.table_to_hdu(out)
    hdu.header['EXTNAME'] = 'ELLIPSE'
    hdu.header.update(hdr)
    hdu.add_checksum()

    hdu0 = fits.PrimaryHDU()
    hdu0.header['EXTNAME'] = 'PRIMARY'
    hx = fits.HDUList([hdu0, hdu])

    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    tmpfile = ellipsefitfile+'.tmp'
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, ellipsefitfile)
    #hx.writeto(ellipsefitfile, overwrite=True, checksum=True)
    
    #out.write(ellipsefitfile, overwrite=True)
    #fitsio.write(ellipsefitfile, out.as_array(), extname='ELLIPSE', header=hdr, clobber=True)

def read_ellipsefit(galaxy, galaxydir, filesuffix='', galaxy_id='', verbose=True):
    """Read the output of write_ellipsefit. Convert the astropy Table into a
    dictionary so we can use a bunch of legacy code.

    """
    if galaxy_id.strip() == '':
        galid = ''
    else:
        galid = '-{}'.format(galaxy_id)
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
            ellipsefit[key.lower()] = val # lowercase!
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

def _get_psfsize_and_depth(tractor, bands, pixscale, incenter=False):
    """Support function for read_multiband. Compute the average PSF size (in arcsec)
    and depth (in 5-sigma AB mags) in each bandpass based on the Tractor
    catalog.

    """
    out = {}

    # Optionally choose sources in the center of the field.
    H = np.max(tractor.bx) - np.min(tractor.bx)
    W = np.max(tractor.by) - np.min(tractor.by)
    if incenter:
        dH = 0.1 * H
        these = np.where((tractor.bx >= np.int(H / 2 - dH)) * (tractor.bx <= np.int(H / 2 + dH)) *
                         (tractor.by >= np.int(H / 2 - dH)) * (tractor.by <= np.int(H / 2 + dH)))[0]
    else:
        #these = np.where(tractor.get(psfdepthcol) > 0)[0]
        these = np.arange(len(tractor))
    
    # Get the average PSF size and depth in each bandpass.
    for filt in bands:
        psfsizecol = 'psfsize_{}'.format(filt.lower())
        psfdepthcol = 'psfdepth_{}'.format(filt.lower())
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfsizecol)[these] > 0)[0]
            if len(good) == 0:
                print('  No good measurements of the PSF size in band {}!'.format(filt))
                out['psfsigma_{}'.format(filt)] = np.float32(0.0)
                out['psfsize_{}'.format(filt)] = np.float32(0.0)
            else:
                # Get the PSF size and image depth.
                psfsize = tractor.get(psfsizecol)[these][good]   # [FWHM, arcsec]
                psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

                out['psfsigma_{}'.format(filt)] = np.median(psfsigma).astype('f4') 
                out['psfsize_{}'.format(filt)] = np.median(psfsize).astype('f4') 
            
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfdepthcol)[these] > 0)[0]
            if len(good) == 0:
                print('  No good measurements of the PSF depth in band {}!'.format(filt))
                out['psfdepth_{}'.format(filt)] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                out['psfdepth_{}'.format(filt)] = (22.5-2.5*np.log10(1/np.sqrt(np.median(psfdepth)))).astype('f4') 
        
    return out

def _read_image_data(data, filt2imfile, starmask=None, fill_value=0.0,
                     filt2pixscale=None, verbose=False):
    """Helper function for the project-specific read_multiband method.

    Read the multi-band images and inverse variance images and pack them into a
    dictionary. Also create an initial pixel-level mask and handle images with
    different pixel scales (e.g., GALEX and WISE images).

    """
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import gaussian_filter
    from skimage.transform import resize

    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    from astrometry.util.util import Tan
    from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs

    bands, refband = data['bands'], data['refband']

    #refhdr = fitsio.read_header(filt2imfile[refband]['image'], ext=1)
    #refsz = (refhdr['NAXIS1'], refhdr['NAXIS2'])

    # Loop on each filter and return the masked data.
    residual_mask = None
    for filt in bands:
        # Read the data and initialize the mask with the inverse variance image,
        # if available.
        if verbose:
            print('Reading {}'.format(filt2imfile[filt]['image']))
            print('Reading {}'.format(filt2imfile[filt]['model']))
        image = fitsio.read(filt2imfile[filt]['image'])
        hdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)
        model = fitsio.read(filt2imfile[filt]['model'])

        sz = image.shape

        ## optional additional (scalar) sky-subtraction
        #if 'sky' in filt2imfile[filt].keys():
        #    #print('Subtracting!!! ', filt2imfile[filt]['sky'])
        #    image += filt2imfile[filt]['sky']
        #    model += filt2imfile[filt]['sky']

        # GALEX, unWISE need to be resized.
        if starmask.shape == sz:
            doresize = False
        else:
            doresize = True

        # Initialize the mask based on the inverse variance
        if 'invvar' in filt2imfile[filt].keys():
            if verbose:
                print('Reading {}'.format(filt2imfile[filt]['invvar']))
            invvar = fitsio.read(filt2imfile[filt]['invvar'])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # Retrieve the PSF and WCS.
        if filt == refband:
            HH, WW = sz
            data['refband_width'] = WW
            data['refband_height'] = HH
            
        if verbose:
            print('Reading {}'.format(filt2imfile[filt]['psf']))
        psfimg = fitsio.read(filt2imfile[filt]['psf'])
        data['{}_psf'.format(filt)] = PixelizedPSF(psfimg)

        wcs = Tan(filt2imfile[filt]['image'], 1)
        if 'MJD_MEAN' in hdr:
            mjd_tai = hdr['MJD_MEAN'] # [TAI]
            wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
        else:
            wcs = ConstantFitsWcs(wcs)
        data['{}_wcs'.format(filt)] = wcs

        # Add in the star mask, resizing if necessary for this image/pixel scale.
        if doresize:
            _starmask = resize(starmask, mask.shape, mode='reflect')
            mask = np.logical_or(mask, _starmask)
        else:
            mask = np.logical_or(mask, starmask)

        # Flag significant residual pixels after subtracting *all* the models
        # (we will restore the pixels of the galaxies of interest later). Only
        # consider the optical (grz) bands here.
        resid = gaussian_filter(image - model, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        data['{}_sigma'.format(filt)] = sig
        if residual_mask is None:
            residual_mask = np.abs(resid) > 5*sig
        else:
            _residual_mask = np.abs(resid) > 5*sig
            # In grz, use a cumulative residual mask. In UV/IR use an
            # individual-band mask.
            if doresize:
                pass
                #residual_mask = resize(_residual_mask, residual_mask.shape, mode='reflect')
            else:
                residual_mask = np.logical_or(residual_mask, _residual_mask)

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

    data['residual_mask'] = residual_mask

    return data
