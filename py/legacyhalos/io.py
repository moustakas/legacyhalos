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

def read_multiband(galaxy, galaxydir, bands=('g', 'r', 'z'), refband='r',
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   sdss_pixscale=0.396, maskfactor=2.0, fill_value=0.0,
                   pipeline=False, sdss=False, verbose=False):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.  Finally,
    convert to surface brightness by dividing by the pixel area.

    This script needs to be refactored to pull out the unWISE + GALEX stuff (see
    ellipse.legacyhalos_ellipse).

    """
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    from astropy.stats import sigma_clipped_stats

    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import ellipse_mask

    # Dictionary mapping between filter and filename coded up in coadds.py,
    # galex.py, and unwise.py (see the LSLGA product, too).
    if sdss:
        masksuffix = 'sdss-mask-gri'
        bands = ('g', 'r', 'i')
        filt2imfile = {
            'g': ['sdss-image', 'sdss-model-nocentral', 'sdss-model'],
            'r': ['sdss-image', 'sdss-model-nocentral', 'sdss-model'],
            'i': ['sdss-image', 'sdss-model-nocentral', 'sdss-model']
            }
        filt2pixscale =  {
            'g': sdss_pixscale,
            'r': sdss_pixscale,
            'i': sdss_pixscale
            }
    else:
        masksuffix = 'custom-mask-grz'
        if pipeline:
            prefix = 'pipeline'
        else:
            prefix = 'custom'
        filt2imfile = {
            'g': ['{}-image'.format(prefix), '{}-model-nocentral'.format(prefix), '{}-model'.format(prefix), 'invvar'],
            'r': ['{}-image'.format(prefix), '{}-model-nocentral'.format(prefix), '{}-model'.format(prefix), 'invvar'],
            'z': ['{}-image'.format(prefix), '{}-model-nocentral'.format(prefix), '{}-model'.format(prefix), 'invvar']
            }
        filt2pixscale =  {
            'g': pixscale,
            'r': pixscale,
            'z': pixscale
            }
            
    filt2imfile.update({
        'FUV': ['image', 'model-nocentral', 'custom-model', 'invvar'],
        'NUV': ['image', 'model-nocentral', 'custom-model', 'invvar'],
        'W1':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
        'W2':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
        'W3':  ['image', 'model-nocentral', 'custom-model', 'invvar'],
        'W4':  ['image', 'model-nocentral', 'custom-model', 'invvar']
        })
        
    filt2pixscale.update({
        'FUV': galex_pixscale,
        'NUV': galex_pixscale,
        'W1':  unwise_pixscale,
        'W2':  unwise_pixscale,
        'W3':  unwise_pixscale,
        'W4':  unwise_pixscale
        })

    found_data = True
    for filt in bands:
        for ii, imtype in enumerate(filt2imfile[filt]):
            for suffix in ('.fz', ''):
                imfile = os.path.join(galaxydir, '{}-{}-{}.fits{}'.format(galaxy, imtype, filt, suffix))
                if os.path.isfile(imfile):
                    filt2imfile[filt][ii] = imfile
                    break
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    tractorfile = os.path.join(galaxydir, '{}-custom-tractor.fits'.format(galaxy))
    if os.path.isfile(tractorfile):
        tractor = Table(fitsio.read(tractorfile, upper=True))
        print('Read {} sources from {}'.format(len(tractor), tractorfile))
    else:
        print('Missing Tractor catalog {}'.format(tractorfile))
        found_data = False

    data = dict()
    if not found_data:
        return data

    # Treat the optical/grz special because we are most sensitive to residuals
    # and poor model fits because of the higher spatial sampling/resolution.
    opt_residual_mask = []
    for filt in bands:
        if verbose:
            print('Reading {}'.format(filt2imfile[filt][0]))
            print('Reading {}'.format(filt2imfile[filt][2]))
        image = fitsio.read(filt2imfile[filt][0])
        allmodel = fitsio.read(filt2imfile[filt][2]) # read the all-model image

        # Get the average PSF size and depth in each bandpass.
        H, W = image.shape

        psfsizecol = 'PSFSIZE_{}'.format(filt.upper())
        psfdepthcol = 'PSFDEPTH_{}'.format(filt.upper())
        if not psfsizecol in tractor.colnames or not psfdepthcol in tractor.colnames:
            print('Warning: PSFSIZE ({}) or PSFDEPTH ({}) column not found in Tractor catalog!'.format(
                psfsizecol, psfdepthcol))
        else:
            dH = 0.1 * H
            these = ( (tractor['BX'] > np.int(H / 2 - dH)) * (tractor['BX'] < np.int(H / 2 + dH)) *
                      (tractor['BY'] > np.int(H / 2 - dH)) * (tractor['BY'] < np.int(H / 2 + dH)) )
            if np.sum(these) == 0:
                print('No sources at the center of the field, sonable to get PSF size!')
            #data['npsfsize_{}'.format(filt)] = np.sum(these).astype(int)
            data['psfsize_{}'.format(filt)] = np.median(tractor[psfsizecol][these]) # [arcsec]
            data['psfsize_min_{}'.format(filt)] = np.min(tractor[psfsizecol])
            data['psfsize_max_{}'.format(filt)] = np.max(tractor[psfsizecol])

            data['psfdepth_{}'.format(filt)] = 22.5-2.5*np.log10(1/np.sqrt(np.median(tractor[psfdepthcol][these]))) # [AB mag, 5-sigma]
            data['psfdepth_min_{}'.format(filt)] = 22.5-2.5*np.log10(1/np.sqrt(np.min(tractor[psfdepthcol])))
            data['psfdepth_max_{}'.format(filt)] = 22.5-2.5*np.log10(1/np.sqrt(np.max(tractor[psfdepthcol])))
        
        resid = gaussian_filter(image - allmodel, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        
        opt_residual_mask.append(np.abs(resid) > 3*sig)
        #opt_residual_mask.append(np.logical_or(resid > 3*sig, resid < 5*sig))
        
        # "Find" the galaxy in the reference band.
        if filt == refband:
            #wcs = ConstantFitsWcs(Tan(filt2imfile[filt][0], 1))
            
            opt_shape = image.shape
            model = fitsio.read(filt2imfile[filt][1]) # model excluding the central

            mgegalaxy = find_galaxy(image-model, nblob=1, binning=3, quiet=True)
            
            H, W = image.shape
            xobj, yobj = np.ogrid[0:H, 0:W] # mask the galaxy
            majoraxis = 1.3*mgegalaxy.majoraxis
            opt_objmask = ellipse_mask(H/2, W/2, majoraxis, majoraxis*(1-mgegalaxy.eps),
                                       np.radians(mgegalaxy.theta-90), xobj, yobj)

            # Read the coadded (custom) mask and flag/mask pixels with bright stars etc.
            maskfile = os.path.join(galaxydir, '{}-{}.fits.gz'.format(galaxy, masksuffix))
            if os.path.isfile(maskfile):
                #print('Reading {}'.format(maskfile))
                opt_custom_mask = fitsio.read(maskfile)
                opt_custom_mask =  opt_custom_mask & 2**0 != 0 # True=masked
                # Restore masked pixels from either mis-identified stars (e.g.,
                # 0000433-033703895) or stars that are too close to the center.
                opt_custom_mask[opt_objmask] = False
            else:
                opt_custom_mask = np.zeros_like(image).astype(bool)

    # Find the union of all residuals but restore pixels centered on the central
    # object.
    opt_residual_mask = np.logical_or.reduce(np.array(opt_residual_mask))
    opt_residual_mask[opt_objmask] = False
    
    #opt_residual_mask = np.logical_or(opt_custom_mask, np.logical_or.reduce(np.array(opt_residual_mask)))

    # Now loop on each filter.
    for filt in bands:
        thispixscale = filt2pixscale[filt]

        image = fitsio.read(filt2imfile[filt][0])
        model = fitsio.read(filt2imfile[filt][1])
        allmodel = fitsio.read(filt2imfile[filt][2])

        # Identify the pixels belonging to the object of interest.
        majoraxis = 0.3*mgegalaxy.majoraxis * filt2pixscale[refband] / thispixscale # [pixels]
        
        H, W = image.shape
        xobj, yobj = np.ogrid[0:H, 0:W] # mask the galaxy
        objmask = ellipse_mask(H/2, W/2, majoraxis, majoraxis*(1-mgegalaxy.eps),
                               np.radians(mgegalaxy.theta-90), xobj, yobj)

        # Initialize the mask with the inverse variance map, if available.
        if len(filt2imfile[filt]) == 4:
            print('Reading {}'.format(filt2imfile[filt][3]))
            invvar = fitsio.read(filt2imfile[filt][3])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # Flag significant pixels (i.e., fitted objects) in the model image,
        # except those that are very close to the center of the object---we just
        # have to live with those.
        _, _, sig = sigma_clipped_stats(image - allmodel, sigma=3.0)
        residual_mask = model > 3*sig
        residual_mask[objmask] = False
        
        mask = np.logical_or(mask, residual_mask)

        # Add the custom mask (based on masked bright stars) to the mask,
        # resizing if necessary for this image/pixel scale.  For grz also add
        # the residual mask.
        if image.shape != opt_shape:
            from skimage.transform import resize
            custom_mask = resize(opt_custom_mask, image.shape, mode='reflect')
            mask = np.logical_or(mask, custom_mask)
        else:
            mask = np.logical_or(mask, opt_custom_mask)
            mask = np.logical_or(mask, opt_residual_mask)

        # Finally restore the pixels of the central galaxy.
        #mask[objmask] = False

        #majoraxis = mgegalaxy.majoraxis * filt2pixscale[refband] / thispixscale # [pixels]
        #these = ellipse_mask(H/2, W/2, majoraxis, majoraxis*(1-mgegalaxy.eps),
        #                     np.radians(mgegalaxy.theta-90), cat.bx, cat.by)
        #srcs = read_fits_catalog(cat[these], fluxPrefix='')
        #
        #test = srcs2image(srcs, wcs, psf_sigma=1.0)
        #import matplotlib.pyplot as plt
        #plt.imshow(np.log10(test), origin='lower') ; plt.savefig('junk2.png')
        #
        #pdb.set_trace()

        # Grow the mask slightly.
        mask = binary_dilation(mask, iterations=2)

        # Finally, pack it in!
        data[filt] = (image - model) / thispixscale**2 # [nanomaggies/arcsec**2]
        
        data['{}_masked'.format(filt)] = ma.masked_array(data[filt], mask)
        ma.set_fill_value(data['{}_masked'.format(filt)], fill_value)
        #data['{}_masked'.format(filt)].filled(fill_value)        

        if invvar is not None:
            var = np.zeros_like(invvar)
            var[~mask] = 1 / invvar[~mask]
            data['{}_var'.format(filt)] = var / thispixscale**4 # [nanomaggies**2/arcsec**4]

    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = pixscale

    if 'NUV' in bands:
        data['galex_pixscale'] = galex_pixscale
    if 'W1' in bands:
        data['unwise_pixscale'] = unwise_pixscale

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

def read_redmapper(rmversion='v6.3.1', sdssdr='dr14', index=None, satellites=False):
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
