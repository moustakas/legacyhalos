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
from astropy.table import Table
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

def paper1_dir(figures=False, data=False):
    pdir = os.path.join(legacyhalos_dir(), 'science', 'paper1')
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

def paper2_dir(figures=False, data=False):
    pdir = os.path.join(legacyhalos_dir(), 'science', 'paper2')
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

def write_ellipsefit(galaxy, galaxydir, ellipsefit, verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit.p'.format(galaxy))
    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    with open(ellipsefitfile, 'wb') as ell:
        pickle.dump(ellipsefit, ell)

def read_ellipsefit(galaxy, galaxydir, verbose=True):
    """Read the output of write_ellipsefit."""

    ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit.p'.format(galaxy))
    try:
        with open(ellipsefitfile, 'rb') as ell:
            ellipsefit = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(ellipsefitfile))
        ellipsefit = dict()

    return ellipsefit

def write_sky_ellipsefit(galaxy, galaxydir, skyellipsefit, verbose=False):
    """Pickle the sky ellipse-fitting results

    """
    skyellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit-sky.p'.format(galaxy))
    if verbose:
        print('Writing {}'.format(skyellipsefitfile))
    with open(skyellipsefitfile, 'wb') as ell:
        pickle.dump(skyellipsefit, ell)

def read_sky_ellipsefit(galaxy, galaxydir, verbose=True):
    """Read the output of write_skyellipsefit."""

    skyellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit-sky.p'.format(galaxy))
    try:
        with open(skyellipsefitfile, 'rb') as ell:
            skyellipsefit = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(skyellipsefitfile))
        skyellipsefit = dict()

    return skyellipsefit

def write_sersic(galaxy, galaxydir, sersic, modeltype='single', verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    sersicfile = os.path.join(galaxydir, '{}-sersic-{}.p'.format(galaxy, modeltype))
    if verbose:
        print('Writing {}'.format(sersicfile))
    with open(sersicfile, 'wb') as ell:
        pickle.dump(sersic, ell)

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
        pickle.dump(mgefit, mge)

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

def read_multiband(galaxy, galaxydir, band=('g', 'r', 'z'), refband='r',
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   maskfactor=2.0):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.  Finally,
    convert to surface brightness by dividing by the pixel area.

    """
    #from scipy.stats import sigmaclip
    from scipy.ndimage.morphology import binary_dilation

    from astropy.stats import sigma_clipped_stats
    #from astropy.stats import sigma_clip
    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import ellipse_mask

    # Dictionary mapping between filter and filename coded up in coadds.py,
    # galex.py, and unwise.py (see the LSLGA product, too).
    filt2imfile = {
        'g':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'r':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'z':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'FUV': ['image', 'model-nocentral'],
        'NUV': ['image', 'model-nocentral'],
        'W1':  ['image', 'model-nocentral'],
        'W2':  ['image', 'model-nocentral'],
        'W3':  ['image', 'model-nocentral'],
        'W4':  ['image', 'model-nocentral']}
        
    filt2pixscale =  {
        'g':   pixscale,
        'r':   pixscale,
        'z':   pixscale,
        'FUV': galex_pixscale,
        'NUV': galex_pixscale,
        'W1':  unwise_pixscale,
        'W2':  unwise_pixscale,
        'W3':  unwise_pixscale,
        'W4':  unwise_pixscale}

    found_data = True
    for filt in band:
        for ii, imtype in enumerate(filt2imfile[filt]):
            for suffix in ('.fz', ''):
                imfile = os.path.join(galaxydir, '{}-{}-{}.fits{}'.format(galaxy, imtype, filt, suffix))
                if os.path.isfile(imfile):
                    filt2imfile[filt][ii] = imfile
                    break
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    #tractorfile = os.path.join(galaxydir, '{}-tractor.fits'.format(galaxy))
    #if os.path.isfile(tractorfile):
    #    cat = Table(fitsio.read(tractorfile, upper=True))
    #    print('Read {} sources from {}'.format(len(cat), tractorfile))
    #else:
    #    print('Missing Tractor catalog {}'.format(tractorfile))
    #    found_data = False

    data = dict()
    if not found_data:
        return data

    # Find the central galaxy in the reference band.
    image = fitsio.read(filt2imfile[refband][0])
    model = fitsio.read(filt2imfile[refband][1])
    mgegalaxy = find_galaxy(image-model, nblob=1, binning=3, quiet=True)

    maskfile = os.path.join(galaxydir, '{}-custom-mask.fits.gz'.format(galaxy))
    if os.path.isfile(maskfile):
        print('Reading {}'.format(maskfile))
        custom_mask = fitsio.read(maskfile)
    else:
        custom_mask = np.zeros_like(image).astype(bool)
   
    for filt in band:
        thispixscale = filt2pixscale[filt]

        image = fitsio.read(filt2imfile[filt][0])
        model = fitsio.read(filt2imfile[filt][1])

        # Identify the pixels belonging to the object of interest.
        majoraxis = mgegalaxy.majoraxis * filt2pixscale[refband] / thispixscale # [pixels]
        
        H, W = image.shape
        xobj, yobj = np.ogrid[0:H, 0:W] # mask the galaxy
        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, majoraxis,
                               mgegalaxy.majoraxis*(1-mgegalaxy.eps),
                               mgegalaxy.theta, xobj, yobj)

        ## Initialize the mask with the inverse variance map, if available.
        #if len(filt2imfile[filt]) == 3:
        #    invvar = fitsio.read(filt2imfile[filt][2])
        #    mask = (invvar <= 0) # True-->bad, False-->good
        #else:
        #    mask = np.zeros_like(image).astype(bool)

        #mask = np.bitwise_or(custom_mask & 2**0, custom_mask & 2**1) > 0
        mask = custom_mask & 2**0 != 0

        # Compute sigma-clipped statistics on the residual image, ignoring the
        # central object.
        #resid = image - model
        #sigma_clip(ma.masked_array(image - model, objmask), sigma=3)
        
        _, _, sig = sigma_clipped_stats(image - model, mask=objmask, sigma=3.0)
        mask = np.logical_or(mask, model > 3*sig)

        ## Can give a divide-by-zero error for, e.g., GALEX imaging
        ##with np.errstate(divide='ignore', invalid='ignore'):
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    with np.errstate(all='ignore'):
        #        model_clipped, _, _ = sigmaclip(model, low=4.0, high=4.0)

        #print(filt, 1-len(model_clipped)/image.size)
        #if filt == 'W1':
        #    pdb.set_trace()
            
        #if len(model_clipped) > 0:
        #    mask = np.logical_or( mask, model > 5 * np.std(model_clipped) )
        #    #model_clipped = model
        
        mask = binary_dilation(mask, iterations=3) # True-->bad

        # Do not mask the central galaxy.
        mask[objmask] = 0

        data[filt] = (image - model) / thispixscale**2 # [nanomaggies/arcsec**2]
        
        #data['{}_mask'.format(filt)] = mask # True->bad
        data['{}_masked'.format(filt)] = ma.masked_array(data[filt], mask)
        ma.set_fill_value(data['{}_masked'.format(filt)], 0)

    data['band'] = band
    data['refband'] = refband
    data['pixscale'] = pixscale

    if 'NUV' in band:
        data['galex_pixscale'] = galex_pixscale
    if 'W1' in band:
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

def _read_paper_sample(paper='paper1', first=None, last=None, dr='dr6-dr7',
                       sfhgrid=1, isedfit_lsphot=False, isedfit_sdssphot=False,
                       isedfit_lhphot=False, candidates=False, kcorr=False,
                       verbose=False):
    """Wrapper to read a sample for a given paper.

    """
    if paper == 'paper1':
        paperdir = paper1_dir(data=True)
    elif paper == 'paper2':
        paperdir = paper2_dir(data=True)
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
        samplefile = os.path.join(paperdir, '{}-{}-{}.fits'.format(paper, prefix, dr))
        
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

def read_paper1_sample(first=None, last=None, dr='dr6-dr7', sfhgrid=1, isedfit_lsphot=False,
                       isedfit_sdssphot=False, isedfit_lhphot=False, candidates=False,
                       kcorr=False, verbose=False):
    """Read the Paper 1 sample.

    """
    sample = _read_paper_sample(paper='paper1', first=first, last=last, dr=dr,
                                sfhgrid=1, isedfit_lsphot=isedfit_lsphot,
                                isedfit_sdssphot=isedfit_sdssphot,
                                isedfit_lhphot=isedfit_lhphot, kcorr=kcorr,
                                candidates=candidates, verbose=verbose)
    return sample
    
def read_paper2_sample(first=None, last=None, dr='dr6-dr7', sfhgrid=1, isedfit_lsphot=False,
                       isedfit_sdssphot=False, isedfit_lhphot=False, candidates=False,
                       kcorr=False, verbose=False):
    """Read the Paper 1 sample.

    """
    sample = _read_paper_sample(paper='paper2', first=first, last=last, dr=dr,
                                sfhgrid=1, isedfit_lsphot=isedfit_lsphot,
                                isedfit_sdssphot=isedfit_sdssphot,
                                isedfit_lhphot=isedfit_lhphot, kcorr=kcorr,
                                candidates=candidates, verbose=verbose)
    return sample
    
def literature(kravtsov=True, gonzalez=False):
    """Assemble some data from the literature here."""

    if kravtsov:
        krav = dict()
        krav['m500'] = np.log10(np.array([15.6,10.3,7,5.34,2.35,1.86,1.34,0.46,0.47])*1e14)
        krav['mbcg'] = np.array([3.12,4.14,3.06,1.47,0.79,1.26,1.09,0.91,1.38])*1e12
        krav['mbcg'] = krav['mbcg']*0.7**2 # ????
        krav['mbcg_err'] = np.array([0.36,0.3,0.3,0.13,0.05,0.11,0.06,0.05,0.14])*1e12
        krav['mbcg_err'] = krav['mbcg_err'] / krav['mbcg'] / np.log(10)
        krav['mbcg'] = np.log10(krav['mbcg'])
        return krav

    if gonzalez:
        gonz = dict()
        gonz['mbcg'] = np.array([0.84,0.87,0.33,0.57,0.85,0.60,0.86,0.93,0.71,0.81,0.70,0.57])*1e12*2.65
        gonz['mbcg'] = gonz['mbcg']*0.7**2 # ????
        gonz['mbcg_err'] = np.array([0.03,0.09,0.01,0.01,0.14,0.03,0.03,0.05,0.07,0.12,0.02,0.01])*1e12*2.65
        gonz['m500'] = np.array([2.26,5.15,0.95,3.46,3.59,0.99,0.95,3.23,2.26,2.41,2.37,1.45])*1e14
        gonz['m500_err'] = np.array([0.19,0.42,0.1,0.32,0.28,0.11,0.1,0.19,0.23,0.18,0.24,0.21])*1e14
        gonz['mbcg_err'] = gonz['mbcg_err'] / gonz['mbcg'] / np.log(10)
        gonz['mbcg'] = np.log10(gonz['mbcg'])
        gonz['m500'] = np.log10(gonz['m500'])
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
