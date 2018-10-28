"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
import os
import pickle, pdb
import numpy as np
import numpy.ma as ma
from glob import glob

import fitsio
from astropy.table import Table
from astropy.io import fits

def get_galaxy_galaxydir(cat, analysisdir=None):
    """Retrieve the galaxy name and the (nested) directory based on CENTRAL_ID. 

    """
    import astropy
    import healpy as hp
    from legacyhalos.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if analysisdir is None:
        analysisdir = analysis_dir()

    def get_healpix_subdir(nside, pixnum, analysisdir):
        subdir = os.path.join(str(pixnum // 100), str(pixnum))
        return os.path.abspath(os.path.join(analysisdir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat['CENTRAL_ID']]
        pixnum = [radec2pix(nside, cat['RA'], cat['DEC'])]
    else:
        ngal = len(cat)
        galaxy = np.array([cc for cc in cat['CENTRAL_ID']])
        #galaxy = np.array([cc.decode('utf-8') for cc in cat['CENTRAL_ID']])
        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, analysisdir), gal)
                          for pix, gal in zip(pixnum, galaxy)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
            
    return galaxy, galaxydir

def _get_galaxy_galaxydir(cat, analysisdir=None):
    """Retrieve the galaxy name and the (nested) directory based on CENTRAL_ID. 

    """
    if analysisdir is None:
        analysisdir = analysis_dir()

    ngal = len(cat)

    galaxy = np.array([cc.decode('utf-8') for cc in cat['CENTRAL_ID'].data])
    galid = galaxy.astype(np.int32)

    galaxydir = np.zeros(ngal, dtype='U{}'.format( len(analysisdir) + 1 + 5 + 1 + 4 + 1 + 6) )

    subdir1 = np.array(['{:05d}'.format(gg // 10000) for gg in galid])
    
    for dir1 in sorted(set(subdir1)):
        indx1 = np.where(dir1 == subdir1)[0]
        subdir2 = np.array(['{:04d}'.format(gg // 1000) for gg in galid[indx1]])
        for dir2 in sorted(set(subdir2)):
            indx2 = np.where(dir2 == subdir2)[0]

            allgaldir = np.array(['{:06d}'.format(gg) for gg in galid[indx1[indx2]]])
            galaxydir[indx1[indx2]] = [os.path.join(analysisdir, dir1, dir2, galdir) for galdir in allgaldir]

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
            
    return galaxy, galaxydir

def get_objid(cat, analysisdir=None):
    """Build a unique object ID based on the redmapper mem_match_id.

    Args:
      cat - must be a redmapper catalog or a catalog that has MEM_MATCH_ID.

    """
    if analysisdir is None:
        analysisdir = analysis_dir()

    ngal = len(np.atleast_1d(cat))
    objid = np.zeros(ngal, dtype='U7')
    objdir = np.zeros(ngal, dtype='U{}'.format(len(analysisdir)+1+7))

    #objid, objdir = list(), list()
    for ii, memid in enumerate(np.atleast_1d(cat['mem_match_id'])):
        objid[ii] = '{:07d}'.format(memid)
        objdir[ii] = os.path.join(analysisdir, objid[ii])
        #objid.append('{:07d}'.format(memid))
        #objdir.append(os.path.join(analysis_dir, objid[ii]))
        if not os.path.isdir(objdir[ii]):
            os.makedirs(objdir[ii], exist_ok=True)
    #objid = np.array(objid)
    #objdir = np.array(objdir)

    if ngal == 1:
        objid = objid[0]
        objdir = objdir[0]
            
    return objid, objdir

def legacyhalos_dir():
    if 'LEGACYHALOS_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DIR environment variable not set.')
        raise EnvironmentError
    return os.path.abspath(os.getenv('LEGACYHALOS_DIR'))

def analysis_dir():
    adir = os.path.join(legacyhalos_dir(), 'analysis')
    if not os.path.isdir(adir):
        os.makedirs(adir, exist_ok=True)
    return adir

def sample_dir():
    sdir = os.path.join(legacyhalos_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def paper1_dir(figures=True):
    pdir = os.path.join(legacyhalos_dir(), 'science', 'paper1')
    if not os.path.ipdir(pdir):
        os.makedirs(pdir, exist_ok=True)
    if figures:
        pdir = os.path.join(pdir, 'figures')
        if not os.path.ipdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    return pdir

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/legacyhalos'
    #else:
    #    htmldir = os.path.join(legacyhalos_dir(), 'html')

    htmldir = os.path.join(legacyhalos_dir(), 'html')

    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

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

def read_parent(extname='LSPHOT', upenn=True, isedfit=False, columns=None, verbose=False):
    """Read the various parent catalogs.

    Args:
      upenn - Restrict to the UPenn-matched catalogs.

    """
    suffix = ''
    if isedfit:
        suffix = '-isedfit'
    elif upenn:
        suffix = '-upenn'

    lsdir = legacyhalos_dir()
    catfile = os.path.join(lsdir, 'legacyhalos-parent{}.fits'.format(suffix))
    
    cat = Table(fitsio.read(catfile, ext=extname, columns=columns, lower=True))
    if verbose:
        print('Read {} objects from {} [{}]'.format(len(cat), catfile, extname))

    return cat

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

def read_multiband(galaxy, galaxydir, band=('g', 'r', 'z'), refband='r', pixscale=0.262):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.  Finally,
    convert to surface brightness by dividing by the pixel area.

    """
    from scipy.ndimage.morphology import binary_dilation

    data = dict()

    found_data = True
    for filt in band:
        for imtype in ('custom-image', 'custom-model-nocentral', 'invvar'):
            imfile = os.path.join(galaxydir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, filt))
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    if not found_data:
        return data
    
    for filt in band:
        image = fitsio.read(os.path.join(galaxydir, '{}-custom-image-{}.fits.fz'.format(galaxy, filt)))
        model = fitsio.read(os.path.join(galaxydir, '{}-custom-model-nocentral-{}.fits.fz'.format(galaxy, filt)))
        invvar = fitsio.read(os.path.join(galaxydir, '{}-invvar-{}.fits.fz'.format(galaxy, filt)))

        # Mask pixels with ivar<=0. Also build an object mask from the model
        # image, to handle systematic residuals.
        mask = (invvar <= 0) # 1=bad, 0=good
        if np.sum(mask) > 0:
            invvar[mask] = 1e-3

        snr = model * np.sqrt(invvar)
        mask = np.logical_or( mask, (snr > 1) ) 

        #sig1 = 1.0 / np.sqrt(np.median(invvar))
        #mask = np.logical_or( mask, (image - model) > (3 * sig1) )

        mask = binary_dilation(mask * 1, iterations=3)

        data[filt] = (image - model) / pixscale**2 # [nanomaggies/arcsec**2]
        
        data['{}_mask'.format(filt)] = mask == 0 # 1->bad
        data['{}_masked'.format(filt)] = ma.masked_array(data[filt], ~data['{}_mask'.format(filt)]) # 0->bad
        ma.set_fill_value(data['{}_masked'.format(filt)], 0)

    data['band'] = band
    data['refband'] = refband
    data['pixscale'] = pixscale

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
                isedfit_lhphot=False, satellites=False,
                kcorr=False, verbose=False):
    """Read the sample.

    """
    if satellites:
        prefix = 'satellites'
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

    info = fitsio.FITS(samplefile)
    nrows = info[1].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
    if first == last:
        last = last + 1

    rows = np.arange(first, last)

    if kcorr:
        ext = 2
    else:
        ext = 1

    sample = Table(info[1].read(rows=rows, ext=ext))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last-1, len(sample), samplefile))
            
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


# For HSC vs DECaLS analysis:

def hsc_vs_decals_dir():
    ddir = os.path.join(legacyhalos_dir(), 'hsc-vs-decals')
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
