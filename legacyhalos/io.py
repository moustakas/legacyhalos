"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
from __future__ import absolute_import, division, print_function

import os
import pickle, pdb
import numpy as np
import numpy.ma as ma
from glob import glob

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

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/legacyhalos'
    #else:
    #    htmldir = os.path.join(legacyhalos_dir(), 'html')

    htmldir = os.path.join(legacyhalos_dir(), 'html')

    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

def write_ellipsefit(objid, objdir, ellipsefit, verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    ellipsefitfile = os.path.join(objdir, '{}-ellipsefit.p'.format(objid))
    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    with open(ellipsefitfile, 'wb') as ell:
        pickle.dump(ellipsefit, ell)

def read_ellipsefit(objid, objdir):
    """Read the output of write_ellipsefit."""

    ellipsefitfile = os.path.join(objdir, '{}-ellipsefit.p'.format(objid))
    try:
        with open(ellipsefitfile, 'rb') as ell:
            ellipsefit = pickle.load(ell)
    except:
        #raise IOError
        print('File {} not found!'.format(ellipsefitfile))
        ellipsefit = dict()

    return ellipsefit

def write_sersic(objid, objdir, sersic, model='single', verbose=False):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    sersicfile = os.path.join(objdir, '{}-sersic-{}.p'.format(objid, model))
    if verbose:
        print('Writing {}'.format(sersicfile))
    with open(sersicfile, 'wb') as ell:
        pickle.dump(sersic, ell)

def read_sersic(objid, objdir, model='single'):
    """Read the output of write_sersic."""

    sersicfile = os.path.join(objdir, '{}-sersic-{}.p'.format(objid, model))
    try:
        with open(sersicfile, 'rb') as ell:
            sersic = pickle.load(ell)
    except:
        #raise IOError
        print('File {} not found!'.format(sersicfile))
        sersic = dict()

    return sersic

def write_mgefit(objid, objdir, mgefit, band='r', verbose=False):
    """Pickle an XXXXX object (see, e.g., ellipse.mgefit_multiband).

    """
    mgefitfile = os.path.join(objdir, '{}-mgefit.p'.format(objid))
    if verbose:
        print('Writing {}'.format(mgefitfile))
    with open(mgefitfile, 'wb') as mge:
        pickle.dump(mgefit, mge)

def read_mgefit(objid, objdir):
    """Read the output of write_mgefit."""

    mgefitfile = os.path.join(objdir, '{}-mgefit.p'.format(objid))
    try:
        with open(mgefitfile, 'rb') as mge:
            mgefit = pickle.load(mge)
    except:
        #raise IOError
        print('File {} not found!'.format(mgefitfile))
        mgefit = dict()

    return mgefit

def read_catalog(extname='LSPHOT', upenn=True, isedfit=False, columns=None):
    """Read the various catalogs.

    Args:
      upenn - Restrict to the UPenn-matched catalogs.

    """
    import fitsio
    from astropy.table import Table

    suffix = ''
    if isedfit:
        suffix = '-isedfit'
    elif upenn:
        suffix = '-upenn'

    lsdir = legacyhalos_dir()
    catfile = os.path.join(lsdir, 'legacyhalos-parent{}.fits'.format(suffix))
    
    cat = Table(fitsio.read(catfile, ext=extname, columns=columns, lower=True))
    print('Read {} objects from {} [{}]'.format(len(cat), catfile, extname))

    return cat

def write_results(results, clobber=False):
    """Write out the output of legacyhalos-results

    """
    lsdir = legacyhalos_dir()
    resultsfilt = os.path.join(lsdir, 'legacyhalos-results.fits')
    if not os.path.isfile(resultsfilt) or clobber:
        print('Writing {}'.format(resultsfilt))
        results.write(resultsfilt, overwrite=True)
    else:
        print('File {} exists.'.format(resultsfilt))

def read_multiband(objid, objdir, band=('g', 'r', 'z'), refband='r', pixscale=0.262):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.  Finally,
    convert to surface brightness by dividing by the pixel area.

    """
    import fitsio
    from scipy.ndimage.morphology import binary_dilation

    data = dict()

    found_data = True
    for filt in band:
        for imtype in ('image', 'model', 'invvar'):
            imfile = os.path.join(objdir, '{}-{}-{}.fits.fz'.format(objid, imtype, filt))
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    if not found_data:
        return data
    
    for filt in band:
        image = fitsio.read(os.path.join(objdir, '{}-image-{}.fits.fz'.format(objid, filt)))
        model = fitsio.read(os.path.join(objdir, '{}-model-nocentral-{}.fits.fz'.format(objid, filt)))
        invvar = fitsio.read(os.path.join(objdir, '{}-invvar-{}.fits.fz'.format(objid, filt)))

        # Mask pixels with ivar<=0. Also build an object mask from the model
        # image, to handle systematic residuals.
        sig1 = 1.0 / np.sqrt(np.median(invvar[invvar > 0]))

        mask = (invvar <= 0)*1 # 1=bad, 0=good
        mask = np.logical_or( mask, ( model > (2 * sig1) )*1 )
        mask = binary_dilation(mask, iterations=5) * 1

        data[filt] = (image - model) / pixscale**2 # [nanomaggies/arcsec**2]
        
        data['{}_mask'.format(filt)] = mask == 0 # 1->bad
        data['{}_masked'.format(filt)] = ma.masked_array(data[filt], ~data['{}_mask'.format(filt)]) # 0->bad
        ma.set_fill_value(data['{}_masked'.format(filt)], 0)

    data['band'] = band
    data['refband'] = refband
    data['pixscale'] = pixscale

    return data

def read_sample(first=None, last=None):
    """Read the sample.

    Temporary hack to add the DR to the catalog.

    """
    from astropy.table import hstack
    import legacyhalos.io

    tractorcols = ('ra', 'dec', 'bx', 'by', 'brickname', 'objid', 'type',
                   'shapeexp_r', 'shapeexp_e1', 'shapeexp_e2',
                   'shapedev_r', 'shapedev_e1', 'shapedev_e2',
                   'fracdev', 'psfsize_g', 'psfsize_r', 'psfsize_z')
        
    rmcols = ('mem_match_id', 'z', 'r_lambda', 'lambda_chisq', 'p_cen')
    sdsscols = ('objid')
        
    sample = legacyhalos.io.read_catalog(extname='LSPHOT', upenn=True,
                                         columns=tractorcols)
    
    rm = legacyhalos.io.read_catalog(extname='REDMAPPER', upenn=True,
                                     columns=rmcols)
    
    sdss = legacyhalos.io.read_catalog(extname='SDSSPHOT', upenn=True,
                                       columns=np.atleast_1d(sdsscols))
    
    sdss.rename_column('objid', 'sdss_objid')
    print('Renaming column objid-->sdss_objid in [SDSSPHOT] extension.')
    sample = hstack( (sample, rm) )
    sample = hstack( (sample, sdss) )

    if first > last:
        print('Index first cannot be greater than index last, {} > {}'.format(first, last))

    if first is None:
        first = 0
    if last is None:
        last = len(sample)-1

    sample = sample[first:last+1]
    print('Sample contains {} objects with first, last indices {}, {}'.format(
        len(sample), first, last))

    return sample
