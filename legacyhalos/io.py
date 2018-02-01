"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
from __future__ import absolute_import, division, print_function

import os
import pickle
import numpy as np
from glob import glob

def get_objid(cat, analysis_dir=None):
    """Build a unique object ID based on the redmapper mem_match_id.

    Args:
      cat - must be a redmapper catalog or a catalog that has MEM_MATCH_ID.

    """
    ngal = len(cat)

    if analysis_dir is None:
        analysis_dir = os.path.join(legacyhalos_dir(), 'analysis')

    objid, objdir = list(), list()
    for ii, memid in enumerate(np.atleast_1d(cat.mem_match_id)):
        objid.append('{:07d}'.format(memid))
        objdir.append(os.path.join(analysis_dir, objid[ii]))
        if not os.path.isdir(objdir[ii]):
            os.makedirs(objdir[ii], exist_ok=True)
    objid = np.array(objid)
    objdir = np.array(objdir)

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
    htmldir = os.path.join(os.sep, 'project', 'projectdirs', 'cosmo',
                           'www', 'temp', 'ioannis', 'html')
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
    from astrometry.util.fits import fits_table

    suffix = ''
    if isedfit:
        suffix = '-isedfit'
    elif upenn:
        suffix = '-upenn'

    lsdir = legacyhalos_dir()
    catfile = os.path.join(lsdir, 'legacyhalos-parent{}.fits'.format(suffix))
    
    cat = fits_table(catfile, ext=extname, columns=columns)

    return cat