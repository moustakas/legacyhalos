"""
legacyhalos.io
==============

Code to read and write the various legacyhalos files.

"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from glob import glob

def legacyhalos_dir():
    key = 'DESI_BASIS_TEMPLATES'
    if 'LEGACYHALOS_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DIR environment variable not set'.format(key))
        raise EnvironmentError
    return os.path.abspath(os.getenv('LEGACYHALOS_DIR'))

def coadds_dir():
    return os.path.abspath(os.path.join(legacyhalos_dir(), 'coadds'))

def findfile(filetype):
    """Return the complete path to a file of interest.

    Args:
        filetype (str): file type, e.g. 'pix' or 'pixsim'

    Returns:
        str: full file path to output file

    """

    topdir = legacyhalos_dir()
    coaddsdir = coadds_dir()

    # Definition of where files go
    location = dict(
        parent = '{topdir:s}/legacyhalos-parent.fits',
        upennparent = '{topdir:s}/legacyhalos-upenn-parent.fits',
        redmapperupenn = '{topdir:s}/sandbox/redmapper-upenn.fits',
    )

    # Check that we know about this kind of file.
    if filetype not in location:
        raise ValueError('Unknown filetype {}; known types are {}'.format(filetype, list(location.keys())))

    # Get the outfile location and cleanup extraneous // from path.
    outfile = location[filetype].format(topdir=legacyhalos_dir(), coaddsdir=coadds_dir())
    outfile = os.path.normpath(outfile)

    return outfile
