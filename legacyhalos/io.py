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
    if 'LEGACYHALOS_DIR' not in os.environ:
        print('Required ${LEGACYHALOS_DIR environment variable not set.')
        raise EnvironmentError
    return os.path.abspath(os.getenv('LEGACYHALOS_DIR'))

def read_catalog(extname='LSPHOT', upenn=True, isedfit=False):
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
    else:
        print('Unrecognized suffix!')
        raise ValueError

    catfile = os.path.join(lsdir, 'legacyhalos-parent{}.fits'.format(suffix))
    
    cat = Table(fitsio.read(catfile, ext=extname))

    return cat
