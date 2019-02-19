"""
legacyhalos.lsbs
================

Code to deal with the LSBs sample and project.

"""
import os
import pdb
import numpy as np

import fitsio
from astropy.table import Table, Column, hstack
from astrometry.util.fits import fits_table

import legacyhalos.qa
import legacyhalos.misc
import legacyhalos.io

def lsbs_dir():
    """Top-level LSBs directory."""
    if 'LEGACYLSBS_DIR' not in os.environ:
        print('Required ${LEGACYLSBS_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYLSBS_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def lsbs_data_dir():
    if 'LEGACYLSBS_DATA_DIR' not in os.environ:
        print('Required ${LEGACYLSBS_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LEGACYLSBS_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def lsbs_html_dir():
    print('Should really have an environment variable for LEGACYLSBS_HTML_DIR!')
    ldir = os.path.join(os.getenv('LEGACYHALOS_HTML_DIR'), 'lsbs')
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from legacyhalos.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = lsbs_data_dir()
    if htmldir is None:
        htmldir = lsbs_html_dir()

    def get_healpix_subdir(nside, pixnum, datadir):
        subdir = os.path.join(str(pixnum // 100), str(pixnum))
        return os.path.abspath(os.path.join(datadir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat['GALAXY']]
    else:
        ngal = len(cat)
        galaxy = cat['GALAXY']

    galaxydir = np.array([os.path.join(datadir, gal) for gal in galaxy])
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, gal) for gal in galaxy])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def read_sample(verbose=False):
    """Read/generate the parent LSBs catalog.
    
    """
    # Really should be reading this from disk.
    sample = Table()
    sample['GALAXY'] = np.array(['NGC1052-F2'])
    sample['RA'] = np.array([40.4455]).astype('f8')
    sample['DEC'] = np.array([-8.4031]).astype('f8')
    sample['RADIUS_MOSAIC'] = np.array([20.0 * 60]).astype('f4') # mosaic half-width and half-height [arcsec]
    sample['RADIUS_GALAXY'] = np.array([20.0]).astype('f4')      # galaxy radius [arcsec]
    sample['RELEASE'] = np.array([7000]).astype(int)

    return sample

