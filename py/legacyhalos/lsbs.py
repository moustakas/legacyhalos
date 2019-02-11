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

from legacyhalos.html import make_plots, _javastring
from legacyhalos.misc import plot_style

sns = plot_style()

RADIUSFACTOR = 10
MANGA_RADIUS = 36.75 # / 2 # [arcsec]

def lsbs_dir():
    """Top-level LSBs directory (should be an environment variable...)."""
    print('Should really have an environment variable for LSBS_DIR!')
    ldir = os.path.join(os.getenv('LEGACYHALOS_DIR'), 'lsbs')
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def lsbs_data_dir():
    print('Should really have an environment variable for LSBS_DATA_DIR!')
    ldir = os.path.join(os.getenv('LEGACYHALOS_DATA_DIR'), 'lsbs')
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def lsbs_html_dir():
    print('Should really have an environment variable for LSBS_HTML_DIR!')
    ldir = os.path.join(os.getenv('LEGACYHALOS_HTML_DIR'), 'lsbs')
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def sample_dir():
    sdir = os.path.join(lsbs_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

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

def read_parent(verbose=False):
    """Read/generate the parent LSBs catalog.
    
    """
    # Really should be reading this from disk.
    sample = Table()
    sample['GALAXY'] = np.array(['NGC1052-F2'])
    sample['RA'] = np.array([40.4455]).astype('f8')
    sample['DEC'] = np.array([-8.4031]).astype('f8')
    sample['RADIUS'] = np.array([20.0 * 60]).astype('f4') # [arcsec]
    sample['RELEASE'] = np.array([7000]).astype(int)

    return sample

def get_samplefile(dr=None, ccds=False):

    suffix = 'fits'
    if dr is not None:
        if ccds:
            samplefile = os.path.join(sample_dir(), 'manga-nsa-{}-ccds.{}'.format(dr, suffix))
        else:
            samplefile = os.path.join(sample_dir(), 'manga-nsa-{}.{}'.format(dr, suffix))
    else:
        samplefile = os.path.join(sample_dir(), 'manga-nsa.{}'.format(suffix))
        
    return samplefile

def read_sample(columns=None, dr='dr67', ccds=False, verbose=False,
                first=None, last=None):
    """Read the sample."""
    samplefile = get_samplefile(dr=dr, ccds=ccds)
    if ccds:
        sample = Table(fitsio.read(samplefile, columns=columns, upper=True))
        if verbose:
            print('Read {} CCDs from {}'.format(len(sample), samplefile))
    else:
        info = fitsio.FITS(samplefile)
        nrows = info[1].get_nrows()
        if first is None:
            first = 0
        if last is None:
            last = nrows
        if first == last:
            last = last + 1
        rows = np.arange(first, last)

        sample = Table(info[1].read(rows=rows))
        if verbose:
            if len(rows) == 1:
                print('Read galaxy index {} from {}'.format(first, samplefile))
            else:
                print('Read galaxy indices {} through {} (N={}) from {}'.format(
                    first, last-1, len(sample), samplefile))

    return sample

