"""
legacyhalos.LSLGA
=================

Code to deal with the LSLGA sample and project.

"""
import os, pdb
import numpy as np
import astropy

import legacyhalos.io

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py.')
    parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                
    args = parser.parse_args()

    return args

def missing_files_groups(args, sample, size, htmldir=None):
    """Simple task-specific wrapper on missing_files.

    """
    if args.coadds:
        suffix = 'coadds'
    elif args.custom_coadds:
        suffix = 'custom-coadds'
    else:
        suffix = ''        

    if suffix != '':
        groups = missing_files(sample, filetype=suffix, size=size,
                               clobber=args.clobber, htmldir=htmldir)
    else:
        groups = []        

    return suffix, groups

def missing_files(sample, filetype='coadds', size=1, htmldir=None,
                  clobber=False):
    """Find missing data of a given filetype."""    

    if filetype == 'coadds':
        filesuffix = '-pipeline-resid-grz.jpg'
    elif filetype == 'custom-coadds':
        filesuffix = '-custom-resid-grz.jpg'
    else:
        print('Unrecognized file type!')
        raise ValueError

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)
    todo = np.ones(ngal, dtype=bool)

    if filetype == 'html':
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    else:
        galaxy, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir)

    for ii, (gal, gdir) in enumerate( zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)) ):
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if os.path.exists(checkfile) and clobber is False:
            todo[ii] = False

    if np.sum(todo) == 0:
        return list()
    else:
        indices = indices[todo]
        
    return np.array_split(indices, size)

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = os.path.join(legacyhalos.io.legacyhalos_data_dir(), 'LSLGA')
    if htmldir is None:
        htmldir = os.path.join(legacyhalos.io.legacyhalos_html_dir(), 'LSLGA')

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

def read_sample(first=None, last=None, verbose=False):
    """Read/generate the parent LSLGA catalog.

    """
    import fitsio
    version = 'v3.0'
    samplefile = os.path.join(os.path.abspath(os.getenv('LSLGA_DIR')), 'sample',
                              version, 'LSLGA-{}.fits'.format(version))

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
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

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))
            
    return sample
