"""
legacyhalos.LSLGA
=================

Code to deal with the LSLGA sample and project.

"""
import os, time, shutil, pdb
import numpy as np
import astropy

import legacyhalos.io

RADIUS_CLUSTER_KPC = 100.0     # default cluster radius
ZCOLUMN = 'Z'

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--d25min', default=0.5, type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--d25max', default=5.0, type=float, help='Maximum diameter (arcmin).')

    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--largegalaxy-coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--largegalaxy-customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')
    #parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')

    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--nomakeplots', action='store_true', help='Do not remake the QA plots for the HTML pages.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-LSLGA', action='store_true', help='Build the LSLGA reference catalog.')
    args = parser.parse_args()

    return args

def _missing_files_one(args):
    """Wrapper for the multiprocessing."""
    return missing_files_one(*args)

def missing_files_one(galaxy, galaxydir, filesuffix, clobber):
    checkfile = os.path.join(galaxydir, '{}{}'.format(galaxy, filesuffix))    
    if os.path.exists(checkfile) and clobber is False:
        return False
    else:
        print('missing_files_one: ', checkfile)
        return True
    
def missing_files(args, sample, size=1, indices_only=False, filesuffix=None):
    from astrometry.util.multiproc import multiproc

    if args.largegalaxy_coadds:
        suffix = 'largegalaxy-coadds'
        if filesuffix is None:
            if args.just_coadds:
                filesuffix = '-largegalaxy-image-grz.jpg'
            else:
                filesuffix = '-largegalaxy-resid-grz.jpg'
        galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if filesuffix is None:
            if args.just_coadds:
                filesuffix = '-pipeline-image-grz.jpg'
            else:
                filesuffix = '-pipeline-resid-grz.jpg'
        galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    elif args.ellipse:
        suffix = 'ellipse'
        if filesuffix is None:
            filesuffix = '-largegalaxy-ellipse.isdone'
        galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    elif args.build_LSLGA:
        suffix = 'build-LSLGA'
        if filesuffix is None:
            filesuffix = '-largegalaxy-ellipse.isdone'
        galaxy, galaxydir = get_galaxy_galaxydir(sample)
    elif args.htmlplots:
        suffix = 'html'
        if filesuffix is None:
            if args.just_coadds:
                filesuffix = '-largegalaxy-grz-montage.png'
            else:
                filesuffix = '-ccdpos.png'
                #filesuffix = '-largegalaxy-maskbits.png'
            galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        if filesuffix is None:
            filesuffix = '-largegalaxy-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        print('Nothing to do.')
        return

    # Always set clobber False for htmlindex because we're not making files,
    # we're just looking for them.
    if args.htmlindex:
        clobber = False
    # Set clobber to True when building the catalog because we're looking for
    # the ellipse files, we're not writing them.
    elif args.build_LSLGA:
        clobber = True
    else:
        clobber = args.clobber

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)

    mp = multiproc(nthreads=args.nproc)
    todo = mp.map(_missing_files_one, [(gal, gdir, filesuffix, clobber)
                                           for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir))])

    if np.sum(todo) == 0:
        if indices_only:
            return list()
        else:
            return suffix, list()
    else:
        indices = indices[todo]

        # Assign the sample to ranks to make the D25 distribution per rank ~flat.

        # https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        weight = np.atleast_1d(sample['D25'])[indices]
        cumuweight = weight.cumsum() / weight.sum()
        idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        weighted_indices = np.array_split(indices, idx)
        indices = np.array_split(indices, size)

        #[print(np.sum(sample['D25'][ww]), np.sum(sample['D25'][vv])) for ww, vv in zip(weighted_indices, indices)]
        if indices_only:
            return weighted_indices
        else:
            return suffix, weighted_indices

def LSLGA_dir():
    if 'LSLGA_DIR' not in os.environ:
        print('Required ${LSLGA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
        #shutil.chown(ldir, group='cosmo')
    return ldir

def LSLGA_data_dir():
    if 'LSLGA_DATA_DIR' not in os.environ:
        print('Required ${LSLGA_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
        #shutil.chown(ldir, group='cosmo')
    return ldir

def LSLGA_html_dir():
    if 'LSLGA_HTML_DIR' not in os.environ:
        print('Required ${LSLGA_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
        #shutil.chown(ldir, group='cosmo')
    return ldir

def get_raslice(ra):
    return '{:06d}'.format(int(ra*1000))[:3]

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = LSLGA_data_dir()
    if htmldir is None:
        htmldir = LSLGA_html_dir()

    # Handle groups.
    if 'GROUP_NAME' in cat.colnames:
        galcolumn = 'GROUP_NAME'
        racolumn = 'GROUP_RA'
    else:
        galcolumn = 'GALAXY'
        racolumn = 'RA'

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat[galcolumn]]
        ra = [cat[racolumn]]
    else:
        ngal = len(cat)
        galaxy = cat[galcolumn]
        ra = cat[racolumn]

    galaxydir = np.array([os.path.join(datadir, get_raslice(ra), gal) for gal, ra in zip(galaxy, ra)])
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, get_raslice(ra), gal) for gal, ra in zip(galaxy, ra)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def LSLGA_version():
    #version = 'v5.0' # dr9e
    version = 'v6.0'  # DR9
    return version

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                customsky=False, preselect_sample=True, d25min=0.5, d25max=5.0):
    """Read/generate the parent LSLGA catalog.

    d25min in arcmin

    big = ss[ss['IN_DESI'] * (ss['GROUP_DIAMETER']>5) * ss['GROUP_PRIMARY']]
    %time bricks = np.hstack([survey.get_bricks_near(bb['GROUP_RA'], bb['GROUP_DEC'], bb['GROUP_DIAMETER']/60).brickname for bb in big])

    """
    import fitsio
    version = LSLGA_version()
    samplefile = os.path.join(LSLGA_dir(), 'sample', version, 'LSLGA-{}.fits'.format(version))

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # Select the galaxies requiring custom sky-subtraction.
    if customsky:
        #customgals = ['NGC4236']
        sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        rows = np.arange(len(sample))

        samplecut = np.where(
            (sample['GROUP_DIAMETER'] > 5) * 
            #(sample['GROUP_DIAMETER'] > 5) * (sample['GROUP_DIAMETER'] < 25) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        #this = np.where(sample['GROUP_NAME'] == 'NGC4448')[0]
        #rows = np.hstack((rows, this))

        rows = rows[samplecut]
        nrows = len(rows)
        print('Selecting {} custom sky galaxies.'.format(nrows))
    elif preselect_sample:
        cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI', 'LSLGA_ID']
        sample = fitsio.read(samplefile, columns=cols)
        rows = np.arange(len(sample))

        samplecut = np.where(
            #(sample['GROUP_DIAMETER'] > d25min) *
            #(sample['GROUP_DIAMETER'] < d25max) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        rows = rows[samplecut]

        if True:
            from legacyhalos.brick import brickname as get_brickname
            #bb = [692770, 232869, 51979, 405760, 1319700, 1387188, 519486, 145096]
            #ww = np.where(np.isin(sample['LSLGA_ID'], bb))[0]
            #ff = get_brickname(sample['GROUP_RA'][ww], sample['GROUP_DEC'][ww])
            
            brickname = get_brickname(sample['GROUP_RA'][samplecut], sample['GROUP_DEC'][samplecut])
            # Test sample-- 1 deg2 patch of sky
            #bricklist = ['0343p012']
            bricklist = ['1948p280', '1951p280', # Coma
                         '1914p307', # NGC4676/Mice
                         '2412p205', # NGC6052=PGC200329
                         '1890p112', # NGC4568 - overlapping spirals in Virgo
                         '0211p037', # NGC520 - train wreck
                         # random galaxies around bright stars
                         '0836m285', '3467p137',
                         '0228m257', '1328m022',
                         '3397m057', '0159m047',
                         '3124m097', '3160p097',
                         # 1 square degree of test bricks--
                         '0341p007', '0341p010', '0341p012', '0341p015', '0343p007', '0343p010',
                         '0343p012', '0343p015', '0346p007', '0346p010', '0346p012', '0346p015',
                         '0348p007', '0348p010', '0348p012', '0348p015',
                         # NGC4448 bricks
                         '1869p287', '1872p285', '1869p285'
                         # NGC2146 bricks
                         '0943p785', '0948p782'
                         ]
            ##nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'bricklist-dr9e-north.txt'), dtype='str')
            ##sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'bricklist-dr9e-south.txt'), dtype='str')
            #nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'bricklist-dr9-north.txt'), dtype='str')
            #sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'bricklist-dr9-south.txt'), dtype='str')
            #bricklist = np.union1d(nbricklist, sbricklist)
            #bricklist = nbricklist

            #rows = np.where([brick in bricklist for brick in brickname])[0]
            brickcut = np.where(np.isin(brickname, bricklist))[0]
            rows = rows[brickcut]

        nrows = len(rows)
        print('Selecting {} galaxies in the DR9 footprint.'.format(nrows))
    else:
        rows = None

    if first is None:
        first = 0
    if last is None:
        last = nrows
        if rows is None:
            rows = np.arange(first, last)
        else:
            rows = rows[np.arange(first, last)]
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        if rows is None:
            rows = np.arange(first, last+1)
        else:
            rows = rows[np.arange(first, last+1)]

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True, columns=columns))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))

    # Add an (internal) index number:
    sample['INDEX'] = rows
    
    #print('Hack the sample!')
    #tt = astropy.table.Table.read('/global/projecta/projectdirs/desi/users/ioannis/dr9d-lslga/dr9d-lslga-south.fits')
    #tt = tt[tt['D25'] > 1]
    #galaxylist = tt['GALAXY'].data

    # strip whitespace
    if 'GALAXY' in sample.colnames:
        sample['GALAXY'] = [gg.strip() for gg in sample['GALAXY']]
    if 'GROUP_NAME' in sample.colnames:
        galcolumn = 'GROUP_NAME'
        sample['GROUP_NAME'] = [gg.strip() for gg in sample['GROUP_NAME']]

    #print('Gigantic hack!!')
    #galaxylist = np.loadtxt('/global/u2/i/ioannis/junk2', dtype=str)
    
    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        sample = sample[np.isin(sample[galcolumn], galaxylist)]

    #print(get_brickname(sample['GROUP_RA'], sample['GROUP_DEC']))

    # Reverse sort by diameter. Actually, don't do this, otherwise there's a
    # significant imbalance between ranks.
    if False:
        sample = sample[np.argsort(sample['GROUP_DIAMETER'])]
    
    return sample

# From TheTractor/code/optimize_mixture_profiles.py
from scipy.special import gammaincinv
def sernorm(n):
	return gammaincinv(2.*n, 0.5)
def sersic_profile(x, n):
    return np.exp(-sernorm(n) * (x ** (1./n) - 1.))

def _build_model_LSLGA_one(args):
    """Wrapper function for the multiprocessing."""
    return build_model_LSLGA_one(*args)

def build_model_LSLGA_one(onegal, pixscale=0.262, minradius=2.0, minsb=25.0, sbcut=25.0,
                          bands=('g', 'r', 'z'), refcat='L5'):
    """Gather the fitting results build a single galaxy.

    minradius in arcsec
    minsb [minimum surface brightness] in r-band AB mag/arcsec**2
    sbcut [minimum surface brightness] in r-band AB mag/arcsec**2

    """
    from glob import glob

    import warnings
    import fitsio
    from astropy.table import Table, vstack
    from scipy.interpolate import interp1d
    from legacypipe.catalog import fits_reverse_typemap
    from astrometry.util.starutil_numpy import arcsec_between
    from tractor.wcs import RaDecPos
    from tractor import NanoMaggies
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor.galaxy import DevGalaxy, ExpGalaxy
    from tractor.sersic import SersicGalaxy
    from legacypipe.survey import LegacySersicIndex
    from legacypipe.bits import IN_BLOB, MASKBITS

    from legacyhalos.io import read_ellipsefit
    from legacyhalos.misc import is_in_ellipse
    
    onegal = Table(onegal)
    galaxy, galaxydir = legacyhalos.LSLGA.get_galaxy_galaxydir(onegal)

    tractorfile = os.path.join(galaxydir, '{}-largegalaxy-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing tractor file {}'.format(tractorfile))
        return None
    
    tractor = Table(fitsio.read(tractorfile))
    print('Temporarily remove the wise light-curve columns!')
    [tractor.remove_column(col) for col in tractor.colnames if 'lc_' in col]

    # See legacyhalos.ellipse.ellipse_cog--
    sbcuts = [23, 24, 25, 25.5, 26]
    radkeys = ['radius_sb{:0g}'.format(sbcut) for sbcut in sbcuts]
        
    tractor['preburned'] = np.ones(len(tractor), bool) # Everything was preburned...
    tractor['freeze'] = np.zeros(len(tractor), bool)   # ...but we only want to freeze the LSLGA galaxies.
    tractor['d25'] = np.zeros(len(tractor), np.float32)
    tractor['pa'] = np.zeros(len(tractor), np.float32)
    tractor['ba'] = np.ones(len(tractor), np.float32)
    for radkey in radkeys:
        tractor[radkey] = np.zeros(len(tractor), np.float32) - 1
        for filt in ['g', 'r', 'z']:
            magkey = radkey.replace('radius_', 'mag_{}_'.format(filt))
            tractor[magkey] = np.zeros(len(tractor), np.float32) - 1

    #islslga = np.array(['L' in refcat for refcat in tractor['ref_cat']])

    # Gather up all the ellipse files, which *define* the sample.
    ellipsefiles = glob(os.path.join(galaxydir, '{}-largegalaxy-*-ellipse.asdf'.format(galaxy)))
    for ellipsefile in ellipsefiles:
        lslga_id = os.path.basename(ellipsefile).split('-')[-2]
        lslga_id = np.int(lslga_id)
        this = np.where((tractor['ref_cat'] == refcat) * (tractor['ref_id'] == lslga_id))[0]
        if len(this) != 1:
            print('Doom has befallen you.')
            pdb.set_trace()
            
        af = read_ellipsefit(galaxy, galaxydir, galaxyid=str(lslga_id), filesuffix='largegalaxy', verbose=True)
        ellipse = af.tree

        # Add the basic ellipse geometry and the aperture photometry--
        tractor['freeze'][this] = True
        tractor['pa'][this] = ellipse['pa']
        tractor['ba'][this] = 1 - ellipse['eps']
        for radkey in radkeys:
            magkey = radkey.replace('radius_', 'mag_{}_'.format(filt))
            tractor[radkey][this] = ellipse[radkey]
            tractor[magkey][this] = ellipse[magkey]
        
        # Now add the radius
        if ellipse['radius_sb25'] > 0:
            tractor['d25'][this] = ellipse['radius_sb25'] * 2 / 60
        elif ellipse['radius_sb24'] > 0:
            tractor['d25'][this] = ellipse['radius_sb24'] * 2 / 60
        else:
            tractor['d25'][this] = ellipse['majoraxis'] * ellipse['refpixscale'] * 2 / 60

        if tractor['d25'][this] == 0:
            print('Doom has befallen you.')
            pdb.set_trace()

        # Next find all the objects in the elliptical "sphere-of-influence" of
        # this galaxy, and freeze those parameters as well.
        logr, e1, e2 = EllipseESoft.rAbPhiToESoft(tractor['d25'][this]*60/2,
                                                  tractor['ba'][this],
                                                  180-tractor['pa'][this]) # note the 180 rotation
        these = np.where(is_in_ellipse(tractor['ra'], tractor['dec'], tractor['ra'][this],
                                       tractor['dec'][this], np.exp(logr), e1, e2))[0]
        #print('Freezing the Tractor parameters of {} non-LSLGA objects.'.format(len(these)))
        tractor['freeze'][these] = True

        #if lslga_id == 278781:
        #    pdb.set_trace()

        #if len(these) > 10:
        #    import matplotlib.pyplot as plt
        #    plt.scatter(tractor['ra'], tractor['dec'], s=5)
        #    plt.scatter(tractor['ra'][these], tractor['dec'][these], s=5, color='red')
        #    plt.savefig('junk.png')
        #    pdb.set_trace()

    return tractor

def build_model_LSLGA(sample, pixscale=0.262, minradius=2.0, minsb=25.0, sbcut=25.0,
                      bands=('g', 'r', 'z'), clobber=False, nproc=1):
    """Gather all the fitting results and build the final model-based LSLGA catalog.

    minradius in arcsec
    minsb [minimum surface brightness] in r-band AB mag/arcsec**2
    sbcut [minimum surface brightness] in r-band AB mag/arcsec**2

    """
    import fitsio
    from astropy.table import Table, vstack, join
    from pydl.pydlutils.spheregroup import spheregroup
    from astrometry.util.multiproc import multiproc
    from legacypipe.reference import get_large_galaxy_version
        
    # This is a little fragile.
    version = legacyhalos.LSLGA.LSLGA_version()
    refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    
    #outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    outdir = '/global/project/projectdirs/cosmo/staging/largegalaxies/{}'.format(version)
    #print('Hack the path!')
    #outdir = '/global/u2/i/ioannis/scratch'
    
    outfile = os.path.join(outdir, 'LSLGA-model-{}.fits'.format(version))
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return

    mp = multiproc(nthreads=nproc)
    cat = mp.map(_build_model_LSLGA_one, [(onegal, pixscale, minradius, minsb, sbcut, bands, refcat) for onegal in sample])
    if len(cat) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    #cat2 = [cc for cc in cat if cc is not None]
    cat = vstack(list(filter(None, cat)))
    #cat = vstack(cat)
    #for d1, d2 in zip(cat[0].dtype.descr, cat[1].dtype.descr):
    #    if d1 != d2:
    #        print(d1, d2)
    [cat.rename_column(col, col.upper()) for col in cat.colnames]
    print('Gathered {} pre-burned and frozen galaxies.'.format(len(cat)))

    # ToDo: deal with pre-burned, but not frozen Tractor sources.
    ifreeze = np.where(cat['FREEZE'])[0]
    ilslga = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]
    cat = cat[ifreeze]
    print('Keeping {} frozen galaxies, of which {} are LSLGA.'.format(len(ifreeze), len(ilslga)))

    # Read the full parent LSLGA catalog and add all the Tractor columns.
    lslgafile = os.getenv('LARGEGALAXIES_CAT')
    lslga, hdr = fitsio.read(lslgafile, header=True)
    lslga = Table(lslga)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    # Move conflicting column names--
    lslga.rename_column('RA', 'LSLGA_RA')
    lslga.rename_column('DEC', 'LSLGA_DEC')
    lslga.rename_column('TYPE', 'MORPHTYPE')
    lslga.rename_column('D25', 'D25_ORIG')
    lslga.rename_column('PA', 'PA_ORIG')
    lslga.rename_column('BA', 'BA_ORIG')
    lslga['REF_ID'] = lslga['LSLGA_ID'] # we will use this for the join

    # Merge the Tractor and LSLGA catalogs, but we have to be careful to treat
    # LSLGA galaxies separately.
    I = np.where(cat['REF_CAT'] == refcat)[0]
    catI = join(lslga, cat[I], keys='REF_ID')

    rem = np.where(np.isin(lslga['LSLGA_ID'], catI['REF_ID']))[0]
    lslga = lslga[np.delete(np.arange(len(lslga)), rem)] # remove duplicates

    J = np.where(cat['REF_CAT'] != refcat)[0]
    catJ = cat[J]
    
    catcols = catJ.colnames
    lslgacols = lslga.colnames

    for col in lslgacols:
        if col in catcols:
            print('Skipping existing column {}'.format(col))
        else:
            if lslga[col].ndim > 1:
                catJ[col] = np.zeros((len(catJ), lslga[col].shape[1]), dtype=lslga[col].dtype)
            else:
                catJ[col] = np.zeros(len(catJ), dtype=lslga[col].dtype)

    out = vstack((catI, catJ)) # reassemble!
    del catI, catJ

    #icat = np.where(np.isin(cat['REF_ID'][I], lslga['LSLGA_ID']))[0]
    #ilslga = np.where(np.isin(lslga['LSLGA_ID'], cat['REF_ID'][I]))[0]
    #for col in cat.colnames:
    #    if col in lslga.colnames:
    #        cat[col][I[icat]] = lslga[ilslga][col]
    #
    #print(cat[cat['REF_ID'] == 278781])
    #pdb.set_trace()
    #
    #lslga = lslga[np.delete(np.arange(len(lslga)), ilslga)] # remove duplicates
    #
    ## Now add Tractor columns to the remaining (unburned, unfrozen) sources in
    ## the parent LSLGA catalog.
    #for col in catcols:
    #    if cat[col].ndim > 1:
    #        lslga[col] = np.zeros((len(lslga), cat[col].shape[1]), dtype=cat[col].dtype)
    #    else:
    #        lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)
    #
    ## Stack the catalogs
    #out = vstack((lslga, cat))
    #del lslga, cat
    
    # Add RA, Dec back in--
    fix = np.where(~out['PREBURNED'])[0]
    if len(fix) > 0:
        out['RA'][fix] = out['LSLGA_RA'][fix]
        out['DEC'][fix] = out['LSLGA_DEC'][fix]
        out['D25'][fix] = out['D25_ORIG'][fix]
        out['PA'][fix] = out['PA_ORIG'][fix]
        out['BA'][fix] = out['BA_ORIG'][fix]

    #chk1 = np.where(out['PREBURNED'] * (out['REF_CAT'] == 'L6'))[0]
    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.scatter(out['D25'], out['D25']/out['D25_ORIG']) ; plt.xscale('log') ; plt.savefig('junk.png')
    #plt.clf() ; plt.scatter(out['D25'], out['PA']-out['PA_ORIG']) ; plt.xscale('log') ; plt.savefig('junk.png')
    
    #print('Selecting just the bricks of interest!')

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    hdrversion = 'L{}-MODEL'.format(version[1:2]) # fragile!
    hdr['LSLGAVER'] = hdrversion
    fitsio.write(outfile, out.as_array(), header=hdr, clobber=True)

    # Write the KD-tree version
    kdoutfile = outfile.replace('.fits', '.kd.fits') # fragile
    cmd = 'startree -i {} -o {} -T -P -k -n largegals'.format(outfile, kdoutfile)
    print(cmd)
    _ = os.system(cmd)

    cmd = 'modhead {} LSLGAVER {}'.format(kdoutfile, hdrversion)
    print(cmd)
    _ = os.system(cmd)

    fix_permissions = True
    if fix_permissions:
        print('Fixing group permissions.')
        shutil.chown(outfile, group='cosmo')
        shutil.chown(kdoutfile, group='cosmo')

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None):
    """Make the HTML pages.

    """
    from glob import glob
    import subprocess
    import fitsio

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width
    #from legacyhalos.misc import cutout_radius_kpc
    #from legacyhalos.misc import HSC_RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    datadir = LSLGA_data_dir()
    htmldir = LSLGA_html_dir()

    if sample is None:
        sample = read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    missing = missing_files(args, sample, indices_only=True)
    #missing = missing_files(sample, filetype='coadds', size=1)[0]
    if len(missing) > 0:
        keep = np.delete(keep, missing)
        print('Keeping {}/{} galaxies with complete montages.'.format(len(keep), len(sample)))
        sample = sample[keep]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    ## group by RA slices
    #raslices = np.array([str(ra)[:3] for ra in sample[racolumn]])
    #rasorted = np.argsort(raslices)
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    rasorted = np.argsort(raslices)

    # Write the last-updated date to a webpage.
    js = legacyhalos.html._javastring()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = gal['D25'] * 60 * 2 / pixscale
        #width = 2 * cutout_radius_kpc(radius_kpc=radius_mosaic_kpc, redshift=gal[zcolumn],
        #                              pixscale=pixscale) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8&lslga'.format(
            baseurl, gal[racolumn], gal[deccolumn], zoom)
        
        return viewer

    def _skyserver_link(gal):
        if 'SDSS_OBJID' in gal.colnames:
            return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['SDSS_OBJID'])
        else:
            return ''

    def _get_mags(cat, rad='10', kpc=False, pipeline=False, cog=False, R24=False, R25=False, R26=False):
        res = []
        for band in ('g', 'r', 'z'):
            mag = None
            if kpc:
                iv = cat['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
                ff = cat['FLUX{}_{}'.format(rad, band.upper())][0]
            elif pipeline:
                iv = cat['flux_ivar_{}'.format(band)]
                ff = cat['flux_{}'.format(band)]
            elif R24:
                mag = cat['mag_{}_sb24'.format(band)]
            elif R25:
                mag = cat['mag_{}_sb25'.format(band)]
            elif R26:
                mag = cat['mag_{}_sb26'.format(band)]
            elif cog:
                mag = cat['cog_params_{}'.format(band)]['mtot']
            else:
                print('Thar be rocks ahead!')
            if mag:
                res.append('{:.3f}'.format(mag))
            else:
                if ff > 0:
                    mag = 22.5-2.5*np.log10(ff)
                    if iv > 0:
                        ee = 1 / np.sqrt(iv)
                        magerr = 2.5 * ee / ff / np.log(10)
                    res.append('{:.3f}'.format(mag))
                    #res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
                else:
                    res.append('...')
        return res
            
    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
        #shutil.chown(htmldir, group='cosmo')
    homehtmlfile = os.path.join(htmldir, homehtml)

    # https://stackoverflow.com/questions/36745577/how-do-you-create-in-python-a-file-with-permissions-other-users-can-write
    with open(homehtmlfile, 'w') as html:
    #with open(os.open(homehtmlfile, os.O_CREAT | os.O_WRONLY, 0o664), 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>Legacy Surveys Large Galaxy Atlas (LSLGA)</h1>\n')
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[inslice], html=True)

            html.write('<h3>RA Slice {}</h3>\n'.format(raslice))
            html.write('<table>\n')

            #html.write('<tr>\n')
            #html.write('<th></th>\n')
            #html.write('<th></th>\n')
            #html.write('<th>RA</th>\n')
            #html.write('<th>Dec</th>\n')
            #html.write('<th>D(25)</th>\n')
            #html.write('<th></th>\n')
            #html.write('</tr>\n')
            #
            #html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            #html.write('<th>Galaxy</th>\n')
            #html.write('<th>(deg)</th>\n')
            #html.write('<th>(deg)</th>\n')
            #html.write('<th>(arcsec)</th>\n')
            #html.write('<th>Viewer</th>\n')

            html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            html.write('<th> </th>\n')
            html.write('<th>Index</th>\n')
            html.write('<th>LSLGA ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')

            html.write('</tr>\n')
            for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-largegalaxy-grz-montage.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-largegalaxy-grz-montage.png'.format(galaxy1))

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['LSLGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.4f}</td>\n'.format(gal[diamcolumn]))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
            html.write('</table>\n')
            #count += 1

        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

        # Build the trends (trends.html) page--
        if maketrends:
            trendshtmlfile = os.path.join(htmldir, trendshtml)
            with open(trendshtmlfile, 'w') as html:
            #with open(os.open(trendshtmlfile, os.O_CREAT | os.O_WRONLY, 0o664), 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
                html.write('</style>\n')

                html.write('<h1>HSC Massive Galaxies: Sample Trends</h1>\n')
                html.write('<p><a href="https://github.com/moustakas/legacyhalos">Code and documentation</a></p>\n')
                html.write('<a href="trends/ellipticity_vs_sma.png"><img src="trends/ellipticity_vs_sma.png" alt="Missing file ellipticity_vs_sma.png" height="auto" width="50%"></a>')
                html.write('<a href="trends/gr_vs_sma.png"><img src="trends/gr_vs_sma.png" alt="Missing file gr_vs_sma.png" height="auto" width="50%"></a>')
                html.write('<a href="trends/rz_vs_sma.png"><img src="trends/rz_vs_sma.png" alt="Missing file rz_vs_sma.png" height="auto" width="50%"></a>')

                html.write('<br /><br />\n')
                html.write('<b><i>Last updated {}</b></i>\n'.format(js))
                html.write('</html></body>\n')
                html.close()

        # Make a separate HTML page for each object.
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)
        
        nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
        prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
        nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
        prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)
        #pdb.set_trace()

        for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate( zip(
            sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir) ) ):

            #radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
            #    redshift=gal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
            #radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

            #ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose, filesuffix='largegalaxy')
            #pipeline_ellipse = {}
            #pipeline_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose,
            #                                                  filesuffix='pipeline')

            if not os.path.exists(htmlgalaxydir1):
                os.makedirs(htmlgalaxydir1)
                #shutil.chown(htmlgalaxydir1, group='cosmo')

            ccdsfile = glob(os.path.join(galaxydir1, '{}-largegalaxy-ccds-*.fits'.format(galaxy1))) # north or south
            if len(ccdsfile) > 0:
                nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()
            else:
                nccds = None

            samplefile = os.path.join(galaxydir1, '{}-largegalaxy-sample.fits'.format(galaxy1))
            tractorfile = os.path.join(galaxydir1, '{}-largegalaxy-tractor.fits'.format(galaxy1))
            if os.path.isfile(tractorfile) and os.path.isfile(samplefile):
                sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
                if verbose:
                    print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))

                cols = ['ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                        'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z']
                tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows
                # Deal with objects potentially dropped from Tractor fitting.
                intractor = np.where(np.isin(tractor['ref_id'], sample['LSLGA_ID']))[0]
                tractor = tractor[intractor]
                tractor = tractor[np.argsort(tractor['flux_r'])[::-1]]

                these = np.where(np.isin(sample['LSLGA_ID'], tractor['ref_id']))[0]
                sample = sample[these]

            nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
            prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

            htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
            with open(htmlfile, 'w') as html:
            #with open(os.open(htmlfile, os.O_CREAT | os.O_WRONLY, 0o664), 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
                html.write('</style>\n')

                html.write('<h1>{}</h1>\n'.format(galaxy1))

                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
                #html.write('<br />\n')
                #html.write('<br />\n')

                html.write('<h2>Group Properties</h2>\n')
                
                # Table of properties
                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th>Index<br />(Primary)</th>\n')
                html.write('<th>LSLGA ID<br />(Primary)</th>\n')
                html.write('<th>Group Name</th>\n')
                html.write('<th>Group RA</th>\n')
                html.write('<th>Group Dec</th>\n')
                html.write('<th>Group Diameter<br />(arcmin)</th>\n')
                #html.write('<th>Richness</th>\n')
                #html.write('<th>Pcen</th>\n')
                html.write('<th>Viewer</th>\n')
                #html.write('<th>SkyServer</th>\n')
                html.write('</tr>\n')

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(ii))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['LSLGA_ID']))
                html.write('<td>{}</td>\n'.format(gal['GROUP_NAME']))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.4f}</td>\n'.format(gal[diamcolumn]))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
                html.write('</table>\n')

                # Add the properties of each galaxy.
                html.write('<h3>Group Members</h3>\n')
                html.write('<table>\n')
                html.write('<tr>\n')
                html.write('<th>LSLGA ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>Morphology</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>D(25)<br />(arcmin)</th>\n')
                #html.write('<th>PA<br />(deg)</th>\n')
                #html.write('<th>e</th>\n')
                html.write('</tr>\n')
                for groupgal in sample:
                    #if '031705' in gal['GALAXY']:
                    #    print(groupgal['GALAXY'])
                    html.write('<tr>\n')
                    html.write('<td>{}</td>\n'.format(groupgal['LSLGA_ID']))
                    html.write('<td>{}</td>\n'.format(groupgal['GALAXY']))
                    html.write('<td>{}</td>\n'.format(groupgal['TYPE']))
                    html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
                    html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
                    html.write('<td>{:.4f}</td>\n'.format(groupgal['D25']))
                    #if np.isnan(groupgal['PA']):
                    #    pa = 0.0
                    #else:
                    #    pa = groupgal['PA']
                    #html.write('<td>{:.2f}</td>\n'.format(pa))
                    #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
                    html.write('</tr>\n')
                html.write('</table>\n')

                html.write('<h2>Image Mosaics</h2>\n')

                if False:
                    html.write('<table>\n')
                    html.write('<tr><th colspan="3">Mosaic radius</th><th colspan="3">Point-source depth<br />(5-sigma, mag)</th><th colspan="3">Image quality<br />(FWHM, arcsec)</th></tr>\n')
                    html.write('<tr><th>kpc</th><th>arcsec</th><th>grz pixels</th><th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th></tr>\n')
                    html.write('<tr><td>{:.0f}</td><td>{:.3f}</td><td>{:.1f}</td>'.format(
                        radius_mosaic_kpc, radius_mosaic_arcsec, radius_mosaic_pixels))
                    if bool(ellipse):
                        html.write('<td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td>'.format(
                            ellipse['psfdepth_g'], ellipse['psfdepth_min_g'], ellipse['psfdepth_max_g'],
                            ellipse['psfdepth_r'], ellipse['psfdepth_min_r'], ellipse['psfdepth_max_r'],
                            ellipse['psfdepth_z'], ellipse['psfdepth_min_z'], ellipse['psfdepth_max_z']))
                        html.write('<td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td></tr>\n'.format(
                            ellipse['psfsize_g'], ellipse['psfsize_min_g'], ellipse['psfsize_max_g'],
                            ellipse['psfsize_r'], ellipse['psfsize_min_r'], ellipse['psfsize_max_r'],
                            ellipse['psfsize_z'], ellipse['psfsize_min_z'], ellipse['psfsize_max_z']))
                    html.write('</table>\n')
                    #html.write('<br />\n')

                pngfile, thumbfile = '{}-largegalaxy-grz-montage.png'.format(galaxy1), 'thumb-{}-largegalaxy-grz-montage.png'.format(galaxy1)
                html.write('<p>Large-galaxy preburn (left) data, (middle) model, and (right) residual image mosaic.</p>\n')
                html.write('<table width="90%">\n')
                html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                    pngfile, thumbfile))
                html.write('</table>\n')

                pngfile, thumbfile = '{}-pipeline-grz-montage.png'.format(galaxy1), 'thumb-{}-pipeline-grz-montage.png'.format(galaxy1)
                if os.path.isfile(os.path.join(htmlgalaxydir1, pngfile)):
                    html.write('<p>Pipeline (left) data, (middle) model, and (right) residual image mosaic.</p>\n')
                    html.write('<table width="90%">\n')
                    html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                        pngfile, thumbfile))
                    html.write('</table>\n')

                html.write('<h2>Elliptical Isophote Analysis</h2>\n')

                html.write('<h3>Geometry</h3>\n')
                if tractor is not None:
                    html.write('<table>\n')
                    html.write('<tr><th></th>\n')
                    html.write('<th colspan="5">Tractor</th>\n')
                    html.write('<th colspan="3">LSLGA</th>\n')
                    html.write('<th colspan="3">Ellipse Moments</th>\n')
                    html.write('<th colspan="5">Ellipse Fitting</th>\n')
                    html.write('</tr>\n')

                    html.write('<tr><th>LSLGA ID</th>\n')
                    html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
                    html.write('<th>R(25)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
                    html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
                    html.write('<th>R(24)<br />(arcsec)</th><th>R(25)<br />(arcsec)</th><th>R(26)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
                    html.write('</tr>\n')
                    
                    for tt in tractor:
                        ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
                        ba = (1 - ee) / (1 + ee)
                        pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
                        pa = pa % 180
                        
                        html.write('<tr><td>{}</td>\n'.format(tt['ref_id']))
                        html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                            tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

                        galaxyid = str(tt['ref_id'])
                        af = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                            galaxyid=galaxyid, verbose=False)
                        if bool(af):
                            ellipse = af.tree
                            html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                                ellipse['lslga_d25']*60/2, ellipse['lslga_pa'], 1-ellipse['lslga_ba']))
                            html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                                ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa'], ellipse['eps']))

                            rr = []
                            for rad in [ellipse['radius_sb24'], ellipse['radius_sb25'], ellipse['radius_sb26']]:
                                if rad < 0:
                                    rr.append('...')
                                else:
                                    rr.append('{:.3f}'.format(rad))
                            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                                rr[0], rr[1], rr[2], ellipse['pa'], ellipse['eps']))
                            af.close()
                        else:
                            html.write('<td>...</td><td>...</td><td>...</td>\n')
                            html.write('<td>...</td><td>...</td><td>...</td>\n')
                            html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                        html.write('</tr>\n')
                    html.write('</table>\n')
                    
                html.write('<h3>Photometry</h3>\n')
                if tractor is not None:
                    html.write('<table>\n')
                    html.write('<tr><th></th><th></th>\n')
                    html.write('<th colspan="3"></th>\n')
                    html.write('<th colspan="12">Curve of Growth</th>\n')
                    html.write('</tr>\n')
                    html.write('<tr><th></th><th></th>\n')
                    html.write('<th colspan="3">Tractor</th>\n')
                    html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
                    html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
                    html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
                    html.write('<th colspan="3">Integrated</th>\n')
                    html.write('</tr>\n')

                    html.write('<tr><th>LSLGA ID</th><th>Galaxy</th>\n')
                    html.write('<th>g</th><th>r</th><th>z</th>\n')
                    html.write('<th>g</th><th>r</th><th>z</th>\n')
                    html.write('<th>g</th><th>r</th><th>z</th>\n')
                    html.write('<th>g</th><th>r</th><th>z</th>\n')
                    html.write('<th>g</th><th>r</th><th>z</th>\n')
                    html.write('</tr>\n')

                    for tt, ss in zip(tractor, sample):
                        g, r, z = _get_mags(tt, pipeline=True)
                        html.write('<tr><td>{}</td><td>{}</td>\n'.format(tt['ref_id'], ss['GALAXY']))
                        html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))

                        galaxyid = str(tt['ref_id'])
                        af = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                            galaxyid=galaxyid, verbose=False)
                        if bool(af):
                            ellipse = af.tree
                            g, r, z = _get_mags(ellipse, R24=True)
                            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                            g, r, z = _get_mags(ellipse, R25=True)
                            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                            g, r, z = _get_mags(ellipse, R26=True)
                            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                            g, r, z = _get_mags(ellipse, cog=True)
                            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                            af.close()
                        else:
                            html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                        html.write('</tr>\n')
                    html.write('</table>\n')

                if bool(ellipse) and False:
                    html.write('<h4>Ellipse-fitting</h4>\n')
                    html.write('<table>\n')
                    html.write('<tr>')
                    html.write('<th colspan="3">Curve of Growth</th><th colspan="2">Mean Geometry</th><th colspan="2">Fitted Geometry</th>')
                    #html.write('<th colspan="3">Curve of growth<br />(custom sky, mag)</th><th colspan="3">Curve of growth<br />(pipeline sky, mag)</th>')
                    html.write('</tr>')

                    html.write('<tr><th>g</th><th>r</th><th>z</th><th>PA<br />(deg)</th><th>e</th><th>PA<br />(deg)</th><th>e</th></tr>\n')
                    #html.write('<th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th>')

                    html.write('<tr>')
                    if bool(ellipse):
                        g, r, z = (ellipse['cog_params_g']['mtot'], ellipse['cog_params_r']['mtot'],
                                   ellipse['cog_params_z']['mtot'])
                        html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(
                            g, r, z, ellipse['mge_pa'], ellipse['mge_eps'], ellipse['pa'], ellipse['eps']))
                    else:
                        html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>')

                    #if bool(pipeline_ellipse):
                    #    g, r, z = (pipeline_ellipse['cog_params_g']['mtot'], pipeline_ellipse['cog_params_r']['mtot'],
                    #               pipeline_ellipse['cog_params_z']['mtot'])
                    #    html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(g, r, z))
                    #else:
                    #    html.write('<td>...</td><td>...</td><td>...</td>')

                    html.write('</tr>')
                    html.write('</table>\n')
                    html.write('<br />\n')
                
                if False:
                    html.write('<h4>Galaxy Geometry</h4>\n')
                    html.write('<p>The ellipses are based on the geometry of the parent LSLGA catalog (blue) and the second moments of the light distribution (orange). </p>\n')

                    html.write('<table width="60%">\n')
                    html.write('<tr><th>Maskbits</th><th>Ellipse Moments</th></tr>\n')
                    html.write('<tr>\n')
                    html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                        '{}-largegalaxy-maskbits.png'.format(galaxy1)))
                    html.write('<td width="50%"><a href="{1}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                        'thumb-{}-largegalaxy-mge-ellipse.png'.format(galaxy1), '{}-largegalaxy-mge-ellipse.png'.format(galaxy1)))
                    html.write('</tr>\n')
                    html.write('</table>\n')

                for igal in np.arange(len(tractor['ref_id'])):
                    galaxyid = str(tractor['ref_id'][igal])
                    af = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                        galaxyid=galaxyid, verbose=verbose)
                    if bool(af):
                        ellipse = af.tree
                        if False:
                            html.write('<table>\n')
                            html.write('<tr><th>Fitting range<br />(arcsec)</th><th>Integration<br />mode</th><th>Clipping<br />iterations</th><th>Clipping<br />sigma</th></tr>')
                            html.write('<tr><td>{:.3f}-{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                                ellipse[refband]['sma'].min()*pixscale, ellipse[refband]['sma'].max()*pixscale,
                                ellipse['integrmode'], ellipse['nclip'], ellipse['sclip']))
                            html.write('</table>\n')
                            html.write('<br />\n')

                            html.write('<table>\n')
                            html.write('<tr><th colspan="5">Mean Geometry</th>')

                            html.write('<th colspan="4">Ellipse-fitted Geometry</th>')
                            if ellipse['input_ellipse']:
                                html.write('<th colspan="2">Input Geometry</th></tr>\n')
                            else:
                                html.write('</tr>\n')

                            html.write('<tr><th>Integer center<br />(x,y, grz pixels)</th><th>Flux-weighted center<br />(x,y grz pixels)</th><th>Flux-weighted size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>')
                            html.write('<th>Semi-major axis<br />(fitting range, arcsec)</th><th>Center<br />(x,y grz pixels)</th><th>PA<br />(deg)</th><th>e</th>')
                            if ellipse['input_ellipse']:
                                html.write('<th>PA<br />(deg)</th><th>e</th></tr>\n')
                            else:
                                html.write('</tr>\n')

                            html.write('<tr><td>({:.0f}, {:.0f})</td><td>({:.3f}, {:.3f})</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(
                                ellipse['x0'], ellipse['y0'], ellipse['mge_xmed'], ellipse['mge_ymed'], ellipse['mge_majoraxis']*pixscale,
                                ellipse['mge_pa'], ellipse['mge_eps']))

                            if 'init_smamin' in ellipse.keys():
                                html.write('<td>{:.3f}-{:.3f}</td><td>({:.3f}, {:.3f})<br />+/-({:.3f}, {:.3f})</td><td>{:.1f}+/-{:.1f}</td><td>{:.3f}+/-{:.3f}</td>'.format(
                                    ellipse['init_smamin']*pixscale, ellipse['init_smamax']*pixscale, ellipse['x0_median'],
                                    ellipse['y0_median'], ellipse['x0_err'], ellipse['y0_err'], ellipse['pa'], ellipse['pa_err'],
                                    ellipse['eps'], ellipse['eps_err']))
                            else:
                                html.write('<td>...</td><td>...</td><td>...</td><td>...</td>')
                            if ellipse['input_ellipse']:
                                html.write('<td>{:.1f}</td><td>{:.3f}</td></tr>\n'.format(
                                    np.degrees(ellipse['geometry'].pa)+90, ellipse['geometry'].eps))
                            else:
                                html.write('</tr>\n')
                            html.write('</table>\n')
                            html.write('<br />\n')

                        html.write('<h4>LSLGA {} - {}</h4>\n'.format(galaxyid, ellipse['galaxy']))
                        html.write('<table width="90%">\n')
                        html.write('<tr>\n')

                        pngfile = '{}-largegalaxy-{}-ellipse-multiband.png'.format(galaxy1, galaxyid)
                        thumbfile = 'thumb-{}-largegalaxy-{}-ellipse-multiband.png'.format(galaxy1, galaxyid)
                        html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" width="100%"></a></td>\n'.format(pngfile, thumbfile))
                        html.write('</tr>\n')
                        html.write('</table>\n')
                        html.write('<br />\n')

                        html.write('<table width="90%">\n')
                        html.write('<tr>\n')
                        pngfile = '{}-largegalaxy-{}-ellipse-sbprofile.png'.format(galaxy1, galaxyid)
                        html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
                        pngfile = '{}-largegalaxy-{}-ellipse-cog.png'.format(galaxy1, galaxyid)
                        html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
                        html.write('</tr>\n')
                        html.write('</table>\n')
                        #html.write('<br />\n')
                        af.close()
                    else:
                        html.write('<p>Ellipse-fitting not done or failed.</p>\n')

                    if intflux:
                        html.write('<h4>Aperture photometry</h4>\n')
                        html.write('<table>\n')
                        html.write('<tr>')
                        html.write('<th colspan="3"><10 kpc (mag)</th>')
                        html.write('<th colspan="3"><30 kpc (mag)</th>')
                        html.write('<th colspan="3"><100 kpc (mag)</th>')
                        html.write('</tr>')

                        html.write('<tr>')
                        html.write('<th>g</th><th>r</th><th>z</th>')
                        html.write('<th>g</th><th>r</th><th>z</th>')
                        html.write('<th>g</th><th>r</th><th>z</th>')
                        html.write('</tr>')

                        html.write('<tr>')
                        g, r, z = _get_mags(intflux[ii], rad='10', kpc=True)
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        g, r, z = _get_mags(intflux[ii], rad='30', kpc=True)
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        g, r, z = _get_mags(intflux[ii], rad='100', kpc=True)
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        html.write('</tr>')

                    html.write('</table>\n')
                    html.write('<br />\n')

                #html.write('<br />\n')
                html.write('<h2>CCD Diagnostics</h2>\n')

                html.write('<table width="90%">\n')
                pngfile = '{}-ccdpos.png'.format(galaxy1)
                html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                    pngfile))
                html.write('</table>\n')
                #html.write('<br />\n')

                if False:
                    html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
                    html.write('<table width="90%">\n')

                    # single-sersic
                    html.write('<tr>\n')
                    html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
                    html.write('</tr>\n')
                    html.write('<tr>\n')
                    html.write('<td><a href="{}-sersic-single-nowavepower.png"><img src="{}-sersic-single-nowavepower.png" alt="Missing file {}-sersic-single-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('<td><a href="{}-sersic-single.png"><img src="{}-sersic-single.png" alt="Missing file {}-sersic-single.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('</tr>\n')

                    # Sersic+exponential
                    html.write('<tr>\n')
                    html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
                    html.write('</tr>\n')
                    html.write('<tr>\n')
                    html.write('<td><a href="{}-sersic-exponential-nowavepower.png"><img src="{}-sersic-exponential-nowavepower.png" alt="Missing file {}-sersic-exponential-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('<td><a href="{}-sersic-exponential.png"><img src="{}-sersic-exponential.png" alt="Missing file {}-sersic-exponential.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('</tr>\n')

                    # double-sersic
                    html.write('<tr>\n')
                    html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
                    html.write('</tr>\n')
                    html.write('<tr>\n')
                    html.write('<td><a href="{}-sersic-double-nowavepower.png"><img src="{}-sersic-double-nowavepower.png" alt="Missing file {}-sersic-double-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('<td><a href="{}-sersic-double.png"><img src="{}-sersic-double.png" alt="Missing file {}-sersic-double.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('</tr>\n')

                    html.write('</table>\n')

                    html.write('<br />\n')

                    if nccds and ccdqa:
                        html.write('<h2>CCD Diagnostics</h2>\n')
                        html.write('<table width="90%">\n')
                        html.write('<tr>\n')
                        pngfile = '{}-ccdpos.png'.format(galaxy1)
                        html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                            pngfile))
                        html.write('</tr>\n')

                        for iccd in range(nccds):
                            html.write('<tr>\n')
                            pngfile = '{}-2d-ccd{:02d}.png'.format(galaxy1, iccd)
                            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                                pngfile))
                            html.write('</tr>\n')
                        html.write('</table>\n')
                        html.write('<br />\n')

                html.write('<br />\n')
                html.write('<br />\n')
                
                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
                html.write('<br />\n')

                html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
                html.write('<br />\n')
                html.write('</html></body>\n')
                html.close()

    # Make the plots.
    if makeplots:
        err = legacyhalos.html.make_plots(sample, datadir=datadir, htmldir=htmldir, refband=refband,
                                          bands=bands, pixscale=pixscale, survey=survey, clobber=clobber,
                                          verbose=verbose, nproc=nproc, ccdqa=ccdqa, maketrends=maketrends,
                                          zcolumn=zcolumn)

    fix_permissions = True
    if fix_permissions:
        print('Fixing group permissions.')
        for topdir, dirs, files in os.walk(htmldir):
            [shutil.chown(os.path.join(topdir, dd), group='cosmo') for dd in dirs]
            [shutil.chown(os.path.join(topdir, ff), group='cosmo') for ff in files]
        
        #cmd = 'chgrp -R cosmo {}'.format(htmldir)
        #print(cmd)
        #err1 = subprocess.call(cmd.split())
        #
        #cmd = 'find {} -type d -exec chmod 775 {{}} +'.format(htmldir)
        #print(cmd)
        #err2 = subprocess.call(cmd.split())
        #
        #cmd = 'find {} -type f -exec chmod 664 {{}} +'.format(htmldir)
        #print(cmd)
        #err3 = subprocess.call(cmd.split())
        #
        #if err1 != 0 or err2 != 0 or err3 != 0:
        #    print('Something went wrong updating permissions; please check the logfile.')
        #    return 0
    
    return 1

