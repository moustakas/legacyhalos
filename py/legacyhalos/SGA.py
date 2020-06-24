"""
legacyhalos.SGA
===============

Code to support the SGA sample and project.

"""
import os, shutil, time, pdb
import numpy as np
import astropy

from legacyhalos.desiutil import brickname as get_brickname
import legacyhalos.io

ZCOLUMN = 'Z'
RACOLUMN = 'GROUP_RA'
DECCOLUMN = 'GROUP_DEC'
DIAMCOLUMN = 'GROUP_DIAMETER'

ELLIPSEBITS = dict(
    largeshift = 2**0, # >10-pixel shift in the flux-weighted center
    )

def SGA_version():
    """Archived versions. We used v2.0 for DR8, v3.0 through v7.0 were originally
    pre-DR9 test catalogs (now archived), and DR9 will use v3.0.

    version = 'v5.0' # dr9e
    version = 'v6.0' # dr9f,g
    version = 'v7.0' # more dr9 testing
    
    """
    version = 'v3.0'  # DR9
    return version

#def SGA_dir():
#    if 'SGA_DIR' not in os.environ:
#        print('Required ${SGA_DIR environment variable not set.')
#        raise EnvironmentError
#    ldir = os.path.abspath(os.getenv('SGA_DIR'))
#    if not os.path.isdir(ldir):
#        os.makedirs(ldir, exist_ok=True)
#    return ldir
#
#def legacyhalos.io.legacyhalos_data_dir():
#    if 'SGA_DATA_DIR' not in os.environ:
#        print('Required ${SGA_DATA_DIR environment variable not set.')
#        raise EnvironmentError
#    ldir = os.path.abspath(os.getenv('SGA_DATA_DIR'))
#    if not os.path.isdir(ldir):
#        os.makedirs(ldir, exist_ok=True)
#    return ldir
#
#def legacyhalos.io.legacyhalos_html_dir():
#    if 'SGA_HTML_DIR' not in os.environ:
#        print('Required ${SGA_HTML_DIR environment variable not set.')
#        raise EnvironmentError
#    ldir = os.path.abspath(os.getenv('SGA_HTML_DIR'))
#    if not os.path.isdir(ldir):
#        os.makedirs(ldir, exist_ok=True)
#    return ldir

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--d25min', default=0.1, type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--d25max', default=100.0, type=float, help='Maximum diameter (arcmin).')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--M33', action='store_true', help='Use a special CCDs file for M33.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--customredux', action='store_true', help='Process the custom reductions of the largest galaxies.')
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')

    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')
    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-SGA', action='store_true', help='Build the SGA reference catalog.')
    args = parser.parse_args()

    return args

def missing_files(args, sample, size=1, clobber_overwrite=None):
    import multiprocessing
    from legacyhalos.io import _missing_files_one

    dependson = None
    if args.htmlplots is False and args.htmlindex is False:
        if args.verbose:
            t0 = time.time()
            print('Getting galaxy names and directories...', end='')
        galaxy, galaxydir = get_galaxy_galaxydir(sample)
        if args.verbose:
            print('...took {:.3f} sec'.format(time.time() - t0))
        
    if args.coadds:
        suffix = 'coadds'
        filesuffix = '-largegalaxy-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        filesuffix = '-largegalaxy-ellipse.isdone'
        dependson = '-largegalaxy-coadds.isdone'
    elif args.build_SGA:
        suffix = 'build-SGA'
        filesuffix = '-largegalaxy-SGA.isdone'
        dependson = '-largegalaxy-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-largegalaxy-grz-montage.png'
        else:
            filesuffix = '-ccdpos.png'
            #filesuffix = '-largegalaxy-maskbits.png'
            dependson = '-largegalaxy-ellipse.isdone'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-largegalaxy-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        print('Nothing to do.')
        return

    # Make clobber=False for build_SGA and htmlindex because we're not making
    # the files here, we're just looking for them. The argument args.clobber
    # gets used downstream.
    if args.htmlindex:
        clobber = False
    elif args.build_SGA:
        clobber = True
    else:
        clobber = args.clobber

    if clobber_overwrite is not None:
        clobber = clobber_overwrite

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)

    pool = multiprocessing.Pool(args.nproc)
    missargs = []
    for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)):
        #missargs.append([gal, gdir, filesuffix, dependson, clobber])
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if dependson:
            missargs.append([checkfile, os.path.join(gdir, '{}{}'.format(gal, dependson)), clobber])
        else:
            missargs.append([checkfile, None, clobber])

    if args.verbose:
        t0 = time.time()
        print('Finding missing files...', end='')
    todo = np.array(pool.map(_missing_files_one, missargs))
    if args.verbose:
        print('...took {:.3f} min'.format((time.time() - t0)/60))

    itodo = np.where(todo == 'todo')[0]
    idone = np.where(todo == 'done')[0]
    ifail = np.where(todo == 'fail')[0]

    if len(ifail) > 0:
        fail_indices = [indices[ifail]]
    else:
        fail_indices = [np.array([])]

    if len(idone) > 0:
        done_indices = [indices[idone]]
    else:
        done_indices = [np.array([])]

    if len(itodo) > 0:
        _todo_indices = indices[itodo]

        # Assign the sample to ranks to make the D25 distribution per rank ~flat.

        # https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        cumuweight = weight.cumsum() / weight.sum()
        idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        if len(idx) < size: # can happen in corner cases or with 1 rank
            todo_indices = np.array_split(_todo_indices, size) # unweighted
        else:
            todo_indices = np.array_split(_todo_indices, idx) # weighted
        for ii in range(size): # sort by weight
            srt = np.argsort(sample[DIAMCOLUMN][todo_indices[ii]])
            todo_indices[ii] = todo_indices[ii][srt]
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices
    
def get_raslice(ra):
    return '{:06d}'.format(int(ra*1000))[:3]

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

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

def _get_brickname_one(args):
    """Wrapper function for the multiprocessing."""
    return get_brickname(*args)

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                customsky=False, preselect_sample=True, customredux=False, nproc=1,
                d25min=0.1, d25max=100.0):
    """Read/generate the parent SGA catalog.

    d25min in arcmin

    big = ss[ss['IN_DESI'] * (ss['GROUP_DIAMETER']>5) * ss['GROUP_PRIMARY']]
    %time bricks = np.hstack([survey.get_bricks_near(bb['GROUP_RA'], bb['GROUP_DEC'], bb['GROUP_DIAMETER']/60).brickname for bb in big])

    """
    import fitsio
            
    version = SGA_version()
    samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample', version, 'SGA-parent-{}.fits'.format(version))

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # Select the galaxies requiring custom sky-subtraction.
    if customsky:
        sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        rows = np.arange(len(sample))

        samplecut = np.where(
            #(sample['GROUP_DIAMETER'] > 5) * 
            (sample['GROUP_DIAMETER'] > 5) * (sample['GROUP_DIAMETER'] < 25) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        #this = np.where(sample['GROUP_NAME'] == 'NGC4448')[0]
        #rows = np.hstack((rows, this))

        rows = rows[samplecut]
        nrows = len(rows)
        print('Selecting {} custom sky galaxies.'.format(nrows))
        
    elif customredux:
        sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        rows = np.arange(len(sample))

        samplecut = np.where(
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        rows = rows[samplecut]
        
        customgals = [
            'NGC3034_GROUP',
            'NGC3077', # maybe?
            'NGC3726', # maybe?
            'NGC3953_GROUP', # maybe?
            'NGC3992_GROUP',
            'NGC4051',
            'NGC4096', # maybe?
            'NGC4125_GROUP',
            'UGC07698',
            'NGC4736_GROUP',
            'NGC5055_GROUP',
            'NGC5194_GROUP',
            'NGC5322_GROUP',
            'NGC5354_GROUP',
            'NGC5866_GROUP',
            'NGC4258',
            'NGC3031_GROUP',
            #'NGC0598_GROUP',
            'NGC5457'
            ]

        these = np.where(np.isin(sample['GROUP_NAME'][samplecut], customgals))[0]
        rows = rows[these]
        nrows = len(rows)

        print('Selecting {} galaxies with custom reductions.'.format(nrows))
        
    elif preselect_sample:
        cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
                'GROUP_PRIMARY', 'GROUP_ID', 'IN_DESI', 'SGA_ID', 'GALAXY', 'RA', 'DEC',
                'BRICKNAME']
        sample = fitsio.read(samplefile, columns=cols)
        rows = np.arange(len(sample))

        samplecut = np.where(
            (sample['GROUP_DIAMETER'] > d25min) *
            (sample['GROUP_DIAMETER'] < d25max) *
            ### custom reductions
            #(np.array(['DR8' not in gg for gg in sample['GALAXY']])) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        rows = rows[samplecut]

        if True: # DR9 bricklist
            nbricklist = np.loadtxt(os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample', 'dr9', 'bricklist-dr9-north.txt'), dtype='str')
            sbricklist = np.loadtxt(os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample', 'dr9', 'bricklist-dr9-south.txt'), dtype='str')

            bricklist = np.union1d(nbricklist, sbricklist)
            #bricklist = nbricklist
            #bricklist = sbricklist

            # Test by Dustin--
            #bricklist = np.array([
            #    '2221p000', '2221p002', '2221p005', '2221p007', '2223p000', '2223p002',
            #    '2223p005', '2223p007', '2226p000', '2226p002', '2226p005', '2226p007',
            #    '2228p000', '2228p002', '2228p005', '2228p007'])

            #bb = [692770, 232869, 51979, 405760, 1319700, 1387188, 519486, 145096]
            #ww = np.where(np.isin(sample['SGA_ID'], bb))[0]
            #ff = get_brickname(sample['GROUP_RA'][ww], sample['GROUP_DEC'][ww])

            ## Testing subtracting galaxy halos before sky-fitting---in Virgo!
            #bricklist = ['1877p122', '1877p125', '1875p122', '1875p125',
            #             '2211p017', '2213p017', '2211p020', '2213p020']
            
            ## Test sample-- 1 deg2 patch of sky
            ##bricklist = ['0343p012']
            #bricklist = ['1948p280', '1951p280', # Coma
            #             '1914p307', # NGC4676/Mice
            #             '2412p205', # NGC6052=PGC200329
            #             '1890p112', # NGC4568 - overlapping spirals in Virgo
            #             '0211p037', # NGC520 - train wreck
            #             # random galaxies around bright stars
            #             '0836m285', '3467p137',
            #             '0228m257', '1328m022',
            #             '3397m057', '0159m047',
            #             '3124m097', '3160p097',
            #             # 1 square degree of test bricks--
            #             '0341p007', '0341p010', '0341p012', '0341p015', '0343p007', '0343p010',
            #             '0343p012', '0343p015', '0346p007', '0346p010', '0346p012', '0346p015',
            #             '0348p007', '0348p010', '0348p012', '0348p015',
            #             # NGC4448 bricks
            #             '1869p287', '1872p285', '1869p285'
            #             # NGC2146 bricks
            #             '0943p785', '0948p782'
            #             ]
            
            brickcut = np.where(np.isin(sample['BRICKNAME'][samplecut], bricklist))[0]
            rows = rows[brickcut]

        if False: # SAGA host galaxies
            from astrometry.libkd.spherematch import match_radec
            saga = astropy.table.Table.read(os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample', 'saga_hosts.csv'))
            #fullsample = legacyhalos.SGA.read_sample(preselect_sample=False)
            m1, m2, d12 = match_radec(sample['RA'][samplecut], sample['DEC'][samplecut],
                                      saga['RA'], saga['DEC'], 5/3600.0, nearest=True)
            #ww = np.where(np.isin(sample['GROUP_ID'], fullsample['GROUP_ID'][m1]))[0]
            #ww = np.hstack([np.where(gid == sample['GROUP_ID'])[0] for gid in fullsample['GROUP_ID'][m1]])
            rows = rows[m1]

        if False: # test fitting of all the DR8 candidates
            fullsample = read_sample(preselect_sample=False, columns=['SGA_ID', 'GALAXY', 'GROUP_ID', 'GROUP_NAME', 'GROUP_DIAMETER', 'IN_DESI'])
            ww = np.where(fullsample['SGA_ID'] >= 5e6)[0]
            these = np.where(np.isin(sample['GROUP_ID'][samplecut], fullsample['GROUP_ID'][ww]))[0]
            rows = rows[these]

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
    sample.add_column(astropy.table.Column(name='INDEX', data=rows), index=0)
    
    #print('Hack the sample!')
    #tt = astropy.table.Table.read('/global/projecta/projectdirs/desi/users/ioannis/dr9d-lslga/dr9d-lslga-south.fits')
    #tt = tt[tt['D25'] > 1]
    #galaxylist = tt['GALAXY'].data

    ## strip whitespace
    #t0 = time.time()
    #if 'GALAXY' in sample.colnames:
    #    sample['GALAXY'] = [gg.strip() for gg in sample['GALAXY']]
    #if 'GROUP_NAME' in sample.colnames:
    #    galcolumn = 'GROUP_NAME'
    #    sample['GROUP_NAME'] = [gg.strip() for gg in sample['GROUP_NAME']]
    #print(time.time() - t0)

    #print('Gigantic hack!!')
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/junk', dtype=str)
    
    if galaxylist is not None:
        galcolumn = 'GROUP_NAME'
        if verbose:
            print('Selecting specific galaxies.')
        these = np.isin(sample[galcolumn], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            sample = sample[these]

    #print(get_brickname(sample['GROUP_RA'], sample['GROUP_DEC']))

    # Reverse sort by diameter. Actually, don't do this, otherwise there's a
    # significant imbalance between ranks.
    #sample = sample[np.argsort(sample['GROUP_DIAMETER'])]
    
    return sample

def _get_diameter(ellipse):
    """Wrapper to get the mean D(26) diameter.

    ellipse - legacyhalos.ellipse dictionary

    diam in arcmin

    """
    if ellipse['radius_sb26'] > 0:
        diam, diamref = 2 * ellipse['radius_sb26'] / 60, 'SB26' # [arcmin]
    elif ellipse['radius_sb25'] > 0:
        diam, diamref = 1.2 * 2 * ellipse['radius_sb25'] / 60, 'SB25' # [arcmin]
    #elif ellipse['radius_sb24'] > 0:
    #    diam, diamref = 1.5 * ellipse['radius_sb24'] * 2 / 60, 'SB24' # [arcmin]
    else:
        diam, diamref = 1.2 * ellipse['d25_leda'], 'LEDA' # [arcmin]
        #diam, diamref = 2 * ellipse['majoraxis'] * ellipse['refpixscale'] / 60, 'WGHT' # [arcmin]

    if diam <= 0:
        raise ValueError('Doom has befallen you.')

    return diam, diamref

def _build_ellipse_SGA_one(args):
    """Wrapper function for the multiprocessing."""
    return build_ellipse_SGA_one(*args)

def build_ellipse_SGA_one(onegal, fullsample, refcat='L3'):
    """Gather the ellipse-fitting results for a single galaxy.

    """
    from glob import glob

    import warnings
    import fitsio
    from astropy.table import Table, vstack, hstack, Column
    from tractor.ellipses import EllipseE # EllipseESoft
    from legacyhalos.io import read_ellipsefit
    from legacyhalos.misc import is_in_ellipse
    from legacyhalos.ellipse import SBTHRESH as sbcuts
    
    onegal = Table(onegal)
    galaxy, galaxydir = get_galaxy_galaxydir(onegal)

    tractorfile = os.path.join(galaxydir, '{}-largegalaxy-tractor.fits'.format(galaxy))
    # These galaxies are missing because we don't have grz coverage. We want to
    # keep them in the SGA catalog, though, so don't add them to the `reject`
    # list here.
    if not os.path.isfile(tractorfile):
        print('Tractor catalog missing: {}'.format(tractorfile))
        return None, None, None, onegal

    # Note: for galaxies on the edge of the footprint we can also sometimes
    # lose 3-band coverage if one or more of the bands is fully masked
    # (these will fail the "found_data" check in
    # legacyhalos.io.read_multiband. Do a similar check here before continuing.
    grzmissing = False
    for band in ['g', 'r', 'z']:
        imfile = os.path.join(galaxydir, '{}-largegalaxy-image-{}.fits.fz'.format(galaxy, band))
        if not os.path.isfile(imfile):
            print('  Missing image {}'.format(imfile))
            grzmissing = True
    if grzmissing:
        return None, None, None, onegal
    
    tractor = Table(fitsio.read(tractorfile, upper=True))

    # Remove Gaia stars immediately, so they're not double-counted.
    notgaia = np.where(tractor['REF_CAT'] != 'G2')[0]
    if len(notgaia) > 0:
        tractor = tractor[notgaia]
    if len(tractor) == 0: # can happen in small fields
        print('Warning: All Tractor sources are Gaia stars in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]))
        return None, None, None, onegal
    assert('G2' not in set(tractor['REF_CAT']))

    # Also remove SGA sources which do not belong to this group, because they
    # will be handled when we deal with *that* group. (E.g., PGC2190838 is in
    # the *mosaic* of NGC5899 but does not belong to the NGC5899 "group").
    ilslga = np.where(tractor['REF_CAT'] == refcat)[0]
    if len(ilslga) == 0:
        print('Warning: No SGA sources in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]))
        return None, None, None, onegal
        
    toss = np.where(np.logical_not(np.isin(tractor['REF_ID'][ilslga], fullsample['SGA_ID'])))[0]
    if len(toss) > 0:
        for tt in toss:
            print('  Removing non-primary SGA_ID={}'.format(tractor[ilslga][tt]['REF_ID']))
        keep = np.delete(np.arange(len(tractor)), ilslga[toss])
        tractor = tractor[keep]

    # Finally toss out Tractor sources which are too small (i.e., are outside
    # the prior range on size in the main pipeline). Actually, the minimum size
    # is 0.01 arcsec, but we cut on 0.1 arcsec to have some margin. If these
    # sources are re-detected in production then so be it.
    keep = np.where(np.logical_or(tractor['TYPE'] == 'PSF', tractor['SHAPE_R'] > 0.1))[0]
    if len(keep) == 0:
        print('All Tractor sources have been dropped in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]))
        return None, None, None, onegal
    tractor = tractor[keep]

    # Next, add all the (new) columns we will need to the Tractor catalog. This
    # is a little wasteful because the non-frozen Tractor sources will be tossed
    # out at the end, but it's much easier and cleaner to do it this way. Also
    # remove the duplicate BRICKNAME column from the Tractor catalog.
    tractor.remove_column('BRICKNAME') # of the form custom-BRICKNAME
    onegal.rename_column('RA', 'SGA_RA')
    onegal.rename_column('DEC', 'SGA_DEC')
    onegal.remove_column('INDEX')
    lslgacols = onegal.colnames
    tractorcols = tractor.colnames
    for col in lslgacols[::-1]: # reverse the order
        if col in tractorcols:
            print('  Skipping existing column {}'.format(col))
        else:
            if onegal[col].ndim > 1:
                # assume no multidimensional strins
                tractor.add_column(Column(name=col, data=np.zeros((len(tractor), onegal[col].shape[1]),
                                                                  dtype=onegal[col].dtype)-1), index=0)
            else:
                typ = onegal[col].dtype.type
                if typ is np.str_ or typ is np.str or typ is np.bool_ or typ is np.bool:
                    tractor.add_column(Column(name=col, data=np.zeros(len(tractor), dtype=onegal[col].dtype)), index=0)
                else:
                    tractor.add_column(Column(name=col, data=np.zeros(len(tractor), dtype=onegal[col].dtype)-1), index=0)
                    
    tractor['GROUP_ID'][:] = onegal['GROUP_ID'] # note that we don't change GROUP_MULT
    tractor['GROUP_NAME'][:] = onegal['GROUP_NAME']

    # add the columns from legacyhalos.ellipse.ellipse_cog
    radkeys = ['RADIUS_SB{:0g}'.format(sbcut) for sbcut in sbcuts]
    for radkey in radkeys:
        tractor[radkey] = np.zeros(len(tractor), np.float32) - 1
    for radkey in radkeys:
        for filt in ['G', 'R', 'Z']:
            magkey = radkey.replace('RADIUS_', '{}_MAG_'.format(filt))
            tractor[magkey] = np.zeros(len(tractor), np.float32) - 1
    for filt in ['G', 'R', 'Z']:
        tractor['{}_MAG_TOT'.format(filt)] = np.zeros(len(tractor), np.float32) - 1

    tractor['PREBURNED'] = np.ones(len(tractor), bool)  # Everything was preburned but we only want to freeze the...
    tractor['FREEZE'] = np.zeros(len(tractor), bool)    # ...SGA galaxies and sources in that galaxy's ellipse.

    # Next, gather up all the ellipse files, which *define* the sample. Also
    # track the galaxies that are dropped by Tractor and, separately, galaxies
    # which fail ellipse-fitting (or are not ellipse-fit because they're too
    # small).
    notractor, noellipse, largeshift = [], [], []
    for igal, lslga_id in enumerate(np.atleast_1d(fullsample['SGA_ID'])):
        ellipsefile = os.path.join(galaxydir, '{}-largegalaxy-{}-ellipse.fits'.format(galaxy, lslga_id))

        # Find this object in the Tractor catalog. If it's not here, it was
        # dropped from Tractor fitting, which most likely means it's spurious!
        # Add it to the 'notractor' catalog.
        match = np.where((tractor['REF_CAT'] == refcat) * (tractor['REF_ID'] == lslga_id))[0]
        if len(match) > 1:
            raise ValueError('Multiple matches should never happen but it did in the field of ID={}?!?'.format(onegal['SGA_ID']))
    
        if not os.path.isfile(ellipsefile):
            if len(match) == 0:
                print('Tractor reject & not ellipse-fit: {} (ID={})'.format(fullsample['GALAXY'][igal], lslga_id))
                notrac = Table(fullsample[igal]['SGA_ID', 'GALAXY', 'RA', 'DEC', 'GROUP_NAME', 'GROUP_ID',
                                                'D25_LEDA', 'PA_LEDA', 'BA_LEDA'])
                notractor.append(notrac)
            else:
                # Objects here were fit by Tractor but *not* ellipse-fit. Keep
                # them but reset ref_cat and ref_id. That way if it's a galaxy
                # that just happens to be too small to pass our size cuts in
                # legacyhalos.io.read_multiband to be ellipse-fit, Tractor will
                # still know about it. In particular, if it's a small galaxy (or
                # point source, I guess) *inside* the elliptical mask of another
                # galaxy (see, e.g., SDSSJ123843.02+092744.0 -
                # http://legacysurvey.org/viewer-dev?ra=189.679290&dec=9.462331&layer=dr8&zoom=14&lslga),
                # we want to be sure it doesn't get forced PSF in production!
                print('Not ellipse-fit: {} (ID={}, type={}, r50={:.2f} arcsec, fluxr={:.3f} nanomaggies)'.format(
                    fullsample['GALAXY'][igal], lslga_id, tractor['TYPE'][match[0]], tractor['SHAPE_R'][match[0]],
                    tractor['FLUX_R'][match[0]]))
                tractor['REF_CAT'][match] = ' '
                #tractor['REF_ID'][match] = 0 # use -1, not zero!
                tractor['REF_ID'][match] = -1 # use -1, not zero!
                tractor['FREEZE'][match] = True

                # Also write out a separate catalog of these objects (including
                # some info from the original Tractor catalog) so we can test
                # and verify that all's well.
                noell = Table(fullsample[igal]['SGA_ID', 'GALAXY', 'RA', 'DEC', 'GROUP_NAME', 'GROUP_ID',
                                               'D25_LEDA', 'PA_LEDA', 'BA_LEDA'])
                noell['TRACTOR_RA'] = tractor['RA'][match]
                noell['TRACTOR_DEC'] = tractor['DEC'][match]
                noell['TRACTOR_TYPE'] = tractor['TYPE'][match]
                noell['FLUX_R'] = tractor['FLUX_R'][match]
                noell['SHAPE_R'] = tractor['SHAPE_R'][match]
                noellipse.append(noell)
        else:
            ellipse = read_ellipsefit(galaxy, galaxydir, galaxyid=str(lslga_id),
                                      filesuffix='largegalaxy', verbose=True)

            # Objects with "largeshift" shifted positions significantly during
            # ellipse-fitting, which *may* point to a problem. Add a bit--
            if ellipse['largeshift']:
                tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['largeshift']
                
                #badcen = Table(fullsample[igal]['SGA_ID', 'GALAXY', 'RA', 'DEC', 'GROUP_NAME', 'GROUP_ID',
                #                               'D25_LEDA', 'PA_LEDA', 'BA_LEDA'])
                #badcen['TRACTOR_RA'] = tractor['RA'][match]
                #badcen['TRACTOR_DEC'] = tractor['DEC'][match]
                #badcen['TRACTOR_TYPE'] = tractor['TYPE'][match]
                #badcen['FLUX_R'] = tractor['FLUX_R'][match]
                #badcen['SHAPE_R'] = tractor['SHAPE_R'][match]
                #largeshift.append(badcen)

            # Get the ellipse-derived geometry, which we'll add to the Tractor
            # catalog below.
            pa, ba = ellipse['pa'], 1 - ellipse['eps']
            diam, diamref = _get_diameter(ellipse)
            
            # Next find all the objects in the "ellipse-of-influence" of this
            # galaxy and freeze them. Note: EllipseE.fromRAbPhi wants semi-major
            # axis (i.e., radius) in arcsec.
            ragal, decgal = tractor['RA'][match], tractor['DEC'][match]
            reff, e1, e2 = EllipseE.fromRAbPhi(diam*60/2, ba, 180-pa) # note the 180 rotation
            inellipse = np.where(is_in_ellipse(tractor['RA'], tractor['DEC'], ragal, decgal, reff, e1, e2))[0]

            #if lslga_id == 278781:
            #    pdb.set_trace()
            #if len(these) > 10:
                #bb = np.where(tractor['objid'] == 33)[0]
                #import matplotlib.pyplot as plt
                #plt.clf()
                #plt.scatter(tractor['ra'], tractor['dec'], s=5)
                #plt.scatter(tractor['ra'][these], tractor['dec'][these], s=5, color='red')
                #plt.savefig('junk.png')
                #plt.scatter(tractor['ra'][bb], tractor['dec'][bb], s=50, color='k')

            # This should never happen since the SGA galaxy itself is in the ellipse!
            if len(inellipse) == 0:
                raise ValueError('No galaxies in the ellipse-of-influence in the field of ID={}?!?'.format(onegal['SGA_ID']))

            #print('Freezing the Tractor parameters of {} objects in the ellipse of ID={}.'.format(len(inellipse), lslga_id))
            tractor['FREEZE'][inellipse] = True

            # Populate the output catalog--
            thisgal = Table(fullsample[igal])
            thisgal.rename_column('RA', 'SGA_RA')
            thisgal.rename_column('DEC', 'SGA_DEC')
            thisgal.remove_column('INDEX')
            for col in thisgal.colnames:
                tractor[col][match] = thisgal[col]

            # Add ellipse geometry and aperture photometry--
            tractor['PA'][match] = pa
            tractor['BA'][match] = ba
            tractor['DIAM'][match] = diam
            tractor['DIAM_REF'][match] = diamref

            for radkey in radkeys:
                tractor[radkey][match] = ellipse[radkey.lower()]
                for filt in ['G', 'R', 'Z']:
                    magkey = radkey.replace('RADIUS_', '{}_MAG_'.format(filt))
                    tractor[magkey][match] = ellipse[magkey.lower()]
                    tractor['{}_MAG_TOT'.format(filt)][match] = ellipse['{}_COG_PARAMS_MTOT'.format(filt).lower()]

    # Keep just frozen sources. Can be empty (e.g., if a field contains just a
    # single dropped source, e.g., DR8-2194p447-894).
    keep = np.where(tractor['FREEZE'])[0]
    if len(keep) == 0:
        #raise ValueError('No frozen galaxies in the field of ID={}?!?'.format(onegal['SGA_ID']))
        tractor = tractor[keep]

    if len(notractor) > 0:
        notractor = vstack(notractor)
    if len(noellipse) > 0:
        noellipse = vstack(noellipse)

    return tractor, notractor, noellipse, None

def build_ellipse_SGA(sample, fullsample, nproc=1, clobber=False, debug=False):
    """Gather all the ellipse-fitting results and build the final SGA catalog.

    """
    import fitsio
    from contextlib import redirect_stdout, redirect_stderr
    from astropy.table import Table, vstack, join
    from astrometry.util.multiproc import multiproc
    #from legacypipe.reference import get_large_galaxy_version

    ## This is a little fragile.
    #version = SGA_version()
    #refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    #lslgafile = os.getenv('LARGEGALAXIES_CAT')
    #print('Using LARGEGALAXIES_CAT={}'.format(lslgafile))
    #if 'ellipse' in lslgafile:
    #    print('Warning: Cannot use $LARGEGALAXIES_CAT with ellipse-fitting results!')
    #    return
    #
    ##outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    ##outdir = '/global/project/projectdirs/cosmo/staging/largegalaxies/{}'.format(version)
    #outdir = legacyhalos.io.legacyhalos_data_dir()
    #outfile = os.path.join(outdir, 'SGA-ellipse-{}.fits'.format(version))
    #if os.path.isfile(outfile) and not clobber:
    #    print('Use --clobber to overwrite existing catalog {}'.format(outfile))
    #    return
    #
    ##if not debug:
    ##    logfile = os.path.join(datadir, '{}-{}.log'.format(galaxy, suffix))
    ##    with open(logfile, 'a') as log:
    ##        with redirect_stdout(log), redirect_stderr(log):
    #            
    #notractorfile = os.path.join(outdir, 'SGA-notractor-{}.fits'.format(version))
    #noellipsefile = os.path.join(outdir, 'SGA-noellipse-{}.fits'.format(version))
    #nogrzfile = os.path.join(outdir, 'SGA-nogrz-{}.fits'.format(version))

    mp = multiproc(nthreads=nproc)
    args = []
    for onegal in sample:
        args.append((onegal, fullsample[fullsample['GROUP_ID'] == onegal['GROUP_ID']], refcat))
    rr = mp.map(_build_ellipse_SGA_one, args)
    rr = list(zip(*rr))

    cat = list(filter(None, rr[0]))
    notractor = list(filter(None, rr[1]))
    noellipse = list(filter(None, rr[2]))
    nogrz = list(filter(None, rr[3]))

    if len(cat) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    cat = vstack(cat)

    if len(notractor) > 0:
        notractor = vstack(notractor)
        print('Writing {} galaxies dropped by Tractor to {}'.format(len(notractor), notractorfile))
        notractor.write(notractorfile, overwrite=True)

    if len(noellipse) > 0:
        noellipse = vstack(noellipse)
        print('Writing {} galaxies not ellipse-fit to {}'.format(len(noellipse), noellipsefile))
        noellipse.write(noellipsefile, overwrite=True)

    if len(nogrz) > 0:
        nogrz = vstack(nogrz)
        print('Writing {} galaxies with no grz coverage to {}'.format(len(nogrz), nogrzfile))
        nogrz.write(nogrzfile, overwrite=True)

    print('Gathered {} pre-burned and frozen galaxies.'.format(len(cat)))
    print('  Frozen (all): {}'.format(np.sum(cat['FREEZE'])))
    print('  Frozen (SGA): {}'.format(np.sum(cat['FREEZE'] * (cat['REF_CAT'] == refcat))))
    print('  Pre-burned: {}'.format(np.sum(cat['PREBURNED'])))

    # We only have frozen galaxies here, but whatever--
    ifreeze = np.where(cat['FREEZE'])[0]
    ilslga = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]

    cat = cat[ifreeze]
    print('Keeping {} frozen galaxies, of which {} are SGA.'.format(len(ifreeze), len(ilslga)))

    # Read the full parent SGA catalog and add all the Tractor columns.
    lslga, hdr = fitsio.read(lslgafile, header=True)
    lslga = Table(lslga)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    # Remove the already-burned SGA galaxies so we don't double-count them--
    ilslga2 = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]
    rem = np.where(np.isin(lslga['SGA_ID'], cat['SGA_ID'][ilslga2]))[0]
    print('Removing {} pre-burned SGA galaxies from the parent catalog, so we do not double-count them.'.format(len(rem)))
    lslga = lslga[np.delete(np.arange(len(lslga)), rem)] # remove duplicates

    # Next, remove galaxies that were either dropped by Tractor in pre-burning
    # or which were not ellipse-fit (the latter are in the Tractor catalog but
    # with REF_CAT='').
    if len(notractor) > 0:
        print('Removing {} SGA galaxies dropped by Tractor.'.format(len(notractor)))
        rem = np.where(np.isin(lslga['SGA_ID'], notractor['SGA_ID']))[0]
        assert(len(rem) == len(notractor))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]
    if len(noellipse) > 0:
        print('Removing {} SGA galaxies not ellipse-fit.'.format(len(noellipse)))
        rem = np.where(np.isin(lslga['SGA_ID'], noellipse['SGA_ID']))[0]
        assert(len(rem) == len(noellipse))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]

    lslga.rename_column('RA', 'SGA_RA')
    lslga.rename_column('DEC', 'SGA_DEC')
    for col in cat.colnames:
        if col in lslga.colnames:
            #print('  Skipping existing column {}'.format(col))
            pass
        else:
            if cat[col].ndim > 1:
                # assume no multidimensional strings or Boolean
                lslga[col] = np.zeros((len(lslga), cat[col].shape[1]), dtype=cat[col].dtype)-1
            else:
                typ = cat[col].dtype.type
                if typ is np.str_ or typ is np.str or typ is np.bool_ or typ is np.bool:
                    lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)
                else:
                    lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)-1
    lslga['RA'][:] = lslga['SGA_RA']
    lslga['DEC'][:] = lslga['SGA_DEC']

    # Stack!
    skipfull = False
    if skipfull:
        print('Temporarily leaving off the original SGA!')
    else:
        out = vstack((lslga, cat))
    del lslga, cat
    out = out[np.argsort(out['SGA_ID'])]
    out = vstack((out[out['SGA_ID'] != -1], out[out['SGA_ID'] == -1]))

    if not skipfull:
        # This may not happen if galaxies are dropped--
        #chk1 = np.where(np.isin(out['SGA_ID'], fullsample['SGA_ID']))[0]
        #assert(len(chk1) == len(fullsample))
        if len(nogrz) > 0:
            chk2 = np.where(np.isin(out['SGA_ID'], nogrz['SGA_ID']))[0]
            assert(len(chk2) == len(nogrz))
        if len(notractor) > 0:
            chk3 = np.where(np.isin(out['SGA_ID'], notractor['SGA_ID']))[0]
            assert(len(chk3) == 0)
        if len(noellipse) > 0:
            chk4 = np.where(np.isin(out['SGA_ID'], noellipse['SGA_ID']))[0]
            assert(len(chk4) == 0)
        if len(nogrz) > 0 and len(notractor) > 0:            
            chk5 = np.where(np.isin(notractor['SGA_ID'], nogrz['SGA_ID']))[0]
            assert(len(chk5) == 0)
    assert(np.all(out['RA'] > 0))
    assert(np.all(np.isfinite(out['PA'])))
    assert(np.all(np.isfinite(out['BA'])))
    ww = np.where(out['SGA_ID'] != -1)[0]
    assert(np.all((out['PA'][ww] >= 0) * (out['PA'][ww] <= 180)))
    assert(np.all((out['BA'][ww] > 0) * (out['BA'][ww] <= 1.0)))

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    hdrversion = 'L{}-ELLIPSE'.format(version[1:2]) # fragile!
    hdr['SGAVER'] = hdrversion
    fitsio.write(outfile, out.as_array(), header=hdr, clobber=True)

    # Write the KD-tree version
    kdoutfile = outfile.replace('.fits', '.kd.fits') # fragile
    cmd = 'startree -i {} -o {} -T -P -k -n largegals'.format(outfile, kdoutfile)
    print(cmd)
    _ = os.system(cmd)

    cmd = 'modhead {} SGAVER {}'.format(kdoutfile, hdrversion)
    print(cmd)
    _ = os.system(cmd)

    fix_permissions = True
    if fix_permissions:
        print('Fixing group permissions.')
        shutil.chown(outfile, group='cosmo')
        shutil.chown(kdoutfile, group='cosmo')

def old_build_ellipse_SGA(sample, fullsample, nproc=1, clobber=False, debug=False):
    """Gather all the ellipse-fitting results and build the final SGA catalog.

    """
    import fitsio
    from contextlib import redirect_stdout, redirect_stderr
    from astropy.table import Table, vstack, join
    from astrometry.util.multiproc import multiproc
    from legacypipe.reference import get_large_galaxy_version

    # This is a little fragile.
    version = SGA_version()
    refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    lslgafile = os.getenv('LARGEGALAXIES_CAT')
    print('Using LARGEGALAXIES_CAT={}'.format(lslgafile))
    if 'ellipse' in lslgafile:
        print('Warning: Cannot use $LARGEGALAXIES_CAT with ellipse-fitting results!')
        return
    
    #outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    #outdir = '/global/project/projectdirs/cosmo/staging/largegalaxies/{}'.format(version)
    outdir = legacyhalos.io.legacyhalos_data_dir()
    outfile = os.path.join(outdir, 'SGA-ellipse-{}.fits'.format(version))
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return

    #if not debug:
    #    logfile = os.path.join(datadir, '{}-{}.log'.format(galaxy, suffix))
    #    with open(logfile, 'a') as log:
    #        with redirect_stdout(log), redirect_stderr(log):
                
    notractorfile = os.path.join(outdir, 'SGA-notractor-{}.fits'.format(version))
    noellipsefile = os.path.join(outdir, 'SGA-noellipse-{}.fits'.format(version))
    nogrzfile = os.path.join(outdir, 'SGA-nogrz-{}.fits'.format(version))

    mp = multiproc(nthreads=nproc)
    args = []
    for onegal in sample:
        args.append((onegal, fullsample[fullsample['GROUP_ID'] == onegal['GROUP_ID']], refcat))
    rr = mp.map(_build_ellipse_SGA_one, args)
    rr = list(zip(*rr))

    cat = list(filter(None, rr[0]))
    notractor = list(filter(None, rr[1]))
    noellipse = list(filter(None, rr[2]))
    nogrz = list(filter(None, rr[3]))

    if len(cat) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    cat = vstack(cat)

    if len(notractor) > 0:
        notractor = vstack(notractor)
        print('Writing {} galaxies dropped by Tractor to {}'.format(len(notractor), notractorfile))
        notractor.write(notractorfile, overwrite=True)

    if len(noellipse) > 0:
        noellipse = vstack(noellipse)
        print('Writing {} galaxies not ellipse-fit to {}'.format(len(noellipse), noellipsefile))
        noellipse.write(noellipsefile, overwrite=True)

    if len(nogrz) > 0:
        nogrz = vstack(nogrz)
        print('Writing {} galaxies with no grz coverage to {}'.format(len(nogrz), nogrzfile))
        nogrz.write(nogrzfile, overwrite=True)

    print('Gathered {} pre-burned and frozen galaxies.'.format(len(cat)))
    print('  Frozen (all): {}'.format(np.sum(cat['FREEZE'])))
    print('  Frozen (SGA): {}'.format(np.sum(cat['FREEZE'] * (cat['REF_CAT'] == refcat))))
    print('  Pre-burned: {}'.format(np.sum(cat['PREBURNED'])))

    # We only have frozen galaxies here, but whatever--
    ifreeze = np.where(cat['FREEZE'])[0]
    ilslga = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]

    cat = cat[ifreeze]
    print('Keeping {} frozen galaxies, of which {} are SGA.'.format(len(ifreeze), len(ilslga)))

    # Read the full parent SGA catalog and add all the Tractor columns.
    lslga, hdr = fitsio.read(lslgafile, header=True)
    lslga = Table(lslga)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    # Remove the already-burned SGA galaxies so we don't double-count them--
    ilslga2 = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]
    rem = np.where(np.isin(lslga['SGA_ID'], cat['SGA_ID'][ilslga2]))[0]
    print('Removing {} pre-burned SGA galaxies from the parent catalog, so we do not double-count them.'.format(len(rem)))
    lslga = lslga[np.delete(np.arange(len(lslga)), rem)] # remove duplicates

    # Next, remove galaxies that were either dropped by Tractor in pre-burning
    # or which were not ellipse-fit (the latter are in the Tractor catalog but
    # with REF_CAT='').
    if len(notractor) > 0:
        print('Removing {} SGA galaxies dropped by Tractor.'.format(len(notractor)))
        rem = np.where(np.isin(lslga['SGA_ID'], notractor['SGA_ID']))[0]
        assert(len(rem) == len(notractor))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]
    if len(noellipse) > 0:
        print('Removing {} SGA galaxies not ellipse-fit.'.format(len(noellipse)))
        rem = np.where(np.isin(lslga['SGA_ID'], noellipse['SGA_ID']))[0]
        assert(len(rem) == len(noellipse))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]

    lslga.rename_column('RA', 'SGA_RA')
    lslga.rename_column('DEC', 'SGA_DEC')
    for col in cat.colnames:
        if col in lslga.colnames:
            #print('  Skipping existing column {}'.format(col))
            pass
        else:
            if cat[col].ndim > 1:
                # assume no multidimensional strings or Boolean
                lslga[col] = np.zeros((len(lslga), cat[col].shape[1]), dtype=cat[col].dtype)-1
            else:
                typ = cat[col].dtype.type
                if typ is np.str_ or typ is np.str or typ is np.bool_ or typ is np.bool:
                    lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)
                else:
                    lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)-1
    lslga['RA'][:] = lslga['SGA_RA']
    lslga['DEC'][:] = lslga['SGA_DEC']

    # Stack!
    skipfull = False
    if skipfull:
        print('Temporarily leaving off the original SGA!')
    else:
        out = vstack((lslga, cat))
    del lslga, cat
    out = out[np.argsort(out['SGA_ID'])]
    out = vstack((out[out['SGA_ID'] != -1], out[out['SGA_ID'] == -1]))

    if not skipfull:
        # This may not happen if galaxies are dropped--
        #chk1 = np.where(np.isin(out['SGA_ID'], fullsample['SGA_ID']))[0]
        #assert(len(chk1) == len(fullsample))
        if len(nogrz) > 0:
            chk2 = np.where(np.isin(out['SGA_ID'], nogrz['SGA_ID']))[0]
            assert(len(chk2) == len(nogrz))
        if len(notractor) > 0:
            chk3 = np.where(np.isin(out['SGA_ID'], notractor['SGA_ID']))[0]
            assert(len(chk3) == 0)
        if len(noellipse) > 0:
            chk4 = np.where(np.isin(out['SGA_ID'], noellipse['SGA_ID']))[0]
            assert(len(chk4) == 0)
        if len(nogrz) > 0 and len(notractor) > 0:            
            chk5 = np.where(np.isin(notractor['SGA_ID'], nogrz['SGA_ID']))[0]
            assert(len(chk5) == 0)
    assert(np.all(out['RA'] > 0))
    assert(np.all(np.isfinite(out['PA'])))
    assert(np.all(np.isfinite(out['BA'])))
    ww = np.where(out['SGA_ID'] != -1)[0]
    assert(np.all((out['PA'][ww] >= 0) * (out['PA'][ww] <= 180)))
    assert(np.all((out['BA'][ww] > 0) * (out['BA'][ww] <= 1.0)))

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    hdrversion = 'L{}-ELLIPSE'.format(version[1:2]) # fragile!
    hdr['SGAVER'] = hdrversion
    fitsio.write(outfile, out.as_array(), header=hdr, clobber=True)

    # Write the KD-tree version
    kdoutfile = outfile.replace('.fits', '.kd.fits') # fragile
    cmd = 'startree -i {} -o {} -T -P -k -n largegals'.format(outfile, kdoutfile)
    print(cmd)
    _ = os.system(cmd)

    cmd = 'modhead {} SGAVER {}'.format(kdoutfile, hdrversion)
    print(cmd)
    _ = os.system(cmd)

    fix_permissions = True
    if fix_permissions:
        print('Fixing group permissions.')
        shutil.chown(outfile, group='cosmo')
        shutil.chown(kdoutfile, group='cosmo')

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
            mag = cat['{}_mag_sb24'.format(band)]
        elif R25:
            mag = cat['{}_mag_sb25'.format(band)]
        elif R26:
            mag = cat['{}_mag_sb26'.format(band)]
        elif cog:
            mag = cat['{}_cog_params_mtot'.format(band)]
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

def build_homehtml(sample, htmldir, homehtml='index.html', pixscale=0.262,
                   racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
                   maketrends=False, fix_permissions=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    import legacyhalos.html
    
    homehtmlfile = os.path.join(htmldir, homehtml)
    print('Building {}'.format(homehtmlfile))

    js = legacyhalos.html.html_javadate()       

    # group by RA slices
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('p {display: inline-block;;}\n')
        html.write('</style>\n')

        html.write('<h1>Siena Galaxy Atlas 2020 (SGA 2020)</h1>\n')

        html.write('<p style="width: 75%">\n')
        html.write("""The Siena Galaxy Atlas (SGA) is an angular diameter-limited sample of
                   galaxies constructed as part of the <a href="http://legacysurvey.org/">DESI Legacy Imaging
                   Surveys.</a> It provides custom, wide-area, optical and infrared
                   mosaics (in grz and W1-W4), azimuthally averaged surface
                   brightness profiles, and both aperture and integrated
                   photometry for a sample of approximately 400,000 galaxies
                   over 20,000 square degrees.</p>\n""")
         
        html.write('<p>The web-page visualizations are organized by one-degree slices of right ascension.</p><br />\n')
        
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr><th>RA Slice</th><th>Number of Galaxies</th></tr>\n')
        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            html.write('<tr><td><a href="RA{0}.html"><h3>{0}</h3></a></td><td>{1}</td></tr>\n'.format(raslice, len(inslice)))
        html.write('</table>\n')

        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(homehtmlfile, group='cosmo')

    # Now build the individual pages (one per RA slice).
    for raslice in sorted(set(raslices)):
        inslice = np.where(raslice == raslices)[0]
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[inslice], html=True)

        slicefile = os.path.join(htmldir, 'RA{}.html'.format(raslice))
        print('Building {}'.format(slicefile))
        
        with open(slicefile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
            html.write('p {width: "75%";}\n')
            html.write('</style>\n')
            
            html.write('<h3>RA Slice {}</h3>\n'.format(raslice))

            html.write('<table>\n')
            html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            html.write('<th> </th>\n')
            html.write('<th>Index</th>\n')
            html.write('<th>ID</th>\n')
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

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, lslga=True)

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.4f}</td>\n'.format(diam1))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
            html.write('</table>\n')
            #count += 1

            html.write('<br /><br />\n')
            html.write('<b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(homehtmlfile, group='cosmo')

    # Optionally build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        print('Building {}'.format(trendshtmlfile))
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

        if fix_permissions:
            shutil.chown(trendshtmlfile, group='cosmo')

def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, homehtml, htmldir,
                       racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy, prevgalaxy,
                       nexthtmlgalaxydir, prevhtmlgalaxydir, verbose, clobber, fix_permissions):
    """Build the web page for a single galaxy.

    """
    import fitsio
    from glob import glob
    import legacyhalos.io
    import legacyhalos.html
    
    if not os.path.exists(htmlgalaxydir1):
        os.makedirs(htmlgalaxydir1)
        if fix_permissions:
            for topdir, dirs, files in os.walk(htmlgalaxydir1):
                for dd in dirs:
                    shutil.chown(os.path.join(topdir, dd), group='cosmo')

    htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
    if os.path.isfile(htmlfile) and not clobber:
        print('File {} exists and clobber=False'.format(htmlfile))
        return
    
    nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
    prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))
    
    js = legacyhalos.html.html_javadate()

    # Support routines--

    def _read_ccds_tractor_sample(prefix):
        nccds, tractor, sample = None, None, None
        
        ccdsfile = glob(os.path.join(galaxydir1, '{}-{}-ccds-*.fits'.format(galaxy1, prefix))) # north or south
        if len(ccdsfile) > 0:
            nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()

        # samplefile can exist without tractorfile when using --just-coadds
        samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))
                
        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # We just care about the galaxies in our sample
            if prefix == 'largegalaxy':
                wt, ws = [], []
                for ii, sid in enumerate(sample['SGA_ID']):
                    ww = np.where(tractor['ref_id'] == sid)[0]
                    if len(ww) > 0:
                        wt.append(ww)
                        ws.append(ii)
                if len(wt) == 0:
                    print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxy1))
                    tractor = None
                else:
                    wt = np.hstack(wt)
                    ws = np.hstack(ws)
                    tractor = tractor[wt]
                    sample = sample[ws]
                    srt = np.argsort(tractor['flux_r'])[::-1]
                    tractor = tractor[srt]
                    sample = sample[srt]
                    assert(np.all(tractor['ref_id'] == sample['SGA_ID']))

        return nccds, tractor, sample

    def _html_group_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, lslga=True)

        html.write('<h2>Group Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        html.write('<th>Index<br />(Primary)</th>\n')
        html.write('<th>ID<br />(Primary)</th>\n')
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
        #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
        html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
        html.write('<td>{}</td>\n'.format(gal['GROUP_NAME']))
        html.write('<td>{:.7f}</td>\n'.format(ra1))
        html.write('<td>{:.7f}</td>\n'.format(dec1))
        html.write('<td>{:.4f}</td>\n'.format(diam1))
        #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
        #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
        #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
        html.write('</tr>\n')
        html.write('</table>\n')

        # Add the properties of each galaxy.
        html.write('<h3>Group Members</h3>\n')
        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>ID</th>\n')
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
            html.write('<td>{}</td>\n'.format(groupgal['SGA_ID']))
            html.write('<td>{}</td>\n'.format(groupgal['GALAXY']))
            typ = groupgal['MORPHTYPE'].strip()
            if typ == '' or typ == 'nan':
                typ = '...'
            html.write('<td>{}</td>\n'.format(typ))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
            html.write('<td>{:.4f}</td>\n'.format(groupgal['D25_LEDA']))
            #if np.isnan(groupgal['PA']):
            #    pa = 0.0
            #else:
            #    pa = groupgal['PA']
            #html.write('<td>{:.2f}</td>\n'.format(pa))
            #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
            html.write('</tr>\n')
        html.write('</table>\n')

    def _html_image_mosaics(html):
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
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
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

    def _html_ellipsefit_and_photometry(html, tractor, sample):
        html.write('<h2>Elliptical Isophote Analysis</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return
            
        html.write('<h3>Geometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="5">Tractor</th>\n')
        html.write('<th colspan="3">ID</th>\n')
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="5">Ellipse Fitting</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>ID</th>\n')
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
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                     galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['d25_leda']*60/2, ellipse['pa_leda'], 1-ellipse['ba_leda']))
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
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
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

        html.write('<tr><th>ID</th><th>Galaxy</th>\n')
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
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                        galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                g, r, z = _get_mags(ellipse, R24=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R25=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R26=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, cog=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            html.write('<h4>{} - {}</h4>\n'.format(galaxyid, sample['GALAXY'][igal]))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                     galaxyid=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue
            #if False:
            #    html.write('<table>\n')
            #    html.write('<tr><th>Fitting range<br />(arcsec)</th><th>Integration<br />mode</th><th>Clipping<br />iterations</th><th>Clipping<br />sigma</th></tr>')
            #    html.write('<tr><td>{:.3f}-{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            #        ellipse[refband]['sma'].min()*pixscale, ellipse[refband]['sma'].max()*pixscale,
            #        ellipse['integrmode'], ellipse['nclip'], ellipse['sclip']))
            #    html.write('</table>\n')
            #    html.write('<br />\n')
            #
            #    html.write('<table>\n')
            #    html.write('<tr><th colspan="5">Mean Geometry</th>')
            #
            #    html.write('<th colspan="4">Ellipse-fitted Geometry</th>')
            #    if ellipse['input_ellipse']:
            #        html.write('<th colspan="2">Input Geometry</th></tr>\n')
            #    else:
            #        html.write('</tr>\n')
            #
            #    html.write('<tr><th>Integer center<br />(x,y, grz pixels)</th><th>Flux-weighted center<br />(x,y grz pixels)</th><th>Flux-weighted size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>')
            #    html.write('<th>Semi-major axis<br />(fitting range, arcsec)</th><th>Center<br />(x,y grz pixels)</th><th>PA<br />(deg)</th><th>e</th>')
            #    if ellipse['input_ellipse']:
            #        html.write('<th>PA<br />(deg)</th><th>e</th></tr>\n')
            #    else:
            #        html.write('</tr>\n')
            #
            #   html.write('<tr><td>({:.0f}, {:.0f})</td><td>({:.3f}, {:.3f})</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(
            #       ellipse['x0'], ellipse['y0'], ellipse['mge_xmed'], ellipse['mge_ymed'], ellipse['mge_majoraxis']*pixscale,
            #       ellipse['mge_pa'], ellipse['mge_eps']))
            #
            #   if 'init_smamin' in ellipse.keys():
            #       html.write('<td>{:.3f}-{:.3f}</td><td>({:.3f}, {:.3f})<br />+/-({:.3f}, {:.3f})</td><td>{:.1f}+/-{:.1f}</td><td>{:.3f}+/-{:.3f}</td>'.format(
            #           ellipse['init_smamin']*pixscale, ellipse['init_smamax']*pixscale, ellipse['x0_median'],
            #           ellipse['y0_median'], ellipse['x0_err'], ellipse['y0_err'], ellipse['pa'], ellipse['pa_err'],
            #           ellipse['eps'], ellipse['eps_err']))
            #   else:
            #       html.write('<td>...</td><td>...</td><td>...</td><td>...</td>')
            #   if ellipse['input_ellipse']:
            #       html.write('<td>{:.1f}</td><td>{:.3f}</td></tr>\n'.format(
            #           np.degrees(ellipse['geometry'].pa)+90, ellipse['geometry'].eps))
            #   else:
            #       html.write('</tr>\n')
            #   html.write('</table>\n')
            #   html.write('<br />\n')

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

    def _html_maskbits(html):
        html.write('<h2>Masking Geometry</h2>\n')
        pngfile = '{}-largegalaxy-maskbits.png'.format(galaxy1)
        html.write('<p>Left panel: color mosaic with the original and final ellipse geometry shown. Middle panel: <i>original</i> maskbits image based on the Hyperleda geometry. Right panel: distribution of all sources and frozen sources (the size of the orange square markers is proportional to the r-band flux).</p>\n')
        html.write('<table width="90%">\n')
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(pngfile))
        html.write('</table>\n')

    def _html_ccd_diagnostics(html):
        html.write('<h2>CCD Diagnostics</h2>\n')

        html.write('<table width="90%">\n')
        pngfile = '{}-ccdpos.png'.format(galaxy1)
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile))
        html.write('</table>\n')
        #html.write('<br />\n')
        
    # Read the catalogs and then build the page--
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='largegalaxy')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>{}</h1>\n'.format(galaxy1))
        raslice = get_raslice(gal[racolumn])
        html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_group_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        _html_maskbits(html)
        _html_ccd_diagnostics(html)

        html.write('<br /><br />\n')
        html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
        html.write('<br />\n')

        html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
        html.write('<br />\n')
        html.write('</html></body>\n')

    if fix_permissions:
        #print('Fixing permissions.')
        shutil.chown(htmlfile, group='cosmo')

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=False,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None, fix_permissions=True):
    """Make the HTML pages.

    """
    import subprocess
    from astrometry.util.multiproc import multiproc

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width

    datadir = legacyhalos.io.legacyhalos_data_dir()
    htmldir = legacyhalos.io.legacyhalos_html_dir()
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)

    if sample is None:
        sample = read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)
        
    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    _, _, done, _ = missing_files(args, sample)
    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return
    
    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_homehtml(sample, htmldir, homehtml=homehtml, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   maketrends=maketrends, fix_permissions=fix_permissions)

    # Now build the individual pages in parallel.
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    rasorted = np.argsort(raslices)
    galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, homehtml, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1
