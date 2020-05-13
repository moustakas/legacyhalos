"""
legacyhalos.LSLGA
=================

Code to deal with the LSLGA sample and project.

"""
import os, shutil, pdb
import numpy as np
import astropy

ZCOLUMN = 'Z'
RACOLUMN = 'GROUP_RA'
DECCOLUMN = 'GROUP_DEC'
DIAMCOLUMN = 'GROUP_DIAMETER'

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

def missing_files_one(galaxy, galaxydir, filesuffix, dependson, clobber):
    checkfile = os.path.join(galaxydir, '{}{}'.format(galaxy, filesuffix))
    #print('missing_files_one: ', checkfile)
    if os.path.exists(checkfile) and clobber is False:
        # Is the stage that this stage depends on done, too?
        if dependson is not None:
            dependsfile = os.path.join(galaxydir, '{}{}'.format(galaxy, dependson))
            if os.path.exists(dependsfile):
                return 'done'
            else:
                return 'todo'
        else:
            return 'done'
    else:
        #print('missing_files_one: ', checkfile)
        # Did this object fail?
        if '.isdone' in checkfile:
            failfile = checkfile.replace('.isdone', '.isfail')
            if os.path.exists(failfile):
                if clobber is False:
                    return 'fail'
                else:
                    os.remove(failfile)
                    return 'todo'
        return 'todo'
    
def missing_files(args, sample, size=1, clobber_overwrite=None):
    from astrometry.util.multiproc import multiproc

    dependson = None
    galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    if args.largegalaxy_coadds:
        suffix = 'largegalaxy-coadds'
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
    elif args.build_LSLGA:
        suffix = 'build-LSLGA'
        filesuffix = '-largegalaxy-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-largegalaxy-grz-montage.png'
        else:
            filesuffix = '-ccdpos.png'
            #filesuffix = '-largegalaxy-maskbits.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-largegalaxy-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        print('Nothing to do.')
        return

    # Make clobber=False for build_LSLGA and htmlindex because we're not making
    # the files here, we're just looking for them. The argument args.clobber
    # gets used downstream.
    if args.htmlindex or args.build_LSLGA:
        clobber = False
    else:
        clobber = args.clobber

    if clobber_overwrite is not None:
        clobber = clobber_overwrite

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)

    mp = multiproc(nthreads=args.nproc)
    args = [(gal, gdir, filesuffix, dependson, clobber) for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir))]
    todo = np.array(mp.map(_missing_files_one, args))

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
        weight = np.atleast_1d(sample['D25'])[_todo_indices]
        cumuweight = weight.cumsum() / weight.sum()
        idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        if len(idx) < size: # can happen in corner cases
            todo_indices = np.array_split(_todo_indices, size) # unweighted
        else:
            todo_indices = np.array_split(_todo_indices, idx) # weighted
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices
    
def LSLGA_version():
    #version = 'v5.0' # dr9e
    #version = 'v6.0'  # dr9f,g
    version = 'v7.0'  # DR9
    return version

def LSLGA_dir():
    if 'LSLGA_DIR' not in os.environ:
        print('Required ${LSLGA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def LSLGA_data_dir():
    if 'LSLGA_DATA_DIR' not in os.environ:
        print('Required ${LSLGA_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def LSLGA_html_dir():
    if 'LSLGA_HTML_DIR' not in os.environ:
        print('Required ${LSLGA_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
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

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                customsky=False, preselect_sample=True, d25min=0.1, d25max=100.0):
    """Read/generate the parent LSLGA catalog.

    d25min in arcmin

    big = ss[ss['IN_DESI'] * (ss['GROUP_DIAMETER']>5) * ss['GROUP_PRIMARY']]
    %time bricks = np.hstack([survey.get_bricks_near(bb['GROUP_RA'], bb['GROUP_DEC'], bb['GROUP_DIAMETER']/60).brickname for bb in big])

    """
    import fitsio
    from legacyhalos.brick import brickname as get_brickname
            
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
        sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        rows = np.arange(len(sample))

        samplecut = np.where(
            #(sample['GROUP_DIAMETER'] > 5) * 
            #(sample['GROUP_DIAMETER'] > 5) * (sample['GROUP_DIAMETER'] < 25) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        #this = np.where(sample['GROUP_NAME'] == 'NGC4448')[0]
        #rows = np.hstack((rows, this))

        rows = rows[samplecut]
        nrows = len(rows)
        print('Selecting {} custom sky galaxies.'.format(nrows))
        
    elif preselect_sample:
        cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
                'GROUP_PRIMARY', 'GROUP_ID', 'IN_DESI', 'LSLGA_ID', 'GALAXY', 'RA', 'DEC']
        sample = fitsio.read(samplefile, columns=cols)
        rows = np.arange(len(sample))

        samplecut = np.where(
            (sample['GROUP_DIAMETER'] > d25min) *
            (sample['GROUP_DIAMETER'] < d25max) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_DESI']))[0]
        rows = rows[samplecut]

        brickname = get_brickname(sample['GROUP_RA'][samplecut], sample['GROUP_DEC'][samplecut])

        if False: # LSLGA-data-DR9-test3 sample
            # Select galaxies containing DR8-supplemented sources
            #ww = []
            #w1 = np.where(sample['GROUP_MULT'] > 1)[0]
            #for I, gid in zip(w1, sample['GROUP_ID'][w1[:50]]):
            #    w2 = np.where(sample['GROUP_ID'] == gid)[0]
            #    if np.any(['DR8-' in nn for nn in sample['GALAXY'][w2]]):
            #        ww.append(I)
            these = np.where(['DR8-' in nn for nn in sample['GROUP_NAME'][samplecut]])[0]
            rows = rows[these]
            
        if False: # LSLGA-data-DR9-test2 sample
            #bb = [692770, 232869, 51979, 405760, 1319700, 1387188, 519486, 145096]
            #ww = np.where(np.isin(sample['LSLGA_ID'], bb))[0]
            #ff = get_brickname(sample['GROUP_RA'][ww], sample['GROUP_DEC'][ww])
            
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
            #rows = np.where([brick in bricklist for brick in brickname])[0]
            brickcut = np.where(np.isin(brickname, bricklist))[0]
            rows = rows[brickcut]

        if False: # largest galaxies which may need reprocessing (just the north)
            #bricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-DR9SV-north.txt'), dtype='str')
            #brickcut = np.where(np.isin(brickname, bricklist))[0]
            #rows = rows[brickcut]
            m1 = sample['GROUP_DEC'][samplecut] > 30 # 32.375
            rows = rows[m1]
            
        if False: # SAGA host galaxies
            from astrometry.libkd.spherematch import match_radec
            saga = astropy.table.Table.read(os.path.join(LSLGA_dir(), 'sample', 'saga_hosts.csv'))
            #fullsample = legacyhalos.LSLGA.read_sample(preselect_sample=False)
            m1, m2, d12 = match_radec(sample['RA'][samplecut], sample['DEC'][samplecut],
                                      saga['RA'], saga['DEC'], 5/3600.0, nearest=True)
            #ww = np.where(np.isin(sample['GROUP_ID'], fullsample['GROUP_ID'][m1]))[0]
            #ww = np.hstack([np.where(gid == sample['GROUP_ID'])[0] for gid in fullsample['GROUP_ID'][m1]])
            rows = rows[m1]

        if False: # test fitting of all the DR8 candidates
            fullsample = read_sample(preselect_sample=False, columns=['LSLGA_ID', 'GALAXY', 'GROUP_ID', 'GROUP_NAME', 'GROUP_DIAMETER', 'IN_DESI'])
            ww = np.where(fullsample['LSLGA_ID'] >= 6e6)[0]
            #galaxylist = np.unique(fullsample['GROUP_NAME'][ww])
            #print(len(ww), len(set(fullsample['GROUP_ID'][ww])))
            #for gid in set(fullsample['GROUP_ID'][ww]):
            #    ii = np.where(gid == sample['GROUP_ID'][samplecut])[0]
            #    jj = np.where(gid == fullsample['GROUP_ID'])[0]
            #    print(astropy.table.Table(sample[samplecut][ii]))
            #    print(fullsample[jj])
            #    pdb.set_trace()
            these = np.where(np.isin(sample['GROUP_ID'][samplecut], fullsample['GROUP_ID'][ww]))[0]
            rows = rows[these]
            
        if True: # DR9 bricklist
            nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-dr9h-north.txt'), dtype='str')
            sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-dr9h-south.txt'), dtype='str')
            
            #nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-dr9-north.txt'), dtype='str')
            #sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-dr9-south.txt'), dtype='str')
            #nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-DR9SV-north.txt'), dtype='str')
            #sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9', 'bricklist-DR9SV-south.txt'), dtype='str')
            bricklist = np.union1d(nbricklist, sbricklist)
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
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/junk', dtype=str)
    
    if galaxylist is not None:
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

def _build_ellipse_LSLGA_one(args):
    """Wrapper function for the multiprocessing."""
    return build_ellipse_LSLGA_one(*args)

def build_ellipse_LSLGA_one(onegal, fullsample, refcat='L7'):
    """Gather the ellipse-fitting results for a single galaxy.

    """
    from glob import glob

    import warnings
    import fitsio
    from astropy.table import Table, vstack, hstack
    from tractor.ellipses import EllipseE # EllipseESoft
    from legacyhalos.io import read_ellipsefit
    from legacyhalos.misc import is_in_ellipse
    from legacyhalos.ellipse import SBTHRESH as sbcuts
    
    onegal = Table(onegal)
    galaxy, galaxydir = get_galaxy_galaxydir(onegal)

    tractorfile = os.path.join(galaxydir, '{}-largegalaxy-tractor.fits'.format(galaxy))
    # These galaxies are missing because we don't have grz coverage. We want to
    # keep them in the LSLGA catalog, though, so don't add them to the `reject`
    # list here.
    if not os.path.isfile(tractorfile):
        print('Tractor catalog missing: {}'.format(tractorfile))
        return None, None, None, None, onegal

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
        return None, None, None, None, onegal
    
    tractor = Table(fitsio.read(tractorfile))

    # Remove Gaia stars immediately, so they're not double-counted.
    notgaia = np.where(tractor['ref_cat'] != 'G2')[0]
    if len(notgaia) > 0:
        tractor = tractor[notgaia]
    assert('G2' not in set(tractor['ref_cat']))

    # Also remove LSLGA sources which do not belong to this group, because they
    # will be handled when we deal with *that* group. (E.g., PGC2190838 is in
    # the *mosaic* of NGC5899 but does not belong to its group).
    ilslga = np.where(tractor['ref_cat'] == refcat)[0]
    toss = np.where(np.logical_not(np.isin(tractor['ref_id'][ilslga], fullsample['LSLGA_ID'])))[0]
    if len(toss) > 0:
        for tt in toss:
            print('  Removing non-primary LSLGA_ID={}'.format(tractor[ilslga][tt]['ref_id']))
        keep = np.delete(np.arange(len(tractor)), ilslga[toss])
        tractor = tractor[keep]

    # Initialize every source with PREBURNED=True, FREEZE=False
    tractor['preburned'] = np.ones(len(tractor), bool)  # Everything was preburned but we only want to freeze the...
    tractor['freeze'] = np.zeros(len(tractor), bool)    # ...LSLGA galaxies and sources in that galaxie's ellipse.
        
    # Add the columns we'll need in legacypipe.reference plus some additional
    # columns that we want to copy over from the original LSLGA.
    tractor['diam'] = np.zeros(len(tractor), np.float32) - 1 
    tractor['diam_ref'] = np.zeros(len(tractor), 'U4')
    tractor['pa'] = np.zeros(len(tractor), np.float32) - 1
    tractor['ba'] = np.ones(len(tractor), np.float32) - 1
    
    #copycols = [
    #    'IN_DESI_NORTH',
    #    'IN_DESI_NORTH_GRZ',
    #    'IN_DESI_SOUTH',
    #    'IN_DESI_SOUTH_GRZ',
    #    'IN_DESI',
    #    'IN_DESI_GRZ',
    #    'GROUP_ID',
    #    'GROUP_NAME',
    #    'GROUP_MULT',
    #    'GROUP_PRIMARY',
    #    'GROUP_RA',
    #    'GROUP_DEC',
    #    'GROUP_DIAMETER',
    #    'NEAR_BRIGHTSTAR']
    #for col in copycols:
    #    tractor[col] = np.repeat(onegal[col].data, len(tractor))
    #tractor['GROUP_PRIMARY'][:] = False

    # Gather up the legacyhalos.ellipse.ellipse_cog results--
    radkeys = ['radius_sb{:0g}'.format(sbcut) for sbcut in sbcuts]
    for radkey in radkeys:
        tractor[radkey] = np.zeros(len(tractor), np.float32) - 1
        for filt in ['g', 'r', 'z']:
            magkey = radkey.replace('radius_', 'mag_{}_'.format(filt))
            tractor[magkey] = np.zeros(len(tractor), np.float32) - 1
    for filt in ['g', 'r', 'z']:
        tractor['mag_{}_tot'.format(filt)] = np.zeros(len(tractor), np.float32) - 1
    
    # Gather up all the ellipse files, which *define* the sample, but also track
    # the galaxies that are dropped by Tractor and, separately, galaxies which
    # fail ellipse-fitting (or are not ellipse-fit because they're too small).
    notractor, noellipse, badcenter = [], [], []
    ellipseisdone = os.path.isfile(os.path.join(galaxydir, '{}-largegalaxy-ellipse.isdone'.format(galaxy)))
    for igal, lslga_id in enumerate(np.atleast_1d(fullsample['LSLGA_ID'])):
        ellipsefile = os.path.join(galaxydir, '{}-largegalaxy-{}-ellipse.asdf'.format(galaxy, lslga_id))

        # Find this object in the Tractor catalog. If it's not here, it was
        # dropped from Tractor fitting, which most likely means it's spurious!
        # Add it to the 'notractor' pile.
        this = np.where((tractor['ref_cat'] == refcat) * (tractor['ref_id'] == lslga_id))[0]
        if len(this) > 1:
            raise ValueError('Multiple matches should never happen!')

        if not os.path.isfile(ellipsefile):
            if len(this) == 0:
                # Add some info from the Tractor catalog so we can debug--
                print('Tractor reject & not ellipse-fit: {} (LSLGA_ID={})'.format(fullsample['GALAXY'][igal], lslga_id))
                notrac = Table(fullsample[igal]['LSLGA_ID', 'GALAXY', 'RA', 'DEC', 'GROUP_NAME', 'GROUP_ID', 'D25', 'PA', 'BA'])
                notrac['TRACTOR_RA'] = np.float64(-1)
                notrac['TRACTOR_DEC'] = np.float64(-1)
                notrac['TRACTOR_TYPE'] = np.array('   ')
                notrac['FLUX_R'] = np.float32(-1)
                notrac['SHAPE_R'] = np.float32(-1)
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
                # we want to be sure it doesn't get forced PSF!
                print('Not ellipse-fit: {} (LSLGA_ID={}, type={}, r50={:.2f} arcsec, fluxr={:.3f} nanomaggies)'.format(
                    fullsample['GALAXY'][igal], lslga_id, tractor['type'][this[0]], tractor['shape_r'][this[0]],
                    tractor['flux_r'][this[0]]))
                tractor['ref_cat'][this] = ' '
                tractor['ref_id'][this] = -1 # use -1, not zero!
                tractor['freeze'][this] = True
                
                noell = Table(fullsample[igal]['LSLGA_ID', 'GALAXY', 'RA', 'DEC', 'GROUP_NAME', 'GROUP_ID', 'D25', 'PA', 'BA'])
                noell['TRACTOR_RA'] = tractor['ra'][this]
                noell['TRACTOR_DEC'] = tractor['dec'][this]
                noell['TRACTOR_TYPE'] = tractor['type'][this]
                noell['FLUX_R'] = tractor['flux_r'][this]
                noell['SHAPE_R'] = tractor['shape_r'][this]
                noellipse.append(noell)
        else:
            af = read_ellipsefit(galaxy, galaxydir, galaxyid=str(lslga_id), filesuffix='largegalaxy', verbose=True)
            ellipse = af.tree
            # Objects with "badcenter" shifted positions significantly during
            # ellipse-fitting, which *may* point to a problem. Inspect those
            # separately.
            if ellipse['badcenter']:
                badcenter.append(Table(fullsample[igal]))

            # Add the basic ellipse geometry and the aperture photometry--
            tractor['freeze'][this] = True
            tractor['pa'][this] = ellipse['pa']
            tractor['ba'][this] = 1 - ellipse['eps']
            for radkey in radkeys:
                tractor[radkey][this] = ellipse[radkey]
                for filt in ['g', 'r', 'z']:
                    magkey = radkey.replace('radius_', 'mag_{}_'.format(filt))
                    if magkey not in ellipse.keys():
                        print('Problem with {}!'.format(galaxy))
                    tractor[magkey][this] = ellipse[magkey]
                    if bool(ellipse['cog_params_{}'.format(filt)]):
                        tractor['mag_{}_tot'.format(filt)][this] = ellipse['cog_params_{}'.format(filt)]['mtot']
                    else:
                        tractor['mag_{}_tot'.format(filt)][this] = -1.0

            # Now add the radius and track where it came from--
            if ellipse['radius_sb26'] > 0:
                tractor['diam_ref'][this] = 'sb26'
                tractor['diam'][this] = ellipse['radius_sb26'] * 2 / 60
            elif ellipse['radius_sb25'] > 0:
                tractor['diam_ref'][this] = 'sb25'
                tractor['diam'][this] = 1.2 * ellipse['radius_sb25'] * 2 / 60
            #elif ellipse['radius_sb24'] > 0:
            #    tractor['diam_ref'][this] = 'sb24'
            #    tractor['diam'][this] = 1.5 * ellipse['radius_sb24'] * 2 / 60
            else:
                tractor['diam_ref'][this] = 'leda'
                tractor['diam'][this] = ellipse['lslga_d25']
                #tractor['d25'][this] = ellipse['majoraxis'] * ellipse['refpixscale'] * 2 / 60

            if tractor['diam'][this] <= 0:
                raise ValueError('Doom has befallen you.')

            # Next find all the objects in the "ellipse-of-influence" of this
            # galaxy, require a minimum size (or PSF), and freeze those
            # parameters as well.
            reff, e1, e2 = EllipseE.fromRAbPhi(tractor['diam'][this].data*60/2,
                                               tractor['ba'][this].data,
                                               180-tractor['pa'][this].data) # note the 180 rotation
            cut1 = is_in_ellipse(tractor['ra'], tractor['dec'], tractor['ra'][this].data,
                                 tractor['dec'][this].data, reff, e1, e2)
            cut2 = np.logical_or(tractor['type'] == 'PSF', tractor['shape_r'] > 0.1)
            #cut3 = (tractor['flux_g'] > 0) * (tractor['flux_r'] > 0) * (tractor['flux_z'] > 0)
            these = np.where(cut1 * cut2)[0]
            #these = np.where(cut1)[0]
            if len(these) > 0: # this should never happen since the LSLGA galaxy itself is in the ellipse!
                #print('Freezing the Tractor parameters of {} non-LSLGA objects.'.format(len(these)))
                tractor['freeze'][these] = True

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
            #   # pdb.set_trace()

    for col in tractor.colnames:
        tractor.rename_column(col, col.upper())
        
    if len(notractor) > 0:
        notractor = vstack(notractor)
        #for col in notractor.colnames:
        #    notractor.rename_column(col, col.upper())
    if len(noellipse) > 0:
        noellipse = vstack(noellipse)
        for col in noellipse.colnames:
            noellipse.rename_column(col, col.upper())
    if len(badcenter) > 0:
        badcenter = vstack(badcenter)
        for col in badcenter.colnames:
            badcenter.rename_column(col, col.upper())

    return tractor, notractor, noellipse, badcenter, None

def build_ellipse_LSLGA(sample, fullsample, nproc=1, clobber=False):
    """Gather all the ellipse-fitting results and build the final LSLGA catalog.

    """
    import fitsio
    from astropy.table import Table, vstack, join
    from pydl.pydlutils.spheregroup import spheregroup
    from astrometry.util.multiproc import multiproc
    from legacypipe.reference import get_large_galaxy_version
        
    # This is a little fragile.
    version = LSLGA_version()
    refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    lslgafile = os.getenv('LARGEGALAXIES_CAT')
    print('Using LARGEGALAXIES_CAT={}'.format(lslgafile))
    if 'ellipse' in lslgafile:
        print('Warning: Cannot use $LARGEGALAXIES_CAT with ellipse-fitting results!')
        return
    
    #outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    #outdir = '/global/project/projectdirs/cosmo/staging/largegalaxies/{}'.format(version)
    outdir = LSLGA_data_dir()
    outfile = os.path.join(outdir, 'LSLGA-ellipse-{}.fits'.format(version))
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return

    notractorfile = os.path.join(outdir, 'LSLGA-notractor-{}.fits'.format(version))
    noellipsefile = os.path.join(outdir, 'LSLGA-noellipse-{}.fits'.format(version))
    badcenterfile = os.path.join(outdir, 'LSLGA-badcenter-{}.fits'.format(version))
    nogrzfile = os.path.join(outdir, 'LSLGA-nogrz-{}.fits'.format(version))

    mp = multiproc(nthreads=nproc)
    args = []
    for onegal in sample:
        args.append((onegal, fullsample[fullsample['GROUP_ID'] == onegal['GROUP_ID']], refcat))
    rr = mp.map(_build_ellipse_LSLGA_one, args)
    rr = list(zip(*rr))

    cat = list(filter(None, rr[0]))
    notractor = list(filter(None, rr[1]))
    noellipse = list(filter(None, rr[2]))
    badcenter = list(filter(None, rr[3]))
    nogrz = list(filter(None, rr[4]))

    if len(cat) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    cat = vstack(cat)
    #print(cat[cat['REF_ID'] == 393385])
    #pdb.set_trace()

    if len(notractor) > 0:
        notractor = vstack(notractor)
        #[reject.rename_column(col, col.upper()) for col in reject.colnames]
        #reject = reject['GALAXY', 'RA', 'DEC', 'LSLGA_ID', 'D25', 'PA', 'BA',
        #                'TRACTOR_TYPE', 'FLUX_R', 'SHAPE_R']
        #notractor.rename_column('GALAXY', 'NAME')
        print('Writing {} galaxies dropped by Tractor to {}'.format(len(notractor), notractorfile))
        notractor.write(notractorfile, overwrite=True)

    if len(noellipse) > 0:
        noellipse = vstack(noellipse)
        print('Writing {} galaxies not ellipse-fit to {}'.format(len(noellipse), noellipsefile))
        noellipse.write(noellipsefile, overwrite=True)

    if len(badcenter) > 0:
        badcenter = vstack(badcenter)
        badcenter = badcenter['GALAXY', 'RA', 'DEC', 'LSLGA_ID', 'D25', 'PA', 'BA']
        #badcenter.rename_column('GALAXY', 'NAME')
        print('Writing {} galaxies with bad ellipse centers to {}'.format(len(badcenter), badcenterfile))
        badcenter.write(badcenterfile, overwrite=True)
        del badcenter

    if len(nogrz) > 0:
        nogrz = vstack(nogrz)
        nogrz = nogrz['GALAXY', 'RA', 'DEC', 'LSLGA_ID', 'D25', 'PA', 'BA']
        #nogrz.rename_column('GALAXY', 'NAME')
        print('Writing {} galaxies with no grz coverage to {}'.format(len(nogrz), nogrzfile))
        nogrz.write(nogrzfile, overwrite=True)
        del nogrz

    #for d1, d2 in zip(cat[0].dtype.descr, cat[1].dtype.descr):
    #    if d1 != d2:
    #        print(d1, d2)
    print('Gathered {} pre-burned and frozen galaxies.'.format(len(cat)))
    print('  Frozen (all): {}'.format(np.sum(cat['FREEZE'])))
    print('  Frozen (LSLGA): {}'.format(np.sum(cat['FREEZE'] * (cat['REF_CAT'] == refcat))))
    print('  Pre-burned: {}'.format(np.sum(cat['PREBURNED'])))

    ifreeze = np.where(cat['FREEZE'])[0]
    ilslga = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]

    cat = cat[ifreeze]
    catcols = cat.colnames
    #print('Tossing out {} sources with FREEZE!=True.'.format(np.sum(~cat['FREEZE'])))
    print('Keeping {} frozen galaxies, of which {} are LSLGA.'.format(len(ifreeze), len(ilslga)))

    # Read the full parent LSLGA catalog and add all the Tractor columns.
    lslga, hdr = fitsio.read(lslgafile, header=True)
    lslga = Table(lslga)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    # Remove galaxies that were either dropped by Tractor in pre-burning or
    # which were not ellipse-fit (the latter are in the Tractor catalog but with
    # REF_CAT='').
    if len(notractor) > 0:
        print('Removing {} LSLGA galaxies dropped by Tractor.'.format(len(notractor)))
        rem = np.where(np.isin(lslga['LSLGA_ID'], notractor['LSLGA_ID']))[0]
        assert(len(rem) == len(notractor))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]
    if len(noellipse) > 0:
        print('Removing {} LSLGA galaxies not ellipse-fit.'.format(len(noellipse)))
        rem = np.where(np.isin(lslga['LSLGA_ID'], noellipse['LSLGA_ID']))[0]
        assert(len(rem) == len(noellipse))
        lslga = lslga[np.delete(np.arange(len(lslga)), rem)]

    lslga['LSLGA_RA'] = lslga['RA']
    lslga['LSLGA_DEC'] = lslga['DEC']
    lslga['MORPHTYPE'] = lslga['TYPE']
    lslga['D25_LEDA'] = lslga['D25']
    lslga['PA_LEDA'] = lslga['PA']
    lslga['BA_LEDA'] = lslga['BA']
    lslga['SB_D25_LEDA'] = lslga['SB_D25']
    lslga.remove_column('SB_D25')
    lslga.rename_column('D25', 'DIAM')
    lslga['DIAM_REF'] = 'leda'
    lslgacols = lslga.colnames

    # Create a temporary catalog and add a temporary column to enable the join--
    _lslga = lslga.copy()
    for col in np.intersect1d(lslga.colnames, cat.colnames):
        _lslga.remove_column(col)
    _lslga['REF_ID'] = lslga['LSLGA_ID']

    # Merge the Tractor and LSLGA catalogs, but we have to be careful to treat
    # LSLGA galaxies separately.
    I = np.where(cat['REF_CAT'] == refcat)[0]
    J = np.where(cat['REF_CAT'] != refcat)[0]

    catI = join(_lslga, cat[I], keys='REF_ID')
    catJ = cat[J]
    del _lslga

    for col in lslgacols:
        if col in catcols:
            print('  Skipping existing column {}'.format(col))
        else:
            if lslga[col].ndim > 1:
                catJ[col] = np.zeros((len(catJ), lslga[col].shape[1]), dtype=lslga[col].dtype)
            else:
                catJ[col] = np.zeros(len(catJ), dtype=lslga[col].dtype)
    catJ['LSLGA_ID'] = -1

    out = vstack((catI, catJ)) # reassemble!
    del catI, catJ

    # Next, remove the already-burned LSLGA galaxies so we don't double-count them.
    rem = np.where(np.isin(lslga['LSLGA_ID'], out['REF_ID']))[0]
    print('Removing {} LSLGA galaxies from the pre-burned catalog, so we do not double-count them.'.format(len(rem)))
    lslga = lslga[np.delete(np.arange(len(lslga)), rem)] # remove duplicates

    #chk1 = np.where(out['PREBURNED'] * (out['REF_CAT'] == 'L6'))[0]
    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.scatter(out['D25'], out['D25']/out['D25_ORIG']) ; plt.xscale('log') ; plt.savefig('junk.png')
    #plt.clf() ; plt.scatter(out['D25'], out['PA']-out['PA_ORIG']) ; plt.xscale('log') ; plt.savefig('junk.png')

    # Add Tractor columns to the original LSLGA catalog.
    for col in catcols:
        if col in lslgacols:
            print('  Skipping existing column {}'.format(col))
        else:
            if cat[col].ndim > 1:
                lslga[col] = np.zeros((len(lslga), cat[col].shape[1]), dtype=cat[col].dtype)
            else:
                lslga[col] = np.zeros(len(lslga), dtype=cat[col].dtype)
    lslga['REF_CAT'][:] = refcat
    lslga['REF_ID'] = lslga['LSLGA_ID']
                
    # Stack!
    skipfull = False
    if skipfull:
        print('Temporarily leaving off the original LSLGA!')
    else:
        out = vstack((lslga, out))
    out = out[np.argsort(out['LSLGA_ID'])]
    out = vstack((out[out['LSLGA_ID'] != -1], out[out['LSLGA_ID'] == -1]))
    del lslga, cat

    # One final check--every galaxy in fullsample should be accounted for in
    # either 'out', 'notractor', or 'noellipse', with no duplication!  Galaxies
    # in 'nogrz' *will* be in 'out' because we still want to use them in
    # production even though we couldn't do ellipse-fitting.
    _out = out[(out['LSLGA_ID'] != -1) * out['PREBURNED']]
    #for dd in _out['LSLGA_ID']:
    #    ww = np.where(dd == _out['LSLGA_ID'])[0]
    #    if len(ww) > 1:
    #        print(dd, ww)
    try:
        if not skipfull:
            chk1 = np.where(np.isin(fullsample['LSLGA_ID'], _out['LSLGA_ID']))[0]
            assert(len(chk1) == len(_out))
            if len(nogrz) > 0:
                chk2 = np.where(np.isin(out['LSLGA_ID'], nogrz['LSLGA_ID']))[0]
                assert(len(chk2) == len(nogrz))
            if len(notractor) > 0:
                chk3 = np.where(np.isin(out['LSLGA_ID'], notractor['LSLGA_ID']))[0]
                assert(len(chk3) == 0)
            if len(noellipse) > 0:
                chk4 = np.where(np.isin(out['LSLGA_ID'], noellipse['LSLGA_ID']))[0]
                assert(len(chk4) == 0)
            if len(nogrz) > 0 and len(notractor) > 0:            
                chk5 = np.where(np.isin(notractor['LSLGA_ID'], nogrz['LSLGA_ID']))[0]
                assert(len(chk5) == 0)
    except:
        pdb.set_trace()

    assert(np.all(np.isfinite(out['PA'])))
    assert(np.all(np.isfinite(out['BA'])))

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    hdrversion = 'L{}-ELLIPSE'.format(version[1:2]) # fragile!
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
            #if 'mag_{}_sb24'.format(band) in cat.keys():
            #    mag = cat['mag_{}_sb24'.format(band)]
            #else:
            #    print('Fix me')
            #    mag = -1
        elif R25:
            mag = cat['mag_{}_sb25'.format(band)]
            #if 'mag_{}_sb25'.format(band) in cat.keys():
            #    mag = cat['mag_{}_sb25'.format(band)]
            #else:
            #    mag = -1
        elif R26:
            mag = cat['mag_{}_sb26'.format(band)]
            #if 'mag_{}_sb26'.format(band) in cat.keys():
            #    mag = cat['mag_{}_sb26'.format(band)]
            #else:
            #    mag = -1
        elif cog:
            if bool(cat['cog_params_{}'.format(band)]):
                mag = cat['cog_params_{}'.format(band)]['mtot']
            else:
                mag = -1
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

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, lslga=True)

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['LSLGA_ID']))
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
                for ii, sid in enumerate(sample['LSLGA_ID']):
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
                    assert(np.all(tractor['ref_id'] == sample['LSLGA_ID']))

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
            typ = groupgal['TYPE'].strip()
            if typ == '' or typ == 'nan':
                typ = '...'
            html.write('<td>{}</td>\n'.format(typ))
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
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            html.write('<h4>LSLGA {} - {}</h4>\n'.format(galaxyid, sample['GALAXY'][igal]))

            af = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                galaxyid=galaxyid, verbose=verbose)
            if not bool(af):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue
            
            ellipse = af.tree
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
            af.close()

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

    datadir = LSLGA_data_dir()
    htmldir = LSLGA_html_dir()
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
