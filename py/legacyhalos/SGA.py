"""
legacyhalos.SGA
===============

Code to support the SGA sample and project.

# Courteau 'test' sample--
# SGA-mpi --htmlindex --galaxylist NGC1566_GROUP NGC4649_GROUP NGC4694 NGC4477 --htmlhome courteau.html --html-noraslices --clobber

rsync -av /global/cscratch1/sd/ioannis/SGA-data-dr9alpha/065/NGC1566_GROUP /global/cscratch1/sd/ioannis/SGA-data-dr9alpha/187/NGC4477 /global/cscratch1/sd/ioannis/SGA-data-dr9alpha/190/NGC4649_GROUP /global/cscratch1/sd/ioannis/SGA-data-dr9alpha/192/NGC4694 .
tar czvf dr9-courteau.tar.gz NGC1566_GROUP NGC4477 NGC4649_GROUP NGC4694

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

SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds

ELLIPSEBITS = dict(
    largeshift = 2**0,      # >10-pixel shift in the flux-weighted center
    rex_toosmall = 2**1,    # type == REX & shape_r < 5
    notrex_toosmall = 2**2, # type != REX & shape_r < 2
    failed = 2**3,          # ellipse-fitting failed
    notfit = 2**4,          # not ellipse-fit
    indropcat = 2**5,       # in the dropcat catalog
    )

DROPBITS = dict(
    notfit = 2**0,    # no Tractor catalog, not fit
    nogrz = 2**1,     # missing grz coverage
    masked = 2**2,    # masked (e.g., due to a bleed trail)
    dropped = 2**3,   # dropped by Tractor (either spurious or a problem with the fitting)
    isPSF = 2**4,     # tractor type=PSF
    veto = 2**5,      # veto the ellipse-fit geometry
    negflux = 2**6,   # flux_r <= 0
    )

# veto the ellipse-fit parameters for these objects
VETO_ELLIPSE = np.array(list(set([
    # LG dwarfs
    'LeoII',
    'LeoI',
    'LGS3',
    'AndromedaVI',
    'AndromedaXXVIII',
    'Phoenix',
    'Cetus',
    'Tucana',
    'KKR25',
    'LeoA',
    'ESO349-031',
    # ~small galaxies with problematic ellipse-fits and no bright non-SGA companions in the Hyperleda mask
    '2MASXJ13161816+0925305', # bleed trail
    'PGC1029274', # bleed
    'PGC1783580', # star
    'PGC091013', # ellipse diameter too big
    'PGC020336',  # ellipse moves to bright star
    'PGC2022273', # ellipse diameter too big
    'PGC091171',  # ellipse diameter too big
    'PGC2665094', # ellipse moves to bright star
    'PGC090861',  # affected by bleed trail
    'PGC1036933', # ellipse diameter too big
    'PGC437802',  # ellipse diameter too big
    'PGC213966',  # ellipse diameter too big (but we may lose a small companion)
    'PGC2046017', # affected by bleed trail
    'UGC09630',   # ellipse moves to bright star
    'PGC2042481', # ellipse diameter too big
    'PGC022276',  # ellipse moves to bright star
    'PGC061640',  # ellipse diameter too big
    'PGC2123520', # ellipse diameter too big
    'PGC2034992', # affected by bleed trail
    'PGC090944', # ellipse diameter too big
    'PGC090808',  # ellipse diameter too big
    'UGC04788',   # ellipse diameter too big
    'PGC147988', # wrong diameter (bleed trail)
    'PGC061380', # diameter and PA very wrong
    'PGC1090829', # ellipse diameter too big
    'PGC2385881', # totally wrong ellipse diameter
    'PGC2283901', # ellipse diameter too big
    'PGC006108', # ellipse diameter too big
    'PGC3288397', # ellipse diameter too big
    'PGC2371341', # diameter too big?
    'PGC711419',  # ellipse diameter too big
    'PGC145889', # ellipse diam far too large
    'UGC03974', # b/a too narrow
    'PGC2631024', # diameter too big
    'UGC10111', # b/a too big
    'UGC10736', # PA affected by star
    'UGC04363', # ellipse b/a is too narrow and diameter too big (bleed trail)
    'PGC883370', # ellipse center moves to a spurious position
    'PGC2742031', # affected by star
    'UGC03998', # affected by star
    'UGC10061', # irregular galaxy; poor SB profile
    'PGC086661', # Hyperleda is better
    'UGC08614', # Hyperleda is better
    'UGC04483',
    'DDO125',
    'UGC08308',
    'UGC07599',
    #'NGC4204',  # ellipse b/a is too narrow
    #'PGC069404', # PA not great
    #'NGC0660', # ellipse PA is wrong
    #'PGC045254', # b/a could be better
    #'PGC2742031', # ellipse b/a, pa affected by star (fit as PSF)
    #'ESO085-014', # ellipse diameter too big
    #'NGC4395', # too big?
    #'NGC2403', # not big enough?
    #'NGC7462', # ellipse diameter too big
    #'NGC0988', # ellipse diameter too big
    #'NGC0134', # ellipse diameter too big
    #'NGC3623', # ellipse diameter too big
    #'NGC1515', # ellipse diameter too big
    #'NGC3972', # ellipse diameter too big
    #'NGC4527', # ellipse diameter too big
    #'NGC4062', # ellipse diameter too big
    #'NGC4570', # ellipse diameter too big
    #'ESO437-020', # ellipse diameter too big
    #'ESO196-013', # ellipse diameter too big
    #'PGC047705',  # ellipse diameter too big
    #'PGC2369617', # ellipse diameter too big
    #'UGC00484', # ellipse diameter too big
    #'NGC1566',  # ellipse PA not quite right
    ])))

def SGA_version():
    """Archived versions. We used v2.0 for DR8, v3.0 through v7.0 were originally
    pre-DR9 test catalogs (now archived), and DR9 will use v3.0.

    version = 'v5.0' # dr9e
    version = 'v6.0' # dr9f,g
    version = 'v7.0' # more dr9 testing
    
    """
    version = 'v3.0'  # DR9
    return version

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
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py).')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--M33', action='store_true', help='Use a special CCDs file for M33.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--html-noraslices', dest='html_raslices', action='store_false',
                        help='Do not organize HTML pages by RA slice (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')

    parser.add_argument('--ubercal-sky', action='store_true', help='Build the largest large-galaxy coadds with custom (ubercal) sky-subtraction.')

    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
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
            #dependson = '-largegalaxy-ellipse.isdone'
            dependson = None
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-largegalaxy-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        raise ValueError('Need at least one keyword argument.')

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

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                preselect_sample=True, nproc=1,
                #customsky=False, customredux=False, 
                d25min=0.1, d25max=100.0):
    """Read/generate the parent SGA catalog.

    d25min in arcmin

    big = ss[ss['IN_FOOTPRINT'] * (ss['GROUP_DIAMETER']>5) * ss['GROUP_PRIMARY']]
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

    if preselect_sample:
        cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
                'GROUP_PRIMARY', 'GROUP_ID', 'IN_FOOTPRINT', 'SGA_ID', 'GALAXY', 'RA', 'DEC',
                'BRICKNAME']
        sample = fitsio.read(samplefile, columns=cols)
        rows = np.arange(len(sample))

        samplecut = np.where(
            (sample['GROUP_DIAMETER'] > d25min) *
            (sample['GROUP_DIAMETER'] < d25max) *
            ### custom reductions
            #(np.array(['DR8' not in gg for gg in sample['GALAXY']])) *
            (sample['GROUP_PRIMARY'] == True) *
            (sample['IN_FOOTPRINT']))[0]
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
            saga = astropy.table.Table.read(os.path.join(legacyhalos.io.legacyhalos_dir(), 'sample', 'catalogs', 'saga_hosts.csv'))
            #fullsample = legacyhalos.SGA.read_sample(preselect_sample=False)
            m1, m2, d12 = match_radec(sample['RA'][samplecut], sample['DEC'][samplecut],
                                      saga['RA'], saga['DEC'], 5/3600.0, nearest=True)
            #ww = np.where(np.isin(sample['GROUP_ID'], fullsample['GROUP_ID'][m1]))[0]
            #ww = np.hstack([np.where(gid == sample['GROUP_ID'])[0] for gid in fullsample['GROUP_ID'][m1]])
            rows = rows[m1]

        if False: # test fitting of all the DR8 candidates
            fullsample = read_sample(preselect_sample=False, columns=['SGA_ID', 'GALAXY', 'GROUP_ID', 'GROUP_NAME', 'GROUP_DIAMETER', 'IN_FOOTPRINT'])
            ww = np.where(fullsample['SGA_ID'] >= 5e6)[0]
            these = np.where(np.isin(sample['GROUP_ID'][samplecut], fullsample['GROUP_ID'][ww]))[0]
            rows = rows[these]

        nrows = len(rows)
        print('Selecting {} galaxies in the DR9 footprint.'.format(nrows))
        
    #elif customsky:
    ## Select the galaxies requiring custom sky-subtraction.
    #    sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_FOOTPRINT'])
    #    rows = np.arange(len(sample))
    #
    #    samplecut = np.where(
    #        #(sample['GROUP_DIAMETER'] > 5) * 
    #        (sample['GROUP_DIAMETER'] > 5) * (sample['GROUP_DIAMETER'] < 25) *
    #        (sample['GROUP_PRIMARY'] == True) *
    #        (sample['IN_FOOTPRINT']))[0]
    #    #this = np.where(sample['GROUP_NAME'] == 'NGC4448')[0]
    #    #rows = np.hstack((rows, this))
    #
    #    rows = rows[samplecut]
    #    nrows = len(rows)
    #    print('Selecting {} custom sky galaxies.'.format(nrows))
    #    
    #elif customredux:
    #    sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_FOOTPRINT'])
    #    rows = np.arange(len(sample))
    #
    #    samplecut = np.where(
    #        (sample['GROUP_PRIMARY'] == True) *
    #        (sample['IN_FOOTPRINT']))[0]
    #    rows = rows[samplecut]
    #    
    #    customgals = [
    #        'NGC3034_GROUP',
    #        'NGC3077', # maybe?
    #        'NGC3726', # maybe?
    #        'NGC3953_GROUP', # maybe?
    #        'NGC3992_GROUP',
    #        'NGC4051',
    #        'NGC4096', # maybe?
    #        'NGC4125_GROUP',
    #        'UGC07698',
    #        'NGC4736_GROUP',
    #        'NGC5055_GROUP',
    #        'NGC5194_GROUP',
    #        'NGC5322_GROUP',
    #        'NGC5354_GROUP',
    #        'NGC5866_GROUP',
    #        'NGC4258',
    #        'NGC3031_GROUP',
    #        'NGC0598_GROUP',
    #        'NGC5457'
    #        ]
    #
    #    these = np.where(np.isin(sample['GROUP_NAME'][samplecut], customgals))[0]
    #    rows = rows[these]
    #    nrows = len(rows)
    #
    #    print('Selecting {} galaxies with custom reductions.'.format(nrows))
        
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
    
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/north-ellipse-outdated.txt', str)
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/north-refit-ispsf-dropped.txt', str)
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/north-refit-ispsf2.txt', str)

    #galaxylist = np.loadtxt('/global/homes/i/ioannis/closepairs.txt', str)
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/south-closepairs2.txt', str)
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/south-refit-newparent.txt', str)
    #galaxylist = np.loadtxt('/global/homes/i/ioannis/south-ispsf.txt', str)

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

    return sample

def _get_diameter(ellipse):
    """Wrapper to get the mean D(26) diameter.

    ellipse - legacyhalos.ellipse dictionary

    diam in arcmin

    """
    if ellipse['radius_sb26'] > 0:
        diam, diamref = 2 * ellipse['radius_sb26'] / 60, 'SB26' # [arcmin]
    elif ellipse['radius_sb25'] > 0:
        diam, diamref = 1.25 * 2 * ellipse['radius_sb25'] / 60, 'SB25' # [arcmin]
    #elif ellipse['radius_sb24'] > 0:
    #    diam, diamref = 1.5 * ellipse['radius_sb24'] * 2 / 60, 'SB24' # [arcmin]
    else:
        diam, diamref = 1.25 * ellipse['d25_leda'], 'LEDA' # [arcmin]
        #diam, diamref = 2 * ellipse['majoraxis'] * ellipse['refpixscale'] / 60, 'WGHT' # [arcmin]

    if diam <= 0:
        raise ValueError('Doom has befallen you.')

    return diam, diamref

def _init_ellipse_SGA(clobber=False):
    import legacyhalos.io
    from legacyhalos.SGA import SGA_version
    from legacypipe.reference import get_large_galaxy_version

    # This is a little fragile.
    version = SGA_version()
    refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    lslgafile = os.getenv('LARGEGALAXIES_CAT')
    print('Using LARGEGALAXIES_CAT={}'.format(lslgafile))
    if 'ellipse' in lslgafile:
        print('Warning: Cannot use $LARGEGALAXIES_CAT with ellipse-fitting results!')
        return None

    #outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    #outdir = '/global/project/projectdirs/cosmo/staging/largegalaxies/{}'.format(version)
    outdir = legacyhalos.io.legacyhalos_data_dir()
    #print('HACKING THE OUTPUT DIRECTORY!!!') ; outdir = os.path.join(outdir, 'test')
    outfile = os.path.join(outdir, 'SGA-ellipse-{}.fits'.format(version))
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return None

    #if not debug:
    #    logfile = os.path.join(datadir, '{}-{}.log'.format(galaxy, suffix))
    #    with open(logfile, 'a') as log:
    #        with redirect_stdout(log), redirect_stderr(log):

    dropfile = os.path.join(outdir, 'SGA-dropped-{}.fits'.format(version))

    return outfile, dropfile, refcat

def _write_ellipse_SGA(cat, dropcat, outfile, dropfile, refcat,
                       exclude_full_sga=False, writekd=True):
    import shutil
    import fitsio
    from astropy.table import Table, vstack, join
    from legacyhalos.SGA import SGA_version
    #from contextlib import redirect_stdout, redirect_stderr
    
    version = SGA_version()
    
    if len(cat) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    cat = vstack(cat)

    if len(dropcat) > 0:
        dropcat = vstack(dropcat)
        print('Writing {} galaxies to {}'.format(len(dropcat), dropfile))
        dropcat.write(dropfile, overwrite=True)

    print('Gathered {} pre-burned and frozen galaxies.'.format(len(cat)))
    print('  Frozen (all): {}'.format(np.sum(cat['FREEZE'])))
    print('  Frozen (SGA): {}'.format(np.sum(cat['FREEZE'] * (cat['REF_CAT'] == refcat))))
    print('  Pre-burned: {}'.format(np.sum(cat['PREBURNED'])))

    # We only have frozen galaxies here, but whatever--
    ifreeze = np.where(cat['FREEZE'])[0]
    isga = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]

    cat = cat[ifreeze]
    print('Keeping {} frozen galaxies, of which {} are SGA.'.format(len(ifreeze), len(isga)))

    # Read the full parent SGA catalog and add all the Tractor columns.
    sgafile = os.getenv('LARGEGALAXIES_CAT')
    sga, hdr = fitsio.read(sgafile, header=True)
    sga = Table(sga)
    print('Read {} galaxies from {}'.format(len(sga), sgafile))

    # Remove the already-burned SGA galaxies so we don't double-count them--
    isga2 = np.where(cat['FREEZE'] * (cat['REF_CAT'] == refcat))[0]
    rem = np.where(np.isin(sga['SGA_ID'], cat['SGA_ID'][isga2]))[0]
    print('Removing {} pre-burned SGA galaxies from the parent catalog, so we do not double-count them.'.format(len(rem)))
    sga = sga[np.delete(np.arange(len(sga)), rem)] # remove duplicates

    # Update the reference diameter for objects that were not pre-burned--
    print('Updating the reference diameter from Hyperleda.')
    sga['DIAM'] *= 1.25

    # Next, remove all galaxies from the 'dropcat' catalog *except* those with
    # DROPBITS[notfit] | DROPBITS[nogrz]. Every other galaxy is spurious (or not
    # large) in some fashion. Update: all 'dropped' galaxies should be kept!
    print('Found {} SGA galaxies in the dropcat catalog.'.format(len(dropcat)))
    if len(dropcat) > 0:
        if False:
            ignore = np.logical_or(dropcat['DROPBIT'] & DROPBITS['notfit'] != 0,
                                   dropcat['DROPBIT'] & DROPBITS['masked'] != 0)
            ignore = np.logical_or(ignore, dropcat['DROPBIT'] & DROPBITS['nogrz'] != 0)
            ignore = np.where(ignore)[0]
        else:
            ignore = np.arange(len(dropcat))
        if len(ignore) > 0:
            print('Not removing {} dropped SGA galaxies.'.format(len(ignore)))
            ignore_dropcat = dropcat[np.delete(np.arange(len(dropcat)), ignore)] # remove duplicates
        if len(ignore_dropcat) > 0:
            print('Removing {} SGA dropped galaxies.'.format(len(ignore_dropcat)))
            rem = np.where(np.isin(sga['SGA_ID'], ignore_dropcat['SGA_ID']))[0]
            assert(len(rem) == len(ignore_dropcat))
            sga = sga[np.delete(np.arange(len(sga)), rem)]

    sga.rename_column('RA', 'SGA_RA')
    sga.rename_column('DEC', 'SGA_DEC')
    for col in cat.colnames:
        if col in sga.colnames:
            #print('  Skipping existing column {}'.format(col))
            pass
        else:
            if cat[col].ndim > 1:
                # assume no multidimensional strings or Boolean
                sga[col] = np.zeros((len(sga), cat[col].shape[1]), dtype=cat[col].dtype)-1
            else:
                typ = cat[col].dtype.type
                if typ is np.str_ or typ is np.str or typ is np.bool_ or typ is np.bool:
                    sga[col] = np.zeros(len(sga), dtype=cat[col].dtype)
                else:
                    sga[col] = np.zeros(len(sga), dtype=cat[col].dtype)-1
    sga['RA'][:] = sga['SGA_RA']
    sga['DEC'][:] = sga['SGA_DEC']
    sga['DROPBIT'][:] = DROPBITS['nogrz'] # outside the footprint
    sga['ELLIPSEBIT'][:] = ELLIPSEBITS['notfit'] # not fit
    if len(dropcat) > 0:
        these = np.where(np.isin(sga['SGA_ID'], dropcat['SGA_ID']))[0]
        assert(len(these) == len(dropcat))
        sga['DROPBIT'][these] = dropcat['DROPBIT']
        sga['ELLIPSEBIT'][these] = ELLIPSEBITS['indropcat']
        
    # Stack!
    if exclude_full_sga:
        #print('Temporarily leaving off the original SGA!')
        out = cat
    else:
        out = vstack((sga, cat))
    del sga, cat
    out = out[np.argsort(out['SGA_ID'])]
    out = vstack((out[out['SGA_ID'] != -1], out[out['SGA_ID'] == -1]))

    # Annoying hack. Leo I had unWISE time-resolved photometry, which we don't
    # need or want; remove it here. Also remove the FITBITS column, since that
    # was added late as well.
    for col in out.colnames:
        if col == 'FITBITS' or 'NEA' in col or 'LC_' in col:
            out.remove_column(col)

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    hdrversion = 'L{}-ELLIPSE'.format(version[1:2]) # fragile!
    hdr['SGAVER'] = hdrversion
    fitsio.write(outfile, out.as_array(), header=hdr, clobber=True)

    # Write the KD-tree version
    if writekd:
        kdoutfile = outfile.replace('.fits', '.kd.fits') # fragile
        cmd = 'startree -i {} -o {} -T -P -k '.format(outfile, kdoutfile)
        print(cmd)
        _ = os.system(cmd)

        cmd = 'modhead {} SGAVER {}'.format(kdoutfile, hdrversion)
        print(cmd)
        _ = os.system(cmd)

    #fix_permissions = True
    #if fix_permissions:
    #    #print('Fixing group permissions.')
    #    shutil.chown(outfile, group='cosmo')
    #    if writekd:
    #        shutil.chown(kdoutfile, group='cosmo')

def _build_ellipse_SGA_one(args):
    """Wrapper function for the multiprocessing."""
    return build_ellipse_SGA_one(*args)

def build_ellipse_SGA_one(onegal, fullsample, refcat='L3', verbose=False):
    """Gather the ellipse-fitting results for a single galaxy.

    """
    from glob import glob

    import warnings
    import fitsio
    from astropy.table import Table, vstack, hstack, Column
    from astrometry.util.util import Tan
    from tractor.ellipses import EllipseE # EllipseESoft
    from legacyhalos.io import read_ellipsefit, get_run
    from legacyhalos.misc import is_in_ellipse
    from legacyhalos.ellipse import SBTHRESH as sbcuts

    onegal = Table(onegal)
    galaxy, galaxydir = get_galaxy_galaxydir(onegal)

    onegal['DROPBIT'] = np.zeros(1, dtype=np.int32)
    fullsample['DROPBIT'] = np.zeros(len(fullsample), dtype=np.int32)

    # An object may be missing a Tractor catalog either because it wasn't fit or
    # because it is missing grz coverage. In both cases, however, we want to
    # keep them in the SGA catalog, not reject them.
    run = get_run(onegal)
    ccdsfile = os.path.join(galaxydir, '{}-ccds-{}.fits'.format(galaxy, run))
    isdonefile = os.path.join(galaxydir, '{}-largegalaxy-coadds.isdone'.format(galaxy))
    tractorfile = os.path.join(galaxydir, '{}-largegalaxy-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        if os.path.isfile(ccdsfile) and os.path.isfile(isdonefile):
            print('Missing grz coverage in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['nogrz']
        elif not os.path.isfile(ccdsfile) and os.path.isfile(isdonefile): # no photometric CCDs touching brick (e.g., PGC2045341)
            print('Missing grz coverage in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['nogrz']
        elif os.path.isfile(ccdsfile) and not os.path.isfile(isdonefile):
            print('Missing fitting results in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['notfit']
        else: # shouldn't happen...I think
            print('Warning: no Tractor catalog in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['notfit']
        return None, onegal

    # Note: for galaxies on the edge of the footprint we can also sometimes lose
    # 3-band coverage if one or more of the bands is fully masked (these will
    # fail the "found_data" check in legacyhalos.io.read_multiband. Do a similar
    # check here before continuing.
    grzmissing = False
    for band in ['g', 'r', 'z']:
        imfile = os.path.join(galaxydir, '{}-largegalaxy-image-{}.fits.fz'.format(galaxy, band))
        if not os.path.isfile(imfile):
            print('  Missing image {}'.format(imfile), flush=True)
            grzmissing = True
    if grzmissing:
        print('Missing grz coverage in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
        onegal['DROPBIT'] |= DROPBITS['nogrz']
        return None, onegal

    # OK, now keep going!    
    tractor = Table(fitsio.read(tractorfile, upper=True))

    # Remove Gaia stars from the Tractor catalog immediately, so they're not
    # double-counted. If this step removes everything, then it means the galaxy
    # is not a galaxy, so return it in the "notractor" catalog.
    isgaia = np.where(tractor['REF_CAT'] == 'G2')[0]
    if len(isgaia) > 0:
        tractor = tractor[np.delete(np.arange(len(tractor)), isgaia)]
        if len(tractor) == 0: # can happen on the edge of the footprint
            print('Warning: All Tractor sources are Gaia stars in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['nogrz']
            #onegal['DROPBIT'] |= DROPBITS['allGaia']
            return None, onegal
        assert('G2' not in set(tractor['REF_CAT']))

    def _check_grz(galaxydir, galaxy, radec=None, just_ivar=False):
        grzmissing, box = False, 2
        for band in ['g', 'r', 'z']:
            imfile = os.path.join(galaxydir, '{}-largegalaxy-image-{}.fits.fz'.format(galaxy, band))
            ivarfile = os.path.join(galaxydir, '{}-largegalaxy-invvar-{}.fits.fz'.format(galaxy, band))
            wcs = Tan(imfile, 1)
            img = fitsio.read(imfile)
            ivar = fitsio.read(ivarfile)
            H, W = img.shape
            if radec is not None:
                _, xcen, ycen = wcs.radec2pixelxy(radec[0], radec[1])
                ycen, xcen = np.int(ycen-1), np.int(xcen-1)
            else:
                ycen, xcen = H//2, W//2
            #print(band, img[ycen-box:ycen+box, xcen-box:xcen+box],
            #      ivar[ycen-box:ycen+box, xcen-box:xcen+box])
            if just_ivar:
                if np.all(ivar[ycen-box:ycen+box, xcen-box:xcen+box] == 0):
                    grzmissing = True
                    break
            else:
                if (np.all(img[ycen-box:ycen+box, xcen-box:xcen+box] == 0) and
                    np.all(ivar[ycen-box:ycen+box, xcen-box:xcen+box] == 0)):
                    grzmissing = True
                    break
        return grzmissing

    # Make sure we have at least one SGA in this field; if not, it means the
    # galaxy is spurious (or *all* the galaxies, in the case of a galaxy group
    # are spurious). Actually, on the edge of the footprint we can also have a
    # catalog without any SGA sources (i.e., if there are no pixels). Capture
    # that case here, too.
    isga = np.where(tractor['REF_CAT'] == refcat)[0]
    if len(isga) == 0:
        #print('Warning: No SGA sources in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]))
        # Are there pixels?
        grzmissing = _check_grz(galaxydir, galaxy)
        if grzmissing:
            print('Missing grz coverage in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
            onegal['DROPBIT'] |= DROPBITS['nogrz']
        else:
            # Is it fully masked because of a bleed trail or just dropped?
            if _check_grz(galaxydir, galaxy, just_ivar=True):
                print('Masked and dropped by Tractor in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
                onegal['DROPBIT'] |= DROPBITS['masked']
            else:
                print('Dropped by Tractor in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
                onegal['DROPBIT'] |= DROPBITS['dropped']
        return None, onegal
        
    # Next, remove SGA sources which do not belong to this group, because they
    # will be handled when we deal with *that* group. (E.g., PGC2190838 is in
    # the *mosaic* of NGC5899 but does not belong to the NGC5899 "group").
    toss = np.where(np.logical_not(np.isin(tractor['REF_ID'][isga], fullsample['SGA_ID'])))[0]
    if len(toss) > 0:
        for tt in toss:
            if verbose:
                print('  Removing non-primary SGA_ID={}'.format(tractor[isga][tt]['REF_ID']), flush=True)
        keep = np.delete(np.arange(len(tractor)), isga[toss])
        tractor = tractor[keep]

    # Finally toss out Tractor sources which are too small (i.e., are outside
    # the prior range on size in the main pipeline). Actually, the minimum size
    # is 0.01 arcsec, but we cut on 0.1 arcsec to have some margin. If these
    # sources are re-detected in production then so be it. If all sources end up
    # being dropped, then this system is spurious---add it to the "notractor"
    # catalog.
    keep = np.where(np.logical_or(tractor['TYPE'] == 'PSF', tractor['SHAPE_R'] > 0.1))[0]
    if len(keep) == 0:
        print('All Tractor sources have been dropped in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
        onegal['DROPBIT'] |= DROPBITS['allsmall']
        return None, onegal
    tractor = tractor[keep]

    # Next, add all the (new) columns we will need to the Tractor catalog. This
    # is a little wasteful because the non-frozen Tractor sources will be tossed
    # out at the end, but it's much easier and cleaner to do it this way. Also
    # remove the duplicate BRICKNAME column from the Tractor catalog.
    tractor.remove_column('BRICKNAME') # of the form custom-BRICKNAME
    onegal.rename_column('RA', 'SGA_RA')
    onegal.rename_column('DEC', 'SGA_DEC')
    onegal.remove_column('INDEX')
    sgacols = onegal.colnames
    tractorcols = tractor.colnames
    for col in sgacols[::-1]: # reverse the order
        if col in tractorcols:
            print('  Skipping existing column {}'.format(col), flush=True)
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
    tractor['ELLIPSEBIT'][:] = np.zeros(len(tractor), dtype=np.int32) # we don't want -1 here

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

    tractor['PREBURNED'] = np.ones(len(tractor), bool)        # Everything was preburned but we only want to freeze the...
    tractor['FREEZE'] = np.zeros(len(tractor), bool)          # ...SGA galaxies and sources in that galaxy's ellipse.

    # Next, gather up all the ellipse files, which *define* the sample. Also
    # track the galaxies that are dropped by Tractor and, separately, galaxies
    # which fail ellipse-fitting (or are not ellipse-fit because they're too
    # small).
    isdonefile = os.path.join(galaxydir, '{}-largegalaxy-ellipse.isdone'.format(galaxy))
    isfailfile = os.path.join(galaxydir, '{}-largegalaxy-ellipse.isfail'.format(galaxy))

    dropcat = []
    for igal, sga_id in enumerate(np.atleast_1d(fullsample['SGA_ID'])):
        ellipsefile = os.path.join(galaxydir, '{}-largegalaxy-{}-ellipse.fits'.format(galaxy, sga_id))

        # Find this object in the Tractor catalog. 
        match = np.where((tractor['REF_CAT'] == refcat) * (tractor['REF_ID'] == sga_id))[0]
        if len(match) > 1:
            raise ValueError('Multiple matches should never happen but it did in the field of ID={}?!?'.format(onegal['SGA_ID']))

        thisgal = Table(fullsample[igal])

        # For some systems (e.g., LG dwarfs), override the ellipse geometry for
        # specific galaxies where we want the (usually larger) Hyperleda
        # ellipse.
        if thisgal['GALAXY'] in VETO_ELLIPSE:
            print('Vetoing ellipse-fitting results for galaxy {}'.format(thisgal['GALAXY'][0]))
            thisgal['DROPBIT'] |= DROPBITS['veto']
            dropcat.append(thisgal)
            continue

        # An object can be missing an ellipsefit file for two reasons:
        if not os.path.isfile(ellipsefile):
             # If the galaxy does not appear in the Tractor catalog, it was
             # dropped during fitting, which means that it's either spurious (or
             # there's a fitting bug) (or we're missing grz coverage, e.g.,
             # PGC1062274)!
            if len(match) == 0:
                # check for grz coverage
                grzmissing = _check_grz(galaxydir, galaxy, radec=(thisgal['RA'], thisgal['DEC']))
                if grzmissing:
                    print('Missing grz coverage for galaxy {} in the field of {} (SGA_ID={})'.format(
                        fullsample['GALAXY'][igal], galaxy, onegal['SGA_ID'][0]), flush=True)
                    thisgal['DROPBIT'] |= DROPBITS['nogrz']
                    dropcat.append(thisgal)
                else:
                    if verbose:
                        print('Dropped by Tractor and not ellipse-fit: {} (ID={})'.format(fullsample['GALAXY'][igal], sga_id), flush=True)
                    thisgal['DROPBIT'] |= DROPBITS['dropped']
                    dropcat.append(thisgal)
            else:
                # Objects here were fit by Tractor but *not* ellipse-fit. Keep
                # them but reset ref_cat and ref_id. That way if it's a galaxy
                # that just happens to be too small to pass our size cuts in
                # legacyhalos.io.read_multiband to be ellipse-fit, Tractor will
                # still know about it. In particular, if it's a small galaxy (or
                # point source, I guess) *inside* the elliptical mask of another
                # galaxy (see, e.g., SDSSJ123843.02+092744.0 -
                # http://legacysurvey.org/viewer-dev?ra=189.679290&dec=9.462331&layer=dr8&zoom=14&sga),
                # we want to be sure it doesn't get forced PSF in production!
                if verbose:
                    print('Not ellipse-fit: {} (ID={}, type={}, r50={:.2f} arcsec, fluxr={:.3f} nanomaggies)'.format(
                        fullsample['GALAXY'][igal], sga_id, tractor['TYPE'][match[0]], tractor['SHAPE_R'][match[0]],
                        tractor['FLUX_R'][match[0]]), flush=True)

                typ = tractor['TYPE'][match]
                r50 = tractor['SHAPE_R'][match]
                rflux = tractor['FLUX_R'][match]

                # Bug in fit_on_coadds: nobs_[g,r,z] is 1 even when missing the
                # band, so use S/N==0 in grz.
                #ng, nr, nz = tractor['NOBS_G'][match], tractor['NOBS_R'][match], tractor['NOBS_Z'][match]
                ng = tractor['FLUX_G'][match] * np.sqrt(tractor['FLUX_IVAR_G'][match]) == 0
                nr = tractor['FLUX_R'][match] * np.sqrt(tractor['FLUX_IVAR_R'][match]) == 0
                nz = tractor['FLUX_Z'][match] * np.sqrt(tractor['FLUX_IVAR_Z'][match]) == 0

                #if ng == 0 or nr == 0 or nz == 0:
                if ng or nr or nz:
                    thisgal['DROPBIT'] |= DROPBITS['masked']
                    dropcat.append(thisgal)
                elif typ == 'PSF' or rflux < 0:
                    # In some corner cases we can end up as PSF or with negative
                    # r-band flux because of missing grz coverage right at the
                    # center of the galaxy, e.g., PGC046314.
                    grzmissing = _check_grz(galaxydir, galaxy, radec=(thisgal['RA'], thisgal['DEC']))
                    if grzmissing:
                        print('Missing grz coverage in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
                        thisgal['DROPBIT'] |= DROPBITS['nogrz']
                        dropcat.append(thisgal)
                    else:
                        # check for fully mask (e.g. bleed trail)--
                        if _check_grz(galaxydir, galaxy, radec=(thisgal['RA'], thisgal['DEC']), just_ivar=True):
                            print('Masked galaxy in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
                            thisgal['DROPBIT'] |= DROPBITS['masked']
                            dropcat.append(thisgal)
                        elif typ == 'PSF':
                            thisgal['DROPBIT'] |= DROPBITS['isPSF']
                            dropcat.append(thisgal)
                        elif rflux < 0:
                            #print('Negative r-band flux in the field of {} (SGA_ID={})'.format(galaxy, onegal['SGA_ID'][0]), flush=True)
                            thisgal['DROPBIT'] |= DROPBITS['negflux']
                            dropcat.append(thisgal)
                        else:
                            pass
                else:
                    # If either of these files exist (but there's not
                    # ellipse.fits catalog) then something has gone wrong. If
                    # *neither* file exists, then this galaxy was never fit!
                    if os.path.isfile(isfailfile):
                        tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['failed']
                    elif os.path.isfile(isdonefile):
                        if typ == 'REX' and r50 < 5.0:
                            tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['rex_toosmall']
                        elif typ != 'REX' and typ != 'PSF' and r50 < 2.0:
                            tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['notrex_toosmall']
                        else:
                            # corner case (e.g., IC1613) where I think I made the done files by fiat
                            tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['notfit']
                    else:
                        tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['notfit']
                        
                    #if os.path.isfile(isdonefile) and not os.path.isfile(isfailfile):
                    #    if typ == 'REX' and r50 < 5.0:
                    #        tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['rex_toosmall']
                    #    elif typ != 'REX' and typ != 'PSF' and r50 < 2.0:
                    #        tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['notrex_toosmall']
                    #    else:
                    #        pass
                    #elif os.path.isfile(isdonefile) and os.path.isfile(isfailfile):
                    #    tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['failed']
                    #elif not os.path.isfile(isdonefile) and not os.path.isfile(isfailfile):
                    #    tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['notfit']
                    #else:
                    #    raise ValueError('This should never happen....galaxy {} in group {}'.format(fullsample['GALAXY'][igal], galaxy))
                    #    #print('Problem: This should never happen....galaxy {} in group {}'.format(fullsample['GALAXY'][igal], galaxy))
                    #    #pdb.set_trace()

                    # Populate the output catalog--

                    # What do we do about ref_cat, ref_id???
                    #tractor['REF_CAT'][match] = ' '
                    #tractor['REF_ID'][match] = -1 # use -1, 0!
                    tractor['FREEZE'][match] = True

                    thisgal.rename_column('RA', 'SGA_RA')
                    thisgal.rename_column('DEC', 'SGA_DEC')
                    thisgal.remove_column('INDEX')
                    for col in thisgal.colnames:
                        if col == 'ELLIPSEBIT': # skip because we filled it, above
                            #print('  Skipping existing column {}'.format(col))
                            pass
                        else:
                            tractor[col][match] = thisgal[col]
                        
            # Update the nominal diameter--
            tractor['DIAM'][match] = 1.25 * tractor['DIAM'][match]
        else:
            ellipse = read_ellipsefit(galaxy, galaxydir, galaxyid=str(sga_id),
                                      filesuffix='largegalaxy', verbose=True)

            # Objects with "largeshift" shifted positions significantly during
            # ellipse-fitting, which *may* point to a problem. Add a bit--
            if ellipse['largeshift']:
                tractor['ELLIPSEBIT'][match] |= ELLIPSEBITS['largeshift']

            # Get the ellipse-derived geometry, which we'll add to the Tractor
            # catalog below. 
            ragal, decgal = tractor['RA'][match], tractor['DEC'][match]
            pa, ba = ellipse['pa'], 1 - ellipse['eps']
            diam, diamref = _get_diameter(ellipse)
            
            # Next find all the objects in the "ellipse-of-influence" of this
            # galaxy and freeze them. Note: EllipseE.fromRAbPhi wants semi-major
            # axis (i.e., radius) in arcsec.
            reff, e1, e2 = EllipseE.fromRAbPhi(diam*60/2, ba, 180-pa) # note the 180 rotation
            #try:
            inellipse = np.where(is_in_ellipse(tractor['RA'], tractor['DEC'], ragal, decgal, reff, e1, e2))[0]
            #except:
            #    print('!!!!!!!!!!!!!!!!!!!!!', onegal['GROUP_NAME'])

            # This should never happen since the SGA galaxy itself is in the ellipse!
            if len(inellipse) == 0:
                raise ValueError('No galaxies in the ellipse-of-influence in the field of ID={}?!?'.format(onegal['SGA_ID']))

            #print('Freezing the Tractor parameters of {} objects in the ellipse of ID={}.'.format(len(inellipse), sga_id))
            tractor['FREEZE'][inellipse] = True

            # Populate the output catalog--
            thisgal.rename_column('RA', 'SGA_RA')
            thisgal.rename_column('DEC', 'SGA_DEC')
            thisgal.remove_column('INDEX')
            for col in thisgal.colnames:
                tractor[col][match] = thisgal[col]

            # RA, Dec and the geometry values can be different for the VETO list--
            tractor['RA'][match] = ragal
            tractor['DEC'][match] = decgal
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
    if len(keep) > 0:
        #raise ValueError('No frozen galaxies in the field of ID={}?!?'.format(onegal['SGA_ID']))
        tractor = tractor[keep]

    if len(dropcat) > 0:
        dropcat = vstack(dropcat)

    #print(dropcat)
    #print(tractor[tractor['SGA_ID'] != -1])
    #ww = tractor['SGA_ID'] != -1 ; tractor['GALAXY', 'GROUP_NAME', 'RA', 'DEC', 'GROUP_RA', 'GROUP_DEC', 'PA', 'BA', 'DIAM', 'DIAM_REF', 'ELLIPSEBIT'][ww]
    # stop here
    #pdb.set_trace()
    
    return tractor, dropcat

def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          verbose=False):
    """Wrapper to prepare the data for the SGA / large-galaxy project.

    """
    import numpy.ma as ma
    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import srcs2image, ellipse_mask

    bands, refband = data['bands'], data['refband']
    residual_mask = data['residual_mask']

    nbox = 5
    box = np.arange(nbox)-nbox // 2
    #box = np.meshgrid(np.arange(nbox), np.arange(nbox))[0]-nbox//2

    xobj, yobj = np.ogrid[0:data['refband_height'], 0:data['refband_width']]

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if 'galaxy_indx' in data.keys():
        galaxy_indx = data['galaxy_indx']
    else:
        galaxy_indx = np.array([np.argmin((tractor.bx - data['refband_height']/2)**2 + (tractor.by - data['refband_width']/2)**2)])

    #print('Import hack!')
    #import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm
    
    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        if verbose:
            print('Building masked image for central {}/{}.'.format(ii+1, len(galaxy_indx)))

        # Build the model image (of every object except the central)
        # on-the-fly. Need to be smarter about Tractor sources of resolved
        # structure (i.e., sources that "belong" to the central).
        nocentral = np.delete(np.arange(len(tractor)), central)
        srcs = tractor.copy()
        srcs.cut(nocentral)
        model_nocentral = srcs2image(srcs, data['wcs'], band=refband.lower(),
                                     pixelized_psf=data['refband_psf'])

        # Mask all previous (brighter) central galaxies, if any.
        img, newmask = ma.getdata(data[refband]) - model_nocentral, ma.getmask(data[refband])
        for jj in np.arange(ii):
            geo = data['mge'][jj] # the previous galaxy

            # Do this step iteratively to capture the possibility where the
            # previous galaxy has masked the central pixels of the *current*
            # galaxy, in each iteration reducing the size of the mask.
            for shrink in np.arange(0.1, 1.05, 0.05)[::-1]:
                maxis = shrink * geo['majoraxis']
                _mask = ellipse_mask(geo['xmed'], geo['ymed'], maxis, maxis * (1-geo['eps']),
                                     np.radians(geo['theta']-90), xobj, yobj)
                notok = False
                for xb in box:
                    for yb in box:
                        if _mask[int(yb+tractor.by[central]), int(xb+tractor.bx[central])]:
                            notok = True
                            break
                if notok:
                #if _mask[int(tractor.by[central]), int(tractor.bx[central])]:
                    print('The previous central has masked the current central with shrink factor {:.2f}'.format(shrink))
                else:
                    break
            newmask = ma.mask_or(_mask, newmask)

        # Next, get the basic galaxy geometry and pack it into a dictionary. If
        # the object of interest has been masked by, e.g., an adjacent star
        # (see, e.g., IC4041), temporarily unmask those pixels using the Tractor
        # geometry.
        
        minsb = 10**(-0.4*(27.5-22.5)) / filt2pixscale[refband]**2
        #import matplotlib.pyplot as plt ; plt.clf()
        #mgegalaxy = find_galaxy(img / filt2pixscale[refband]**2, nblob=1, binning=3, quiet=not verbose, plot=True, level=minsb)
        #mgegalaxy = find_galaxy(img / filt2pixscale[refband]**2, nblob=1, fraction=0.1, binning=3, quiet=not verbose, plot=True)
        notok, val = False, []
        for xb in box:
            for yb in box:
                #print(xb, yb, val)
                val.append(newmask[int(yb+tractor.by[central]), int(xb+tractor.bx[central])])
                
        # Use np.any() here to capture the case where a handful of the central
        # pixels are masked due to, e.g., saturation, which if we don't do, will
        # cause issues in the ellipse-fitting (specifically with
        # CentralEllipseFitter(censamp).fit() if the very central pixel is
        # masked).  For a source masked by a star, np.all() would have worked
        # fine.
        if np.any(val):
            notok = True
            
        if notok:
            print('Central position has been masked, possibly by a star (or saturated core).')
            xmed, ymed = tractor.by[central], tractor.bx[central]
            #if largegalaxy:
            #    ba = tractor.ba_leda[central]
            #    pa = tractor.pa_leda[central]
            #    maxis = tractor.d25_leda[central] * 60 / 2 / filt2pixscale[refband] # [pixels]
            ee = np.hypot(tractor.shape_e1[central], tractor.shape_e2[central])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[central], tractor.shape_e1[central]) / 2))
            pa = pa % 180
            maxis = 1.5 * tractor.shape_r[central] / filt2pixscale[refband] # [pixels]
            theta = (270 - pa) % 180
                
            fixmask = ellipse_mask(xmed, ymed, maxis, maxis*ba, np.radians(theta-90), xobj, yobj)
            newmask[fixmask] = ma.nomask
        
        #import matplotlib.pyplot as plt ; plt.clf()
        mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), 
                                nblob=1, binning=3, level=minsb)#, plot=True)#, quiet=not verbose
        #plt.savefig('junk.png') ; pdb.set_trace()

        # Above, we used the Tractor positions, so check one more time here with
        # the light-weighted positions, which may have shifted into a masked
        # region (e.g., check out the interacting pair PGC052639 & PGC3098317).
        val = []
        for xb in box:
            for yb in box:
                val.append(newmask[int(xb+mgegalaxy.xmed), int(yb+mgegalaxy.ymed)])
        if np.any(val):
            notok = True

        # If we fit the geometry by unmasking pixels using the Tractor fit then
        # we're probably sitting inside the mask of a bright star, so call
        # find_galaxy a couple more times to try to grow the "unmasking".
        if notok:
            print('Iteratively unmasking pixels:')
            maxis = 1.0 * mgegalaxy.majoraxis # [pixels]
            print('  r={:.2f} pixels'.format(maxis))
            prevmaxis, iiter, maxiter = 0.0, 0, 4
            while (maxis > prevmaxis) and (iiter < maxiter):
                #print(prevmaxis, maxis, iiter, maxiter)
                print('  r={:.2f} pixels'.format(maxis))
                fixmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed,
                                       maxis, maxis * (1-mgegalaxy.eps), 
                                       np.radians(mgegalaxy.theta-90), xobj, yobj)
                newmask[fixmask] = ma.nomask
                mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), 
                                        nblob=1, binning=3, quiet=True, plot=False, level=minsb)
                prevmaxis = maxis.copy()
                maxis = 1.2 * mgegalaxy.majoraxis # [pixels]
                iiter += 1

        #plt.savefig('junk.png') ; pdb.set_trace()
        print(mgegalaxy.xmed, tractor.by[central], mgegalaxy.ymed, tractor.bx[central])
        maxshift = 10
        if (np.abs(mgegalaxy.xmed-tractor.by[central]) > maxshift or # note [xpeak,ypeak]-->[by,bx]
            np.abs(mgegalaxy.ymed-tractor.bx[central]) > maxshift):
            print('Peak position has moved by more than {} pixels---falling back on Tractor geometry!'.format(maxshift))
            #import matplotlib.pyplot as plt ; plt.clf()
            #mgegalaxy = find_galaxy(ma.masked_array(img/filt2pixscale[refband]**2, newmask), nblob=1, binning=3, quiet=False, plot=True, level=minsb)
            #plt.savefig('junk.png') ; pdb.set_trace()
            #pdb.set_trace()
            largeshift = True
            
            ee = np.hypot(tractor.shape_e1[central], tractor.shape_e2[central])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[central], tractor.shape_e1[central]) / 2))
            mgegalaxy.xmed = tractor.by[central]
            mgegalaxy.ymed = tractor.bx[central]
            mgegalaxy.xpeak = tractor.by[central]
            mgegalaxy.ypeak = tractor.bx[central]
            mgegalaxy.eps = 1 - ba
            mgegalaxy.pa = pa % 180
            mgegalaxy.theta = (270 - pa) % 180
            mgegalaxy.majoraxis = 2 * tractor.shape_r[central] / filt2pixscale[refband] # [pixels]
            print('  r={:.2f} pixels'.format(mgegalaxy.majoraxis))
            fixmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed,
                                   mgegalaxy.majoraxis, mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
                                   np.radians(mgegalaxy.theta-90), xobj, yobj)
            newmask[fixmask] = ma.nomask
        else:
            largeshift = False

        #if tractor.ref_id[central] == 474614:
        #    import matplotlib.pyplot as plt
        #    plt.imshow(mask, origin='lower')
        #    plt.savefig('junk.png')
        #    pdb.set_trace()
            
        radec_med = data['wcs'].pixelToPosition(mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        radec_peak = data['wcs'].pixelToPosition(mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {'largeshift': largeshift,
            'ra': tractor.ra[central], 'dec': tractor.dec[central],
            'bx': tractor.bx[central], 'by': tractor.by[central],
            'mw_transmission_g': tractor.mw_transmission_g[central],
            'mw_transmission_r': tractor.mw_transmission_r[central],
            'mw_transmission_z': tractor.mw_transmission_z[central],
            'ra_x0': radec_med[0], 'dec_y0': radec_med[1],
            #'ra_peak': radec_med[0], 'dec_peak': radec_med[1]
            }
        for key in ('eps', 'majoraxis', 'pa', 'theta', 'xmed', 'ymed', 'xpeak', 'ypeak'):
            mge[key] = np.float32(getattr(mgegalaxy, key))
            if key == 'pa': # put into range [0-180]
                mge[key] = mge[key] % np.float32(180)
        data['mge'].append(mge)

        # Now, loop on each filter and build a custom image and mask for each
        # central. Specifically, pack the model-subtracted images images
        # corresponding to each (unique) central into a list. Note that there's
        # a little bit of code to deal with different pixel scales but this case
        # requires more work.
        
        #for filt in [refband]:
        for filt in bands:
            thispixscale = filt2pixscale[filt]
            
            imagekey, varkey = '{}_masked'.format(filt), '{}_var'.format(filt)
            if imagekey not in data.keys():
                data[imagekey], data[varkey] = [], []

            factor = filt2pixscale[refband] / filt2pixscale[filt]
            majoraxis = 1.5 * factor * mgegalaxy.majoraxis # [pixels]

            # Grab the pixels belonging to this galaxy so we can unmask them below.
            central_mask = ellipse_mask(mge['xmed'] * factor, mge['ymed'] * factor, 
                                        majoraxis, majoraxis * (1-mgegalaxy.eps), 
                                        np.radians(mgegalaxy.theta-90), xobj, yobj)
            if np.sum(central_mask) == 0:
                print('No pixels belong to the central galaxy---this is bad!')
                data['failed'] = True
                break

            # Build the mask from the (cumulative) residual-image mask and the
            # inverse variance mask for this galaxy, but then "unmask" the
            # pixels belonging to the central.
            _residual_mask = residual_mask.copy()
            _residual_mask[central_mask] = ma.nomask
            mask = ma.mask_or(_residual_mask, newmask, shrink=False)

            #import matplotlib.pyplot as plt
            #plt.clf() ; plt.imshow(central_mask, origin='lower') ; plt.savefig('junk2.png')
            #pdb.set_trace()

            # Need to be smarter about the srcs list...
            srcs = tractor.copy()
            srcs.cut(nocentral)
            model_nocentral = srcs2image(srcs, data['wcs'], band=filt.lower(),
                                         pixelized_psf=data['refband_psf'])

            # Convert to surface brightness and 32-bit precision.
            img = (ma.getdata(data[filt]) - model_nocentral) / thispixscale**2 # [nanomaggies/arcsec**2]
            img = ma.masked_array(img.astype('f4'), mask)
            var = data['{}_var_'.format(filt)] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #img.filled(fill_value)
            data[imagekey].append(img)
            data[varkey].append(var)

            #if tractor.ref_id[central] == 474614:
            #    import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm ; plt.clf()
            #    thisimg = np.log10(data[imagekey][ii]) ; norm = simple_norm(thisimg, 'log') ; plt.imshow(thisimg, origin='lower', norm=norm) ; plt.savefig('junk{}.png'.format(ii+1))
            #    pdb.set_trace()

    # Cleanup?
    for filt in bands:
        del data[filt]
        del data['{}_var_'.format(filt)]

    return data            

def read_multiband(galaxy, galaxydir, filesuffix='largegalaxy', refband='r', 
                   bands=['g', 'r', 'z'], pixscale=0.262, fill_value=0.0,
                   verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    from astrometry.util.fits import fits_table
    from legacypipe.bits import MASKBITS
    from legacyhalos.io import _get_psfsize_and_depth, _read_image_data

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data, filt2imfile, filt2pixscale = {}, {}, {}

    for band in bands:
        filt2imfile.update({band: {'image': '{}-image'.format(filesuffix),
                                   'model': '{}-model'.format(filesuffix),
                                   'invvar': '{}-invvar'.format(filesuffix),
                                   'psf': '{}-psf'.format(filesuffix),
                                   }})
        filt2pixscale.update({band: pixscale})
    filt2imfile.update({'tractor': '{}-tractor'.format(filesuffix),
                        'sample': '{}-sample'.format(filesuffix),
                        'maskbits': '{}-maskbits'.format(filesuffix),
                        })

    # Do all the files exist? If not, bail!
    missing_data = False
    for filt in bands:
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            #if imtype == 'sky': # this is a dictionary entry
            #    continue
            imfile = os.path.join(galaxydir, '{}-{}-{}.fits.fz'.format(galaxy, filt2imfile[filt][imtype], filt))
            #print(imtype, imfile)
            if os.path.isfile(imfile):
                filt2imfile[filt][imtype] = imfile
            else:
                if verbose:
                    print('File {} not found.'.format(imfile))
                missing_data = True
                break
    
    if missing_data:
        return data

    # Pack some preliminary info into the output dictionary.
    data['filesuffix'] = filesuffix
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)
    data['failed'] = False # be optimistic!

    # Add ellipse parameters like maxsma, delta_logsma, etc.?


    
    # We ~have~ to read the tractor catalog using fits_table because we will
    # turn these catalog entries into Tractor sources later.
    tractorfile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['tractor']))
    if verbose:
        print('Reading {}'.format(tractorfile))
        
    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2',
            'flux_g', 'flux_r', 'flux_z',
            'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
            'nobs_g', 'nobs_r', 'nobs_z',
            'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 
            'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
            'psfsize_g', 'psfsize_r', 'psfsize_z']
    #if galex:
    #    cols = cols+['flux_fuv', 'flux_nuv']
    #if unwise:
    #    cols = cols+['flux_w1', 'flux_w1', 'flux_w1', 'flux_w1']
    tractor = fits_table(tractorfile, columns=cols)
    hdr = fitsio.read_header(tractorfile)
    if verbose:
        print('Read {} sources from {}'.format(len(tractor), tractorfile))
    data.update(_get_psfsize_and_depth(tractor, bands, pixscale, incenter=False))

    # Read the maskbits image and build the starmask.
    maskbitsfile = os.path.join(galaxydir, '{}-{}.fits.fz'.format(galaxy, filt2imfile['maskbits']))
    if verbose:
        print('Reading {}'.format(maskbitsfile))
    maskbits = fitsio.read(maskbitsfile)
    # initialize the mask using the maskbits image
    starmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) | (maskbits & MASKBITS['MEDIUM'] != 0) |
                 (maskbits & MASKBITS['CLUSTER'] != 0) | (maskbits & MASKBITS['ALLMASK_G'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_R'] != 0) | (maskbits & MASKBITS['ALLMASK_Z'] != 0) )

    # Read the basic imaging data and masks.
    data = _read_image_data(data, filt2imfile, starmask=starmask,
                            fill_value=fill_value, verbose=verbose)

    # Figure out which galaxies we are going to ellipse-fit by iterating on all
    # the SGA sources in the field and gather the data we need.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    sample = Table(fitsio.read(samplefile, upper=True))    
    if sample:
        print('Read {} sources from {}'.format(len(sample), samplefile))
    
    # Be pedantic to be sure we get it right (np.isin doens't preserve order)-- 
    msg = []
    islslga = ['L' in refcat for refcat in tractor.ref_cat] # e.g., L6
    minsize = 2.0     # [arcsec]
    minsize_rex = 5.0 # minimum size for REX [arcsec]
    galaxy_indx, reject_galaxy, keep_galaxy = [], [], []
    data['tractor_flags'] = {}
    for ii, sid in enumerate(sample['SGA_ID']):
        I = np.where((sid == tractor.ref_id) * islslga)[0]
        if len(I) == 0: # dropped by Tractor
            reject_galaxy.append(ii)
            data['tractor_flags'].update({str(sid): 'dropped'})
            msg.append('Dropped by Tractor (spurious?)')
        else:
            r50 = tractor.shape_r[I][0]
            refflux = tractor.get('flux_{}'.format(refband))[I][0]
            # Bug in fit_on_coadds: nobs_[g,r,z] is 1 even when missing the
            # band, so use flux_ivar_[g,r,z].
            #ng, nr, nz = tractor.nobs_g[I][0], tractor.nobs_z[I][0], tractor.nobs_z[I][0]
            #if ng < 1 or nr < 1 or nz < 1:
            ng = tractor.flux_g[I][0] * np.sqrt(tractor.flux_ivar_g[I][0]) == 0
            nr = tractor.flux_r[I][0] * np.sqrt(tractor.flux_ivar_r[I][0]) == 0
            nz = tractor.flux_z[I][0] * np.sqrt(tractor.flux_ivar_z[I][0]) == 0
            if ng or nr or nz:
                reject_galaxy.append(ii)
                data['tractor_flags'].update({str(sid): 'nogrz'})
                msg.append('Missing 3-band coverage')
            elif tractor.type[I] == 'PSF': # always reject
                reject_galaxy.append(ii)
                data['tractor_flags'].update({str(sid): 'psf'})
                msg.append('Tractor type=PSF')
            elif refflux < 0:
                reject_galaxy.append(ii)
                data['tractor_flags'].update({str(sid): 'negflux'})
                msg.append('{}-band flux={:.3g} (<=0)'.format(refband, refflux))
            elif r50 < minsize:
                reject_galaxy.append(ii)
                data['tractor_flags'].update({str(sid): 'anytype_toosmall'})
                msg.append('type={}, r50={:.3f} (<{:.1f}) arcsec'.format(tractor.type[I], r50, minsize))
            elif tractor.type[I] == 'REX':
                if r50 < minsize_rex: # REX must have a minimum size
                    reject_galaxy.append(ii)
                    data['tractor_flags'].update({str(sid): 'rex_toosmall'})
                    msg.append('Tractor type=REX & r50={:.3f} (<{:.1f}) arcsec'.format(r50, minsize_rex))
                else:
                    keep_galaxy.append(ii)
                    galaxy_indx.append(I)
            else:
                keep_galaxy.append(ii)
                galaxy_indx.append(I)

    if len(reject_galaxy) > 0:
        reject_galaxy = np.hstack(reject_galaxy)
        for jj, rej in enumerate(reject_galaxy):
            print('  Dropping {} (SGA_ID={}, RA, Dec = {:.7f} {:.7f}): {}'.format(
                sample[rej]['GALAXY'], sample[rej]['SGA_ID'], sample[rej]['RA'], sample[rej]['DEC'], msg[jj]))

    if len(galaxy_indx) > 0:
        keep_galaxy = np.hstack(keep_galaxy)
        galaxy_indx = np.hstack(galaxy_indx)
        sample = sample[keep_galaxy]
    else:
        data['failed'] = True
        return data

    #sample = sample[np.searchsorted(sample['SGA_ID'], tractor.ref_id[galaxy_indx])]
    assert(np.all(sample['SGA_ID'] == tractor.ref_id[galaxy_indx]))

    tractor.d25_leda = np.zeros(len(tractor), dtype='f4')
    tractor.pa_leda = np.zeros(len(tractor), dtype='f4')
    tractor.ba_leda = np.zeros(len(tractor), dtype='f4')
    if 'D25_LEDA' in sample.colnames and 'PA_LEDA' in sample.colnames and 'BA_LEDA' in sample.colnames:
        tractor.d25_leda[galaxy_indx] = sample['D25_LEDA']
        tractor.pa_leda[galaxy_indx] = sample['PA_LEDA']
        tractor.ba_leda[galaxy_indx] = sample['BA_LEDA']

    # Do we need to take into account the elliptical mask of each source??
    srt = np.argsort(tractor.flux_r[galaxy_indx])[::-1]
    galaxy_indx = galaxy_indx[srt]
    print('Sort by flux! ', tractor.flux_r[galaxy_indx])
    galaxy_id = tractor.ref_id[galaxy_indx]

    data['galaxy_id'] = galaxy_id
    data['galaxy_indx'] = galaxy_indx

    # Now build the multiband mask.
    data = _build_multiband_mask(data, tractor, filt2pixscale,
                                 fill_value=fill_value,
                                 verbose=verbose)

    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][0]), origin='lower') ; plt.savefig('junk1.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][1]), origin='lower') ; plt.savefig('junk2.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][2]), origin='lower') ; plt.savefig('junk3.png')
    #pdb.set_trace()

    # Gather some additional info that we want propagated to the output ellipse
    # catalogs.
    if data['galaxy_id'] is not None:
        central_galaxy_id_all = data['central_galaxy_id']
    else:
        central_galaxy_id_all = np.atleast_1d(1)

    for igal in np.arange(len(central_galaxy_id_all)):
        central_galaxy_id = central_galaxy_id_all[igal]
        galaxy_id = str(central_galaxy_id)
        print('Starting ellipse-fitting for galaxy {} with {} core(s)'.format(galaxy_id, nproc))
        if largegalaxy:
            import astropy.units as u
            # Supplement the fit results dictionary with some additional info.
            samp = sample[sample['SGA_ID'] == central_galaxy_id]
            galaxyinfo = {'sga_id': (central_galaxy_id, ''),
                          'galaxy': (str(np.atleast_1d(samp['GALAXY'])[0]), '')}
            for key, unit in zip(['ra', 'dec', 'pgc', 'pa_leda', 'ba_leda', 'd25_leda'],
                                 [u.deg, u.deg, '', u.deg, '', u.arcmin]):
                galaxyinfo[key] = (np.atleast_1d(samp[key.upper()])[0], unit)
            # Specify the fitting range
            maxis = data['mge'][igal]['majoraxis'] # [pixels]
            if samp['D25_LEDA'] > 10 or samp['PGC'] == 31968: # e.g., NGC5457
                if samp['PGC'] == 31968: # special-case NGC3344
                    print('Special-casing PGC031968==NGC3344!')
                maxsma = 1.5 * maxis # [pixels]
                if delta_sma is None:
                    delta_sma = 0.003 * maxsma
            else:
                maxsma = 2 * maxis # [pixels]
                if delta_sma is None:
                    delta_sma = 0.0015 * maxsma
            if delta_sma < 1:
                delta_sma = 1.0
            print('  majoraxis={:.2f} pix, maxsma={:.2f} pix, delta_sma={:.1f} pix'.format(maxis, maxsma, delta_sma))

    return data, galaxyinfo

def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1,
                 filesuffix='largegalaxy', bands=['g', 'r', 'z'], refband='r',
                 unwise=False, verbose=False, debug=False, logfile=None):
    """Wrapper on legacyhalos.mpi.call_ellipse but with specific preparatory work
    and hooks for the SGA project.

    """
    from legacyhalos.mpi import call_ellipse as mpi_call_ellipse


    data, galaxyinfo = read_multiband(galaxy, galaxydir, bands=bands,
                                      filesuffix=filesuffix,
                                      refband=refband, pixscale=pixscale,
                                      verbose=verbose)

    mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                     pixscale=pixscale, nproc=nproc, 
                     bands=bands, refband=refband, sbthresh=SBTHRESH,
                     verbose=verbose, debug=debug, logfile=logfile)

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

def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
                   maketrends=False, fix_permissions=True, html_raslices=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    import legacyhalos.html
    
    htmlhomefile = os.path.join(htmldir, htmlhome)
    print('Building {}'.format(htmlhomefile))

    js = legacyhalos.html.html_javadate()       

    # group by RA slices
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(htmlhomefile, 'w') as html:
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

        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        # The default is to organize the sample by RA slice, but support both options here.
        if html_raslices:
            html.write('<p>The web-page visualizations are organized by one-degree slices of right ascension.</p><br />\n')

            html.write('<table>\n')
            html.write('<tr><th>RA Slice</th><th>Number of Galaxies</th></tr>\n')
            for raslice in sorted(set(raslices)):
                inslice = np.where(raslice == raslices)[0]
                html.write('<tr><td><a href="RA{0}.html"><h3>{0}</h3></a></td><td>{1}</td></tr>\n'.format(raslice, len(inslice)))
            html.write('</table>\n')
        else:
            html.write('<br /><br />\n')
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th> </th>\n')
            html.write('<th>Index</th>\n')
            html.write('<th>ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')
            
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)
            for gal, galaxy1, htmlgalaxydir1 in zip(sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-largegalaxy-grz-montage.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-largegalaxy-grz-montage.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.4f}</td>\n'.format(diam1))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                html.write('</tr>\n')
            html.write('</table>\n')
            
        # close up shop
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(htmlhomefile, group='cosmo')
        
    # Optionally build the individual pages (one per RA slice).
    if html_raslices:
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
                    viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

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
            shutil.chown(htmlhomefile, group='cosmo')

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

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
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
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

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

        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
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
        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
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
              htmlhome='index.html', html_raslices=True,
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
    _, missing, done, _ = missing_files(args, sample)
    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return
    
    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    trendshtml = 'trends.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   maketrends=maketrends, fix_permissions=fix_permissions,
                   html_raslices=html_raslices)

    # Now build the individual pages in parallel.
    if html_raslices:
        raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
        rasorted = np.argsort(raslices)
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)
    else:
        rasorted = np.arange(len(sample))
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1
