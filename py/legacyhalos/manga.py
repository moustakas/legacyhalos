"""
legacyhalos.manga
=================

Code to deal with the MaNGA-NSA sample and project.

"""
import os, shutil, pdb
import numpy as np
import astropy

import legacyhalos.io

ZCOLUMN = 'Z'
RACOLUMN = 'IFURA' # 'RA'
DECCOLUMN = 'IFUDEC' # 'DEC'
GALAXYCOLUMN = 'PLATEIFU'
REFIDCOLUMN = 'MANGANUM' # 'MANGAID_INT'

RADIUSFACTOR = 4 # 10
MANGA_RADIUS = 36.75 # / 2 # [arcsec]

ELLIPSEBITS = dict(
    largeshift = 2**0, # >10-pixel shift in the flux-weighted center
    )

SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds
APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] # multiples of MAJORAXIS

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')
    #parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--resampled-phot', action='store_true', help='Do photometry on the resampled images.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--resampled-pixscale', default=0.75, type=float, help='resampled pixel scale (arcsec/pix).')

    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--nomakeplots', action='store_true', help='Do not remake the QA plots for the HTML pages.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final photometric catalog.')
    args = parser.parse_args()

    return args

def missing_files(args, sample, size=1, clobber_overwrite=None):
    from astrometry.util.multiproc import multiproc
    from legacyhalos.io import _missing_files_one

    dependson = None
    galaxy, galaxydir = get_galaxy_galaxydir(sample, resampled=args.resampled_phot)        
    if args.coadds:
        suffix = 'coadds'
        filesuffix = '-custom-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        filesuffix = '-custom-ellipse.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.resampled_phot:
        if args.htmlplots:
            suffix = 'resampled-htmlplots'
            filesuffix = '-resampled-htmlplots.isdone'
        elif args.build_catalog:
            suffix = 'resampled-build-catalog'
            filesuffix = None # '-resampled-ellipse.isdone'        
        else:
            suffix = 'resampled-ellipse'
            filesuffix = '-resampled-ellipse.isdone'        
    elif args.build_catalog:
        suffix = 'build-catalog'
        filesuffix = None # '-custom-ellipse.isdone'        
        #dependson = '-custom-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-custom-montage-grz.png'
        else:
            filesuffix = '-custom-montage-grz.png'
            #filesuffix = '-ccdpos.png'
            #filesuffix = '-custom-maskbits.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-custom-montage-grz.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        print('Nothing to do.')
        return

    # Make clobber=False for build_SGA and htmlindex because we're not making
    # the files here, we're just looking for them. The argument args.clobber
    # gets used downstream.
    if args.htmlindex or args.build_catalog:
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

    if filesuffix is None:
        todo = np.repeat('done', ngal)
    else:
        mp = multiproc(nthreads=args.nproc)
        missargs = []
        for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)):
            #missargs.append([gal, gdir, filesuffix, dependson, clobber])
            checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
            if dependson:
                missargs.append([checkfile, os.path.join(gdir, '{}{}'.format(gal, dependson)), clobber])
            else:
                missargs.append([checkfile, None, clobber])
            
        todo = np.array(mp.map(_missing_files_one, missargs))

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
        todo_indices = np.array_split(_todo_indices, size) # unweighted

        ## Assign the sample to ranks to make the D25 distribution per rank ~flat.
        #
        ## https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        #weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        #cumuweight = weight.cumsum() / weight.sum()
        #idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        #if len(idx) < size: # can happen in corner cases
        #    todo_indices = np.array_split(_todo_indices, size) # unweighted
        #else:
        #    todo_indices = np.array_split(_todo_indices, idx) # weighted
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices
    
def get_raslice(ra):
    return '{:06d}'.format(int(ra*1000))[:3]

def get_plate(plate):
    return '{:05d}'.format(plate)

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False, resampled=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat[GALAXYCOLUMN]]
        plate = [cat['PLATE']]
    else:
        ngal = len(cat)
        galaxy = cat[GALAXYCOLUMN]
        plate = cat['PLATE']

    if resampled:
        # need to fix the plate!
        galaxydir = np.array([os.path.join(datadir, 'resampled', str(plt), gal) for gal, plt in zip(galaxy, plate)])
        #galaxydir = np.array([os.path.join(datadir, 'resampled', get_plate(plt), gal) for gal, plt in zip(galaxy, plate)])
    else:
        galaxydir = np.array([os.path.join(datadir, str(plt), gal) for gal, plt in zip(galaxy, plate)])
        #galaxydir = np.array([os.path.join(datadir, get_plate(plt), gal) for gal, plt in zip(galaxy, plate)])
        
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, str(plt), gal) for gal, plt in zip(galaxy, plate)])
        #htmlgalaxydir = np.array([os.path.join(htmldir, get_plate(plt), gal) for gal, plt in zip(galaxy, plate)])

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
                use_testbed=True, ellipse=False, resampled_phot=False, htmlplots=False,
                fullsample=False):
    """Read/generate the parent SGA catalog.

    """
    import fitsio
    from legacyhalos.desiutil import brickname as get_brickname

    if use_testbed:
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-testbed100.fits')
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-testbed.fits')
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'mn-dr17-v0.1sub-summary.fits')
        samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-0.2.0.testbed.fits')
    else:
        samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-0.3.0.fits')
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'mn-0.2.0.testbed-summary.fits')
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-dr17-v0.1.fits')
        #samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-v2_4_3.fits')

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()
    rows = np.arange(nrows)

    if not use_testbed and False: # the drpall-dr17-v0.1.fits file already has the selection we want
        # See here to select unique Manga galaxies--
        # https://www.sdss.org/dr16/manga/manga-tutorials/drpall/#py-uniq-gals
        tbdata = fitsio.read(samplefile, lower=True, columns=['mngtarg1', 'mngtarg3', 'mangaid'])

        keep = np.where(
            np.logical_and(
                np.logical_or((tbdata['mngtarg1'] != 0), (tbdata['mngtarg3'] != 0)),
                ((tbdata['mngtarg3'] & 1<<19) == 0) * ((tbdata['mngtarg3'] & 1<<20) == 0) * ((tbdata['mngtarg3'] & 1<<21) == 0)
                ))[0]
        rows = rows[keep]

        _, uindx = np.unique(tbdata['mangaid'][rows], return_index=True)
        rows = rows[uindx]

        ## Find galaxies excluding those from the Coma, IC342, M31 ancillary programs (bits 19,20,21)
        #cube_bools = (tbdata['mngtarg1'] != 0) | (tbdata['mngtarg3'] != 0)
        #cubes = tbdata[cube_bools]
        #
        #targ3 = tbdata['mngtarg3']
        #galaxies = tbdata[cube_bools & ((targ3 & 1<<19) == 0) & ((targ3 & 1<<20) == 0) & ((targ3 & 1<<21) == 0)]
        #
        #uniq_vals, uniq_idx = np.unique(galaxies['mangaid'], return_index=True)
        #uniq_galaxies = galaxies[uniq_idx]

        #for ii in np.arange(len(rows)):
        #    print(tbdata['mangaid'][rows[ii]], uniq_galaxies['mangaid'][ii])

    nrows = len(rows)
    
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

    # a little bit of clean-up
    # 12767-1902 is the globular cluster NGC6341=M92
    # https://www.legacysurvey.org/viewer?ra=259.28079&dec=43.135944&layer=ls-dr9&zoom=13&sga&manga
    
    # * 12187-12705 is the globular cluster NGC7078=M15
    # https://www.legacysurvey.org/viewer?ra=322.49304&dec=12.167&layer=ls-dr9&zoom=13&sga&manga

    # * 11843-12705 is the globular cluster NGC6229
    # https://www.legacysurvey.org/viewer?ra=251.74496&dec=47.52775&layer=ls-dr9&zoom=13&sga&manga
    
    # * 11842-12705 is the globular cluster NGC5634
    # https://www.legacysurvey.org/viewer?ra=217.40512&dec=-5.9764167&layer=ls-dr9&zoom=13&sga&manga

    # * 11841-12705 is the globular cluster NGC4147
    # https://www.legacysurvey.org/viewer?ra=182.52625&dec=18.542639&layer=ls-dr9&zoom=13&sga&manga

    # * 9673-3703 is off the footprint
    # https://www.legacysurvey.org/viewer?ra=56.232562&dec=67.787128&layer=ls-dr9&zoom=13&sga&manga
    if ellipse or resampled_phot:
        if htmlplots:
            DOCOLUMN = 'DO_MANGA'
        else:
            DOCOLUMN = 'DO_ELLIPSE'
    else:
        DOCOLUMN = 'DO_MANGA'

    #bb = sample[np.isin(sample[REFIDCOLUMN], np.array([
    #    10506012702, 10504009102,
    #    10843012704, 11866009101,
    #    7443001901, 9872001901,
    #    9872003702, 7443003701,
    #    8983003703, 7495012704,
    #    7963003701, 8651003701,
    #    8329003701, 8333012701,
    #    8239006104, 8567012702,
    #    8588003701, 8603012701,
    #    9046006104, 9048003703]))]['PLATEIFU', RACOLUMN, DECCOLUMN, REFIDCOLUMN]
    #bb = bb[np.argsort(bb[RACOLUMN])]
    #print(bb)

    # duplicates
    #   PLATEIFU    IFURA     IFUDEC     MANGANUM
    # ----------- --------- ---------- -----------
    # 10506-12702 139.48427   34.50929 10506012702 - missing Tractor/ellipse
    #  10504-9102 139.48431  34.509323 10504009102
    # 10843-12704 149.70772 0.83665769 10843012704 - missing Tractor/ellipse
    #  11866-9101 149.70772 0.83665771 11866009101
    #   7443-1901 231.04261  42.068721  7443001901 - missing Tractor/ellipse
    #   9872-1901 231.04264  42.068723  9872001901
    #   7443-3701 230.57438   42.28702  7443003701 - missing Tractor/ellipse
    #   9872-3702 230.57439  42.287013  9872003702
    #  7495-12704  205.4384  27.004754  7495012704 - missing Tractor/ellipse
    #   8983-3703 205.43841  27.004738  8983003703
    #   7963-3701 313.57074 -1.2491752  7963003701 - missing Tractor/ellipse
    #   8651-3701 313.57074 -1.2491752  8651003701    
    #  8567-12702 118.53567   48.70984  8567012702 - missing Tractor/ellipse
    #   8239-6104  118.5407  48.708898  8239006104
    #   8329-3701 213.43219  43.662481  8329003701 - missing Tractor/ellipse
    #  8333-12701 213.43219  43.662481  8333012701
    #   8588-3701 248.14056  39.131021  8588003701 - missing Tractor/ellipse
    #  8603-12701 248.14056  39.131021  8603012701
    #   9046-6104 246.17117   25.32936  9046006104 - missing Tractor/ellipse
    #   9048-3703 246.17224  25.328235  9048003703

    print('Flagging sources outside the DR9 footprint or with missing grz coverage.')
    remgals = np.array([
        # missing z-band
        '8086-12703', 
        '8156-12703',
        '8156-1901', 
        '8156-1902',
        # no CCDs touching footprint
        '8086-6104', # tiny bit of data in the corner, but nothing usable
        '10480-12701', '10480-12702', '10480-12703', '10480-12704', '10480-12705', '10481-12701', 
        '10481-12702', '10481-12703', '10481-12704', '10481-12705', '10482-12701', '10482-12702', 
        '10482-12703', '10482-12704', '10482-12705', '10483-12701', '10483-12702', '10483-12703', 
        '10483-12704', '10483-12705', '10484-12701', '10484-12702', '10484-12703', '10484-12704', 
        '10484-12705', '10485-12701', '10485-12702', '10485-12703', '10480-1901',
        '10480-1902', '10480-3701', '10480-3702', '10480-3703', '10480-3704', '10480-6101', 
        '10480-6102', '10480-6103', '10480-6104', '10480-9101', '10480-9102', '10481-1902', 
        '10481-3701', '10481-3703', '10481-3704', '10481-6101', '10481-6102', '10481-6104', 
        '10481-9101', '10481-9102', '10482-1902', '10482-3701', '10482-3703', '10482-3704', 
        '10482-6101', '10482-6102', '10482-6104', '10482-9101', '10482-9102', '10483-1902', 
        '10483-3701', '10483-3703', '10483-3704', '10483-6101', '10483-6102', '10483-6104', 
        '10483-9101', '10483-9102', '10484-1902', '10484-3701', '10484-3703', '10484-3704',
        '10484-6101', '10484-6102', '10484-6104', '10484-9101', '10484-9102', 
        '10141-12702', '10141-12703', '10141-12704', '10141-12705', '10142-12701', '10142-12702',
        '10142-12703', '10142-12704', '10142-12705', '10143-12701', '10143-12702', '10143-12703',
        '10143-12704', '10143-12705', '10144-12701', '10144-12702', '10144-12703', '10144-12704',
        '10144-12705', '10145-12701', '10145-12702', '10145-12703', '10145-12704', '10145-12705',
        '10146-12701', '10146-12702', '10146-12703', '10146-12704', '10146-12705', '10147-12701',
        '10147-12702', '10147-12703', '10147-12704', '10147-12705', '10148-12701', '10148-12702',
        '10148-12703', '10148-12704', '10148-12705', '10149-12701', '10149-12702', '10149-12703',
        '10149-12704', '10149-12705', '10150-12701', '10150-12702', '10150-12703', '10150-12704',
        '10150-12705', '10141-1901', '10141-1902', '10141-3701', '10141-3702', '10141-3703',
        '10141-3704', '10141-6101', '10141-6102', '10141-6103', '10141-6104', '10141-9101',
        '10141-9102', '10142-1901', '10142-1902', '10142-3701', '10142-3702', '10142-3703',
        '10142-3704', '10142-6101', '10142-6102', '10142-6103', '10142-6104', '10142-9101',
        '10142-9102', '10143-1901', '10143-1902', '10143-3701', '10143-3702', '10143-3703',
        '10143-3704', '10143-6101', '10143-6102', '10143-6103', '10143-6104', '10143-9101',
        '10143-9102', '10144-1901', '10144-1902', '10144-3701', '10144-3702', '10144-3703',
        '10144-3704', '10144-6101', '10144-6102', '10144-6103', '10144-6104', '10144-9101',
        '10144-9102', '10145-1901', '10145-1902', '10145-3701', '10145-3702', '10145-3703',
        '10145-3704', '10145-6101', '10145-6102', '10145-6103', '10145-6104', '10145-9101',
        '10145-9102', '10146-1901', '10146-1902', '10146-3701', '10146-3702', '10146-3703',
        '10146-3704', '10146-6101', '10146-6102', '10146-6103', '10146-6104', '10146-9101',
        '10146-9102', '10147-1901', '10147-1902', '10147-3701', '10147-3702', '10147-3703',
        '10147-3704', '10147-6101', '10147-6102', '10147-6103', '10147-6104', '10147-9101',
        '10147-9102', '10148-1901', '10148-1902', '10148-3701', '10148-3702', '10148-3703',
        '10148-3704', '10148-6101', '10148-6102', '10148-6103', '10148-6104', '10148-9101',
        '10148-9102', '10149-1901', '10149-1902', '10149-3701', '10149-3702', '10149-3703',
        '10149-3704', '10149-6101', '10149-6102', '10149-6103', '10149-6104', '10149-9101',
        '10149-9102', '10150-1901', '10150-1902', '10150-3701', '10150-3702', '10150-3703',
        '10150-3704', '10150-6101', '10150-6102', '10150-6103', '10150-6104', '10150-9101',
        '10150-9102', '11986-6102', '11986-9102', '11987-3701', '11987-3703', '10141-12701', 
        '8086-1902', '8086-3702', '8086-3703', '8086-3704', '8086-6101', '8086-6103', 
        '8086-9101', '8086-9102', '9677-1901', '9677-1902', '9677-3701', '9677-3702', 
        '9677-3703', '9677-3704', '9677-6101', '9677-6104', '9678-1901', '9678-1902', 
        '9678-3701', '9678-3702', '9678-3703', '9678-3704', '9678-6101', '9678-6104',
        '10485-12704', '10485-12705', '10485-1901', '10485-1902', '10485-3701', '10485-3702',
        '10485-3703', '10485-3704', '10485-6101', '10485-6102', '10485-6103', '10485-6104', '10485-9101',
        '10485-9102', '10486-12701', '10486-12702', '10486-12703', '10486-12704', '10486-12705',
        '10486-1902', '10486-3701', '10486-3703', '10486-3704', '10486-6101', '10486-6102',
        '10486-6104', '10486-9101', '10486-9102', '10487-12701', '10487-12702', '10487-12703',
        '10487-12704', '10487-12705', '10487-1902', '10487-3701', '10487-3703', '10487-3704',
        '10487-6101', '10487-6102', '10487-6104', '10487-9101', '10487-9102', '10488-12701',
        '10488-12702', '10488-12703', '10488-12704', '10488-12705', '10488-1902', '10488-3701',
        '10488-3703', '10488-3704', '10488-6101', '10488-6102', '10488-6104', '10488-9101',
        '10488-9102', '10489-12701', '10489-12702', '10489-12703', '10489-12704', '10489-12705',
        '10489-1901', '10489-1902', '10489-3701', '10489-3702', '10489-3703', '10489-3704',
        '10489-6101', '10489-6102', '10489-6103', '10489-6104', '10489-9101', '10489-9102',
        '10490-12702', '10490-12703', '10490-12704', '10490-12705', '10490-1901', '10490-1902',
        '10490-3701', '10490-3702', '10490-3703', '10490-3704', '10490-6101', '10490-6102',
        '10490-6103', '10490-6104', '10490-9101', '10490-9102', '10491-12701', '10491-12702',
        '10491-12703', '10491-12704', '10491-12705', '10491-1901', '10491-1902', '10491-3701',
        '10491-3702', '10491-3703', '10491-3704', '10491-6101', '10491-6102', '10491-6103',
        '10491-6104', '10491-9101', '10491-9102', '11986-12703', '11986-12704', '11986-12705',
        '11987-12701', '11987-12702', '11987-12703', '11987-12704', '11987-12705', '12027-12701',
        '12027-12702', '12027-12703', '12027-12704', '12027-12705', '12028-12701', '12028-12702',
        '12028-12703', '12028-12704', '12028-12705', '12029-12701', '12029-12702', '12029-12703',
        '12029-12704', '12029-12705', '12030-12701', '12030-12702', '12030-12703', '12030-12704',
        '12030-12705', '12031-12701', '12031-12702', '12031-12703', '12031-12704', '12031-12705',
        '12032-12701', '12032-12702', '12032-12703', '12032-12704', '12032-12705', '12033-12701',
        '12033-12702', '12033-12703', '12033-12704', '12033-12705', '12034-12701', '12034-12702',
        '12034-12703', '12034-12704', '12034-12705', '12035-12701', '12035-12702', '12035-12703',
        '12035-12704', '12035-12705', '12036-12701', '12036-12702', '12036-12703', '12036-12704',
        '12036-12705', '12037-12701', '12037-12702', '12037-12703', '12037-12704', '12037-12705',
        '12038-12701', '12038-12702', '12038-12703', '12038-12704', '12038-12705', '12039-12701',
        '12039-12702', '12039-12703', '12039-12704', '12039-12705', '12040-12701', '12040-12702',
        '12040-12703', '12040-12704', '12040-12705', '12041-12701', '12041-12702', '12041-12703',
        '12041-12704', '12041-12705', '12042-12701', '12042-12702', '12042-12703', '12042-12704',
        '12042-12705', '12043-12701', '12043-12702', '12043-12703', '12043-12704', '12043-12705',
        '12044-12701', '12044-12702', '12044-12703', '12044-12704', '12044-12705', '12045-12701',
        '12045-12702', '12045-12703', '12045-12704', '12045-12705', '12046-12701', '12046-12702',
        '12046-12703', '12046-12704', '12046-12705', '12047-12701', '12047-12702', '12047-12703',
        '12047-12704', '12047-12705', '12048-12701', '12048-12702', '12048-12703', '12048-12704',
        '12048-12705', '12049-12701', '12049-12702', '12049-12703', '12049-12704', '12049-12705',
        '12050-12701', '12050-12702', '12050-12703', '12050-12704', '12050-12705',
        '11986-1902', '11986-3703', '11986-3704', '11986-6104', '11987-1901', '11987-1902',
        '11987-3702', '11987-3704', '11987-6101', '11987-6102', '11987-6103', '11987-6104',
        '11987-9101', '11987-9102', '12027-1901', '12027-1902', '12027-3701', '12027-3702',
        '12027-3703', '12027-3704', '12027-6101', '12027-6102', '12027-6103', '12027-6104',
        '12027-9101', '12027-9102', '12028-1902', '12028-3701', '12028-3703', '12028-3704',
        '12028-6101', '12028-6102', '12028-6104', '12028-9101', '12028-9102', '12029-1901',
        '12029-1902', '12029-3701', '12029-3702', '12029-3703', '12029-3704', '12029-6101',
        '12029-6102', '12029-6103', '12029-6104', '12029-9101', '12029-9102', '12030-1901',
        '12030-1902', '12030-3701', '12030-3702', '12030-3703', '12030-3704', '12030-6101',
        '12030-6102', '12030-6103', '12030-6104', '12030-9101', '12030-9102', '12031-1901',
        '12031-1902', '12031-3701', '12031-3702', '12031-3703', '12031-3704', '12031-6101',
        '12031-6102', '12031-6103', '12031-6104', '12031-9101', '12031-9102', '12032-1901',
        '12032-1902', '12032-3701', '12032-3702', '12032-3703', '12032-3704', '12032-6101',
        '12032-6102', '12032-6103', '12032-6104', '12032-9101', '12032-9102', '12033-1901',
        '12033-1902', '12033-3701', '12033-3702', '12033-3703', '12033-3704', '12033-6101',
        '12033-6102', '12033-6103', '12033-6104', '12033-9101', '12033-9102', '12034-1901',
        '12034-1902', '12034-3701', '12034-3702', '12034-3703', '12034-3704', '12034-6101',
        '12034-6102', '12034-6103', '12034-6104', '12034-9101', '12034-9102', '12035-1901',
        '12035-1902', '12035-3701', '12035-3702', '12035-3703', '12035-3704', '12035-6101',
        '12035-6102', '12035-6103', '12035-6104', '12035-9101', '12035-9102', '12036-1901',
        '12036-1902', '12036-3701', '12036-3702', '12036-3703', '12036-3704', '12036-6101',
        '12036-6102', '12036-6103', '12036-6104', '12036-9101', '12036-9102', '12037-1901',
        '12037-1902', '12037-3701', '12037-3702', '12037-3703', '12037-3704', '12037-6101',
        '12037-6102', '12037-6103', '12037-6104', '12037-9101', '12037-9102', '12038-1901',
        '12038-1902', '12038-3701', '12038-3702', '12038-3703', '12038-3704', '12038-6101',
        '12038-6102', '12038-6103', '12038-6104', '12038-9101', '12038-9102', '12039-1901',
        '12039-1902', '12039-3701', '12039-3702', '12039-3703', '12039-3704', '12039-6101',
        '12039-6102', '12039-6103', '12039-6104', '12039-9101', '12039-9102', '12040-1901',
        '12040-1902', '12040-3701', '12040-3702', '12040-3703', '12040-3704', '12040-6101',
        '12040-6102', '12040-6103', '12040-6104', '12040-9101', '12040-9102', '12041-1901',
        '12041-1902', '12041-3701', '12041-3702', '12041-3703', '12041-3704', '12041-6101',
        '12041-6102', '12041-6103', '12041-6104', '12041-9101', '12041-9102', '12042-1901',
        '12042-1902', '12042-3701', '12042-3702', '12042-3703', '12042-3704', '12042-6101',
        '12042-6102', '12042-6103', '12042-6104', '12042-9101', '12042-9102', '12043-1901',
        '12043-1902', '12043-3701', '12043-3702', '12043-3703', '12043-3704', '12043-6101',
        '12043-6102', '12043-6103', '12043-6104', '12043-9101', '12043-9102', '12044-1901',
        '12044-1902', '12044-3701', '12044-3702', '12044-3703', '12044-3704', '12044-6101',
        '12044-6102', '12044-6103', '12044-6104', '12044-9101', '12044-9102', '12045-1901',
        '12045-1902', '12045-3701', '12045-3702', '12045-3703', '12045-3704', '12045-6101',
        '12045-6102', '12045-6103', '12045-6104', '12045-9101', '12045-9102', '12046-1901',
        '12046-1902', '12046-3701', '12046-3702', '12046-3703', '12046-3704', '12046-6101',
        '12046-6102', '12046-6103', '12046-6104', '12046-9101', '12046-9102', '12047-1901',
        '12047-1902', '12047-3701', '12047-3702', '12047-3703', '12047-3704', '12047-6101',
        '12047-6102', '12047-6103', '12047-6104', '12047-9101', '12047-9102', '12048-1901',
        '12048-1902', '12048-3701', '12048-3702', '12048-3703', '12048-3704', '12048-6101',
        '12048-6102', '12048-6103', '12048-6104', '12048-9101', '12048-9102', '12049-1901',
        '12049-1902', '12049-3701', '12049-3702', '12049-3703', '12049-3704', '12049-6101',
        '12049-6102', '12049-6103', '12049-6104', '12049-9101', '12049-9102', '12050-1901',
        '12050-1902', '12050-3701', '12050-3702', '12050-3703', '12050-3704', '12050-6101',
        '12050-6102', '12050-6103', '12050-6104', '12050-9101', '12050-9102', '9673-12701',
        '9673-12702', '9673-12703', '9673-12704', '9673-12705', '9674-12701', '9674-12702',
        '9674-12703', '9674-12704', '9674-12705', '9675-12701', '9675-12702', '9675-12703',
        '9675-12704', '9675-12705', '9677-12701', '9677-12702', '9677-12703', '9677-12704',
        '9677-12705', '9678-12701', '9678-12702', '9678-12703', '9678-12704', '9678-12705',
        '9673-1901', '9673-1902', '9673-3701', '9673-3702', '9673-3703', '9673-3704',
        '9673-6101', '9673-6102', '9673-6103', '9673-6104', '9673-9101', '9673-9102',
        '9674-1901', '9674-1902', '9674-3701', '9674-3702', '9674-3703', '9674-3704',
        '9674-6101', '9674-6102', '9674-6104', '9674-9101', '9674-9102', '9675-1901',
        '9675-1902', '9675-3701', '9675-3702', '9675-3703', '9675-3704', '9675-6101',
        '9675-6102', '9675-6103', '9675-6104', '9675-9101', '9675-9102', '9677-6102',
        '9677-6103', '9677-9101', '9677-9102', '9678-6102', '9678-6103', '9678-9101', '9678-9102'])  
    sample['DO_IMAGING'] = np.ones(len(sample), bool)
    sample['DO_IMAGING'][np.isin(sample[GALAXYCOLUMN], remgals)] = False

    if not fullsample:
        print('Cleaning up the sample; keeping objects with {}==True and DO_IMAGING==True.'.format(DOCOLUMN))
        rem = sample[DOCOLUMN] * sample['DO_IMAGING']
        if np.sum(rem) > 0:
            sample = sample[rem]

    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        these = np.isin(sample[GALAXYCOLUMN], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            sample = sample[these]

    # add some geometric columns we can use for the ellipse-fitting
    ngal = len(sample)
    sample['BA_INIT'] = np.repeat(1.0, ngal).astype('f4') # fixed b/a (circular)
    sample['PA_INIT'] = np.repeat(0.0, ngal).astype('f4') # fixed position angle
    sample['DIAM_INIT'] = np.repeat(2 * MANGA_RADIUS / 60.0, ngal).astype('f4') # fixed diameter [arcmin]
    
    igood = np.where((sample['NSA_NSAID'] != -9999) * (sample['NSA_NSAID'] > 0))[0]
    if len(igood) > 0:
        sample['BA_INIT'][igood] = sample['NSA_SERSIC_BA'][igood]
        sample['PA_INIT'][igood] = sample['NSA_SERSIC_PHI'][igood]
        sample['DIAM_INIT'][igood] = 2 * 2 * sample['NSA_SERSIC_TH50'][igood] / 60 # [2*half-light radius, arcmin]

    sample[REFIDCOLUMN] = sample['PLATE'] * 1000000 + np.int32(sample['IFUDSGN'])

    ## add the dust
    #from legacyhalos.dust import SFDMap, mwdust_transmission
    #ebv = SFDMap().ebv(sample[RACOLUMN], sample[DECCOLUMN])
    #for band in ['fuv', 'nuv', 'g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:
    #    sample['MW_TRANSMISSION_{}'.format(band.upper())] = mwdust_transmission(ebv, band, 'N', match_legacy_surveys=True).astype('f4')

    return sample

def build_catalog(sample, nproc=1, refcat='R1', resampled=False, verbose=False, clobber=False):
    import time
    import fitsio
    from astropy.io import fits
    from astropy.table import Table, vstack
    from legacyhalos.io import read_ellipsefit

    version = '0.3.0' # '0.2.0.testbed' # 'v1.0'

    ngal = len(sample)

    if resampled:
        outfile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'manga-legacyphot-{}.fits'.format(version))
    else:
        outfile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'manga-legacyphot-native-{}.fits'.format(version))
        
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return None

    galaxy, galaxydir = get_galaxy_galaxydir(sample)
    if resampled:
        _, resamp_galaxydir = get_galaxy_galaxydir(sample, resampled=resampled)
    else:
        resamp_galaxydir = [None] * len(galaxy)

    t0 = time.time()
    tractor, parent, ellipse, phot = [], [], [], []
    for gal, gdir, rdir, onegal in zip(galaxy, galaxydir, resamp_galaxydir, sample):
        refid = onegal[REFIDCOLUMN]
        tractorfile = os.path.join(gdir, '{}-custom-tractor.fits'.format(gal))
        ellipsefile = os.path.join(gdir, '{}-custom-ellipse-{}.fits'.format(gal, refid))
        if not os.path.isfile(tractorfile) and onegal['DO_MANGA'] and onegal['DO_IMAGING']:
            print('Missing Tractor catalog {}'.format(tractorfile))
        if not os.path.isfile(ellipsefile) and onegal['DO_ELLIPSE'] and onegal['DO_IMAGING']:
            print('Missing ellipse file {}'.format(ellipsefile))
        
        if os.path.isfile(tractorfile) and os.path.isfile(ellipsefile):
            _ellipse = read_ellipsefit(gal, gdir, galaxy_id=str(refid), asTable=True,
                                       filesuffix='custom', verbose=True)
            for col in _ellipse.colnames:
                if _ellipse[col].ndim > 1:
                    _ellipse.remove_column(col)

            _tractor = Table(fitsio.read(tractorfile, upper=True))
            match = np.where((_tractor['REF_CAT'] == refcat) * (_tractor['REF_ID'] == refid))[0]
            if len(match) != 1:
                raise ValueError('Problem here!')

            if rdir is None:
                ellipse.append(_ellipse)
                tractor.append(_tractor[match])
                parent.append(onegal)
            else:
                resampfile = os.path.join(rdir, '{}-resampled-ellipse-{}.fits'.format(gal, refid))
                if os.path.isfile(resampfile):
                    _phot = read_ellipsefit(gal, rdir, galaxy_id=str(refid), asTable=True,
                                            filesuffix='resampled', verbose=True)
                    for col in _phot.colnames:
                        if _phot[col].ndim > 1:
                            _phot.remove_column(col)
        
                    ellipse.append(_ellipse)
                    tractor.append(_tractor[match])
                    parent.append(onegal)
                    phot.append(_phot)
                else:
                    print('Missing resampled photometry {}'.format(resampfile))

    if len(tractor) == 0:
        print('Need at least one fitted object to create a merged catalog.')
        return

    assert(len(ellipse) == len(parent))
    assert(len(tractor) == len(parent))
    ellipse = vstack(ellipse, metadata_conflicts='silent')
    tractor = vstack(tractor, metadata_conflicts='silent')
    parent = vstack(parent, metadata_conflicts='silent')
    if resampled:
        assert(len(phot) == len(parent))
        phot = vstack(phot, metadata_conflicts='silent')
    print('Merging {} galaxies took {:.2f} min.'.format(len(tractor), (time.time()-t0)/60.0))
    assert(len(tractor) == len(parent))

    # row-match the merged catalogs to the input sample catalog
    I = np.isin(sample[REFIDCOLUMN], parent[REFIDCOLUMN])
    assert(np.all(sample[I] == parent))
    
    out_ellipse = Table()
    for col in ellipse.colnames:
        out_ellipse[col] = np.zeros(ngal, dtype=ellipse[col].dtype)
    out_ellipse[I] = ellipse

    out_tractor = Table()
    for col in tractor.colnames:
        if len(tractor[col].shape) > 1:
            out_tractor[col] = np.zeros((ngal, tractor[col].shape[1]), dtype=tractor[col].dtype)
        else:
            out_tractor[col] = np.zeros(ngal, dtype=tractor[col].dtype)
    out_tractor[I] = tractor

    if resampled:
        out_phot = Table()
        for col in phot.colnames:
            out_phot[col] = np.zeros(ngal, dtype=phot[col].dtype)
        out_phot[I] = phot

    del phot, tractor, ellipse

    # write out
    hdu_primary = fits.PrimaryHDU()
    hdu_parent = fits.convenience.table_to_hdu(sample) # parent)
    hdu_parent.header['EXTNAME'] = 'PARENT'
        
    hdu_ellipse = fits.convenience.table_to_hdu(out_ellipse)
    hdu_ellipse.header['EXTNAME'] = 'ELLIPSE'

    hdu_tractor = fits.convenience.table_to_hdu(out_tractor)
    hdu_tractor.header['EXTNAME'] = 'TRACTOR'
        
    if resampled:
        hdu_phot = fits.convenience.table_to_hdu(out_phot)
        hdu_phot.header['EXTNAME'] = 'RESAMPLED'
        hx = fits.HDUList([hdu_primary, hdu_parent, hdu_ellipse, hdu_phot, hdu_tractor])
    else:
        hx = fits.HDUList([hdu_primary, hdu_parent, hdu_ellipse, hdu_tractor])
        
    hx.writeto(outfile, overwrite=True, checksum=True)

    print('Wrote {} galaxies to {}'.format(len(sample), outfile))

def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          threshmask=0.01, r50mask=0.05, maxshift=0.0,
                          sigmamask=3.0, neighborfactor=1.0, verbose=False):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    r50mask - mask satellites whose r50 radius (arcsec) is > r50mask 

    threshmask - mask satellites whose flux ratio is > threshmmask relative to
    the central galaxy.

    """
    import numpy.ma as ma
    from copy import copy
    from skimage.transform import resize
    from legacyhalos.mge import find_galaxy
    from legacyhalos.misc import srcs2image, ellipse_mask

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    bands, refband = data['bands'], data['refband']
    #residual_mask = data['residual_mask']

    #nbox = 5
    #box = np.arange(nbox)-nbox // 2
    #box = np.meshgrid(np.arange(nbox), np.arange(nbox))[0]-nbox//2

    xobj, yobj = np.ogrid[0:data['refband_height'], 0:data['refband_width']]
    dims = data[refband].shape
    assert(dims[0] == dims[1])

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if 'galaxy_indx' in data.keys():
        galaxy_indx = np.atleast_1d(data['galaxy_indx'])
    else:
        galaxy_indx = np.array([np.argmin((tractor.bx - data['refband_height']/2)**2 +
                                          (tractor.by - data['refband_width']/2)**2)])
        data['galaxy_indx'] = np.atleast_1d(galaxy_indx)
        data['galaxy_id'] = ''

    #print('Import hack!')
    #norm = simple_norm(img, 'log', min_percent=0.05, clip=True)
    #import matplotlib.pyplot as plt ; from astropy.visualization import simple_norm

    ## Get the PSF sources.
    #psfindx = np.where(tractor.type == 'PSF')[0]
    #if len(psfindx) > 0:
    #    psfsrcs = tractor.copy()
    #    psfsrcs.cut(psfindx)
    #else:
    #    psfsrcs = None

    def tractor2mge(indx, factor=1.0):
    #def tractor2mge(indx, majoraxis=None):
        # Convert a Tractor catalog entry to an MGE object.
        class MGEgalaxy(object):
            pass

        if tractor.type[indx] == 'PSF' or tractor.shape_r[indx] < 5:
            pa = tractor.pa_init[indx]
            ba = tractor.ba_init[indx]
            # take away the extra factor of 2 we put in in read_sample()
            r50 = tractor.diam_init[indx] * 60 / 2 / 2 # [arcsec]
            if r50 < 5:
                r50 = 5.0 # minimum size, arcsec
            majoraxis = factor * r50 / filt2pixscale[refband] # [pixels]
        else:
            ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2))
            pa = pa % 180
            #majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]

            # can be zero (or very small) if fit as a PSF or REX
            if tractor.shape_r[indx] > 1:
                majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]
            else:
                majoraxis = factor * tractor.diam_init[indx] * 60 / 2 / 2 / filt2pixscale[refband] # [pixels]

        mgegalaxy = MGEgalaxy()

        # force the central pixels to be at the center of the mosaic because all
        # MaNGA sources were visually inspected and we want to have consistency
        # between the center used for the IFU and the center used for photometry.
        mgegalaxy.xmed = dims[0] / 2
        mgegalaxy.ymed = dims[0] / 2
        mgegalaxy.xpeak = dims[0] / 2
        mgegalaxy.ypeak = dims[0] / 2
        #mgegalaxy.xmed = tractor.by[indx]
        #mgegalaxy.ymed = tractor.bx[indx]
        #mgegalaxy.xpeak = tractor.by[indx]
        #mgegalaxy.ypeak = tractor.bx[indx]
        mgegalaxy.eps = 1-ba
        mgegalaxy.pa = pa
        mgegalaxy.theta = (270 - pa) % 180
        mgegalaxy.majoraxis = majoraxis

        # by default, restore all the pixels within 10% of the nominal IFU
        # footprint, assuming a circular geometry.
        default_majoraxis = 1.1 * MANGA_RADIUS / 2 / filt2pixscale[refband] # [pixels]
        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                               default_majoraxis, default_majoraxis, 0.0, xobj, yobj)
        #objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
        #                       mgegalaxy.majoraxis,
        #                       mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
        #                       np.radians(mgegalaxy.theta-90), xobj, yobj)
    
        return mgegalaxy, objmask

    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        print('Determing the geometry for galaxy {}/{}.'.format(
                ii+1, len(galaxy_indx)))

        #if tractor.ref_cat[galaxy_indx] == 'R1' and tractor.ref_id[galaxy_indx] == 8587006103:
        #    neighborfactor = 1.0

        # [1] Determine the non-parametricc geometry of the galaxy of interest
        # in the reference band. First, subtract all models except the galaxy
        # and galaxies "near" it. Also restore the original pixels of the
        # central in case there was a poor deblend.
        largeshift = False
        mge, centralmask = tractor2mge(central, factor=1.0)
        #plt.clf() ; plt.imshow(centralmask, origin='lower') ; plt.savefig('junk-mask.png') ; pdb.set_trace()

        iclose = np.where([centralmask[np.int(by), np.int(bx)]
                           for by, bx in zip(tractor.by, tractor.bx)])[0]
        
        srcs = tractor.copy()
        srcs.cut(np.delete(np.arange(len(tractor)), iclose))
        model = srcs2image(srcs, data['{}_wcs'.format(refband.lower())],
                           band=refband.lower(),
                           pixelized_psf=data['{}_psf'.format(refband.lower())])

        img = data[refband].data - model
        img[centralmask] = data[refband].data[centralmask]

        # the "residual mask" is initialized in legacyhalos.io._read_image_data
        # and it includes pixels which are significant residuals (data minus
        # model), pixels with invvar==0, and pixels belonging to maskbits
        # BRIGHT, MEDIUM, CLUSTER, or ALLMASK_[GRZ]
        
        mask = np.logical_or(ma.getmask(data[refband]), data['residual_mask'])
        #mask = np.logical_or(data[refband].mask, data['residual_mask'])
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)

        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('desi-users/ioannis/tmp/debug.png')

        # force the center
        mgegalaxy.xmed = dims[0] / 2
        mgegalaxy.ymed = dims[0] / 2
        mgegalaxy.xpeak = dims[0] / 2
        mgegalaxy.ypeak = dims[0] / 2
        print('Enforcing galaxy centroid to the center of the mosaic: (x,y)=({:.3f},{:.3f})'.format(
            mgegalaxy.xmed, mgegalaxy.ymed))
        
        #if True:
        #    import matplotlib.pyplot as plt
        #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/debug.png')
        ##    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #    pdb.set_trace()

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            print('Large centroid shift! (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                mgegalaxy.xmed, mgegalaxy.ymed, mge.xmed, mge.ymed))
            largeshift = True

            # For the MaNGA project only, check to make sure the Tractor
            # position isn't far from the center of the mosaic, which can happen
            # near bright stars, e.g., 8133-12705
            mgegalaxy = copy(mge)
            sz = img.shape
            if np.abs(mgegalaxy.xmed-sz[1]/2) > maxshift or np.abs(mgegalaxy.ymed-sz[0]/2) > maxshift:
                print('Large centroid shift in Tractor coordinates! (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                    mgegalaxy.xmed, mgegalaxy.ymed, sz[1]/2, sz[0]/2))
                mgegalaxy.xmed = sz[1]/2
                mgegalaxy.ymed = sz[0]/2
            
        radec_med = data['{}_wcs'.format(refband.lower())].pixelToPosition(
            mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        radec_peak = data['{}_wcs'.format(refband.lower())].pixelToPosition(
            mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {
            'largeshift': largeshift,
            'ra': tractor.ra[central], 'dec': tractor.dec[central],
            'bx': tractor.bx[central], 'by': tractor.by[central],
            #'mw_transmission_g': tractor.mw_transmission_g[central],
            #'mw_transmission_r': tractor.mw_transmission_r[central],
            #'mw_transmission_z': tractor.mw_transmission_z[central],
            'ra_moment': radec_med[0], 'dec_moment': radec_med[1],
            #'ra_peak': radec_med[0], 'dec_peak': radec_med[1]
            }

        # add the dust
        from legacyhalos.dust import SFDMap, mwdust_transmission
        ebv = SFDMap().ebv(radec_peak[0], radec_peak[1])
        mge['ebv'] = np.float32(ebv)
        for band in ['fuv', 'nuv', 'g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:
            mge['mw_transmission_{}'.format(band.lower())] = mwdust_transmission(ebv, band, 'N', match_legacy_surveys=True).astype('f4')
            
        for key in ('eps', 'majoraxis', 'pa', 'theta', 'xmed', 'ymed', 'xpeak', 'ypeak'):
            mge[key] = np.float32(getattr(mgegalaxy, key))
            if key == 'pa': # put into range [0-180]
                mge[key] = mge[key] % np.float32(180)
        data['mge'].append(mge)

        #if False:
        #    #plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #    plt.clf() ; mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=True, plot=True)
        #    plt.savefig('/mnt/legacyhalos-data/debug.png')

        # [2] Create the satellite mask in all the bandpasses. Use srcs here,
        # which has had the satellites nearest to the central galaxy trimmed
        # out.
        print('Building the satellite mask.')
        #srcs = tractor.copy()
        satmask = np.zeros(data[refband].shape, bool)
        for filt in bands:
            # do not let GALEX and WISE contribute to the satellite mask
            if data[filt].shape != satmask.shape:
                continue
            
            cenflux = getattr(tractor, 'flux_{}'.format(filt.lower()))[central]
            satflux = getattr(srcs, 'flux_{}'.format(filt.lower()))
            if cenflux <= 0.0:
                print('Central galaxy flux is negative! Proceed with caution...')
                #pdb.set_trace()
                #raise ValueError('Central galaxy flux is negative!')
            
            satindx = np.where(np.logical_or(
                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
                srcs.ref_cat == 'R1'))[0]
            #satindx = np.where(srcs.ref_cat == 'R1')[0]
            #if np.isin(central, satindx):
            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
            if len(satindx) == 0:
                #raise ValueError('All satellites have been dropped!')
                print('Warning! All satellites have been dropped from band {}!'.format(filt))
            else:
                satsrcs = srcs.copy()
                #satsrcs = tractor.copy()
                satsrcs.cut(satindx)
                satimg = srcs2image(satsrcs, data['{}_wcs'.format(filt.lower())],
                                    band=filt.lower(),
                                    pixelized_psf=data['{}_psf'.format(filt.lower())])
                thissatmask = satimg > sigmamask*data['{}_sigma'.format(filt.lower())]
                #if filt == 'FUV':
                #    plt.clf() ; plt.imshow(thissatmask, origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
                #    #plt.clf() ; plt.imshow(data[filt], origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
                #    pdb.set_trace()
                if satmask.shape != satimg.shape:
                    thissatmask = resize(thissatmask*1.0, satmask.shape, mode='reflect') > 0

                satmask = np.logical_or(satmask, thissatmask)
                #if True:
                #    import matplotlib.pyplot as plt
                ##    plt.clf() ; plt.imshow(np.log10(satimg), origin='lower') ; plt.savefig('debug.png')
                #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/debug.png')
                ###    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
                #    pdb.set_trace()

            #print(filt, np.sum(satmask), np.sum(thissatmask))

        #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('junk-satmask.png')
        
        # [3] Build the final image (in each filter) for ellipse-fitting. First,
        # subtract out the PSF sources. Then update the mask (but ignore the
        # residual mask). Finally convert to surface brightness.
        #for filt in ['W1']:
        for filt in bands:
            thismask = ma.getmask(data[filt])
            if satmask.shape != thismask.shape:
                _satmask = (resize(satmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
                _centralmask = (resize(centralmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
                mask = np.logical_or(thismask, _satmask)
                mask[_centralmask] = False
            else:
                mask = np.logical_or(thismask, satmask)
                mask[centralmask] = False
            #if filt == 'W1':
            #    plt.imshow(_satmask, origin='lower') ; plt.savefig('junk-satmask-{}.png'.format(filt))
            #    plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask-{}.png'.format(filt))
            #    pdb.set_trace()

            varkey = '{}_var'.format(filt.lower())
            imagekey = '{}_masked'.format(filt.lower())
            psfimgkey = '{}_psfimg'.format(filt.lower())
            thispixscale = filt2pixscale[filt]
            if imagekey not in data.keys():
                data[imagekey], data[varkey], data[psfimgkey] = [], [], []

            img = ma.getdata(data[filt]).copy()
            
            # Get the PSF sources.
            psfindx = np.where((tractor.type == 'PSF') * (getattr(tractor, 'flux_{}'.format(filt.lower())) / cenflux > threshmask))[0]
            if len(psfindx) > 0 and filt.upper() != 'W3' and filt.upper() != 'W4':
                psfsrcs = tractor.copy()
                psfsrcs.cut(psfindx)
            else:
                psfsrcs = None
            
            if psfsrcs:
                psfimg = srcs2image(psfsrcs, data['{}_wcs'.format(filt.lower())],
                                    band=filt.lower(),
                                    pixelized_psf=data['{}_psf'.format(filt.lower())])
                if False:
                    #import fitsio ; fitsio.write('junk-psf-{}.fits'.format(filt.lower()), data['{}_psf'.format(filt.lower())].img, clobber=True)
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    im = ax1.imshow(np.log10(img), origin='lower') ; fig.colorbar(im, ax=ax1)
                    im = ax2.imshow(np.log10(psfimg), origin='lower') ; fig.colorbar(im, ax=ax2)
                    im = ax3.imshow(np.log10(data['{}_psf'.format(filt.lower())].img), origin='lower') ; fig.colorbar(im, ax=ax3)
                    im = ax4.imshow(img-psfimg, origin='lower') ; fig.colorbar(im, ax=ax4)
                    plt.savefig('qa-psf-{}.png'.format(filt.lower()))
                    #if filt == 'W4':# or filt == 'r':
                    #    pdb.set_trace()
                img -= psfimg
            else:
                psfimg = np.zeros((2, 2), 'f4')

            data[psfimgkey].append(psfimg)

            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
            var = data['{}_var_'.format(filt.lower())] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #if filt == 'r':# or filt == 'r':
            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt.lower()))
            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt.lower()))
            ##    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-thismask-{}.png'.format(filt.lower()))
            #    pdb.set_trace()
                
            data[imagekey].append(img)
            data[varkey].append(var)

        #test = data['r_masked'][0]
        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #pdb.set_trace()

    # Cleanup?
    for filt in bands:
        del data[filt]
        del data['{}_var_'.format(filt.lower())]

    return data            

def read_multiband(galaxy, galaxydir, filesuffix='custom',
                   refband='r', bands=['g', 'r', 'z'], pixscale=0.262,
                   galex_pixscale=1.5, unwise_pixscale=2.75,
                   galaxy_id=None, galex=False, unwise=False,
                   redshift=None, fill_value=0.0, sky_tests=False, verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    import astropy.units as u    
    from astrometry.util.fits import fits_table
    from legacypipe.bits import MASKBITS
    from legacyhalos.io import _get_psfsize_and_depth, _read_image_data

    #galaxy_id = np.atleast_1d(galaxy_id)
    #if len(galaxy_id) > 1:
    #    raise ValueError('galaxy_id in read_multiband cannot be a >1-element vector for now!')
    #galaxy_id = galaxy_id[0]
    #assert(np.isscalar(galaxy_id))

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
                        'sample': 'sample',
                        'maskbits': '{}-maskbits'.format(filesuffix),
                        })

    if galex:
        galex_bands = ['FUV', 'NUV']
        #galex_bands = ['fuv', 'nuv'] # ['FUV', 'NUV']
        bands = bands + galex_bands
        for band in galex_bands:
            filt2imfile.update({band: {'image': '{}-image'.format(filesuffix),
                                       'model': '{}-model'.format(filesuffix),
                                       'invvar': '{}-invvar'.format(filesuffix),
                                       'psf': '{}-psf'.format(filesuffix)}})
            filt2pixscale.update({band: galex_pixscale})
        
    if unwise:
        unwise_bands = ['W1', 'W2', 'W3', 'W4']
        #unwise_bands = ['w1', 'w2', 'w3', 'w4'] # ['W1', 'W2', 'W3', 'W4']
        bands = bands + unwise_bands
        for band in unwise_bands:
            filt2imfile.update({band: {'image': '{}-image'.format(filesuffix),
                                       'model': '{}-model'.format(filesuffix),
                                       'invvar': '{}-invvar'.format(filesuffix),
                                       'psf': '{}-psf'.format(filesuffix)}})
            filt2pixscale.update({band: unwise_pixscale})

    data.update({'filt2pixscale': filt2pixscale})

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
    
    data['failed'] = False # be optimistic!
    data['missingdata'] = False
    data['filesuffix'] = filesuffix
    if missing_data:
        data['missingdata'] = True
        return data, None

    # Pack some preliminary info into the output dictionary.
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)

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
    if galex:
        cols = cols+['flux_fuv', 'flux_nuv', 'flux_ivar_fuv', 'flux_ivar_nuv']
    if unwise:
        cols = cols+['flux_w1', 'flux_w2', 'flux_w3', 'flux_w4',
                     'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
        
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

    # Are we doing sky tests? If so, build the dictionary of sky values here.

    # subsky - dictionary of additional scalar value to subtract from the imaging,
    #   per band, e.g., {'g': -0.01, 'r': 0.002, 'z': -0.0001}
    if sky_tests:
        #imfile = os.path.join(galaxydir, '{}-{}-{}.fits.fz'.format(galaxy, filt2imfile[refband]['image'], refband))
        hdr = fitsio.read_header(filt2imfile[refband]['image'], ext=1)
        nskyaps = hdr['NSKYANN'] # number of annuli

        # Add a list of dictionaries to iterate over different sky backgrounds.
        data.update({'sky': []})
        
        for isky in np.arange(nskyaps):
            subsky = {}
            subsky['skysuffix'] = '{}-skytest{:02d}'.format(filesuffix, isky)
            for band in bands:
                refskymed = hdr['{}SKYMD00'.format(band.upper())]
                skymed = hdr['{}SKYMD{:02d}'.format(band.upper(), isky)]
                subsky[band] = refskymed - skymed # *add* the new correction
            print(subsky)
            data['sky'].append(subsky)

    # Read the basic imaging data and masks.
    data = _read_image_data(data, filt2imfile, starmask=starmask,
                            filt2pixscale=filt2pixscale,
                            fill_value=fill_value, verbose=verbose)
    
    # Find the galaxies of interest.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    sample = Table(fitsio.read(samplefile))
    print('Read {} sources from {}'.format(len(sample), samplefile))

    # keep all objects
    galaxy_indx = []
    galaxy_indx = np.hstack([np.where(sid == tractor.ref_id)[0] for sid in sample[REFIDCOLUMN]])
    #if len(galaxy_indx

    #sample = sample[np.searchsorted(sample['VF_ID'], tractor.ref_id[galaxy_indx])]
    assert(np.all(sample[REFIDCOLUMN] == tractor.ref_id[galaxy_indx]))

    tractor.diam_init = np.zeros(len(tractor), dtype='f4')
    tractor.pa_init = np.zeros(len(tractor), dtype='f4')
    tractor.ba_init = np.zeros(len(tractor), dtype='f4')
    if 'DIAM_INIT' in sample.colnames and 'PA_INIT' in sample.colnames and 'BA_INIT' in sample.colnames:
        tractor.diam_init[galaxy_indx] = sample['DIAM_INIT']
        tractor.pa_init[galaxy_indx] = sample['PA_INIT']
        tractor.ba_init[galaxy_indx] = sample['BA_INIT']
 
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
    allgalaxyinfo = []
    for igal, (galaxy_id, galaxy_indx) in enumerate(zip(data['galaxy_id'], data['galaxy_indx'])):
        samp = sample[sample[REFIDCOLUMN] == galaxy_id]
        galaxyinfo = {'mangaid': (str(galaxy_id), None)}
        #for band in ['fuv', 'nuv', 'g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:
        #    galaxyinfo['mw_transmission_{}'.format(band)] = (samp['MW_TRANSMISSION_{}'.format(band.upper())][0], None)
        
        #              'galaxy': (str(np.atleast_1d(samp['GALAXY'])[0]), '')}
        #for key, unit in zip(['ra', 'dec'], [u.deg, u.deg]):
        #    galaxyinfo[key] = (np.atleast_1d(samp[key.upper()])[0], unit)
        allgalaxyinfo.append(galaxyinfo)
        
    return data, allgalaxyinfo

def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1,
                 filesuffix='custom', bands=['g', 'r', 'z'], refband='r',
                 galex_pixscale=1.5, unwise_pixscale=2.75,
                 sky_tests=False, unwise=False, galex=False, verbose=False,
                 clobber=False, debug=False, logfile=None):
    """Wrapper on legacyhalos.mpi.call_ellipse but with specific preparatory work
    and hooks for the legacyhalos project.

    """
    import astropy.table
    from copy import deepcopy
    from legacyhalos.mpi import call_ellipse as mpi_call_ellipse

    if type(onegal) == astropy.table.Table:
        onegal = onegal[0] # create a Row object

    if logfile:
        from contextlib import redirect_stdout, redirect_stderr
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                data, galaxyinfo = read_multiband(galaxy, galaxydir, bands=bands,
                                                  filesuffix=filesuffix, refband=refband,
                                                  pixscale=pixscale,
                                                  galex_pixscale=galex_pixscale, unwise_pixscale=unwise_pixscale,
                                                  unwise=unwise, galex=galex,
                                                  sky_tests=sky_tests, verbose=verbose)
    else:
        data, galaxyinfo = read_multiband(galaxy, galaxydir, bands=bands,
                                          filesuffix=filesuffix, refband=refband,
                                          pixscale=pixscale,
                                          galex_pixscale=galex_pixscale, unwise_pixscale=unwise_pixscale,
                                          unwise=unwise, galex=galex,
                                          sky_tests=sky_tests, verbose=verbose)

    maxsma = None
    #maxsma = 5 * MANGA_RADIUS # None
    delta_logsma = 3 # 3.0

    # don't pass logfile and set debug=True because we've already opened the log
    # above!
    mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                     pixscale=pixscale, nproc=nproc, 
                     bands=bands, refband=refband, sbthresh=SBTHRESH,
                     apertures=APERTURES,
                     logsma=True, delta_logsma=delta_logsma, maxsma=maxsma,
                     clobber=clobber, verbose=verbose,
                     copy_mw_transmission=True,
                     #debug=True,
                     debug=debug, logfile=logfile)

def qa_multiwavelength_sed(ellipsefit, resamp_ellipsefit=None, tractor=None,
                           png=None, verbose=True):
    """Plot up the multiwavelength SED.

    """
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import matplotlib.ticker as ticker
    from legacyhalos.qa import _sbprofile_colors
    
    bands, refband = ellipsefit['bands'], ellipsefit['refband']

    galex = 'FUV' in bands
    unwise = 'W1' in bands
    colors = _sbprofile_colors(galex=galex, unwise=unwise)
        
    if 'redshift' in ellipsefit.keys():
        redshift = ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
    else:
        redshift, smascale = None, None

    # see also Morrisey+05
    effwave_north = {
        'fuv': 1528.0, 'nuv': 2271.0,
        'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
        'g': 4815.95363513, 'r': 6437.79282937, 'z': 9229.65786449}
    effwave_south = {
        'fuv': 1528.0, 'nuv': 2271.0,
        'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
        'g': 4890.03670428, 'r': 6469.62203811, 'z': 9196.46396394}

    run = 'north' # hack
    if run == 'north':
        effwave = effwave_north
    else:
        effwave = effwave_south

    # build the arrays
    nband = len(bands)
    bandwave = np.array([effwave[filt.lower()] for filt in bands])

    _phot = {'abmag': np.zeros(nband, 'f4')-1,
             'abmagerr': np.zeros(nband, 'f4')+0.5,
             'lower': np.zeros(nband, bool)}
    phot = {'tractor': deepcopy(_phot), 'mag_tot': deepcopy(_phot), 'mag_sb25': deepcopy(_phot),
            'resamp_mag_tot': deepcopy(_phot), 'resamp_mag_sb25': deepcopy(_phot),
            'manga': deepcopy(_phot)}

    for ifilt, filt in enumerate(bands):
        # original photometry
        mtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        if mtot > 0:
            phot['mag_tot']['abmag'][ifilt] = mtot
            phot['mag_tot']['abmagerr'][ifilt] = 0.1 # hack!!
            phot['mag_tot']['lower'][ifilt] = False

        flux = ellipsefit['flux_sb25_{}'.format(filt.lower())]
        ivar = ellipsefit['flux_ivar_sb25_{}'.format(filt.lower())]
        if flux > 0 and ivar > 0:
            mag = 22.5 - 2.5 * np.log10(flux)
            ferr = 1.0 / np.sqrt(ivar)
            magerr = 2.5 * ferr / flux / np.log(10)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = magerr
            phot['mag_sb25']['lower'][ifilt] = False
        if flux <=0 and ivar > 0:
            ferr = 1.0 / np.sqrt(ivar)
            mag = 22.5 - 2.5 * np.log10(ferr)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = 0.75
            phot['mag_sb25']['lower'][ifilt] = True

        # resampled photometry
        if resamp_ellipsefit:
            mtot = resamp_ellipsefit['cog_mtot_{}'.format(filt.lower())]
            if mtot > 0:
                phot['resamp_mag_tot']['abmag'][ifilt] = mtot
                phot['resamp_mag_tot']['abmagerr'][ifilt] = 0.1 # hack!!
                phot['resamp_mag_tot']['lower'][ifilt] = False
    
            flux = resamp_ellipsefit['flux_sb25_{}'.format(filt.lower())]
            ivar = resamp_ellipsefit['flux_ivar_sb25_{}'.format(filt.lower())]
            if flux > 0 and ivar > 0:
                mag = 22.5 - 2.5 * np.log10(flux)
                ferr = 1.0 / np.sqrt(ivar)
                magerr = 2.5 * ferr / flux / np.log(10)
                phot['resamp_mag_sb25']['abmag'][ifilt] = mag
                phot['resamp_mag_sb25']['abmagerr'][ifilt] = magerr
                phot['resamp_mag_sb25']['lower'][ifilt] = False
            if flux <=0 and ivar > 0:
                ferr = 1.0 / np.sqrt(ivar)
                mag = 22.5 - 2.5 * np.log10(ferr)
                phot['resamp_mag_sb25']['abmag'][ifilt] = mag
                phot['resamp_mag_sb25']['abmagerr'][ifilt] = 0.75
                phot['resamp_mag_sb25']['lower'][ifilt] = True

            flux = resamp_ellipsefit['flux_apmanga_{}'.format(filt.lower())]
            ivar = resamp_ellipsefit['flux_ivar_apmanga_{}'.format(filt.lower())]
            if flux > 0 and ivar > 0:
                mag = 22.5 - 2.5 * np.log10(flux)
                ferr = 1.0 / np.sqrt(ivar)
                magerr = 2.5 * ferr / flux / np.log(10)
                phot['manga']['abmag'][ifilt] = mag
                phot['manga']['abmagerr'][ifilt] = magerr
                phot['manga']['lower'][ifilt] = False
            if flux <=0 and ivar > 0:
                ferr = 1.0 / np.sqrt(ivar)
                mag = 22.5 - 2.5 * np.log10(ferr)
                phot['manga']['abmag'][ifilt] = mag
                phot['manga']['abmagerr'][ifilt] = 0.75
                phot['manga']['lower'][ifilt] = True

        if tractor is not None:
            flux = tractor['flux_{}'.format(filt.lower())]
            ivar = tractor['flux_ivar_{}'.format(filt.lower())]
            if flux > 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(flux)
                phot['tractor']['abmagerr'][ifilt] = 0.1
            if flux <= 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(1/np.sqrt(ivar))
                phot['tractor']['abmagerr'][ifilt] = 0.75
                phot['tractor']['lower'][ifilt] = True

    def _addphot(thisphot, color, marker, alpha, label):
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        lolims=True, linestyle='none', alpha=alpha)#, lolims=True)
                        
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        label=label, linestyle='none', alpha=alpha)
    
    # make the plot
    fig, ax = plt.subplots(figsize=(9, 7))

    # get the plot limits
    good = np.where(phot['mag_tot']['abmag'] > 0)[0]
    ymax = np.min(phot['mag_tot']['abmag'][good])
    ymin = np.max(phot['mag_tot']['abmag'][good])

    good = np.where(phot['tractor']['abmag'] > 0)[0]
    if np.min(phot['tractor']['abmag'][good]) < ymax:
        ymax = np.min(phot['tractor']['abmag'][good])
    if np.max(phot['tractor']['abmag']) > ymin:
        ymin = np.max(phot['tractor']['abmag'][good])
    #print(ymin, ymax)

    ymin += 1.5
    ymax -= 1.5

    wavemin, wavemax = 0.1, 30

    # have to set the limits before plotting since the axes are reversed
    if np.abs(ymax-ymin) > 15:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim(ymin, ymax)
    
    _addphot(phot['mag_tot'], color='red', marker='s', alpha=1.0, label=r'$m_{\mathrm{tot}}$')
    if resamp_ellipsefit:    
        _addphot(phot['resamp_mag_tot'], color='green', marker='o', alpha=0.5, label=r'$m_{\mathrm{tot}}^{\prime}$')        
    _addphot(phot['mag_sb25'], color='orange', marker='^', alpha=0.9, label=r'$m(r<R_{25})$')
    if resamp_ellipsefit:    
        _addphot(phot['resamp_mag_sb25'], color='purple', marker='s', alpha=0.5, label=r'$m^{\prime}(r<R_{25})$')
    _addphot(phot['manga'], color='k', marker='*', alpha=0.75, label='MaNGA Hex')
    _addphot(phot['tractor'], color='blue', marker='o', alpha=0.75, label='Tractor')

    ax.set_xlabel(r'Observed-frame Wavelength ($\mu$m)') 
    ax.set_ylabel(r'Apparent Brightness (AB mag)') 
    ax.set_xlim(wavemin, wavemax)
    ax.set_xscale('log')
    ax.legend(loc='lower right')

    def _frmt(value, _):
        if value < 1:
            return '{:.1f}'.format(value)
        else:
            return '{:.0f}'.format(value)

    ax.set_xticks([0.1, 0.2, 0.4, 1.0, 3.0, 5.0, 10, 20])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_frmt))

    if smascale:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.85, right=0.95)
    else:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.95, right=0.95)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir, 
                                filesuffix='resampled', 
                                barlen=None, barlabel=None, just_coadds=False,
                                clobber=False, verbose=False):
    """Montage the GALEX and WISE coadds into a nice QAplot.

    barlen - pixels

    """
    import subprocess
    import shutil
    from legacyhalos.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    bandsuffix = 'multiwavelength'
    
    montagefile = os.path.join(htmlgalaxydir, '{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
    thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
    if not os.path.isfile(montagefile) or clobber:
        # make a copy of the QA so we can add a scale bar
        pngfiles = []
        for suffix in ('galex', 'dlis', 'wise'):
            _pngfile = os.path.join(galaxydir, 'resampled-{}-{}.png'.format(galaxy, suffix))
            if not os.path.isfile(_pngfile):
                print('Missing {}'.format(_pngfile))
                continue
            
            pngfile = os.path.join(htmlgalaxydir, 'resampled-{}-{}.png'.format(galaxy, suffix))
            tmpfile = pngfile+'.tmp'
            shutil.copyfile(_pngfile, tmpfile)
            os.rename(tmpfile, pngfile)
            pngfiles.append(pngfile)

        if len(pngfiles) == 0:
            print('There was a problem writing {}'.format(montagefile))
        else:
            cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
            if barlen and len(pngfiles) > 0:
                barpngfile = pngfiles[0]
                addbar_to_png(pngfiles[0], barlen, barlabel, None, barpngfile,
                              scaledfont=False, pixscalefactor=1.0, fntsize=8)
                cmd = cmd+' '+barpngfile+' '
                cmd = cmd+' '.join(ff for ff in pngfiles[1:])
            else:
                cmd = cmd+' '.join(ff for ff in pngfiles)
                
            cmd = cmd+' {}'.format(montagefile)
            print(cmd)
    
            print('Writing {}'.format(montagefile))
            subprocess.call(cmd.split())
            if not os.path.isfile(montagefile):
                raise IOError('There was a problem writing {}'.format(montagefile))

        ## Create a couple smaller thumbnail images
        #with Image.open(pngfiles[0]) as im:
        #    sz = im.size
        #thumbsz = sz[0] // 3
        #
        #cmd = 'convert -thumbnail {0} {1} {2}'.format(thumbsz, montagefile, thumbfile)
        ##print(cmd)
        #if os.path.isfile(thumbfile):
        #    os.remove(thumbfile)                
        #print('Writing {}'.format(thumbfile))
        #subprocess.call(cmd.split())

def htmlplots_resampled_phot(onegal, galaxy, galaxydir, orig_galaxydir,
                             htmlgalaxydir, resampled_pixscale=0.75,
                             barlen=None, barlabel=None,
                             filesuffix='resampled',
                             verbose=False, debug=False, clobber=False):
    """Build QA from the resampled images. This script is very roughly based on
    legacyhalos.html.make_plots, html.make_ellipse_qa, 

    """
    import fitsio
    import astropy.table
    from astropy.table import Table    
    from PIL import Image
    from legacyhalos.io import read_ellipsefit
    from legacyhalos.qa import (display_multiband, display_ellipsefit,
                                display_ellipse_sbprofile, qa_curveofgrowth)

    Image.MAX_IMAGE_PIXELS = None

    if type(onegal) == astropy.table.Table:
        onegal = onegal[0] # create a Row object

    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True, mode=0o775)

    galaxy_id = onegal['MANGANUM']
    galid = str(galaxy_id)

    # Read the original and resampled ellipse-fitting results.
    ellipsefit = read_ellipsefit(galaxy, orig_galaxydir, filesuffix='custom', galaxy_id=galid)
    if not bool(ellipsefit):
        return 1 # missing
        
    resamp_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix, galaxy_id=galid)

    # optionally read the Tractor catalog
    tractorfile = os.path.join(orig_galaxydir, '{}-custom-tractor.fits'.format(galaxy)) 
    if os.path.isfile(tractorfile):
        tractor = Table(fitsio.read(tractorfile, lower=True))
    else:
        tractor = None

    qasuffix = 'resampled'

    # multiwavelength montage
    make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
                                filesuffix=qasuffix,
                                barlen=barlen, barlabel=barlabel,
                                clobber=clobber, verbose=verbose)

    
    # SED
    sedfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}-sed.png'.format(galaxy, qasuffix, galid))
    _tractor = tractor[(tractor['ref_cat'] != '  ')*np.isin(tractor['ref_id'], galaxy_id)] # fragile...
    if not os.path.isfile(sedfile) or clobber:
        qa_multiwavelength_sed(ellipsefit, resamp_ellipsefit=resamp_ellipsefit, tractor=_tractor,
                               png=sedfile, verbose=verbose)

    sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}-sbprofile.png'.format(galaxy, qasuffix, galid))
    if not os.path.isfile(sbprofilefile) or clobber:
        display_ellipse_sbprofile(resamp_ellipsefit, plot_radius=False, plot_sbradii=False,
                                  png=sbprofilefile, verbose=verbose, minerr=0.0)

    cogfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}-cog.png'.format(galaxy, qasuffix, galid))
    if not os.path.isfile(cogfile) or clobber:
        qa_curveofgrowth(resamp_ellipsefit, plot_sbradii=False, png=cogfile, verbose=verbose)

    #multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}-multiband.png'.format(galaxy, qasuffix, galid))
    #thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}-multiband.png'.format(galaxy, qasuffix, galid))
    #if not os.path.isfile(multibandfile) or clobber:
    #    with Image.open(os.path.join(galaxydir, '{}-{}-image-FUVNUVgrzW1W4.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
    #        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
    #                          igal=igal, barlen=barlen, barlabel=barlabel,
    #                          png=multibandfile, verbose=verbose, scaledfont=scaledfont)
    #
    #    # Create a thumbnail.
    #    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
    #    if os.path.isfile(thumbfile):
    #        os.remove(thumbfile)
    #    print('Writing {}'.format(thumbfile))
    #    subprocess.call(cmd.split())

    return 1

def call_htmlplots_resampled_phot(onegal, galaxy, galaxydir, orig_galaxydir,
                                  htmlgalaxydir, resampled_pixscale=0.75, filesuffix='resampled',
                                  barlen=None, barlabel=None, verbose=False, debug=False,
                                  clobber=False, write_donefile=True, logfile=None):
    """Wrapper script to do photometry on the resampled images.

    """
    import time
    from contextlib import redirect_stdout, redirect_stderr
    from legacyhalos.mpi import _done, _start

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = htmlplots_resampled_phot(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                       orig_galaxydir=orig_galaxydir,
                                       htmlgalaxydir=htmlgalaxydir,
                                       barlabel=barlabel, barlen=barlen, 
                                       resampled_pixscale=resampled_pixscale,
                                       verbose=verbose, debug=debug, clobber=clobber)
        if write_donefile:
            _done(galaxy, galaxydir, err, t0, 'resampled-htmlplots', filesuffix=None)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = htmlplots_resampled_phot(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                               orig_galaxydir=orig_galaxydir,
                                               htmlgalaxydir=htmlgalaxydir,
                                               barlabel=barlabel, barlen=barlen, 
                                               resampled_pixscale=resampled_pixscale,
                                               verbose=verbose, debug=debug, clobber=clobber)
                if write_donefile:
                    _done(galaxy, galaxydir, err, t0, 'resampled-htmlplots', None, log=log)

    return 1

def call_resampled_phot(onegal, galaxy, galaxydir, orig_galaxydir,
                        resampled_pixscale=0.75, refband='r', filesuffix='resampled',
                        nproc=1, fill_value=0.0, verbose=False, debug=False,
                        clobber=False, write_donefile=True, logfile=None):
    """Wrapper script to do photometry on the resampled images.

    """
    import time
    from contextlib import redirect_stdout, redirect_stderr
    from legacyhalos.mpi import _done, _start

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = resampled_phot(onegal, galaxy=galaxy, galaxydir=galaxydir,
                             orig_galaxydir=orig_galaxydir, filesuffix=filesuffix,
                             resampled_pixscale=resampled_pixscale,
                             nproc=nproc, verbose=verbose, debug=debug,
                             clobber=clobber)
        if write_donefile:
            _done(galaxy, galaxydir, err, t0, 'resampled-ellipse', None)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = resampled_phot(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                     orig_galaxydir=orig_galaxydir, filesuffix=filesuffix,
                                     resampled_pixscale=resampled_pixscale,
                                     nproc=nproc, verbose=verbose, debug=debug,
                                     clobber=clobber)
                if write_donefile:
                    _done(galaxy, galaxydir, err, t0, 'resampled-ellipse', None, log=log)

    return err

def resampled_phot(onegal, galaxy, galaxydir, orig_galaxydir,
                   resampled_pixscale=0.75, refband='r',
                   filesuffix='resampled', nproc=1,
                   fill_value=0.0, verbose=False, debug=False, clobber=False):
    """Do photometry on the resampled images.

    """
    import fitsio
    import astropy.units as u
    import astropy.table
    import numpy.ma as ma
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from legacyhalos.io import read_ellipsefit, write_ellipsefit
    from legacyhalos.ellipse import legacyhalos_ellipse

    if type(onegal) == astropy.table.Table:
        onegal = onegal[0] # create a Row object

    bands = ['FUV', 'NUV', 'g', 'r', 'z', 'W1', 'W2', 'W3', 'W4']

    galaxy_id = str(onegal['MANGANUM'])
    galaxyinfo = {'mangaid': (str(galaxy_id), None)}

    # https://www.legacysurvey.org/dr9/description/#photometry
    vega2ab = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

    # Read the original ellipse-fitting results and pull out the parameters we
    # need to run ellipse-fitting on the resampled images.
    odata = read_ellipsefit(galaxy, orig_galaxydir, filesuffix='custom', galaxy_id=galaxy_id)
    if not bool(odata):
        print('No imaging or native-resolution photometry.')
        return 1

    integrmode, sclip, nclip, fitgeometry = odata['integrmode'], odata['sclip'], odata['nclip'], odata['fitgeometry']

    data = {'bands': bands, 'refband': refband, 'refpixscale': np.float32(resampled_pixscale),
            'missingdata': False, 'failed': False,
            'galaxy_id': galaxy_id, 'filesuffix': filesuffix,
            }
    for key in ['psfsize_g', 'psfdepth_g', 'psfsize_r', 'psfdepth_r', 'psfsize_z', 'psfdepth_z']:
        data[key] = odata[key]

    # Unfortunately, we need to create an MGE dictionary, which then gets
    # unpacked in ellipse.legacyhalos_ellipse.
    mge = {}
    for key, newkey in zip(['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa', 'eps'], #'xmed', 'ymed'],
                           ['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa_moment', 'eps_moment']):#, 'y0_moment', 'x0_moment']):
        if key == 'majoraxis':
            #mge['majoraxis'] = odata['sma_moment'] / odata['refpixscale'] # [pixels]
            mge['majoraxis'] = odata['sma_moment'] / resampled_pixscale # [resampled pixels!]
        else:
            mge[key] = odata[newkey]

    mge['ebv'] = odata['ebv']
    for band in bands:
        mge['mw_transmission_{}'.format(band.lower())] = odata['mw_transmission_{}'.format(band.lower())]

    mangaphot = {}
    for filt in bands:
        fitsfile = os.path.join(galaxydir, 'resampled-{}-{}.fits'.format(galaxy, filt))
        if not os.path.isfile(fitsfile):
            print('Missing {}'.format(fitsfile))
            continue

        F = fitsio.FITS(fitsfile)

        image = F['IMAGE'].read()
        hdr = F['IMAGE'].read_header()
        invvar = F['IVAR'].read()
        mangamask = F['MASK'].read() # manga mask
        #psfimg = F['PSF'].read()
        
        #if 'IVAR' in F:
        #    image = F['IMAGE'].read()
        #    hdr = F['IMAGE'].read_header()
        #    invvar = F['IVAR'].read()
        #    #psfimg = F['PSF'].read()
        #else:
        #    image = F[0].read()
        #    hdr = F[0].read_header()
        #    invvar = F[1].read()

        if np.any(invvar < 0):
            raise ValueError('Warning! Negative pixels in the {}-band inverse variance map!'.format(filt))

        # convert WISE images from Vega nanomaggies to AB nanomaggies
        # https://www.legacysurvey.org/dr9/description/#photometry
        if filt.lower() == 'w1' or filt.lower() == 'w2' or filt.lower() == 'w3' or filt.lower() == 'w4':
            image *= 10**(-0.4*vega2ab[filt])
            invvar /= (10**(-0.4*vega2ab[filt]))**2
            
        mask = invvar <= 0 # True-->bad, False-->good
        var = np.zeros_like(invvar)
        ok = invvar > 0
        var[ok] = 1 / invvar[ok]

        # compute the total flux and variance in the manga footprint
        mangaflux = np.sum(image * mangamask) # nanomaggies
        mangavar = np.sum(var * mangamask)
        if mangavar <= 0:
            print('Warning: MaNGA variance is zero or negative in band {}!'.format(filt))
            mangaivar = 0.0
        else:
            mangaivar = 1 / mangavar

        mangaphot['flux_apmanga_{}'.format(filt.lower())] = mangaflux # nanomaggies
        mangaphot['flux_ivar_apmanga_{}'.format(filt.lower())] = mangaivar

        data['{}_masked'.format(filt.lower())] = [ma.masked_array(image / resampled_pixscale**2, mask)] # [nanomaggies/arcsec**2]
        data['{}_var'.format(filt.lower())] = [var / resampled_pixscale**4]                           # [nanomaggies**2/arcsec**4]

        ma.set_fill_value(data['{}_masked'.format(filt.lower())], fill_value)

        if filt == refband:
            WW, HH = image.shape
            data['refband_width'] = WW
            data['refband_height'] = HH

            #wcshdr = fitsio.FITSHDR()
            #wcshdr['NAXIS'] = 2
            #wcshdr['NAXIS1'] = WW
            #wcshdr['NAXIS2'] = HH
            #wcshdr['CTYPE1'] = 'RA---TAN'
            #wcshdr['CTYPE2'] = 'DEC--TAN'
            #wcshdr['CRVAL1'] = hdr['CRVAL1']
            #wcshdr['CRVAL2'] = hdr['CRVAL2']
            #wcshdr['CRPIX1'] = hdr['CRPIX1']
            #wcshdr['CRPIX2'] = hdr['CRPIX2']
            #wcshdr['CD1_2'] = 0.0
            #wcshdr['CD2_1'] = 0.0
            #wcshdr['CD1_1'] = -resampled_pixscale / 3600.0
            #wcshdr['CD2_2'] =  resampled_pixscale / 3600.0
            #wcs = WCS(wcshdr)
            wcs = WCS(hdr)

            # convert the central coordinates using the new WCS
            ymed, xmed = wcs.world_to_pixel(SkyCoord(odata['ra_moment']*u.degree, odata['dec_moment']*u.degree))
            mge['xmed'] = xmed
            mge['ymed'] = ymed
            
    data['mge'] = [mge]

    input_ellipse = {'eps': odata['eps_moment'], 'pa': odata['pa_moment']}

    maxsma = None
    logsma = True
    delta_logsma = 1.2

    err = legacyhalos_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                              bands=bands, refband=refband,
                              pixscale=resampled_pixscale, nproc=nproc,
                              input_ellipse=input_ellipse,
                              sbthresh=SBTHRESH, apertures=APERTURES, 
                              delta_logsma=delta_logsma, maxsma=maxsma, logsma=logsma,
                              copy_mw_transmission=True,
                              integrmode=integrmode, sclip=sclip, nclip=nclip, fitgeometry=fitgeometry,
                              verbose=verbose, clobber=clobber)

    # read the ellipse file back in, add more data, and write back out
    resampled_ellipse = legacyhalos.io.read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix,
                                                       galaxy_id=galaxy_id, verbose=verbose)
        

    try:
        add_datamodel_cols = []
        for key in mangaphot:
            resampled_ellipse[key] = np.float32(mangaphot[key])
            if 'ivar' in key:
                unit = 'nanomaggies-2'
            else:
                unit = 'nanomaggies'
            add_datamodel_cols.append((key, unit))
              
        write_ellipsefit(galaxy, galaxydir, resampled_ellipse,
                         galaxy_id=galaxy_id,
                         galaxyinfo=galaxyinfo,
                         refband=refband,
                         sbthresh=SBTHRESH, apertures=APERTURES, 
                         bands=bands, verbose=False,
                         copy_mw_transmission=True,
                         filesuffix=filesuffix,
                         add_datamodel_cols=add_datamodel_cols)
    except:
        print('Problem adding MaNGA aperture photometry to resampled ellipse file!')
        err = 0

    #phot = read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix,
    #                       galaxy_id=galaxy_id, verbose=verbose, asTable=True)

    return err

def _get_mags(cat, rad='10', bands=['FUV', 'NUV', 'g', 'r', 'z', 'W1', 'W2', 'W3', 'W4'],
              kpc=False, pipeline=False, cog=False, R24=False, R25=False, R26=False):
    res = []
    for band in bands:
        mag = None
        if kpc:
            iv = cat['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
            ff = cat['FLUX{}_{}'.format(rad, band.upper())][0]
        elif pipeline:
            iv = cat['flux_ivar_{}'.format(band.lower())]
            ff = cat['flux_{}'.format(band.lower())]
        elif R24:
            mag = cat['{}_mag_sb24'.format(band.lower())]
        elif R25:
            mag = cat['{}_mag_sb25'.format(band.lower())]
        elif R26:
            mag = cat['{}_mag_sb26'.format(band.lower())]
        elif cog:
            mag = cat['cog_mtot_{}'.format(band.lower())]
        else:
            print('Thar be rocks ahead!')
        if mag is not None:
            if mag > 0:
                res.append('{:.3f}'.format(mag))
            else:
                res.append('...')
        else:
            if ff > 0:
                mag = 22.5-2.5*np.log10(ff)
                if iv > 0:
                    ee = 1 / np.sqrt(iv)
                    magerr = 2.5 * ee / ff / np.log(10)
                res.append('{:.3f}'.format(mag))
                #res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
            elif ff < 0 and iv > 0:
                # upper limit
                mag = 22.5-2.5*np.log10(1/np.sqrt(iv))
                res.append('>{:.3f}'.format(mag))
            else:
                res.append('...')
    return res

def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn='RA', deccolumn='DEC', #diamcolumn='GROUP_DIAMETER',
                   maketrends=False, fix_permissions=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    import legacyhalos.html
    
    htmlhomefile = os.path.join(htmldir, htmlhome)
    print('Building {}'.format(htmlhomefile))

    js = legacyhalos.html.html_javadate()       

    # group by RA slices
    #raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(htmlhomefile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>MaNGA-NSF</h1>\n')
        html.write('<p style="width: 75%">\n')
        html.write("""Multiwavelength analysis of the MaNGA sample.</p>\n""")
        
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        # The default is to organize the sample by RA slice, but support both options here.
        if False:
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
            #html.write('<th>Index</th>\n')
            html.write('<th>MaNGA ID</th>\n')
            html.write('<th>PLATE-IFU</th>\n')
            #html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')
            
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)
            for gal, galaxy1, htmlgalaxydir1 in zip(sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], 5 * MANGA_RADIUS / pixscale
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1, manga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['MANGAID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.5f}</td>\n'.format(gal[ZCOLUMN]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                html.write('</tr>\n')
            html.write('</table>\n')
            
        # close up shop
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(htmlhomefile, group='cosmo')

def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, resampled_galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                       racolumn, deccolumn, pixscale, nextgalaxy, prevgalaxy,
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
        samplefile = os.path.join(galaxydir1, '{}-sample.fits'.format(galaxy1))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))
                
        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                    'flux_fuv', 'flux_nuv', 'flux_ivar_fuv', 'flux_ivar_nuv', 
                    'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4',
                    'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # We just care about the galaxies in our sample
            #if prefix == 'largegalaxy':
            wt, ws = [], []
            for ii, sid in enumerate(sample[REFIDCOLUMN]):
                ww = np.where(tractor['ref_id'] == sid)[0]
                if len(ww) > 0:
                    wt.append(ww)
                    ws.append(ii)
            if len(wt) == 0:
                print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxydir1))
                tractor = None
            else:
                wt = np.hstack(wt)
                ws = np.hstack(ws)
                tractor = tractor[wt]
                sample = sample[ws]
                srt = np.argsort(tractor['flux_r'])[::-1]
                tractor = tractor[srt]
                sample = sample[srt]
                assert(np.all(tractor['ref_id'] == sample[REFIDCOLUMN]))

        return nccds, tractor, sample

    def _html_galaxy_properties(html, gal):
        """Build the table of group properties.

        """
        galaxy1, ra1, dec1, diam1 = gal[GALAXYCOLUMN], gal[racolumn], gal[deccolumn], 5 * MANGA_RADIUS / pixscale
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1, manga=True)

        html.write('<h2>Galaxy Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Index</th>\n')
        html.write('<th>MaNGA ID</th>\n')
        html.write('<th>PLATE-IFU</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
        #html.write('<td>{:g}</td>\n'.format(ii))
        #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
        #html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal['MANGAID']))
        html.write('<td>{}</td>\n'.format(galaxy1))
        html.write('<td>{:.7f}</td>\n'.format(ra1))
        html.write('<td>{:.7f}</td>\n'.format(dec1))
        html.write('<td>{:.5f}</td>\n'.format(gal[ZCOLUMN]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
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

        html.write('<p>Color mosaics showing (from left to right) the data, Tractor model, and residuals and (from top to bottom), GALEX, <i>grz</i>, and unWISE.</p>\n')
        html.write('<table width="90%">\n')
        for bandsuffix in ('grz', 'FUVNUV', 'W1W2'):
            pngfile, thumbfile = '{}-custom-montage-{}.png'.format(galaxy1, bandsuffix), 'thumb-{}-custom-montage-{}.png'.format(galaxy1, bandsuffix)
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
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="3">Surface Brightness<br /> Threshold Radii<br />(arcsec)</th>\n')
        html.write('<th colspan="3">Half-light Radii<br />(arcsec)</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(24)</th><th>R(25)</th><th>R(26)</th>\n')
        html.write('<th>g(50)</th><th>r(50)</th><th>z(50)</th>\n')
        html.write('</tr>\n')

        for ss, tt in zip(sample, tractor):
            ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
            pa = pa % 180

            html.write('<tr><td>{}</td>\n'.format(ss[GALAXYCOLUMN]))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['sma_moment'], ellipse['pa_moment'], ellipse['eps_moment']))

                rr = []
                if 'sma_sb24' in ellipse.keys():
                    for rad in [ellipse['sma_sb24'], ellipse['sma_sb25'], ellipse['sma_sb26']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')

                rr = []
                if 'cog_sma50_g' in ellipse.keys():
                    for rad in [ellipse['cog_sma50_g'], ellipse['cog_sma50_r'], ellipse['cog_sma50_z']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')                
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
        html.write('<table>\n')
        #html.write('<tr><th></th><th></th>\n')
        #html.write('<th colspan="3"></th>\n')
        #html.write('<th colspan="12">Curve of Growth</th>\n')
        #html.write('</tr>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="9">Tractor</th>\n')
        html.write('<th colspan="9">Curve of Growth</th>\n')
        #html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        #html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(ss[GALAXYCOLUMN]))
            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                fuv, nuv, g, r, z, w1, w2, w3, w4))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                        galaxy_id=galaxyid, verbose=False)
            if bool(ellipse) and 'cog_mtot_fuv' in ellipse.keys():
                #g, r, z = _get_mags(ellipse, R24=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R25=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R26=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)
                #try:
                #    fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)
                #except:
                #    pdb.set_trace()
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                    fuv, nuv, g, r, z, w1, w2, w3, w4))
                #g, r, z = _get_mags(ellipse, cog=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            #html.write('<h4>{}</h4>\n'.format(galaxyid))
            html.write('<h4>{}</h4>\n'.format(sample[GALAXYCOLUMN][igal]))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxy_id=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue

            html.write('<table width="90%">\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="60%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="80%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="100%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-custom-ellipse-{}-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sed.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_resampled_photometry(html, tractor, sample):
        html.write('<h2>Resampled Mosaics & Photometry</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return
            
        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            #html.write('<h4>{}</h4>\n'.format(galaxyid))
            html.write('<h4>{}</h4>\n'.format(sample[GALAXYCOLUMN][igal]))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxy_id=galaxyid, verbose=verbose)
            resampled_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, resampled_galaxydir1, filesuffix='custom',
                                                               galaxy_id=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue

            html.write('<table width="90%">\n')

            html.write('<tr>\n')
            pngfile = '{}-resampled-montage-multiwavelength.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" align="left" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-resampled-ellipse-{}-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-resampled-ellipse-{}-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-resampled-ellipse-{}-sed.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_maskbits(html):
        html.write('<h2>Masking Geometry</h2>\n')
        pngfile = '{}-custom-maskbits.png'.format(galaxy1)
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
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='custom')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>PLATE-IFU {}</h1>\n'.format(galaxy1))
        #raslice = get_raslice(gal[racolumn])
        #html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_galaxy_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        _html_resampled_photometry(html, tractor, sample)
        #_html_maskbits(html)
        #_html_ccd_diagnostics(html)

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
              refband='r', pixscale=0.262, zcolumn=ZCOLUMN, intflux=None,
              racolumn=RACOLUMN, deccolumn=DECCOLUMN, #diamcolumn='GROUP_DIAMETER',
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
    htmlhome = 'index.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, #diamcolumn=diamcolumn,
                   maketrends=maketrends, fix_permissions=fix_permissions)

    # Now build the individual pages in parallel.
    if False:
        raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
        rasorted = np.argsort(raslices)
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)
    else:
        plateifusorted = np.arange(len(sample))
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)
        _, resampled_galaxydir = get_galaxy_galaxydir(sample, resampled=True)
    
    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, resampled_galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[plateifusorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir),
        np.atleast_1d(resampled_galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, resampled_galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1
