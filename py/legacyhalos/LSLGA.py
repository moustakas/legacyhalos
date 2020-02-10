"""
legacyhalos.LSLGA
=================

Code to deal with the LSLGA sample and project.

"""
import os, time, pdb
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

    parser.add_argument('--coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py.')
    #parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')
    #parser.add_argument('--LSLGA', action='store_true', help='Special code for large galaxies.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
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

def missing_files_groups(args, sample, size, htmldir=None):
    """Simple task-specific wrapper on missing_files.

    """
    if args.coadds:
        suffix = 'coadds'
    #elif args.custom_coadds:
    #    suffix = 'custom-coadds'
    #elif args.LSLGA:
    #    suffix = 'pipeline-coadds'
    elif args.htmlplots:
        suffix = 'html'
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
    #elif filetype == 'custom-coadds':
    #    filesuffix = '-custom-resid-grz.jpg'
    #elif filetype == 'LSLGA':
    #    filesuffix = '-custom-resid-grz.jpg'
    elif filetype == 'html':
        filesuffix = '-maskbits.png'
        #filesuffix = '-ccdpos.png'
        #filesuffix = '-sersic-exponential-nowavepower.png'
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

def LSLGA_version():
    version = 'v5.0'
    return version

def read_sample(first=None, last=None, galaxylist=None, verbose=False, preselect_sample=True,
                d25min=0.5, d25max=5.0):
    """Read/generate the parent LSLGA catalog.

    d25min in arcmin

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

    # Choose the parent sample here.
    if preselect_sample:
        from legacyhalos.brick import brickname as get_brickname

        sample = fitsio.read(samplefile, columns=['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        bigcut = np.where((sample['GROUP_DIAMETER'] > d25min) * (sample['GROUP_DIAMETER'] < d25max) *
                          (sample['GROUP_PRIMARY'] == 1) * (sample['IN_DESI']))[0]

        brickname = get_brickname(sample['GROUP_RA'][bigcut], sample['GROUP_DEC'][bigcut])
        nbricklist = np.loadtxt('/global/cscratch1/sd/desimpp/dr9e/image_lists/dr9e_bricks_north.txt', dtype='str')
        sbricklist = np.loadtxt('/global/cscratch1/sd/desimpp/dr9e/image_lists/dr9e_bricks_south.txt', dtype='str')
        #nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9e-north-bricklist.txt'), dtype='str')
        #sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9e-south-bricklist.txt'), dtype='str')
        bricklist = np.union1d(nbricklist, sbricklist)
        #rows = np.where([brick in bricklist for brick in brickname])[0]
        brickcut = np.where(np.isin(brickname, bricklist))[0]

        rows = np.arange(len(sample))
        rows = rows[bigcut][brickcut]
        # Add in specific galaxies for testing:
        if False:
            this = np.where(sample['GROUP_NAME'] == 'NGC4448')[0]
            rows = np.hstack((rows, this))
            rows = np.sort(rows)
        nrows = len(rows)
        print('Selecting {} galaxies in the dr9e footprint.'.format(nrows))
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

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True))
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
    sample['GALAXY'] = [gg.strip() for gg in sample['GALAXY']]
    if 'GROUP_NAME' in sample.colnames:
        galcolumn = 'GROUP_NAME'
        sample['GROUP_NAME'] = [gg.strip() for gg in sample['GROUP_NAME']]
    
    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        sample = sample[np.isin(sample[galcolumn], galaxylist)]

    #print(get_brickname(sample['GROUP_RA'], sample['GROUP_DEC']))

    # Reverse sort by diameter
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

def build_model_LSLGA_one(onegal, pixscale=0.262, minradius=5.0, minsb=25.0, sbcut=25.0,
                          bands=('g', 'r', 'z'), refcat='L5'):
    """Gather the fitting results build a single galaxy.

    minradius in arcsec
    minsb [minimum surface brightness] in r-band AB mag/arcsec**2
    sbcut [minimum surface brightness] in r-band AB mag/arcsec**2

    """
    import warnings
    import fitsio
    from astropy.table import Table, vstack
    from scipy.interpolate import interp1d
    from legacypipe.catalog import fits_reverse_typemap
    from astrometry.util.starutil_numpy import arcsec_between
    from tractor.wcs import RaDecPos
    from tractor import NanoMaggies
    from tractor.ellipses import EllipseE
    from tractor.galaxy import DevGalaxy, ExpGalaxy
    from tractor.sersic import SersicGalaxy
    from legacypipe.survey import LegacySersicIndex
        
    onegal = Table(onegal)
    galaxy, galaxydir = legacyhalos.LSLGA.get_galaxy_galaxydir(onegal)
    catfile = os.path.join(galaxydir, '{}-pipeline-tractor.fits'.format(galaxy))
    if not os.path.isfile(catfile):
        print('Skipping missing file {}'.format(catfile))
        return Table()
    cat = Table(fitsio.read(catfile))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sb = 22.5-2.5*np.log10(cat['flux_r'] / (np.pi*cat['shape_r']**2))
        #print(cat['shape_r'], sb)
        cut = np.where(np.logical_or((cat['type'] != 'REX') * (cat['type'] != 'PSF') *
                                     (cat['shape_r'] > minradius) * (sb < minsb), cat['ref_cat'] == refcat))[0]
        #cut = np.where(np.logical_or((cat['ref_cat'] != 'G2') * (cat['type'] != 'REX') *
        #                             (cat['shape_r'] > minradius), cat['ref_cat'] == refcat))[0]
    if len(cut) == 0:
        print('No large, high surface-brightness sources in galaxy {} field!'.format(galaxy))
        return Table()

    I = np.where(cat['ref_cat'][cut] == refcat)[0]
    if len(I) == 0:
        print('Large galaxy {} was dropped!'.format(galaxy))
        #print('This should not happen!')
        #pdb.set_trace()
        return Table()

    #print('Analyzing {}/{} galaxies (of which {} are LSLGA) in the {} footprint.'.format(
    #    len(cut), len(cat), len(I), galaxy))
    cat = cat[cut]

    # Include the (angular) distance of each source to the center of the
    # mosaic/group.
    cat['theta_center'] = arcsec_between(cat['ra'], cat['dec'], onegal['RA'], onegal['DEC']) # [arcsec]

    cat['d25_model'] = np.zeros(len(cat), np.float32)
    cat['pa_model'] = np.zeros(len(cat), np.float32)
    cat['ba_model'] = np.ones(len(cat), np.float32)
    cat['preburned'] = np.ones(len(cat), bool) # Everything was preburned...
    cat['freeze'] = np.zeros(len(cat), bool)   # ...but we only want to freeze the LSLGA galaxies.

    # For each galaxy, compute the position angle and "size" based
    # on the r-band surface brightness limit.
    for ii, g in zip(I, cat[I]):
        if g['type'].strip() == 'PSF':
            print('Large galaxy {} is PSF--skipping!'.format(galaxy))
            continue

        typ = fits_reverse_typemap[g['type'].strip()]
        pos = RaDecPos(g['ra'], g['dec'])
        fluxes = dict([(band, g['flux_{}'.format(band)]) for band in bands])
        bright = NanoMaggies(order=bands, **fluxes)
        shape = None
        if issubclass(typ, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
            shape = EllipseE(g['shape_r'], g['shape_e1'], g['shape_e2'])

        if issubclass(typ, (DevGalaxy, ExpGalaxy)):
            if issubclass(typ, DevGalaxy):
                serindex = 4.
            else:
                serindex = 1.
        elif issubclass(typ, (SersicGalaxy)):
            serindex = g['sersic']
            sersic = LegacySersicIndex(g['sersic'])
        else:
            print('Unknown type {}'.format(typ))

        # Masking radius based on surface brightness--
        # Normalize by total flux and then divide by the
        # pixelscale to get the surface brightness.
        radius_pixel = np.arange(20*g['shape_r'] / pixscale)   # [pixels]
        radius = pixscale * radius_pixel                       # [arcsec]
        pro = sersic_profile(radius / g['shape_r'], serindex) # 1D profile
        pro /= np.sum(pro * 2. * np.pi * radius_pixel)
        for band in ('r', 'z', 'g'):
            if g['flux_{}'.format(band)] > 0:
                pro *= (g['flux_{}'.format(band)] / pixscale**2) # [nanomaggies / arcsec^2]
                break

        # Interpolate to get the radius at which the surface
        # brightness profile crosses the desired threshold. Note:
        # 0.1 nanomaggies = 25 mag/arcsec^2
        try:
            r25 = interp1d(pro, radius)(10**(-0.4*(sbcut-22.5))) # [arcsec]
        except:
            print('Warning: extrapolating r25 for galaxy {}'.format(galaxy))
            r25 = interp1d(pro, radius, fill_value='extrapolate')(10**(-0.4*(sbcut-22.5))) # [arcsec]

        cat['d25_model'][ii] = 2 * r25 / 60.0                # [arcmin]
        if cat['d25_model'][ii] == 0.0:
            print('Warning: radius too small for galaxy {} ({}, r={:.3f})!'.format(
                galaxy, g['type'], g['shape_r']))
            pdb.set_trace()

        if shape is not None:
            e = shape.e
            cat['ba_model'][ii] = (1. - e) / (1. + e)
            pa = -np.rad2deg(np.arctan2(shape.e2, shape.e1) / 2.) # [degrees]
            if pa != 0.0:
                cat['pa_model'][ii] = 180-pa
        cat['freeze'][ii] = True

        #pdb.set_trace()
        return cat

    return Table()

def build_model_LSLGA(sample, pixscale=0.262, minradius=5.0, minsb=25.0, sbcut=25.0,
                      bands=('g', 'r', 'z'), clobber=False, nproc=1):
    """Gather all the fitting results and build the final model-based LSLGA catalog.

    minradius in arcsec
    minsb [minimum surface brightness] in r-band AB mag/arcsec**2
    sbcut [minimum surface brightness] in r-band AB mag/arcsec**2

    """
    import fitsio
    from astropy.table import Table, vstack
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
    out = mp.map(_build_model_LSLGA_one, [(onegal, pixscale, minradius, minsb, sbcut, bands, refcat) for onegal in sample])
    if len(out) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return

    #[print(ii) for ii, oo in enumerate(out) if oo is None]
    out = vstack(out)
    [out.rename_column(col, col.upper()) for col in out.colnames]
    print('Gathered {} pre-burned galaxies.'.format(len(out)))

    # Now read the full parent LSLGA catalog and supplement it with the
    # pre-burned galaxies.
    lslgafile = os.getenv('LARGEGALAXIES_CAT')
    lslga, hdr = fitsio.read(lslgafile, header=True)
    lslga = Table(lslga)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    # Add all the Tractor columns to the parent LSLGA.
    lslga.rename_column('RA', 'LSLGA_RA')
    lslga.rename_column('DEC', 'LSLGA_DEC')
    lslga.rename_column('TYPE', 'MORPHTYPE')
    for col in out.colnames:
        if out[col].ndim > 1:
            lslga[col] = np.zeros((len(lslga), out[col].shape[1]), dtype=out[col].dtype)
        else:
            lslga[col] = np.zeros(len(lslga), dtype=out[col].dtype)

    # First deal with pre-burned, but not frozen Tractor sources.  Remove
    # duplicates by choosing the source closest to the center of the mosaic.
    lslga_sup = []
    jindx = np.where(~out['FREEZE'])[0]
    if len(jindx) > 0:
        out_sup = out[jindx]
        grp, mult, first, next = spheregroup(out_sup['RA'], out_sup['DEC'], 0.1/3600.0)
        keep = np.where(mult == 1)[0]
        for ii in np.where(mult > 1)[0]:
            these = np.where(grp == grp[ii])[0]
            keep = np.hstack((keep, these[np.argmin(out_sup[these]['THETA_CENTER'])]))
        print('Removing {}/{} duplicate non-LSLGA Tractor sources.'.format(len(out_sup)-len(keep), len(out_sup)))
        out_sup = out_sup[np.sort(keep)]

        lslga_sup = Table()
        for col in lslga.colnames:
            if lslga[col].ndim > 1:
                lslga_sup[col] = np.zeros((len(out_sup), lslga[col].shape[1]), dtype=lslga[col].dtype)
            else:
                lslga_sup[col] = np.zeros(len(out_sup), dtype=lslga[col].dtype)
        for col in out.colnames:
            lslga_sup[col] = out_sup[col]
        lslga_sup['LSLGA_ID'] = -1
        del out_sup

    # Now deal with LSLGA galaxies.
    iindx = np.where(out['FREEZE'])[0] # should all have REF_CAT == refcat
    out_lslga = out[iindx]

    # Deal with duplicates by keeping the entry closest to the center of its
    # group.
    #grp, mult, first, next = spheregroup(out_lslga['RA'], out_lslga['DEC'], 0.1/3600.0)
    #keep = np.where(mult == 1)[0]
    #for ii in np.where(mult > 1)[0]:
    #    these = np.where(grp == grp[ii])[0]
    #    keep = np.hstack((keep, these[np.argmin(out_lslga[these]['THETA_CENTER'])]))

    keep = []
    for refid in set(out_lslga['REF_ID']):
        these = np.where(out_lslga['REF_ID'] == refid)[0]
        if len(these) == 1:
            keep.append(these)
        else:
            print('Found {} occurrences of {}'.format(len(these), refid))
            keep.append(these[np.argmin(out_lslga[these]['THETA_CENTER'])])
    keep = np.hstack(keep)

    #refid, cnt = np.unique(out_lslga['REF_ID'], return_counts=True)
    #keep = np.where(np.isin(out_lslga['REF_ID'], refid))[0]
    #keep = np.where(cnt == 1)[0]
    #for idup in np.where(cnt > 1)[0]:
    #    these = np.where(out_lslga['REF_ID'] == refid[idup])[0]
    #    keep = np.hstack((keep, these[np.argmin(out_lslga[these]['THETA_CENTER'])]))
    print('Removing {}/{} duplicate LSLGA Tractor sources.'.format(len(out_lslga)-len(keep), len(out_lslga)))
    out_lslga = out_lslga[keep]

    # Finally, populate the Tractor columns.
    kindx = np.where(np.isin(lslga['LSLGA_ID'], out_lslga['REF_ID']))[0]
    for col in out.colnames:
        lslga[col][kindx] = out_lslga[col]
    del out_lslga

    if len(lslga_sup) > 0:
        out = vstack((lslga, lslga_sup))
    else:
        out = lslga

    # Add RA, Dec back in--
    fix = np.where(~out['PREBURNED'])[0]
    if len(fix) > 0:
        out['RA'][fix] = out['LSLGA_RA'][fix]
        out['DEC'][fix] = out['LSLGA_DEC'][fix]

    #print(out[out['FREEZE']]['PA', 'PA_MODEL'])
    #pdb.set_trace()

    print('Writing {} galaxies to {}'.format(len(out), outfile))
    #out.write(outfile, overwrite=True)
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

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
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

    def _get_mags(cat, rad='10'):
        res = []
        for band in ('g', 'r', 'z'):
            iv = intflux['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
            ff = intflux['FLUX{}_{}'.format(rad, band.upper())][0]
            if iv > 0:
                ee = 1 / np.sqrt(iv)
                mag = 22.5-2.5*np.log10(ff)
                magerr = 2.5 * ee / ff / np.log(10)
                res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
            else:
                res.append('...')
        return res
            
    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    homehtmlfile = os.path.join(htmldir, homehtml)

    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>Legacy Survey Large Galaxy Atlas (LSLGA)</h1>\n')
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

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['LSLGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.2f}</td>\n'.format(gal[diamcolumn]))
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
            #
            #ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
            ##if 'psfdepth_g' not in ellipse.keys():
            ##    pdb.set_trace()
            #pipeline_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose,
            #                                                  filesuffix='pipeline')

            if not os.path.exists(htmlgalaxydir1):
                os.makedirs(htmlgalaxydir1)

            ccdsfile = os.path.join(galaxydir1, '{}-ccds.fits'.format(galaxy1))
            if os.path.isfile(ccdsfile):
                nccds = fitsio.FITS(ccdsfile)[1].get_nrows()
            else:
                nccds = None

            nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
            prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

            htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
            with open(htmlfile, 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
                html.write('</style>\n')

                html.write('<h1>Galaxy {}</h1>\n'.format(galaxy1))

                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
                html.write('<br />\n')
                html.write('<br />\n')

                # Table of properties
                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th>Index</th>\n')
                html.write('<th>LSLGA ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
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
                html.write('<td>{}</td>\n'.format(galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.2f}</td>\n'.format(gal[diamcolumn]))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
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

                html.write('<p>(Left) data, (middle) model, and (right) residual image mosaic.</p>\n')
                #html.write('<br />\n')

                html.write('<table width="90%">\n')
                pngfile = '{}-grz-montage.png'.format(galaxy1)
                thumbfile = 'thumb-{}-grz-montage.png'.format(galaxy1)
                html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                    pngfile, thumbfile))
                #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
                html.write('</table>\n')

                #html.write('<br />\n')
                html.write('<p>Maskbits.</p>\n')
                
                html.write('<table width="30%">\n')
                pngfile = '{}-maskbits.png'.format(galaxy1)
                html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                    pngfile))
                html.write('</table>\n')

                #html.write('<br />\n')
                html.write('<p>Spatial distribution of CCDs.</p>\n')

                html.write('<table width="90%">\n')
                pngfile = '{}-ccdpos.png'.format(galaxy1)
                html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                    pngfile))
                html.write('</table>\n')
                #html.write('<br />\n')

                if False:
                    html.write('<h2>Elliptical Isophote Analysis</h2>\n')
                    if bool(ellipse):
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

                        html.write('<table>\n')
                        html.write('<tr><th>Fitting range<br />(arcsec)</th><th>Integration<br />mode</th><th>Clipping<br />iterations</th><th>Clipping<br />sigma</th></tr>')
                        html.write('<tr><td>{:.3f}-{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                            ellipse[refband]['sma'].min()*pixscale, ellipse[refband]['sma'].max()*pixscale,
                            ellipse['integrmode'], ellipse['nclip'], ellipse['sclip']))
                        html.write('</table>\n')
                        html.write('<br />\n')
                    else:
                        html.write('<p>Ellipse-fitting not done or failed.</p>\n')

                    html.write('<table width="90%">\n')
                    html.write('<tr>\n')
                    html.write('<td><a href="{}-ellipse-multiband.png"><img src="{}-ellipse-multiband.png" alt="Missing file {}-ellipse-multiband.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    html.write('</tr>\n')
                    html.write('</table>\n')
                    html.write('<br />\n')

                    html.write('<table width="90%">\n')
                    html.write('<tr>\n')
                    #html.write('<td><a href="{}-ellipse-ellipsefit.png"><img src="{}-ellipse-ellipsefit.png" alt="Missing file {}-ellipse-ellipsefit.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                    pngfile = '{}-ellipse-sbprofile.png'.format(galaxy1)
                    html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
                    pngfile = '{}-ellipse-cog.png'.format(galaxy1)
                    html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
                    #html.write('<td></td>\n')
                    html.write('</tr>\n')
                    html.write('</table>\n')
                    html.write('<br />\n')

                    html.write('<h2>Observed & rest-frame photometry</h2>\n')

                    html.write('<h4>Integrated photometry</h4>\n')
                    html.write('<table>\n')
                    html.write('<tr>')
                    html.write('<th colspan="3">Curve of growth<br />(custom sky, mag)</th><th colspan="3">Curve of growth<br />(pipeline sky, mag)</th>')
                    html.write('</tr>')

                    html.write('<tr>')
                    html.write('<th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th>')
                    html.write('</tr>')

                    html.write('<tr>')
                    if bool(ellipse):
                        g, r, z = (ellipse['cog_params_g']['mtot'], ellipse['cog_params_r']['mtot'],
                                   ellipse['cog_params_z']['mtot'])
                        html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(g, r, z))
                    else:
                        html.write('<td>...</td><td>...</td><td>...</td>')

                    if bool(pipeline_ellipse):
                        g, r, z = (pipeline_ellipse['cog_params_g']['mtot'], pipeline_ellipse['cog_params_r']['mtot'],
                                   pipeline_ellipse['cog_params_z']['mtot'])
                        html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(g, r, z))
                    else:
                        html.write('<td>...</td><td>...</td><td>...</td>')

                    html.write('</tr>')
                    html.write('</table>\n')
                    html.write('<br />\n')

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

                    if intflux:
                        html.write('<tr>')
                        g, r, z = _get_mags(intflux[ii], rad='10')
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        g, r, z = _get_mags(intflux[ii], rad='30')
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        g, r, z = _get_mags(intflux[ii], rad='100')
                        html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                        html.write('</tr>')

                    html.write('</table>\n')
                    html.write('<br />\n')

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

                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
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

    cmd = 'chgrp -R cosmo {}'.format(htmldir)
    print(cmd)
    err1 = subprocess.call(cmd.split())

    cmd = 'find {} -type d -exec chmod 775 {{}} +'.format(htmldir)
    print(cmd)
    err2 = subprocess.call(cmd.split())

    cmd = 'find {} -type f -exec chmod 664 {{}} +'.format(htmldir)
    print(cmd)
    err3 = subprocess.call(cmd.split())

    if err1 != 0 or err2 != 0 or err3 != 0:
        print('Something went wrong updating permissions; please check the logfile.')
        return 0
    
    return 1

