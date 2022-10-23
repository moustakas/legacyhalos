"""
legacyhalos.virgofilaments
==========================

Code to deal with the Virgo filaments sample and project.

"""
import os, shutil, pdb
import numpy as np
import astropy

import legacyhalos.io

#ZCOLUMN = 'Z'
#RACOLUMN = 'RA'
#DECCOLUMN = 'DEC'
#DIAMCOLUMN = 'DIAM'
GALAXYCOLUMN = 'GALAXY'
RACOLUMN = 'GROUP_RA'
DECCOLUMN = 'GROUP_DEC'
DIAMCOLUMN = 'GROUP_DIAMETER'
GALAXYCOLUMN = 'GROUP_NAME'
REFIDCOLUMN = 'VF_ID'

SBTHRESH = [22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0] # surface brightness thresholds
APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] # multiples of MAJORAXIS

ELLIPSEBITS = dict(
    largeshift = 2**0, # >10-pixel shift in the flux-weighted center
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

def get_raslice(ra):
    return '{:06d}'.format(int(ra*1000))[:3]

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
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
        galaxy = [cat[GALAXYCOLUMN]]
        ra = [cat[racolumn]]
    else:
        ngal = len(cat)
        galaxy = cat[GALAXYCOLUMN]
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

def missing_files(args, sample, size=1, clobber_overwrite=None):
    from astrometry.util.multiproc import multiproc
    from legacyhalos.io import _missing_files_one

    dependson = None
    galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    if args.coadds:
        suffix = 'coadds'
        filesuffix = '-custom-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.rebuild_unwise:
        suffix = 'rebuild-unwise'
        filesuffix = '-rebuild-unwise.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        filesuffix = '-custom-ellipse.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.build_catalog:
        suffix = 'build-catalog'
        filesuffix = '-custom-ellipse.isdone'
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
        raise ValueError('Need at least one keyword argument.')

    # Make clobber=False for build_catalog and htmlindex because we're not
    # making the files here, we're just looking for them. The argument
    # args.clobber gets used downstream.
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
        #todo_indices = np.array_split(_todo_indices, size) # unweighted

        # Assign the sample to ranks to make the D25 distribution per rank ~flat.
        
        # https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        cumuweight = weight.cumsum() / weight.sum()
        idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        if len(idx) < size: # can happen in corner cases or with 1 rank
            #todo_indices = np.array_split(_todo_indices, size) # unweighted
            todo_indices = np.array_split(_todo_indices[np.argsort(weight)], size) # unweighted but sorted
        else:
            todo_indices = np.array_split(_todo_indices, idx) # weighted
        for ii in range(size): # sort by weight
            srt = np.argsort(sample[DIAMCOLUMN][todo_indices[ii]])
            todo_indices[ii] = todo_indices[ii][srt]            
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices
    
def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--d25min', default=0.0, type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--d25max', default=100.0, type=float, help='Maximum diameter (arcmin).')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--rebuild-unwise', action='store_true', help='Rebuild the unWISE mosaics.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--html-raslices', action='store_true',
                        help='Organize HTML pages by RA slice (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')

    parser.add_argument('--no-galex-ceres', action='store_true', help='Do not use Ceres solver to perform GALEX forced photometry.')
    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-galex', action='store_false', dest='galex', help='Do not build GALEX coadds or do forced GALEX photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final catalog.')
    args = parser.parse_args()

    return args

def get_version():
    return 'v2'

def read_sample(first=None, last=None, galaxylist=None, verbose=False, fullsample=False,
                d25min=0.1, d25max=100.0, version=None):
    """Read/generate the parent catalog.

    d25min,d25max in arcmin

    """
    import fitsio

    if version is None:
        version = get_version()
    
    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()

    samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'vf_north_{}_main_groups.fits'.format(version))    
    if not os.path.isfile(samplefile):
        raise IOError('Sample file not found! {}'.format(samplefile))

    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # Select primary group members with an SGA match--
    if not fullsample:
        cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY', 'GROUP_MULT', REFIDCOLUMN]
        sample = fitsio.read(samplefile, columns=cols, upper=True)
        rows = np.arange(len(sample))

        #testvfid = fitsio.read(os.path.join(legacyhalos.io.legacyhalos_dir(),
        #                                    'vf_north_v1_main_groups_testsample2.fits'),
        #                                    columns=REFIDCOLUMN)
        samplecut = np.where(
            #np.isin(sample[REFIDCOLUMN], testvfid) *
            sample['GROUP_PRIMARY'] *
            (sample['GROUP_DIAMETER'] > d25min) *
            (sample['GROUP_DIAMETER'] < d25max)
            )[0]
        rows = rows[samplecut]
        nrows = len(rows)
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
    #sample.add_column(astropy.table.Column(name='INDEX', data=rows), index=0)

    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        these = np.isin(sample[GALAXYCOLUMN], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            sample = sample[these]

    return sample

def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          threshmask=0.01, r50mask=0.05, maxshift=10,
                          relmaxshift=0.1,
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
        # Convert a Tractor catalog entry to an MGE object.
        class MGEgalaxy(object):
            pass

        default_majoraxis = tractor.diam_init[indx] * 60 / 2 / filt2pixscale[refband] # [pixels]
        default_pa = tractor.pa_init[indx]
        default_ba = tractor.ba_init[indx]
        #default_theta = (270 - default_pa) % 180
        #default_eps = 1 - tractor.ba_init[indx]

        #if tractor.sga_id[indx] > -1:
        if tractor.type[indx] == 'PSF' or tractor.shape_r[indx] < 5:
            pa = tractor.pa_init[indx]
            ba = tractor.ba_init[indx]
            # take away the extra factor of 2 we put in in read_sample()
            r50 = tractor.diam_init[indx] * 60 / 2 / 2
            if r50 < 5:
                r50 = 5.0 # minimum size, arcsec
            majoraxis = factor * r50 / filt2pixscale[refband] # [pixels]
            #majoraxis = factor * tractor.diam_init[indx] * 60 / 2 / 2 / filt2pixscale[refband] # [pixels]
        else:
            ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2))
            pa = pa % 180

            # can be zero (or very small) if fit as a PSF or REX
            if tractor.shape_r[indx] > 1:
                majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]
            else:
                majoraxis = factor * tractor.diam_init[indx] * 60 / 2 / 2 / filt2pixscale[refband] # [pixels]

        mgegalaxy = MGEgalaxy()
        
        mgegalaxy.xmed = tractor.by[indx]
        mgegalaxy.ymed = tractor.bx[indx]
        mgegalaxy.xpeak = tractor.by[indx]
        mgegalaxy.ypeak = tractor.bx[indx]

        # never use the Tractor geometry (only the centroid)
        # https://portal.nersc.gov/project/cosmo/temp/ioannis/virgofilaments-html/215/NGC5584/NGC5584.html
        if False:
            mgegalaxy.eps = 1-ba
            mgegalaxy.pa = pa
            mgegalaxy.theta = (270 - pa) % 180
            mgegalaxy.majoraxis = majoraxis
        else:
            mgegalaxy.eps = 1 - default_ba
            mgegalaxy.pa = default_pa
            mgegalaxy.theta = (270 - default_pa) % 180
            mgegalaxy.majoraxis = default_majoraxis

        # always restore all pixels within the nominal / initial size of the galaxy
        #objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
        #                       default_majoraxis,
        #                       default_majoraxis * (1-default_eps), 
        #                       np.radians(default_theta-90), xobj, yobj)
        #objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
        #                       default_majoraxis, default_majoraxis, 0.0, xobj, yobj)

        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                               mgegalaxy.majoraxis,
                               mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
                               np.radians(mgegalaxy.theta-90), xobj, yobj)

        # central 10% pixels can override the starmask
        objmask_center = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                                      0.1*mgegalaxy.majoraxis,
                                      0.1*mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
                                      np.radians(mgegalaxy.theta-90), xobj, yobj)

        return mgegalaxy, objmask, objmask_center

    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        print('Determing the geometry for galaxy {}/{}.'.format(
                ii+1, len(galaxy_indx)))

        # [1] Determine the non-parametric geometry of the galaxy of interest
        # in the reference band. First, subtract all models except the galaxy
        # and galaxies "near" it. Also restore the original pixels of the
        # central in case there was a poor deblend.
        largeshift = False
        mge, centralmask, centralmask2 = tractor2mge(central, factor=1.0)
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

        mask = np.logical_or(ma.getmask(data[refband]), data['residual_mask'])
        #mask = np.logical_or(data[refband].mask, data['residual_mask'])

        # restore the central pixels but not the masked stellar pixels
        centralmask[np.logical_and(data['starmask'], np.logical_not(centralmask2))] = False
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)
        #if ii == 1:
        #    pdb.set_trace()

        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('cosmo-www/tmp/junk-mge.png')
        #plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask.png')
        ##plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #pdb.set_trace()

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            print('Large centroid shift (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                mgegalaxy.xmed, mgegalaxy.ymed, mge.xmed, mge.ymed))
            print('  Reverting to the default geometry and the Tractor centroid.')
            largeshift = True
            mgegalaxy = copy(mge)

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
        satmask = np.zeros(data[refband].shape, bool)
        for filt in bands:
            # do not let GALEX and WISE contribute to the satellite mask
            if data[filt].shape != satmask.shape:
                continue
            
            cenflux = getattr(tractor, 'flux_{}'.format(filt.lower()))[central]
            satflux = getattr(srcs, 'flux_{}'.format(filt.lower()))
            if cenflux <= 0.0:
                #raise ValueError('Central galaxy flux is negative!')
                print('Central galaxy flux is negative! Proceed with caution...')
                #pdb.set_trace()
                
            satindx = np.where(np.logical_or(
                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
                srcs.ref_cat == 'R1'))[0]
            #satindx = np.where(srcs.ref_cat == 'R1')[0]
            #if np.isin(central, satindx):
            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
            if len(satindx) == 0:
                #raise ValueError('All satellites have been dropped!')
                #print('Warning! All satellites have been dropped from band {}!'.format(filt))
                print('Note: no satellites to mask in band {}.'.format(filt))
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
                #    plt.clf() ; plt.imshow(np.log10(satimg), origin='lower') ; plt.savefig('debug.png')
                #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('debug.png')
                ##    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
                #    pdb.set_trace()

            #print(filt, np.sum(satmask), np.sum(thissatmask))

        #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-satmask.png')
        
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
            #if filt == 'r':
            #    #plt.imshow(_satmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-satmask-{}.png'.format(filt))
            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt))
            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt))
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
            #if len(psfindx) > 0 and filt.upper() != 'NUV' and filt.upper() != 'FUV' and filt.upper() != 'W3' and filt.upper() != 'W4':
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
                    plt.savefig('desi-users/ioannis/tmp/qa-psf-{}.png'.format(filt.lower()))
                    if filt == 'r':# or filt == 'r':
                        pdb.set_trace()
                img -= psfimg
            else:
                psfimg = np.zeros((2, 2), 'f4')

            data[psfimgkey].append(psfimg)
            
            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
            var = data['{}_var_'.format(filt.lower())] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #if ii == 0 and filt == 'r': #filt == 'W1' or 
            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt.lower()))
            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt.lower()))
            #####    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('junk-thismask-{}.png'.format(filt.lower()))
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
            #'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 
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
    galaxy_indx = np.hstack([np.where((tractor.ref_id == sid) * (tractor.ref_cat != '  '))[0] for sid in sample[REFIDCOLUMN]])

    #sample = sample[np.searchsorted(sample[REFIDCOLUMN], tractor.ref_id[galaxy_indx])]
    assert(np.all(sample[REFIDCOLUMN] == tractor.ref_id[galaxy_indx]))

    tractor.sga_id = np.zeros(len(tractor), dtype=np.int64)-1
    tractor.diam_init = np.zeros(len(tractor), dtype='f4')
    tractor.pa_init = np.zeros(len(tractor), dtype='f4')
    tractor.ba_init = np.zeros(len(tractor), dtype='f4')
    if 'DIAM_INIT' in sample.colnames and 'PA_INIT' in sample.colnames and 'BA_INIT' in sample.colnames:
        tractor.sga_id[galaxy_indx] = sample['SGA_ID']
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
        galaxyinfo = {'vf_id': (np.int32(galaxy_id), ''),
                      'galaxy': (str(np.atleast_1d(samp['GALAXY'])[0]), '')}
        #for key, unit in zip(['ra', 'dec', 'diam_init', 'pa_init', 'ba_init'],
        #                     [u.deg, u.deg, u.arcmin, u.deg, '']):
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
    delta_logsma = 4 # 10

    #igal = 0
    #maxis = data['mge'][igal]['majoraxis'] # [pixels]
    #
    #if galaxyinfo[igal]['diam'] > 10:
    #    maxsma = 1.5 * maxis # [pixels]
    #    delta_logsma = 10.0
    #else:
    #    maxsma = 2 * maxis # [pixels]
    #    delta_logsma = 6.0

    # don't pass logfile and set debug=True because we've already opened the log
    # above!
    mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                     pixscale=pixscale, nproc=nproc, 
                     bands=bands, refband=refband, sbthresh=SBTHRESH,
                     apertures=APERTURES,
                     logsma=True, delta_logsma=delta_logsma, maxsma=maxsma,
                     verbose=verbose, clobber=clobber,
                     #debug=True,
                     debug=debug, logfile=logfile)

def call_rebuild_unwise(onegal, galaxy, galaxydir, filesuffix='custom',
                        pixscale=0.262, radius_mosaic=None,
                        racolumn='RA', deccolumn='DEC', 
                        nproc=1, verbose=False, debug=False,
                        clobber=False, write_donefile=True, logfile=None):
    """Wrapper script to rebuild the unWISE mosaics and forced photometry.

    """
    import time
    from contextlib import redirect_stdout, redirect_stderr
    from legacyhalos.mpi import _done, _start

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = rebuild_unwise(onegal, galaxy=galaxy, galaxydir=galaxydir,
                             pixscale=pixscale, radius_mosaic=radius_mosaic,
                             racolumn=racolumn, deccolumn=deccolumn, 
                             filesuffix=filesuffix, nproc=nproc, verbose=verbose,
                             debug=debug, clobber=clobber)
        if write_donefile:
            _done(galaxy, galaxydir, err, t0, 'rebuild-unwise', None)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = rebuild_unwise(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                     pixscale=pixscale, radius_mosaic=radius_mosaic,
                                     racolumn=racolumn, deccolumn=deccolumn, 
                                     filesuffix=filesuffix, nproc=nproc, verbose=verbose,
                                     debug=debug, clobber=clobber, log=log)
                if write_donefile:
                    _done(galaxy, galaxydir, err, t0, 'rebuild-unwise', None, log=log)

    return err

def rebuild_unwise(onegal, galaxy, galaxydir, filesuffix='custom',
                   pixscale=0.262, radius_mosaic=None, racolumn='RA', deccolumn='DEC',
                   nproc=1, verbose=False, debug=False, clobber=False, log=None):
    """Rebuild the unWISE mosaics.

    """
    import shutil
    import subprocess
    from legacyhalos.coadds import _mosaic_width

    stagesuffix = 'rebuild-unwise'

    def _copyfile(infile, outfile, clobber=False, update_header=False):
        if os.path.isfile(outfile) and not clobber:
            return 1
        if os.path.isfile(infile):
            tmpfile = outfile+'.tmp'
            shutil.copy2(infile, tmpfile)
            #shutil.copyfile(infile, tmpfile)
            os.rename(tmpfile, outfile)
            if update_header:
                pass
            return 1
        else:
            print('Missing file {}; please check the logfile.'.format(infile))
            return 0

    # if coadds aren't done, skip this galaxy
    if not os.path.isfile(os.path.join(galaxydir, '{}-custom-coadds.isdone'.format(galaxy))):
        print('Custom coadds not done; skipping rebuilding WISE coadds.', flush=True, file=log)
        return 1, stagesuffix

    # A small fraction of the sample doesn't have grz coverage. These will have
    # .isdone files but no montages.
    if (os.path.isfile(os.path.join(galaxydir, '{}-custom-coadds.isdone'.format(galaxy))) and
        not os.path.isfile(os.path.join(galaxydir, '{}-custom-resid-W1W2.jpg'.format(galaxy)))):
        print('No WISE data to process; skipping rebuilding WISE coadds.', flush=True, file=log)
        return 1, stagesuffix

    # backup the existing files *once* so we can rerun this stage if necessary
    backupdir = os.path.join(galaxydir, 'rebuild-unwise-backup')
    if not os.path.isdir(backupdir):
        os.makedirs(backupdir)
        ok = _copyfile(os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, filesuffix)),
                       os.path.join(backupdir, '{}-{}-tractor.fits'.format(galaxy, filesuffix)), clobber=True)
        if not ok:
            raise ValueError('Problem backing up Tractor catalog!')
        
        for band in ('W1', 'W2', 'W3', 'W4'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(galaxydir, '{}-{}-{}-{}.fits.fz'.format(galaxy, filesuffix, imtype, band)),
                    os.path.join(backupdir, '{}-{}-{}-{}.fits.fz'.format(galaxy, filesuffix, imtype, band)), clobber=True)
                if not ok:
                    raise ValueError('Problem backing up {} {}!'.format(imtype, band))
            ok = _copyfile(
                os.path.join(galaxydir, '{}-{}-model-{}.fits.fz'.format(galaxy, filesuffix, band)),
                os.path.join(backupdir, '{}-{}-model-{}.fits.fz'.format(galaxy, filesuffix, band)), clobber=True)
            if not ok:
                raise ValueError('Problem backing up {} {}!'.format(imtype, band))
        for imtype in ('image', 'model', 'resid'):
            ok = _copyfile(
                os.path.join(galaxydir, '{}-{}-{}-W1W2.jpg'.format(galaxy, filesuffix, imtype)),
                os.path.join(backupdir, '{}-{}-{}-W1W2.jpg'.format(galaxy, filesuffix, imtype)), clobber=True)
            if not ok:
                raise ValueError('Problem backing up {}!'.format(imtype))

    # now copy from the backup directory
    ok = _copyfile(os.path.join(backupdir, '{}-{}-tractor.fits'.format(galaxy, filesuffix)),
                   os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, filesuffix)), clobber=True)
    if not ok:
        raise ValueError('Problem backing up Tractor catalog!')
    
    for band in ('W1', 'W2', 'W3', 'W4'):
        for imtype in ('image', 'invvar', 'model'):
            ok = _copyfile(
                os.path.join(backupdir, '{}-{}-{}-{}.fits.fz'.format(galaxy, filesuffix, imtype, band)),
                os.path.join(galaxydir, '{}-{}-{}-{}.fits.fz'.format(galaxy, filesuffix, imtype, band)), clobber=True)
            if not ok:
                raise ValueError('Problem backing up {} {}!'.format(imtype, band))
    for imtype in ('image', 'model', 'resid'):
        ok = _copyfile(
            os.path.join(backupdir, '{}-{}-{}-W1W2.jpg'.format(galaxy, filesuffix, imtype)),
            os.path.join(galaxydir, '{}-{}-{}-W1W2.jpg'.format(galaxy, filesuffix, imtype)), clobber=True)
        if not ok:
            raise ValueError('Problem backing up {}!'.format(imtype))

    width = _mosaic_width(radius_mosaic, pixscale)    

    cmd = 'python {legacypipe_dir}/py/legacyanalysis/rerun-wise-phot.py '
    cmd += '--catalog {galaxydir}/{galaxy}-custom-tractor.fits '
    cmd += '--unwise-dir {unwise_dir} '
    cmd += '--radec {ra} {dec} --width {width} --height {width} '
    cmd += '--pixscale {pixscale} --threads {threads} --outdir {outdir} '

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_CODE_DIR'), galaxy=galaxy,
                     galaxydir=galaxydir, unwise_dir=os.environ.get('UNWISE_COADDS_DIR'),
                     ra=onegal[racolumn], dec=onegal[deccolumn], width=width,
                     pixscale=pixscale, threads=nproc, outdir=galaxydir)
    print(cmd, flush=True, file=log)

    err = subprocess.call(cmd.split(), stdout=log, stderr=log)

    if err != 0:
        print('Something went wrong; please check the logfile.')
        return 0, stagesuffix
    else:
        # Move (rename) files into the desired output directory and clean up.
        # tractor catalog
        brickname = 'custom_%.4f_%.4f' % (onegal[RACOLUMN], onegal[DECCOLUMN])
        ok = _copyfile(
            os.path.join(galaxydir, 'tractor.fits'),
            os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, filesuffix)), clobber=True)
        if not ok:
            raise ValueError('Problem copying Tractor catalog!')
        for band in ('W1', 'W2', 'W3', 'W4'):
            for imtype in ('image', 'invvar', 'model'):
                ok = _copyfile(
                    os.path.join(galaxydir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(galaxydir, '{}-{}-{}-{}.fits.fz'.format(galaxy, filesuffix, imtype, band)),
                    clobber=True)
                if not ok:
                    raise ValueError('Problem copying {} {}!'.format(imtype, band))
        for imtype, suffix in zip(('wise', 'wisemodel', 'wiseresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(galaxydir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(galaxydir, '{}-{}-{}-W1W2.jpg'.format(galaxy, filesuffix, suffix)),
                clobber=True)
            if not ok:
                raise ValueError('Problem backing up {}!'.format(imtype))                

        shutil.rmtree(os.path.join(galaxydir, 'coadd'), ignore_errors=True)
        os.remove(os.path.join(galaxydir, 'tractor.fits'))

        return ok, stagesuffix

def _datarelease_table(ellipse):
    """Convert the ellipse table into a data release catalog."""

    from copy import deepcopy

    out = deepcopy(ellipse)
    for col in out.colnames:
        if out[col].ndim > 1:
            out.remove_column(col)
        if 'REFBAND' in col or 'PSFSIZE' in col or 'PSFDEPTH' in col:
            out.remove_column(col)
    remcols = ('REFPIXSCALE', 'SUCCESS', 'FITGEOMETRY', 'LARGESHIFT',
               'MAXSMA', 'MAJORAXIS', 'EPS_MOMENT', 'INTEGRMODE', 'INPUT_ELLIPSE', 'SCLIP', 'NCLIP')
    for col in remcols:
        out.remove_column(col)

    out['ELLIPSEBIT'] = np.zeros(1, dtype=np.int32) # we don't want -1 here                

    return out

def _build_catalog_one(args):
    """Wrapper function for the multiprocessing."""
    return build_catalog_one(*args)

def build_catalog_one(galaxy, galaxydir, fullsample, refcat='R1', verbose=False):
    """Gather the ellipse-fitting results for a single group."""
    import fitsio
    from astropy.table import Table, vstack
    from legacyhalos.io import read_ellipsefit

    tractor, parent, ellipse = [], [], []

    tractorfile = os.path.join(galaxydir, '{}-custom-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return None, None, None #tractor, parent, ellipse
        #return tractor, parent, ellipse

    for onegal in fullsample:
        refid = onegal[REFIDCOLUMN]
        
        ellipsefile = os.path.join(galaxydir, '{}-custom-ellipse-{}.fits'.format(galaxy, refid))
        if not os.path.isfile(ellipsefile):
            print('Missing ellipse file {}'.format(ellipsefile))
            return None, None, None #tractor, parent, ellipse

        _ellipse = read_ellipsefit(galaxy, galaxydir, galaxy_id=str(refid), asTable=True,
                                  filesuffix='custom', verbose=True)
        _ellipse = _datarelease_table(_ellipse)
        #for col in _ellipse.colnames:
        #    if _ellipse[col].ndim > 1:
        #        _ellipse.remove_column(col)

        _tractor = Table(fitsio.read(tractorfile, upper=True))
        match = np.where((_tractor['REF_CAT'] == refcat) * (_tractor['REF_ID'] == refid))[0]
        if len(match) != 1:
            raise ValueError('Problem here!')

        ellipse.append(_ellipse)
        tractor.append(_tractor[match])
        parent.append(onegal)

    tractor = vstack(tractor, metadata_conflicts='silent')
    parent = vstack(parent, metadata_conflicts='silent')
    ellipse = vstack(ellipse, metadata_conflicts='silent')

    return tractor, parent, ellipse

def build_catalog(sample, fullsample, nproc=1, refcat='R1', verbose=False, clobber=False):
    import time
    import multiprocessing
    from astropy.io import fits
    from astropy.table import vstack

    version = get_version()
    
    outfile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'virgofilaments-{}-legacyphot.fits'.format(version))
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return

    galaxy, galaxydir = get_galaxy_galaxydir(sample)
    
    buildargs = []
    for gal, gdir, onegal in zip(galaxy, galaxydir, sample):
        _fullsample = fullsample[fullsample['GROUP_ID'] == onegal['GROUP_ID']]
        buildargs.append((gal, gdir, _fullsample, refcat, verbose))

    t0 = time.time()
    if nproc > 1:
        with multiprocessing.Pool(nproc) as P:
            results = P.map(_build_catalog_one, buildargs)
    else:
        results = [build_catalog_one(*_buildargs) for _buildargs in buildargs]

    results = list(zip(*results))

    tractor1 = list(filter(None, results[0]))
    parent1 = list(filter(None, results[1]))
    ellipse1 = list(filter(None, results[2]))

    tractor = vstack(tractor1, metadata_conflicts='silent')
    parent = vstack(parent1, metadata_conflicts='silent')
    ellipse = vstack(ellipse1, metadata_conflicts='silent')

    #results = list(zip(*results))
    #tractor = vstack(results[0], metadata_conflicts='silent')
    #parent = vstack(results[1], metadata_conflicts='silent')
    #ellipse = vstack(results[2], metadata_conflicts='silent')
    print('Merging {} galaxies took {:.2f} min.'.format(len(tractor), (time.time()-t0)/60.0))

    if len(tractor) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    assert(len(tractor) == len(parent))
    assert(np.all(tractor['REF_ID'] == parent[REFIDCOLUMN]))

    # write out
    hdu_primary = fits.PrimaryHDU()
    hdu_parent = fits.convenience.table_to_hdu(parent)
    hdu_parent.header['EXTNAME'] = 'PARENT'

    hdu_ellipse = fits.convenience.table_to_hdu(ellipse)
    hdu_ellipse.header['EXTNAME'] = 'ELLIPSE'

    hdu_tractor = fits.convenience.table_to_hdu(tractor)
    hdu_tractor.header['EXTNAME'] = 'TRACTOR'
        
    hx = fits.HDUList([hdu_primary, hdu_parent, hdu_ellipse, hdu_tractor])
    hx.writeto(outfile, overwrite=True, checksum=True)

    print('Wrote {} galaxies to {}'.format(len(parent), outfile))

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
                   racolumn=RACOLUMN, deccolumn=DECCOLUMN, diamcolumn=DIAMCOLUMN,
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

        html.write('<h1>Virgo Filaments</h1>\n')

        html.write('<p style="width: 75%">\n')
        html.write("""This project is super neat.</p>\n""")

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
            #html.write('<th>Index</th>\n')
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
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
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
                #html.write('<th>Index</th>\n')
                html.write('<th>ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
                html.write('<th>Viewer</th>\n')

                html.write('</tr>\n')
                for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                    htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                    pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                    thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                    ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                    viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                    html.write('<tr>\n')
                    #html.write('<td>{:g}</td>\n'.format(count))
                    #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
                    html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                    #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                    html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
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
        samplefile = os.path.join(galaxydir1, '{}-sample.fits'.format(galaxy1))
        #samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
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
            if prefix == 'custom':
                wt, ws = [], []
                for ii, sid in enumerate(sample[REFIDCOLUMN]):
                    ww = np.where((tractor['ref_cat'] != '  ') * (tractor['ref_id'] == sid))[0]
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

    def _html_group_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

        html.write('<h2>Group Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        #html.write('<th>Index<br />(Primary)</th>\n')
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
        #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
        #html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
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
        #html.write('<th>Morphology</th>\n')
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
            html.write('<td>{}</td>\n'.format(groupgal[REFIDCOLUMN]))
            html.write('<td>{}</td>\n'.format(groupgal['GALAXY']))
            #typ = groupgal['MORPHTYPE'].strip()
            #if typ == '' or typ == 'nan':
            #    typ = '...'
            #html.write('<td>{}</td>\n'.format(typ))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
            html.write('<td>{:.4f}</td>\n'.format(groupgal['DIAM_INIT']))
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

        pngfile, thumbfile = '{}-custom-montage-grz.png'.format(galaxy1), 'thumb-{}-custom-montage-grz.png'.format(galaxy1)
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
        html.write('<table width="90%">\n')
        for bandsuffix in ('grz', 'FUVNUV', 'W1W2'):
            pngfile, thumbfile = '{}-custom-montage-{}.png'.format(galaxy1, bandsuffix), 'thumb-{}-custom-montage-{}.png'.format(galaxy1, bandsuffix)
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

            html.write('<tr><td>{}</td>\n'.format(ss['GALAXY']))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['sma_moment'], ellipse['pa_moment'], ellipse['eps_moment']))
                    #ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa_moment'], ellipse['eps_moment']))

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
            html.write('<tr><td>{}</td>\n'.format(ss['GALAXY']))
            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                fuv, nuv, g, r, z, w1, w2, w3, w4))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                        galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):# and 'cog_mtot_fuv' in ellipse.keys():
                #g, r, z = _get_mags(ellipse, R24=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R25=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R26=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)                
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
            html.write('<h4>{}</h4>\n'.format(sample['GALAXY'][igal]))

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
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=False,
              htmlhome='index.html', html_raslices=False,
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
