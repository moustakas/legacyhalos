"""
legacyhalos.hizea
=================

Code to deal with the HizEA sample and project.

"""
import os, shutil, pdb
import numpy as np
import astropy
from astropy.table import Table

import legacyhalos.io

ZCOLUMN = 'Z'
RACOLUMN = 'RA' # 'RA'
DECCOLUMN = 'DEC' # 'DEC'
GALAXYCOLUMN = 'GALAXY'
REFIDCOLUMN = 'REFID'

MOSAICRADIUS = 30.0 # [arcsec]

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

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final photometric catalog.')
    args = parser.parse_args()

    return args

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
    
def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat[GALAXYCOLUMN]]
    else:
        ngal = len(cat)
        galaxy = cat[GALAXYCOLUMN]

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

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None):
    """Read the parent catalog.

    """
    import fitsio
    from legacyhalos.desiutil import brickname as get_brickname

    if True:
        sample = Table()
        sample['GALAXY'] = ['orc4']
        sample['REFID'] = [np.int64(100001)]
        sample['RA'] = [238.852625]
        sample['DEC'] = [27.44286111]
        sample['Z'] = [np.float32(0.4512)]
    else:
        samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'hizea-sample.fits')

        if first and last:
            if first > last:
                print('Index first cannot be greater than index last, {} > {}'.format(first, last))
                raise ValueError()
        ext = 1
        info = fitsio.FITS(samplefile)
        nrows = info[ext].get_nrows()
        rows = np.arange(nrows)
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

def build_catalog(sample, nproc=1, verbose=False):
    import time
    import fitsio
    import multiprocessing
    from astropy.io import fits
    from astropy.table import Table, vstack
    from astrometry.libkd.spherematch import match_radec
    
    outfile = os.path.join(legacyhalos.io.legacyhalos_data_dir(), 'hizea-legacyphot.fits')
    if os.path.isfile(outfile) and not clobber:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))
        return None

    galaxy, galaxydir = get_galaxy_galaxydir(sample)

    t0 = time.time()
    tractor, parent, dist = [], [], []
    for gal, gdir, onegal in zip(galaxy, galaxydir, sample):
        tractorfile = os.path.join(gdir, '{}-pipeline-tractor.fits'.format(gal))
        if os.path.isfile(tractorfile): # just in case
            parent.append(onegal)
            cat = fitsio.read(tractorfile, upper=True)
            m1, m2, d12 = match_radec(cat['RA'], cat['DEC'], onegal['RA'], onegal['DEC'], 1.0/3600, nearest=True)
            if len(m1) != 1:
                print('Multiple matches within 1 arcsec for {}'.format(onegal['GALAXY_FULL']))
                m1 = m1[0]
            tractor.append(Table(cat)[m1])
            dist.append(d12[0]*3600)
    tractor = vstack(tractor)
    parent = vstack(parent)
    dist = np.array(dist)
    print('Merging {} galaxies took {:.2f} min.'.format(len(tractor), (time.time()-t0)/60.0))

    if len(tractor) == 0:
        print('Something went wrong and no galaxies were fitted.')
        return
    assert(len(tractor) == len(parent))

    # write out
    hdu_primary = fits.PrimaryHDU()
    hdu_parent = fits.convenience.table_to_hdu(parent)
    hdu_parent.header['EXTNAME'] = 'PARENT'
        
    hdu_tractor = fits.convenience.table_to_hdu(tractor)
    hdu_tractor.header['EXTNAME'] = 'TRACTOR'
        
    hx = fits.HDUList([hdu_primary, hdu_parent, hdu_tractor])
    hx.writeto(outfile, overwrite=True, checksum=True)

    print('Wrote {} galaxies to {}'.format(len(parent), outfile))

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
        if tractor.type[indx] == 'PSF' or tractor.shape_r[indx] < 2:
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
        if True:
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
        #plt.clf() ; plt.imshow(centralmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask.png') ; pdb.set_trace()

        iclose = np.where([centralmask[int(by), int(bx)]
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
                   refband='r', bands=['g', 'r', 'i', 'z'], pixscale=0.262,
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

    optbands = bands
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
            'sersic', 'shape_r', 'shape_e1', 'shape_e2']
    for band in bands:
        cols = cols + ['flux_{}'.format(band.lower()), 'flux_ivar_{}'.format(band.lower())]
        cols = cols + ['mw_transmission_{}'.format(band.lower())]
    for band in optbands:
        cols = cols + ['nobs_{}'.format(band.lower()), 'psfdepth_{}'.format(band.lower()),
                       'psfsize_{}'.format(band.lower())]
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
        galaxyinfo = {'refid': (str(galaxy_id), None)}
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
    delta_logsma = 2 # 3.0

    # don't pass logfile and set debug=True because we've already opened the log
    # above!
    mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                     pixscale=pixscale, nproc=nproc, 
                     bands=bands, refband=refband, sbthresh=SBTHRESH,
                     apertures=APERTURES,
                     logsma=True, delta_logsma=delta_logsma, maxsma=maxsma,
                     verbose=verbose, debug=True, clobber=clobber)#debug, logfile=logfile)

def qa_multiwavelength_sed(ellipsefit, tractor=None, png=None, verbose=True):
    """Plot up the multiwavelength SED.

    """
    import matplotlib.pyplot as plt    
    from copy import deepcopy
    import matplotlib.ticker as ticker
    from astropy.table import Table
    from legacyhalos.io import get_run
    from legacyhalos.qa import _sbprofile_colors    
    
    if ellipsefit['success'] is False or np.atleast_1d(ellipsefit['sma_r'])[0] == -1:
        return
    
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
    effwave_north = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4815.95363513, 'r': 6437.79282937, 'i': 7847.78249813, 'z': 9229.65786449}
    effwave_south = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4890.03670428, 'r': 6469.62203811, 'i': 7847.78249813, 'z': 9196.46396394}

    _tt = Table()
    _tt['RA'] = [ellipsefit['ra_moment']]
    _tt['DEC'] = [ellipsefit['dec_moment']]
    run = get_run(_tt)

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
    phot = {'mag_tot': deepcopy(_phot), 'tractor': deepcopy(_phot), 'mag_sb25': deepcopy(_phot)}

    for ifilt, filt in enumerate(bands):
        mtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        if mtot > 0:
            phot['mag_tot']['abmag'][ifilt] = mtot
            phot['mag_tot']['abmagerr'][ifilt] = 0.1
            phot['mag_tot']['lower'][ifilt] = False

        flux = ellipsefit['flux_sb25_{}'.format(filt.lower())]
        ivar = ellipsefit['flux_ivar_sb25_{}'.format(filt.lower())]
        #print(filt, mag)

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

        if tractor is not None:
            flux = tractor['flux_{}'.format(filt.lower())]
            ivar = tractor['flux_ivar_{}'.format(filt.lower())]
            #if filt == 'FUV':
            #    pdb.set_trace()
            if flux > 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(flux)
                phot['tractor']['abmagerr'][ifilt] = 0.1
            if flux <= 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(1/np.sqrt(ivar))
                phot['tractor']['abmagerr'][ifilt] = 0.75
                phot['tractor']['lower'][ifilt] = True

    #print(phot['mag_tot']['abmag'])
    #print(phot['mag_sb25']['abmag'])
    #print(phot['tractor']['abmag'])

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
    #_addphot(phot['mag_sb25'], color='orange', marker='^', alpha=0.7, label=r'$m(r<R_{25})$')
    _addphot(phot['tractor'], color='blue', marker='o', alpha=0.5, label='Tractor')

    #thisphot = phot['tractor']
    #color='blue'
    #marker='o'
    #label='Tractor'

    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                label=label, linestyle='none')
    
    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
    ##ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good],
    ##            marker='o', uplims=thisphot['lower'][good], linestyle='none')
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good][0],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                uplims=thisphot['lower'][good], linestyle='none')#, lolims=True)
                    
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

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xticks([0.1, 0.2, 0.4, 1.0, 3.0, 5.0, 10, 20])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_frmt))

    if smascale:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.85, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.85, right=0.88)
    else:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.95, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.95, right=0.88)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def _get_mags(cat, rad='10', bands=['FUV', 'NUV', 'g', 'r', 'i', 'z', 'W1', 'W2', 'W3', 'W4'],
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
            html.write('<th>Galaxy</th>\n')
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

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], 5 * MOSAICRADIUS / pixscale
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1, dr10=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                #html.write('<td>{}</td>\n'.format(gal['INDEX']))
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

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
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
                    'flux_g', 'flux_r', 'flux_i', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_i', 'flux_ivar_z',
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
                assert(np.all(tractor['ref_id'] == sample[REFIDCOLUMN]))

        return nccds, tractor, sample

    def _html_galaxy_properties(html, gal):
        """Build the table of group properties.

        """
        galaxy1, ra1, dec1, diam1 = gal[GALAXYCOLUMN], gal[racolumn], gal[deccolumn], 5 * MOSAICRADIUS / pixscale
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1, dr10=True)

        html.write('<h2>Galaxy Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Index</th>\n')
        html.write('<th>Galaxy</th>\n')
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
        for filesuffix in ('FUVNUV', 'grz', 'W1W2'):
            pngfile, thumbfile = '{}-custom-montage-{}.png'.format(galaxy1, filesuffix), 'thumb-{}-custom-montage-{}.png'.format(galaxy1, filesuffix)
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
        html.write('<th colspan="10">Tractor</th>\n')
        html.write('<th colspan="10">Curve of Growth</th>\n')
        #html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        #html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>i</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>i</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            fuv, nuv, g, r, i, z, w1, w2, w3, w4 = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(ss[GALAXYCOLUMN]))
            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                fuv, nuv, g, r, i, z, w1, w2, w3, w4))

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
                fuv, nuv, g, r, i, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)
                #try:
                #    fuv, nuv, g, r, i, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)
                #except:
                #    pdb.set_trace()
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                    fuv, nuv, g, r, i, z, w1, w2, w3, w4))
                #g, r, z = _get_mags(ellipse, cog=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...<td>...</td></td><td>...</td><td>...</td><td>...</td>\n')
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
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="100%"></a></td>\n'.format(pngfile, thumbfile))
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
        html.write('<h1>{}</h1>\n'.format(galaxy1))
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

def make_html(sample=None, datadir=None, htmldir=None, bands=['g', 'r', 'i', 'z'],
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
        
    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[plateifusorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1

