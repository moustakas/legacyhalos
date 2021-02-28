"""
legacyhalos.hsc
===============

Miscellaneous code pertaining to the project comparing HSC and DECaLS surface
brightness profiles.

"""
import os, time, shutil, subprocess, pdb
import numpy as np
import astropy
import fitsio

import legacyhalos.io

ZCOLUMN = 'Z_BEST'
RACOLUMN = 'RA'
DECCOLUMN = 'DEC'
DIAMCOLUMN = 'RADIUS_MOSAIC' # [radius, arcsec]
GALAXYCOLUMN = 'ID_S16A'

RADIUS_CLUSTER_KPC = 250.0 # default cluster radius
RADIUS_CLUSTER_LOWZ_KPC = 150.0 # default cluster radius

SBTHRESH = [23.0, 24.0, 25.0, 26.0] # surface brightness thresholds

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    #import healpy as hp
    from legacyhalos.misc import radec2pix
    
    #nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if not 'ID_S16A' in cat.colnames:
        # need to handle the lowz catalog
        print('Missing ID_S16A and NAME in catalog!')
        raise ValueError()

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = ['{:017d}'.format(cat[GALAXYCOLUMN])]
        subdir = [galaxy[0][:4]]
        #pixnum = [radec2pix(nside, cat[RACOLUMN], cat[DECCOLUMN])]
    else:
        ngal = len(cat)
        galaxy = np.array(['{:017d}'.format(gid) for gid in cat[GALAXYCOLUMN]])
        subdir = [gal[:4] for gal in galaxy]
        #pixnum = radec2pix(nside, cat[RACOLUMN], cat[DECCOLUMN]).data
        
    galaxydir = np.array([os.path.join(datadir, sdir, gal) for sdir, gal in zip(subdir, galaxy)])
    #galaxydir = np.array([os.path.join(datadir, '{}'.format(nside), '{}'.format(pix), gal)
    #                      for pix, gal in zip(pixnum, gal)])

    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, sdir, gal) for sdir, gal in zip(subdir, galaxy)])
        #htmlgalaxydir = np.array([os.path.join(htmldir, '{}'.format(nside), '{}'.format(pix), gal)
        #                          for pix, gal in zip(pixnum, galaxy)])

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
        filesuffix = '-custom-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        #filesuffix = '-custom-skytest03-*-ellipse.fits'
        filesuffix = '-custom-ellipse.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.build_catalog:
        suffix = 'build-catalog'
        filesuffix = '-custom-catalog.isdone'
        dependson = '-custom-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-custom-grz-montage.png'
        else:
            filesuffix = '-ccdpos.png'
            #filesuffix = '-custom-maskbits.png'
            #dependson = '-custom-ellipse.isdone'
            dependson = None
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-custom-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
        #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        raise ValueError('Need at least one keyword argument.')

    # Make clobber=False for build_catalog and htmlindex because we're not
    # making the files here, we're just looking for them. The argument
    # args.clobber gets used downstream.
    if args.htmlindex:
        clobber = False
    elif args.build_catalog:
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

        # Assign the sample to ranks to make the diameter distribution per rank ~flat.

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

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the custom coadds.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipelinecoadds.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--integrate', action='store_true', help='Integrate the surface brightness profiles.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    
    parser.add_argument('--unwise', action='store_true', help='Build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')
    parser.add_argument('--sky-tests', action='store_true', help='Test choice of sky apertures in ellipse-fitting.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final photometric catalog.')
    args = parser.parse_args()

    return args

def legacyhsc_cosmology(WMAP=False, Planck=False):
    """Establish the default cosmology for the project."""

    if WMAP:
        from astropy.cosmology import WMAP9 as cosmo
    elif Planck:
        from astropy.cosmology import Planck15 as cosmo
    else:
        from astropy.cosmology import FlatLambdaCDM
        #params = dict(H0=70, Om0=0.3, Ob0=0.0457, Tcmb0=2.7255, Neff=3.046)
        #sigma8 = 0.82
        #ns = 0.96
        #cosmo = FlatLambdaCDM(name='FlatLambdaCDM', **params)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, name='FlatLambdaCDM')

    return cosmo

def cutout_radius_kpc(redshift, pixscale=None, radius_kpc=RADIUS_CLUSTER_KPC, cosmo=None):
    """Get a cutout radius of RADIUS_KPC [in pixels] at the redshift of the cluster.

    """
    if cosmo is None:
        cosmo = legacyhsc_cosmology()
        
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
    radius = radius_kpc * arcsec_per_kpc # [float arcsec]
    if pixscale:
        radius = np.rint(radius / pixscale).astype(int) # [integer/rounded pixels]
        
    return radius

def get_integrated_filename():
    """Return the name of the file containing the integrated photometry."""
    integratedfile = os.path.join(hsc_dir(), 'integrated-flux.fits')
    return integratedfile

def read_sample(first=None, last=None, galaxylist=None, verbose=False):
    """Read/generate the parent HSC sample by combining the low-z and intermediate-z
    samples.

    from astropy.table import Table, Column, vstack
    s1 = Table(fitsio.read('low-z-shape-for-john.fits', upper=True))
    s2 = Table(fitsio.read('s16a_massive_z_0.5_logm_11.4_decals_full_fdfc_bsm_ell.fits', upper=True))

    s1out = s1['NAME', 'RA', 'DEC', 'Z', 'MEAN_E', 'MEAN_PA']
    s1out.rename_column('Z', 'Z_BEST')
    s1out.add_column(Column(name='ID_S16A', dtype=s2['ID_S16A'].dtype, length=len(s1out)), index=1)
    s1out['ID_S16A'] = -1
    s2out = s2['ID_S16A', 'RA', 'DEC', 'Z_BEST', 'MEAN_E', 'MEAN_PA']
    s2out.add_column(Column(name='NAME', dtype=s1['NAME'].dtype, length=len(s2out)), index=0)
    sout = vstack((s1out, s2out))
    sout.write('hsc-sample-s16a-lowz.fits', overwrite=True)

    """
    cosmo = legacyhsc_cosmology()
    
    # Hack for MUSE proposal
    #samplefile = os.path.join(hdir, 's18a_z0.07_0.12_rcmod_18.5_etg_muse_massive_0313.fits')
    
    # intermediate-z sample only
    samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 's16a_massive_z_0.5_logm_11.4_decals_full_fdfc_bsm_ell.fits')
    #samplefile = os.path.join(hdir, 's16a_massive_z_0.5_logm_11.4_dec_30_for_john.fits')

    # low-z sample only
    #samplefile = os.path.join(hdir, 'low-z-shape-for-john.fits')

    # Investigate a subset of galaxies.
    #cat1 = fitsio.read(os.path.join(hdir, 'hsc-sample-s16a-lowz.fits'), upper=True)
    #cat2 = fitsio.read(os.path.join(hdir, 'DECaLS_negative_gal.fits'), upper=True)
    #keep = np.isin(cat1['ID_S16A'], cat2['ID_S16A'])
    #fitsio.write(os.path.join(hdir, 'temp-hsc-sample-s16a-lowz.fits'), cat1[keep], clobber=True)

    # combined sample (see comment block above)
    #if False:
    #print('Temporary sample!!')
    #samplefile = os.path.join(hdir, 'temp-hsc-sample-s16a-lowz.fits')
    #samplefile = os.path.join(hdir, 'hsc-sample-s16a-lowz.fits')
    
    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # select a "test" subset
    if False:
        nrows = 200
        rows = np.arange(nrows)
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

    # pre-compute the diameter of the mosaic (=2*RADIUS_CLUSTER_KPC kpc) for each cluster
    sample[DIAMCOLUMN] = cutout_radius_kpc(redshift=sample[ZCOLUMN], cosmo=cosmo, # diameter, [arcsec]
                                           radius_kpc=2 * RADIUS_CLUSTER_KPC) 

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
                          threshmask=0.001, r50mask=0.1, maxshift=5,
                          verbose=False):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    r50mask - mask satellites whose r50 radius (arcsec) is > r50mask 

    threshmask - mask satellites whose flux ratio is > threshmmask relative to
    the central galaxy.

    """
    import numpy.ma as ma
    from copy import copy
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

    # Get the PSF sources.
    psfindx = np.where(tractor.type == 'PSF')[0]
    if len(psfindx) > 0:
        psfsrcs = tractor.copy()
        psfsrcs.cut(psfindx)
    else:
        psfsrcs = None

    def tractor2mge(indx, factor=1.0):
    #def tractor2mge(indx, majoraxis=None):
        # Convert a Tractor catalog entry to an MGE object.
        class MGEgalaxy(object):
            pass

        ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
        ba = (1 - ee) / (1 + ee)
        pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2))
        pa = pa % 180

        mgegalaxy = MGEgalaxy()
        mgegalaxy.xmed = tractor.by[indx]
        mgegalaxy.ymed = tractor.bx[indx]
        mgegalaxy.xpeak = tractor.by[indx]
        mgegalaxy.ypeak = tractor.bx[indx]
        mgegalaxy.eps = 1-ba
        mgegalaxy.pa = pa
        mgegalaxy.theta = (270 - pa) % 180
        mgegalaxy.majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]

        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                               mgegalaxy.majoraxis,
                               mgegalaxy.majoraxis * (1-mgegalaxy.eps), 
                               np.radians(mgegalaxy.theta-90), xobj, yobj)

        return mgegalaxy, objmask

    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        print('Determing the geometry for galaxy {}/{}.'.format(
                ii+1, len(galaxy_indx)))

        # [1] Determine the non-parametricc geometry of the galaxy of interest
        # in the reference band. First, subtract all models except the galaxy
        # and galaxies "near" it. Also restore the original pixels of the
        # central in case there was a poor deblend.
        largeshift = False
        mge, centralmask = tractor2mge(central, factor=5.0)

        iclose = np.where([centralmask[np.int(by), np.int(bx)]
                           for by, bx in zip(tractor.by, tractor.bx)])[0]
        
        srcs = tractor.copy()
        srcs.cut(np.delete(np.arange(len(tractor)), iclose))
        model = srcs2image(srcs, data['{}_wcs'.format(refband)],
                           band=refband.lower(),
                           pixelized_psf=data['{}_psf'.format(refband)])

        img = data[refband].data - model
        img[centralmask] = data[refband].data[centralmask]

        mask = np.logical_or(ma.getmask(data[refband]), data['residual_mask'])
        #mask = np.logical_or(data[refband].mask, data['residual_mask'])
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)

        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('debug.png')
        #if True:
        #    import matplotlib.pyplot as plt
        #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('debug.png')
        #    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #    pdb.set_trace()
        #pdb.set_trace()

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            print('Large centroid shift! (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                mgegalaxy.xmed, mgegalaxy.ymed, mge.xmed, mge.ymed))
            largeshift = True
            mgegalaxy = copy(mge)

        radec_med = data['{}_wcs'.format(refband)].pixelToPosition(
            mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        radec_peak = data['{}_wcs'.format(refband)].pixelToPosition(
            mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {
            'largeshift': largeshift,
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

        if False:
            #plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
            plt.clf() ; mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=True, plot=True)
            plt.savefig('/mnt/legacyhalos-data/debug.png')

        # [2] Create the satellite mask in all the bandpasses. Use srcs here,
        # which has had the satellites nearest to the central galaxy trimmed
        # out.
        print('Building the satellite mask.')
        satmask = np.zeros(data[refband].shape, bool)
        for filt in bands:
            cenflux = getattr(tractor, 'flux_{}'.format(filt))[central]
            satflux = getattr(srcs, 'flux_{}'.format(filt))
            if cenflux <= 0.0:
                raise ValueError('Central galaxy flux is negative!')
            
            satindx = np.where(np.logical_or(
                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
                srcs.ref_cat == 'R1'))[0]
            #if np.isin(central, satindx):
            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
            if len(satindx) == 0:
                raise ValueError('All satellites have been dropped!')

            satsrcs = srcs.copy()
            #satsrcs = tractor.copy()
            satsrcs.cut(satindx)
            satimg = srcs2image(satsrcs, data['{}_wcs'.format(filt)],
                                band=filt.lower(),
                                pixelized_psf=data['{}_psf'.format(filt)])
            satmask = np.logical_or(satmask, satimg > 10*data['{}_sigma'.format(filt)])
            #if True:
            #    import matplotlib.pyplot as plt
            #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('debug.png')
            #    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
            #    pdb.set_trace()

        # [3] Build the final image (in each filter) for ellipse-fitting. First,
        # subtract out the PSF sources. Then update the mask (but ignore the
        # residual mask). Finally convert to surface brightness.
        for filt in bands:
            mask = np.logical_or(ma.getmask(data[filt]), satmask)
            mask[centralmask] = False
            #plt.imshow(mask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')

            varkey = '{}_var'.format(filt)
            imagekey = '{}_masked'.format(filt)
            psfimgkey = '{}_psfimg'.format(filt)
            thispixscale = filt2pixscale[filt]
            if imagekey not in data.keys():
                data[imagekey], data[varkey], data[psfimgkey] = [], [], []

            img = ma.getdata(data[filt]).copy()
            if psfsrcs:
                psfimg = srcs2image(psfsrcs, data['{}_wcs'.format(filt)],
                                    band=filt.lower(),
                                    pixelized_psf=data['{}_psf'.format(filt)])
                #data[psfimgkey].append(psfimg)
                img -= psfimg

            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
            var = data['{}_var_'.format(filt)] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)

            data[imagekey].append(img)
            data[varkey].append(var)

        #test = data['r_masked'][0]
        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #pdb.set_trace()

    # Cleanup?
    for filt in bands:
        del data[filt]
        del data['{}_var_'.format(filt)]

    return data            

def read_multiband(galaxy, galaxydir, galaxy_id, filesuffix='custom',
                   refband='r', bands=['g', 'r', 'z'], pixscale=0.262,
                   redshift=None, fill_value=0.0, sky_tests=False,
                   verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
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
        return data, None

    # Pack some preliminary info into the output dictionary.
    data['filesuffix'] = filesuffix
    data['bands'] = bands
    data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)
    data['failed'] = False # be optimistic!

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
                            fill_value=fill_value, verbose=verbose)
    
    # Find the central.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    sample = Table(fitsio.read(samplefile))
    print('Read {} sources from {}'.format(len(sample), samplefile))

    data['galaxy_indx'] = []
    data['galaxy_id'] = []
    for galid in np.atleast_1d(galaxy_id):
        galindx = np.where((tractor.ref_cat == 'R1') * (tractor.ref_id == galid))[0]
        if len(galindx) != 1:
            raise ValueError('Problem finding the central galaxy {} in the tractor catalog!'.format(galid))
        data['galaxy_indx'].append(galindx[0])
        data['galaxy_id'].append(galid)

        # Is the flux and/or ivar negative (and therefore perhaps off the
        # footprint?) If so, drop it here.
        for filt in bands:
            cenflux = getattr(tractor, 'flux_{}'.format(filt))[galindx[0]]
            cenivar = getattr(tractor, 'flux_ivar_{}'.format(filt))[galindx[0]]
            if cenflux <= 0.0 or cenivar <= 0.0:
                print('Central galaxy flux is negative. Off footprint or gap in coverage?')
                data['failed'] = True
                return data, []

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
    if redshift:
        allgalaxyinfo = []
        for galaxy_id, galaxy_indx in zip(data['galaxy_id'], data['galaxy_indx']):
            galaxyinfo = { # (value, units) tuple for the FITS header
                'id_cent': (str(galaxy_id), ''),
                'redshift': (redshift, '')}
            allgalaxyinfo.append(galaxyinfo)
    else:
        allgalaxyinfo = None

    return data, allgalaxyinfo

def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1,
                 filesuffix='custom', bands=['g', 'r', 'z'], refband='r',
                 input_ellipse=None, 
                 sky_tests=False, unwise=False, verbose=False,
                 debug=False, logfile=None):
    """Wrapper on legacyhalos.mpi.call_ellipse but with specific preparatory work
    and hooks for the legacyhalos project.

    """
    import astropy.table
    from copy import deepcopy
    from legacyhalos.mpi import call_ellipse as mpi_call_ellipse

    if type(onegal) == astropy.table.Table:
        onegal = onegal[0] # create a Row object
    galaxy_id = onegal[GALAXYCOLUMN]

    if logfile:
        from contextlib import redirect_stdout, redirect_stderr
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id, bands=bands,
                                                  filesuffix=filesuffix, refband=refband,
                                                  pixscale=pixscale, redshift=onegal[ZCOLUMN],
                                                  sky_tests=sky_tests, verbose=verbose)
    else:
        data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id, bands=bands,
                                          filesuffix=filesuffix, refband=refband,
                                          pixscale=pixscale, redshift=onegal[ZCOLUMN],
                                          sky_tests=sky_tests, verbose=verbose)

    maxsma, delta_logsma = None, 6
    #maxsma, delta_logsma = 200, 10

    if sky_tests:
        from legacyhalos.mpi import _done

        def _wrap_call_ellipse():
            skydata = deepcopy(data) # necessary?
            for isky in np.arange(len(data['sky'])):
                skydata['filesuffix'] = data['sky'][isky]['skysuffix']
                for band in bands:
                    # We want to *add* this delta-sky because it is defined as
                    #   sky_annulus_0 - sky_annulus_isky
                    delta_sky = data['sky'][isky][band] 
                    print('  Adjusting {}-band sky backgroud by {:4g} nanomaggies.'.format(band, delta_sky))
                    for igal in np.arange(len(np.atleast_1d(data['galaxy_indx']))):
                        skydata['{}_masked'.format(band)][igal] = data['{}_masked'.format(band)][igal] + delta_sky

                err = mpi_call_ellipse(galaxy, galaxydir, skydata, galaxyinfo=galaxyinfo,
                                       pixscale=pixscale, nproc=nproc, 
                                       bands=bands, refband=refband, sbthresh=SBTHRESH,
                                       delta_logsma=delta_logsma, maxsma=maxsma,
                                       write_donefile=False,
                                       input_ellipse=input_ellipse,
                                       verbose=verbose, debug=True)#, logfile=logfile)# no logfile and debug=True, otherwise this will crash

                # no need to redo the nominal ellipse-fitting
                if isky == 0:
                    inellipsefile = os.path.join(galaxydir, '{}-{}-{}-ellipse.fits'.format(galaxy, skydata['filesuffix'], galaxy_id))
                    outellipsefile = os.path.join(galaxydir, '{}-{}-{}-ellipse.fits'.format(galaxy, data['filesuffix'], galaxy_id))
                    print('Copying {} --> {}'.format(inellipsefile, outellipsefile))
                    shutil.copy2(inellipsefile, outellipsefile)
            return err

        t0 = time.time()
        if logfile:
            with open(logfile, 'a') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    # Capture corner case of missing data / incomplete coverage (see also
                    # ellipse.legacyhalos_ellipse).
                    if bool(data):
                        if data['failed']:
                            err = 1
                        else:
                            err = _wrap_call_ellipse()
                    else:
                        if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, filesuffix))):
                            err = 1 # successful failure
                        else:
                            err = 0 # failed failure
                    _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix, log=log)
        else:
            log = None
            if bool(data):
                if data['failed']:
                    err = 1
                else:
                    err = _wrap_call_ellipse()
            else:
                if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, filesuffix))):
                    err = 1 # successful failure
                else:
                    err = 0 # failed failure
            _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix, log=log)
    else:
        mpi_call_ellipse(galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                         pixscale=pixscale, nproc=nproc, 
                         bands=bands, refband=refband, sbthresh=SBTHRESH,
                         delta_logsma=delta_logsma, maxsma=maxsma,
                         input_ellipse=input_ellipse,
                         verbose=verbose, debug=debug, logfile=logfile)

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              first=None, last=None, nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
    import subprocess
    import fitsio

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width
    from legacyhalos.misc import cutout_radius_kpc
    from legacyhalos.misc import HSC_RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    datadir = hsc_data_dir()
    htmldir = hsc_html_dir()

    if sample is None:
        sample = read_sample(first=first, last=last)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    # Write the last-updated date to a webpage.
    js = legacyhalos.html._javastring()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * cutout_radius_kpc(radius_kpc=radius_mosaic_kpc, redshift=gal[zcolumn],
                                      pixscale=pixscale) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8'.format(
            baseurl, gal['RA'], gal['DEC'], zoom)
        
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
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>HSC Massive Galaxies</h1>\n')
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        #html.write('<th>Richness</th>\n')
        #html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')
        for ii, (gal, galaxy1, htmlgalaxydir1) in enumerate(zip(
            sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir) )):

            htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
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

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    # Make a separate HTML page for each object.
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate( zip(
        sample, np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir) ) ):

        radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
            redshift=gal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

        ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
        #if 'psfdepth_g' not in ellipse.keys():
        #    pdb.set_trace()
        pipeline_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose,
                                                          filesuffix='pipeline')
        
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

            html.write('<h1>HSC Galaxy {}</h1>\n'.format(galaxy1))

            html.write('<a href="../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            #html.write('<th>Richness</th>\n')
            #html.write('<th>Pcen</th>\n')
            html.write('<th>Viewer</th>\n')
            #html.write('<th>SkyServer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Image Mosaics</h2>\n')

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

            html.write('<p>(Left) data, (middle) model of every object in the field except the central galaxy, (right) residual image containing just the central galaxy.</p>\n')
            #html.write('<br />\n')
            
            html.write('<table width="90%">\n')
            pngfile = '{}-grz-montage.png'.format(galaxy1)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            html.write('<p>Spatial distribution of CCDs.</p>\n')

            html.write('<table width="90%">\n')
            pngfile = '{}-ccdpos.png'.format(galaxy1)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            html.write('</table>\n')
            #html.write('<br />\n')

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

            html.write('<a href="../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
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

