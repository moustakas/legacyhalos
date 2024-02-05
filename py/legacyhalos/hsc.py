"""
legacyhalos.hsc
===============

Miscellaneous code pertaining to the project comparing HSC and DECaLS surface
brightness profiles.

"""
import os, time, shutil, pdb
import numpy as np
import astropy
import fitsio

import legacyhalos.io

ZCOLUMN = 'Z_BEST'
RACOLUMN = 'RA'
DECCOLUMN = 'DEC'
DIAMCOLUMN = 'RADIUS_MOSAIC' # [radius, arcsec]
GALAXYCOLUMN = 'ID_S16A'
REFIDCOLUMN = 'ID_S16A'

RADIUS_CLUSTER_KPC = 250.0 # default cluster radius
RADIUS_CLUSTER_LOWZ_KPC = 150.0 # default cluster radius

SBTHRESH = [23.0, 24.0, 25.0, 26.0] # surface brightness thresholds
APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] # multiples of MAJORAXIS

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
        #dependson = '-custom-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-custom-montage-grz.png'
        else:
            filesuffix = '-ccdpos.png'
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
    
def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=int, nargs='*', default=None, help='List of galaxy names to process.')

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
        these = np.in1d(sample[GALAXYCOLUMN], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            sample = sample[these]

    return sample

def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          threshmask=0.001, r50mask=0.1, maxshift=5,
                          sigmamask=3.0, neighborfactor=3.0, verbose=False):
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

        ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
        ba = (1 - ee) / (1 + ee)
        pa = 180 - (-np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2))
        pa = pa % 180

        if tractor.shape_r[indx] < 1:
            print('Galaxy half-light radius is < 1 arcsec!')
            raise ValueError

        majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]

        mgegalaxy = MGEgalaxy()
        mgegalaxy.xmed = tractor.by[indx]
        mgegalaxy.ymed = tractor.bx[indx]
        mgegalaxy.xpeak = tractor.by[indx]
        mgegalaxy.ypeak = tractor.bx[indx]
        mgegalaxy.eps = 1 - ba
        mgegalaxy.pa = pa
        mgegalaxy.theta = (270 - pa) % 180
        mgegalaxy.majoraxis = majoraxis

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
        mge, centralmask = tractor2mge(central, factor=neighborfactor)
        #plt.clf() ; plt.imshow(centralmask, origin='lower') ; plt.savefig('junk-mask.png') ; pdb.set_trace()
        
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
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)

        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('debug.png')
        #plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask.png')
        ##plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
        #pdb.set_trace()

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            print('Large centroid shift! (x,y)=({:.3f},{:.3f})-->({:.3f},{:.3f})'.format(
                mgegalaxy.xmed, mgegalaxy.ymed, mge.xmed, mge.ymed))
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
                print('Warning! All satellites have been dropped from band {}!'.format(filt))
            else:
                satsrcs = srcs.copy()
                #satsrcs = tractor.copy()
                satsrcs.cut(satindx)
                satimg = srcs2image(satsrcs, data['{}_wcs'.format(filt)],
                                    band=filt.lower(),
                                    pixelized_psf=data['{}_psf'.format(filt)])
                thissatmask = satimg > sigmamask*data['{}_sigma'.format(filt.lower())]
                satmask = np.logical_or(satmask, thissatmask)                

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

            # Get the PSF sources.
            psfindx = np.where((tractor.type == 'PSF') * (getattr(tractor, 'flux_{}'.format(filt.lower())) / cenflux > threshmask))[0]
            if len(psfindx) > 0:
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
            var = data['{}_var_'.format(filt)] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #if ii == 1 and filt == 'r': #filt == 'W1' or 
            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('junk-img-{}.png'.format(filt.lower()))
            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask-{}.png'.format(filt.lower()))
            ###    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('junk-thismask-{}.png'.format(filt.lower()))
            #    pdb.set_trace()

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
                   redshift=None, galex=False, unwise=False,
                   fill_value=0.0, sky_tests=False, verbose=False):
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
                            filt2pixscale=filt2pixscale,
                            fill_value=fill_value, verbose=verbose)
    
    # Find the galaxies of interest.
    samplefile = os.path.join(galaxydir, '{}-{}.fits'.format(galaxy, filt2imfile['sample']))
    sample = Table(fitsio.read(samplefile))
    print('Read {} sources from {}'.format(len(sample), samplefile))

    # keep all objects
    galaxy_indx = []
    galaxy_indx = np.hstack([np.where(sid == tractor.ref_id)[0] for sid in sample[REFIDCOLUMN]])
    assert(np.all(sample[REFIDCOLUMN] == tractor.ref_id[galaxy_indx]))
    
    #data['galaxy_indx'] = []
    #data['galaxy_id'] = []
    #for galid in np.atleast_1d(galaxy_id):
    #    galindx = np.where((tractor.ref_cat == 'R1') * (tractor.ref_id == galid))[0]
    #    if len(galindx) != 1:
    #        raise ValueError('Problem finding the central galaxy {} in the tractor catalog!'.format(galid))
    #    data['galaxy_indx'].append(galindx[0])
    #    data['galaxy_id'].append(galid)
    #
    #    # Is the flux and/or ivar negative (and therefore perhaps off the
    #    # footprint?) If so, drop it here.
    #    for filt in bands:
    #        cenflux = getattr(tractor, 'flux_{}'.format(filt))[galindx[0]]
    #        cenivar = getattr(tractor, 'flux_ivar_{}'.format(filt))[galindx[0]]
    #        if cenflux <= 0.0 or cenivar <= 0.0:
    #            print('Central galaxy flux is negative. Off footprint or gap in coverage?')
    #            data['failed'] = True
    #            return data, []

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

    # Gather some additional info that we want propagated to the output ellipse
    # catalogs.
    if redshift:
        allgalaxyinfo = []
        for galaxy_id, galaxy_indx in zip(data['galaxy_id'], data['galaxy_indx']):
            galaxyinfo = { # (value, units) tuple for the FITS header
                'id_cent': (galaxy_id, ''),
                'redshift': (redshift, '')}
            allgalaxyinfo.append(galaxyinfo)
    else:
        allgalaxyinfo = None

    return data, allgalaxyinfo

def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1,
                 filesuffix='custom', bands=['g', 'r', 'z'], refband='r',
                 input_ellipse=None, 
                 sky_tests=False, unwise=False, verbose=False,
                 clobber=False, debug=False, logfile=None):
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

    maxsma, delta_logsma = None, 4
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
                         apertures=APERTURES,
                         delta_logsma=delta_logsma, maxsma=maxsma,
                         input_ellipse=input_ellipse,
                         verbose=verbose, clobber=clobber, debug=debug, logfile=logfile)

def _get_mags(cat, rad='10', bands=['g', 'r', 'z'],
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
            elif ff < 0 and iv > 0:
                # upper limit
                mag = 22.5-2.5*np.log10(1/np.sqrt(iv))
                res.append('>{:.3f}'.format(mag))
            else:
                res.append('...')
    return res

def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn=RACOLUMN, deccolumn=DECCOLUMN, diamcolumn=DIAMCOLUMN,
                   fix_permissions=True):
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

        html.write('<h1>LegacyHalos Demo</h1>\n')
        html.write('<p style="width: 75%">\n')
        html.write("""Central galaxies are neat.</p>\n""")
        
        html.write('<br /><br />\n')
        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th> </th>\n')
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

            ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
            viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1)

            html.write('<tr>\n')
            html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
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
        galaxy1, ra1, dec1, diam1 = gal[GALAXYCOLUMN], gal[racolumn], gal[deccolumn], gal[diamcolumn]
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1)

        html.write('<h2>Galaxy Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
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

        html.write('<p>Color mosaics showing (from left to right) the data, Tractor model, and residuals.</p>\n')
        html.write('<table width="90%">\n')
        for bandsuffix in ['grz']:
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
        html.write('<tr><th></th>\n')
        html.write('<th colspan="3">Tractor</th>\n')
        html.write('<th colspan="3">Curve of Growth</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            g, r, z = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(ss[GALAXYCOLUMN]))
            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                        galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):
                g, r, z = _get_mags(ellipse, cog=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            html.write('<h4>{}</h4>\n'.format(sample[GALAXYCOLUMN][igal]))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                                     galaxy_id=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue

            html.write('<table width="90%">\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="80%"></a></td>\n'.format(pngfile, thumbfile))
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
        html.write('<h1>Galaxy {}</h1>\n'.format(galaxy1))
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
              racolumn=RACOLUMN, deccolumn=DECCOLUMN, diamcolumn=DIAMCOLUMN,
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=False,
              clobber=False, verbose=True, ccdqa=False,
              mpiargs=None, fix_permissions=True):
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
    _, _, done, _ = missing_files(mpiargs, sample)
    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return
    
    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    htmlhome = 'index.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   fix_permissions=fix_permissions)

    # Now build the individual pages in parallel.
    galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)
        
    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample, np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1
