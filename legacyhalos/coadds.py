"""
legacyhalos.coadds
==================

Code to generate grzW1W2 coadds.

Note:
 * We do not yet build unWISE coadds.
 * The code currently only supports DR5 data (e.g., we assume the DECam pixel
   scale).
 * The code will not handle central galaxies that span more than one brick.  We
   should define custom bricks and generate custom Tractor catalogs, which would
   also remove much of the DR5 dependency.

python -u legacyanalysis/extract-calibs.py --drdir /project/projectdirs/cosmo/data/legacysurvey/dr5 --radec 342.4942 -0.6706 --width 300 --height 300

"""
from __future__ import absolute_import, division, print_function

import os
import shutil
import numpy as np

def custom_brickname(galaxycat, prefix='custom-'):
    brickname = 'custom-{:06d}{}{:05d}'.format(
        int(1000*galaxycat['ra']), 'm' if galaxycat['dec'] < 0 else 'p',
        int(1000*np.abs(galaxycat['dec'])))
    return brickname

def cutout_radius_100kpc(redshift, pixscale=0.262, radius_kpc=100):
    """Get a cutout radius of 100 kpc [in pixels] at the redshift of the cluster.

    """
    from astropy.cosmology import WMAP9 as cosmo
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
    radius = np.rint(radius_kpc * arcsec_per_kpc / pixscale).astype(int) # [pixels]
    return radius

def cutout_radius_cluster(redshift, cluster_radius, pixscale=0.262, factor=1.0,
                          rmin=50, rmax=500, bound=False):
    """Get a cutout radius which depends on the richness radius (in h^-1 Mpc)
    R_LAMBDA of each cluster (times an optional fudge factor).

    Optionally bound the radius to (rmin, rmax).

    """
    from astropy.cosmology import WMAP9 as cosmo

    radius_kpc = cluster_radius * 1e3 * cosmo.h # cluster radius in kpc
    radius = np.rint(factor * radius_kpc * cosmo.arcsec_per_kpc_proper(redshift).value / pixscale)

    if bound:
        radius[radius < rmin] = rmin
        radius[radius > rmax] = rmax

    return radius

def _custom_brick(galaxycat, objid, survey=None, radius=100, ncpu=1, pixscale=0.262):
    """Run legacypipe on a custom "brick" centered on the central.

    """
    from legacypipe.runbrick import run_brick

    run_brick(None, survey, radec=(galaxycat['ra'], galaxycat['dec']), pixscale=pixscale,
              width=2*radius, height=2*radius, threads=ncpu, normalizePsf=True,
              do_calibs=False, wise=False, depth_cut=False, splinesky=True,
              early_coadds=False, pixPsf=True, hybridPsf=True, ceres=False,
              rex=True, forceall=True, write_pickles=False, write_metrics=False,
              gaia_stars=True, stages=['writecat'])

    # Move (rename) files into the desired output directory and clean up.
    brickname = custom_brickname(galaxycat, prefix='custom-')
    oldfile = os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname))
    newfile = os.path.join(survey.output_dir, '{}-tractor.fits'.format(objid))
    shutil.copy(oldfile, newfile)

    oldfile = os.path.join(survey.output_dir, 'coadd', 'cus', brickname, 'legacysurvey-{}-ccds.fits'.format(brickname))
    newfile = os.path.join(survey.output_dir, '{}-ccds.fits'.format(objid))
    shutil.copy(oldfile, newfile)

    if True:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

def _coadds_stage_tims(galaxycat, survey=None, mp=None, radius=100,
                       brickname=None, pixscale=0.262, custom=False):
    """Initialize the first step of the pipeline, returning
    a dictionary with the following keys:
    
    ['brickid', 'target_extent', 'version_header', 
     'targetrd', 'brickname', 'pixscale', 'bands', 
     'survey', 'brick', 'ps', 'H', 'ccds', 'W', 
     'targetwcs', 'tims']

    """
    from legacypipe.runbrick import stage_tims

    if custom:
        P = stage_tims(ra=galaxycat['ra'], dec=galaxycat['dec'], brickname=brickname,
                       survey=survey, W=2*radius, H=2*radius, pixscale=pixscale,
                       mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                       depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                       splinesky=True)
    else:
        bbox = [galaxycat['bx']-radius, galaxycat['bx']+radius,
                galaxycat['by']-radius, galaxycat['by']+radius]
        P = stage_tims(brickname=galaxycat['brickname'], survey=survey, target_extent=bbox,
                       pixscale=pixscale, mp=mp, normalizePsf=True, hybridPsf=True,
                       depth_cut=False, apodize=False, do_calibs=False,
                       splinesky=True)
    return P

def _read_tractor(galaxycat, objid=None, targetwcs=None, survey=None, 
                  custom=False, verbose=False):
    """Read the full Tractor catalog for a given brick 
    and remove the BCG.
    
    """
    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec

    # Read the newly-generated Tractor catalog
    if custom:
        fn = os.path.join(survey.output_dir, '{}-tractor.fits'.format(objid))
        cat = fits_table(fn)
        if verbose:
            print('Read {} sources from {}'.format(len(cat), fn))

        # Find and remove the central.  For some reason, match_radec
        # occassionally returns two matches, even though nearest=True.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, galaxycat['ra'], galaxycat['dec'],
                                  1/3600.0, nearest=True)
        if len(d12) == 0:
            print('No matching central found -- definitely a problem.')
        elif len(d12) > 1:
            m1 = m1[np.argmin(d12)]
            
        if verbose:
            print('Removed central galaxy with objid = {}'.format(cat[m1].objid))

        cat.cut( ~np.in1d(cat.objid, m1) )
    else:
        # Read the full Tractor catalog.
        fn = survey.find_file('tractor', brick=galaxycat['brickname'])
        cat = fits_table(fn)
        if verbose:
            print('Read {} sources from {}'.format(len(cat), fn))

        # Restrict to just the sources in our little box. 
        H, W = targetwcs.shape
        ok, xx, yy = targetwcs.radec2pixelxy(cat.ra, cat.dec)
        cat.cut( np.flatnonzero((xx > 0) * (xx < W) * (yy > 0) * (yy < H)) )
        if verbose:
            print('Cut to {} sources within our box.'.format(len(cat)))

        # Remove the central galaxy.
        cat.cut( np.flatnonzero(cat.objid != galaxycat['objid']) )
        if verbose:
            print('Removed central galaxy with objid = {}'.format(galaxycat['objid']))
        
    return cat

def _build_model_image(cat, tims, survey=None, verbose=False):
    """Generate a model image by rendering each source.
    
    """
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.runbrick import _get_mod
    
    if verbose:
        print('Creating tractor sources...')
    srcs = read_fits_catalog(cat, fluxPrefix='')
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    if verbose:
        print('Rendering model images...')
    mods = [_get_mod((tim, srcs)) for tim in tims]

    return mods

def _tractor_coadds(galaxycat, targetwcs, tims, mods, version_header, objid=None,
                    brickname=None, survey=None, mp=None, verbose=False,
                    bands=['g','r','z']):
    """Generate individual-band FITS and color coadds for each central using
    Tractor.

    """
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.runbrick import rgbkwargs, rgbkwargs_resid
    from legacypipe.survey import get_rgb, imsave_jpeg

    if brickname is None:
        brickname = galaxycat['brickname']

    if verbose:
        print('Producing coadds...')
    C = make_coadds(tims, bands, targetwcs, mods=mods, mp=mp,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, 
                                   tims, targetwcs)
                    )
    
    # Move (rename) the coadds into the desired output directory.
    for suffix in ('chi2', 'image', 'invvar', 'model'):
        for band in bands:
            oldfile = os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band))
            newfile = os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(objid, suffix, band))
            shutil.copy(oldfile, newfile)

    if True:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
    
    # Build png postage stamps of the coadds.
    coadd_list = [('image', C.coimgs,   rgbkwargs),
                  ('model', C.comods,   rgbkwargs),
                  ('resid', C.coresids, rgbkwargs_resid)]

    for name, ims, rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        outfn = os.path.join(survey.output_dir, '{}-{}.jpg'.format(objid, name))
        if verbose:
            print('Writing {}'.format(outfn))
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb

def legacyhalos_coadds(galaxycat, survey=None, objid=None, objdir=None,
                       ncpu=1, pixscale=0.262, cluster_radius=False,
                       verbose=False):
    """Top-level wrapper script to generate coadds for a single galaxy.

    """ 
    from astrometry.util.multiproc import multiproc
    mp = multiproc(nthreads=ncpu)

    if objid is None and objdir is None:
        objid, objdir = get_objid(galaxycat)

    survey.output_dir = objdir

    # Step 0 - Get the cutout radius.
    if cluster_radius:
        radius = cutout_radius_cluster(redshift=galaxycat['z'], pixscale=pixscale,
                                        cluster_radius=galaxycat['r_lambda'])
    else:
        radius = cutout_radius_100kpc(redshift=galaxycat['z'], pixscale=pixscale)

    # Step 1 - Set up the first stage of the pipeline.
    P = _coadds_stage_tims(galaxycat, survey=survey, mp=mp, radius=radius,
                           pixscale=pixscale)

    # Step 2 - Read the Tractor catalog for this brick and remove the central.
    cat = _read_tractor(galaxycat, P['targetwcs'], survey=survey, verbose=verbose)

    # Step 3 - Render the model images without the central.
    mods = _build_model_image(cat, tims=P['tims'], survey=survey, verbose=verbose)

    # Step 4 - Generate and write out the coadds.
    _tractor_coadds(galaxycat, P['targetwcs'], P['tims'], mods, P['version_header'],
                    objid=objid, survey=survey, mp=mp, verbose=verbose)

    return 1 # success!

def legacyhalos_custom_coadds(galaxycat, survey=None, objid=None, objdir=None,
                              ncpu=1, pixscale=0.262, cluster_radius=False,
                              verbose=False):
    """Top-level wrapper script to generate coadds for a single galaxy.

    """ 
    from astrometry.util.multiproc import multiproc
    mp = multiproc(nthreads=ncpu)
    
    if objid is None and objdir is None:
        objid, objdir = get_objid(galaxycat)
    brickname = custom_brickname(galaxycat, prefix='')

    survey.output_dir = objdir

    # Step 0 - Get the cutout radius.
    if cluster_radius:
        radius = cutout_radius_cluster(redshift=galaxycat['z'], pixscale=pixscale,
                                       cluster_radius=galaxycat['r_lambda'])
    else:
        radius = cutout_radius_100kpc(redshift=galaxycat['z'], pixscale=pixscale)

    # Step 1 - Run legacypipe to generate a custom "brick" and tractor catalog
    # centered on the central.
    _custom_brick(galaxycat, objid=objid, survey=survey, radius=radius,
                  ncpu=ncpu, pixscale=pixscale)

    # Step 2 - Read the Tractor catalog for this brick and remove the central.
    cat = _read_tractor(galaxycat, objid=objid, survey=survey, custom=True,
                        verbose=verbose)

    # Step 3 - Set up the first stage of the pipeline.
    P = _coadds_stage_tims(galaxycat, survey=survey, mp=mp, radius=radius,
                           brickname=brickname, pixscale=pixscale, custom=True)

    # Step 4 - Render the model images without the central.
    mods = _build_model_image(cat, tims=P['tims'], survey=survey, verbose=verbose)
    
    # Step 3 - Generate and write out the coadds.
    _tractor_coadds(galaxycat, P['targetwcs'], P['tims'], mods, P['version_header'],
                    objid=objid, brickname=brickname, survey=survey, mp=mp,
                    verbose=verbose)

    return 1
