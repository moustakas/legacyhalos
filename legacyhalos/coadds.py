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

import os, pdb
import shutil
import numpy as np

def custom_brickname(galaxycat, prefix='custom-'):
    brickname = 'custom-{:06d}{}{:05d}'.format(
        int(1000*galaxycat['ra']), 'm' if galaxycat['dec'] < 0 else 'p',
        int(1000*np.abs(galaxycat['dec'])))
    return brickname

def cutout_radius_100kpc(redshift, pixscale=0.262, radius_kpc=150):
    """Get a cutout radius of 150 kpc [in pixels] at the redshift of the cluster.

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

def _custom_brick(galaxycat, objid, survey=None, radius=100, ncpu=1,
                  pixscale=0.262, log=None):
    """Run legacypipe on a custom "brick" centered on the central.

    """
    import subprocess

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} --unwise-coadds --skip-metrics '
    #cmd += '--force-stage coadds '
    #cmd += '--force-all '
    cmd += '--write-stage srcs --no-write --skip --skip-calibs --no-wise-ceres '
    cmd += '--checkpoint {outdir}/checkpoint-{objid}.pickle '
    cmd += '--pickle "{outdir}/{objid}-%%(stage)s.pickle" '
    
    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'),
                     ra=galaxycat['ra'], dec=galaxycat['dec'],
                     width=2*radius, pixscale=pixscale,
                     threads=ncpu, outdir=survey.output_dir,
                     objid=objid)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)

    #from legacypipe.runbrick import run_brick
    #run_brick(None, survey, radec=(galaxycat['ra'], galaxycat['dec']), pixscale=pixscale,
    #          width=2*radius, height=2*radius, threads=ncpu, normalizePsf=True,
    #          do_calibs=False, wise=True, depth_cut=False, splinesky=True,
    #          early_coadds=True, pixPsf=True, hybridPsf=True, ceres=False, wise_ceres=False,
    #          rex=True, forceall=True, write_pickles=True, write_metrics=False,
    #          gaia_stars=True, stages=['writecat'])

    # Move (rename) files into the desired output directory and clean up.
    brickname = custom_brickname(galaxycat, prefix='custom-')

    # tractor catalog
    shutil.copy(
        os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
        os.path.join(survey.output_dir, '{}-tractor.fits'.format(objid))
        )

    # data and model images
    for band in ('g', 'r', 'z'):
        for imtype in ('image', 'model', 'invvar'):
            shutil.copy(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(objid, imtype, band))
                )
    for band in ('W1', 'W2'):
        for imtype in ('image', 'model'):
            shutil.copy(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(objid, imtype, band))
                )

    # jpg images
    for imtype in ('image', 'model', 'resid', 'wise', 'wisemodel'):
        shutil.copy(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
            os.path.join(survey.output_dir, '{}-{}.jpg'.format(objid, imtype))
            )

    # CCDs, maskbits, and depth images
    shutil.copy(
        os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-ccds.fits'.format(brickname)),
        os.path.join(survey.output_dir, '{}-ccds.fits'.format(objid))
        )
    shutil.copy(
        os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-maskbits.fits.gz'.format(brickname)),
        os.path.join(survey.output_dir, '{}-maskbits.fits.gz'.format(objid))
        )
    for band in ('g', 'r', 'z'):
        shutil.copy(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
            os.path.join(survey.output_dir, '{}-depth-{}.fits.fz'.format(objid, band))
            )
        
    if True:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

def _coadds_stage_tims(galaxycat, survey=None, mp=None, radius=100,
                       brickname=None, pixscale=0.262):
    """Initialize the first step of the pipeline, returning
    a dictionary with the following keys:
    
    ['brickid', 'target_extent', 'version_header', 
     'targetrd', 'brickname', 'pixscale', 'bands', 
     'survey', 'brick', 'ps', 'H', 'ccds', 'W', 
     'targetwcs', 'tims']

    """
    from legacypipe.runbrick import stage_tims

    unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)
    P = stage_tims(ra=galaxycat['ra'], dec=galaxycat['dec'], brickname=brickname,
                   survey=survey, W=2*radius, H=2*radius, pixscale=pixscale,
                   mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                   depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                   splinesky=True, unwise_dir=unwise_dir)
    return P

def _read_tractor(galaxycat, objid=None, targetwcs=None,
                  survey=None, verbose=False):
    """Read the full Tractor catalog for a given brick 
    and remove the BCG.
    
    """
    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec

    # Read the newly-generated Tractor catalog
    fn = os.path.join(survey.output_dir, '{}-tractor.fits'.format(objid))
    cat = fits_table(fn)
    if verbose:
        print('Read {} sources from {}'.format(len(cat), fn))

    # Find and remove the central.  For some reason, match_radec
    # occassionally returns two matches, even though nearest=True.
    m1, m2, d12 = match_radec(cat.ra, cat.dec, galaxycat['ra'], galaxycat['dec'],
                              1/3600.0, nearest=True)
    if len(d12) == 0:
        raise ValueError('No matching central found -- definitely a problem.')
    elif len(d12) > 1:
        m1 = m1[np.argmin(d12)]

    if verbose:
        print('Removed central galaxy with objid = {}'.format(cat[m1].objid))

    # To prevent excessive masking, leave the central galaxy and any source
    # who's center is within a half-light radius intact.
    if False:
        fracdev = cat[m1].fracdev[0]
        radius = fracdev * cat[m1].shapedev_r[0] + (1-fracdev) * cat[m1].shapeexp_r[0] # [arcsec]
        if radius > 0:
            n1, n2, nd12 = match_radec(cat.ra, cat.dec, galaxycat['ra'], galaxycat['dec'],
                                       radius/3600.0, nearest=False)
            m1 = np.hstack( (m1, n1) )
            m1 = np.unique(m1)

    cat.cut( ~np.in1d(cat.objid, m1) )
        
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
    #from legacypipe.runbrick import rgbkwargs, rgbkwargs_resid
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
    for suffix in np.atleast_1d('model'):
        for band in bands:
            shutil.copy(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-{}-nocentral-{}.fits.fz'.format(objid, suffix, band))
                )

    if True:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
    
    # Build png postage stamps of the coadds.
    rgbkwargs = dict(mnmx=(-1, 100), arcsinh=1)
    #rgbkwargs_resid = dict(mnmx=(0.1, 2), arcsinh=1)
    rgbkwargs_resid = dict(mnmx=(-1, 100), arcsinh=1)

    #coadd_list = [('image-central', C.coimgs,   rgbkwargs),
    #              ('model-central', C.comods,   rgbkwargs),
    #              ('resid-central', C.coresids, rgbkwargs_resid)]
    coadd_list = [('model-nocentral', C.comods,   rgbkwargs),
                  ('image-central', C.coresids, rgbkwargs_resid)]

    for name, ims, rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        outfn = os.path.join(survey.output_dir, '{}-{}.jpg'.format(objid, name))
        if verbose:
            print('Writing {}'.format(outfn))
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb

def legacyhalos_custom_coadds(galaxycat, survey=None, objid=None, objdir=None,
                              ncpu=1, pixscale=0.262, log=None,
                              cluster_radius=False, verbose=False):
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
                  ncpu=ncpu, pixscale=pixscale, log=log)

    # Step 2 - Read the Tractor catalog for this brick and remove the central.
    cat = _read_tractor(galaxycat, objid=objid, survey=survey, verbose=verbose)

    # Step 3 - Set up the first stage of the pipeline.
    P = _coadds_stage_tims(galaxycat, survey=survey, mp=mp, radius=radius,
                           brickname=brickname, pixscale=pixscale)

    # Step 4 - Render the model images without the central.
    mods = _build_model_image(cat, tims=P['tims'], survey=survey, verbose=verbose)
    
    # Step 3 - Generate and write out the coadds.
    _tractor_coadds(galaxycat, P['targetwcs'], P['tims'], mods, P['version_header'],
                    objid=objid, brickname=brickname, survey=survey, mp=mp,
                    verbose=verbose)

    return 1
