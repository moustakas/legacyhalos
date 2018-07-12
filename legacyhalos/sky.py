"""
legacyhalos.sky
===============

Code to measure the sky variance for each object.

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io

def sky_coadd(ra, dec, size=100, outdir='.', survey=None, ncpu=1, band=('g', 'r', 'z'),
              pixscale=0.262, log=None, force=False):
    """Run legacypipe to generate a coadd in a "blank" part of sky near / around a
    given central.

    """
    import subprocess
    from legacyhalos.misc import custom_brickname

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {size} --height {size} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} '
    cmd += '--skip-calibs --stage image_coadds --blob-image '
    if force:
        cmd += '--force-all '
    
    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), 
                     ra=ra, dec=dec, size=size, pixscale=pixscale,
                     threads=ncpu, outdir=outdir)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)

    # Move (rename) files into the desired output directory and clean up.
    brickname = custom_brickname(ra, dec, prefix='custom-')

    

    pdb.set_trace()

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
        
    if True:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
        shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

def sky_positions(ra_cluster, dec_cluster, redshift, r_lambda, nsky, rand):
    """Choose sky positions that are a minimum physical distance away from the
    central, to ensure we're not affected by ICL.  Here, cutout_radius_cluster
    returns the physical size / radius of the cluster in degrees, based on
    redMaPPer's richness-based estimate, r_lambda.

    We choose positions that are 5-10 times away from the central galaxy,
    uniformly distributed in an annulus around the central.

    """
    from legacyhalos.misc import cutout_radius_cluster

    rcluster = cutout_radius_cluster(redshift, r_lambda, pixscale=1, # [degrees]
                                     factor=1.0) / 3600
    radius = rand.uniform(5 * rcluster, 10 * rcluster, nsky) # [degrees]

    dra = radius / np.cos(np.deg2rad(dec_cluster))
    ddec = radius

    angles = rand.uniform(0, 2.*np.pi, nsky)
    ra = (ra_cluster + dra * np.sin(angles) % 360.0) # enforce 0 < ra < 360
    dec = dec_cluster + ddec * np.cos(angles)

    #import matplotlib.pyplot as plt
    #plt.plot(ra, dec, 'gs') ; plt.show()

    return ra, dec

def legacyhalos_sky(sample, survey=None, objid=None, objdir=None, ncpu=1, nsky=10,
                    pixscale=0.262, log=None, seed=None, verbose=False,
                    debug=False, force=False):
    """Top-level wrapper script to measure the sky variance around a given galaxy.

    """
    if objid is None and objdir is None:
        objid, objdir = get_objid(sample)
        
    outdir = os.path.join(objdir, 'sky')

    rand = np.random.RandomState(seed)

    # Read the ellipse-fitting results and 
    ellipsefit = legacyhalos.io.read_ellipsefit(objid, objdir)
    if bool(ellipsefit):
        if ellipsefit['success']:

            # Get sky coordinates.
            ra, dec = sky_positions(sample['ra'], sample['dec'], sample['z'],
                                    sample['r_lambda'], nsky, rand)


            # Set the size of the sky cutout to 2.2 times the length of the
            # semi-major axis of the central galaxy.
            size = np.ceil(2.2 * ellipsefit['geometry'].sma).astype('int')

            for ii in range(nsky):
                sky_coadd(ra[ii], dec[ii], size=size, outdir=outdir, survey=survey, ncpu=ncpu,
                          pixscale=pixscale, log=log, force=force)
