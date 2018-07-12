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

def sky_coadd(ra, dec, outdir='.', survey=None, ncpu=1, ellipsefit=None,
              pixscale=0.262, log=None):
    """Run legacypipe to generate a coadd in a "blank" part of sky near / around a
    given central.

    """
    import subprocess
    import fitsio
    from photutils.isophote import EllipseSample, Isophote, IsophoteList
    from legacyhalos.misc import custom_brickname

    # Check whether the coadd has already been generated.
    brickname = custom_brickname(ra, dec, prefix='custom-')

    alldone = True
    for band in ('g', 'r', 'z'):
        imfile = os.path.join(outdir, 'coadd', 'cus', brickname, 'legacysurvey-{}-image-{}.fits.fz'.format(brickname, band))
        blobfile = os.path.join(outdir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname))
        if not os.path.isfile(imfile) or not os.path.isfile(blobfile):
            alldone = False

    # Make the coadds, if necessary.
    if not alldone:
        # Set the size of the sky cutout to 2.2 times the length of the
        # semi-major axis of the central galaxy.
        size = np.ceil(2.2 * ellipsefit['geometry'].sma).astype('int')

        # Set up runbrick
        cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
        cmd += '--radec {ra} {dec} --width {size} --height {size} --pixscale {pixscale} '
        cmd += '--threads {threads} --outdir {outdir} '
        cmd += '--skip-calibs --stage image_coadds --blob-image '

        cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), 
                         ra=ra, dec=dec, size=size, pixscale=pixscale,
                         threads=ncpu, outdir=outdir)

        print(cmd, flush=True, file=log)
        err = subprocess.call(cmd.split(), stdout=log, stderr=log)
        
    # Now measure the blank-sky surface brightness profile.
    refband = ellipsefit['refband']
    isophot = ellipsefit[refband]

    for band in ('g', 'r', 'z'):
        imfile = os.path.join(outdir, 'coadd', 'cus', brickname, 'legacysurvey-{}-image-{}.fits.fz'.format(brickname, band))
        blobfile = os.path.join(outdir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname, band))

        print('Reading {}'.format(imfile))
        image = fitsio.read(imfile)
        print('Reading {}'.format(blobfile))
        blobs = fitsio.read(blobfile)

        img = ma.masked_array(image, (blobs != -1))

        # Loop on the reference band isophotes
        isobandfit = []
        for iso in isophot:
            g = iso.sample.geometry # fixed geometry
            g.x0, g.y0 = size / 2, size / 2

            # Use the same integration mode and clipping parameters.
            sample = EllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()

            # Create an Isophote instance with the sample.
            isobandfit.append(Isophote(sample, 0, True, 0))

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)

    if False:
        shutil.rmtree(os.path.join(survey.outdir, 'coadd'))

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
                    debug=False):
    """Top-level wrapper script to measure the sky variance around a given galaxy.

    """
    if objid is None and objdir is None:
        objid, objdir = get_objid(sample)
        
    rand = np.random.RandomState(seed)

    # Read the ellipse-fitting results and 
    ellipsefit = legacyhalos.io.read_ellipsefit(objid, objdir)
    if bool(ellipsefit):
        if ellipsefit['success']:

            # Get sky coordinates.
            ra, dec = sky_positions(sample['ra'], sample['dec'], sample['z'],
                                    sample['r_lambda'], nsky, rand)

            # Build each sky coadd and measure the null surface brightness
            # profile.
            for ii in range(nsky):
                outdir = os.path.join(objdir.replace('analysis', 'analysis-archive'), 'sky-{:03d}'.format(ii))
                sky_coadd(ra[ii], dec[ii], ellipsefit=ellipsefit, outdir=outdir,
                          survey=survey, ncpu=ncpu, pixscale=pixscale, log=log)
