"""
legacyhalos.sky
===============

Code to measure the sky variance for each object.

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings
import shutil

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io

def sky_coadd(ra, dec, outdir='.', size=100, prefix='', survey=None, ncpu=1, ellipsefit=None,
              band=('g', 'r', 'z'), pixscale=0.262, log=None, force=False):
    """Run legacypipe to generate a coadd in a "blank" part of sky near / around a
    given central.

    """
    import subprocess
    import fitsio
    from photutils.isophote import EllipseSample, Isophote, IsophoteList
    from legacyhalos.misc import custom_brickname

    # Check whether the coadd has already been generated.
    brickname = custom_brickname(ra, dec, prefix='custom-')

    blobfile = os.path.join(outdir, '{}-blobs.fits.gz'.format(prefix))
    imfiles = [os.path.join(outdir, '{}-{}.fits.fz'.format(prefix, filt)) for filt in band]

    # Are we already done?
    alldone = True
    if not os.path.isfile(blobfile):
        alldone = False
    for imfile in imfiles:
        if not os.path.isfile(imfile):
            alldone = False

    # Make the coadds, if necessary.
    if not alldone or force:
         # Clean up any old coadd and metrics directories:
        for dd in ('coadd', 'metrics'):
            if os.path.isdir(os.path.join(outdir, dd)):
                shutil.rmtree(os.path.join(outdir, dd))

        # Set up runbrick
        cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
        cmd += '--radec {ra} {dec} --width {size} --height {size} --pixscale {pixscale} '
        cmd += '--threads {threads} --outdir {outdir} '
        cmd += '--skip-calibs --no-write --stage image_coadds --blob-image '

        cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), 
                         ra=ra, dec=dec, size=size, pixscale=pixscale,
                         threads=ncpu, outdir=outdir)

        print(cmd, flush=True, file=log)
        err = subprocess.call(cmd.split(), stdout=log, stderr=log)
        if err != 0:
            print('Something we wrong; please check the logfile.')
            return dict()
        else:

            # Move the files we want and clean up.
            print('Writing {}'.format(blobfile))
            shutil.copy(os.path.join(outdir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)), blobfile)
            for filt, imfile in zip( band, imfiles ):
                print('Writing {}'.format(imfile))
                shutil.copy(os.path.join(outdir, 'coadd', 'cus', brickname,
                                         'legacysurvey-{}-image-{}.fits.fz'.format(brickname, filt)),
                                         imfile)

            for dd in ('coadd', 'metrics'):
                if os.path.isdir(os.path.join(outdir, dd)):
                    shutil.rmtree(os.path.join(outdir, dd))

    # Now measure the blank-sky surface brightness profile.
    refband = ellipsefit['refband']
    isophot = ellipsefit[refband]

    integrmode, sclip, nclip = ellipsefit['integrmode'], ellipsefit['sclip'], ellipsefit['nclip']
    skyellipsefit = dict()

    print('Reading {}'.format(blobfile))
    blobs = fitsio.read(blobfile)
    for filt, imfile in zip( band, imfiles ):
        print('Reading {}'.format(imfile))
        image = fitsio.read(imfile)

        img = ma.masked_array(image, mask=(blobs != -1), fill_value=0)
        #fitsio.write('junk.fits', img.filled(img.fill_value), clobber=True)
        #fitsio.write('blobs.fits', blobs, clobber=True)

        # Loop on the reference band isophotes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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
        skyellipsefit[filt] = IsophoteList(isobandfit)

    ## Clean up
    #for dd in ('coadd', 'metrics'):
    #    if os.path.isdir(os.path.join(outdir, dd)):
    #        shutil.rmtree(os.path.join(outdir, dd), ignore_errors=True)

    return skyellipsefit

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
    ra = (ra_cluster + dra * np.sin(angles)) % 360.0 # enforce 0 < ra < 360
    dec = dec_cluster + ddec * np.cos(angles)
    dec[dec > 90] = 90

    #import matplotlib.pyplot as plt
    #plt.plot(ra, dec, 'gs') ; plt.show()

    return ra, dec

def legacyhalos_sky(sample, survey=None, objid=None, objdir=None, ncpu=1, nsky=30,
                    pixscale=0.262, log=None, seed=1, verbose=False, band=('g', 'r', 'z'),
                    debug=False, force=False):
    """Top-level wrapper script to measure the sky variance around a given galaxy.

    """
    if objid is None and objdir is None:
        objid, objdir = legacyhalos.io.get_objid(sample)
        
    rand = np.random.RandomState(seed)

    # Read the ellipse-fitting results and 
    ellipsefit = legacyhalos.io.read_ellipsefit(objid, objdir)
    if bool(ellipsefit):
        if ellipsefit['success']:

            refband = ellipsefit['refband']

            # Set the size of the sky cutout to 2.2 times the length of the
            # semi-major axis of the central galaxy.
            size = np.ceil(2.5 * ellipsefit['geometry'].sma).astype('int')

            # get the (random) sky coordinates
            ra, dec = sky_positions(sample['ra'], sample['dec'], sample['z'],
                                    sample['r_lambda'], nsky, rand)

            # initialize the output dictionary
            sky = dict()
            sky['seed'] = seed
            sky['size'] = size
            sky['ra'] = ra
            sky['dec'] = dec
            sky['sma'] = ellipsefit[refband].sma
            nsma = len(sky['sma'])
            for filt in band:
                #sma = 
                #sky['{}_sma'.format(filt)] = np.zeros( (nsma, nsky) ).astype('f4')
                sky[filt] = np.zeros( (nsma, nsky) ).astype('f4')

            # Build each sky coadd and measure the null surface brightness
            # profile.
            skyellipsefit = dict()
            for ii in range(nsky):
                prefix = 'sky-{:03d}'.format(ii)
                print('Working on {}'.format(prefix))
                outdir = os.path.join(objdir.replace('analysis', 'analysis-archive'))

                # Do it!
                skyellipsefit = sky_coadd(ra[ii], dec[ii], ellipsefit=ellipsefit,
                                          size=size, outdir=outdir, prefix=prefix,
                                          survey=survey, ncpu=ncpu, pixscale=pixscale,
                                          log=log, force=force)
                if bool(skyellipsefit):
                    for filt in band:
                        sky[filt][:, ii] = skyellipsefit[filt].intens
                else:
                    print('Bailing out.')
                    break 

            if bool(skyellipsefit):
                # write out!
                legacyhalos.io.write_sky_ellipsefit(objid, objdir, sky, verbose=True)
                return 1
            else:
                return 0
            
