"""
legacyhalos.coadds
==================

Code to generate grzW1W2 custom coadds / mosaics.

python -u legacyanalysis/extract-calibs.py --drdir /project/projectdirs/cosmo/data/legacysurvey/dr5 --radec 342.4942 -0.6706 --width 300 --height 300

"""
from __future__ import absolute_import, division, print_function

import os, sys, pdb
import shutil
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

from legacyhalos.misc import custom_brickname

def _copyfile(infile, outfile):
    if os.path.isfile(infile):
        shutil.copy(infile, outfile)
        return 1
    else:
        print('Missing file {}; please check the logfile.'.format(infile))
        return 0

def runbrick(onegal, galaxy=None, survey=None, radius=100, ncpu=1, pixscale=0.262,
             splinesky=True, log=None, force=False, archivedir=None, cleanup=True):
    """Run legacypipe.runbrick on a custom "brick" centered on the galaxy.

    """
    import subprocess

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if archivedir is None:
        archivedir = survey.output_dir

    if galaxy is None:
        galaxy = 'galaxy'

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} --unwise-coadds '
    #cmd += '--force-stage coadds '
    cmd += '--write-stage srcs --no-write --skip --no-wise-ceres '
    cmd += '--checkpoint {archivedir}/{galaxy}-runbrick-checkpoint.p --checkpoint-period 600 '
    cmd += '--pickle {archivedir}/{galaxy}-runbrick-%%(stage)s.p ' 
    if force:
        cmd += '--force-all '
    if not splinesky:
        cmd += '--no-splinesky '
    
    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), galaxy=galaxy,
                     ra=onegal['RA'], dec=onegal['DEC'], width=2*radius,
                     pixscale=pixscale, threads=ncpu, outdir=survey.output_dir,
                     archivedir=archivedir)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)
    if err != 0:
        print('Something we wrong; please check the logfile.')
        return 0
    else:
        # Move (rename) files into the desired output directory and clean up.
        brickname = 'custom-{}'.format(custom_brickname(onegal['RA'], onegal['DEC']))

        # tractor catalog
        ok = _copyfile(
            os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy)) )
        if not ok:
            return ok

        # CCDs, maskbits, blob images, and depth images
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-ccds.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-maskbits.fits.gz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-maskbits.fits.gz'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-blobs.fits.gz'.format(galaxy)) )
        if not ok:
            return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-depth-{}.fits.fz'.format(galaxy, band)) )
        if not ok:
            return ok
        
        # Data and model images
        for band in ('g', 'r', 'z'):
            for imtype in ('image', 'model'):
                ok = _copyfile(
                    os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(survey.output_dir, '{}-pipeline-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                if not ok:
                    return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-invvar-{}.fits.fz'.format(brickname, imtype, band)),
                os.path.join(survey.output_dir, '{}-invvar-{}.fits.fz'.format(galaxy, imtype, band)) )
            if not ok:
                return ok

        for band in ('W1', 'W2'):
            for imtype in ('image', 'model'):
                ok = _copyfile(
                    os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                if not ok:
                    return ok

        # JPG images
        for imtype in ('wise', 'wisemodel'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(survey.output_dir, '{}-{}.jpg'.format(galaxy, imtype)) )
            if not ok:
                return ok

        for imtype in ('image', 'model', 'resid'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(survey.output_dir, '{}-pipeline-{}.jpg'.format(galaxy, imtype)) )
            if not ok:
                return ok

        if cleanup:
            shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
            shutil.rmtree(os.path.join(survey.output_dir, 'metrics'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

        return 1

def custom_coadds(onegal, galaxy=None, survey=None, radius=100, ncpu=1,
                  pixscale=0.262, log=None, plots=False, verbose=False,
                  cleanup=True):
    """Build a custom set of coadds for a single galaxy, with a custom mask and sky
    model.

    """
    from scipy.ndimage.morphology import binary_dilation
    from astropy.io import fits

    from astrometry.util.multiproc import multiproc
    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec

    from legacypipe.runbrick import stage_tims
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.runbrick import _get_mod
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.survey import get_rgb, imsave_jpeg

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if galaxy is None:
        galaxy = 'galaxy'
        
    brickname = custom_brickname(onegal['RA'], onegal['DEC'])

    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('qa-{}'.format(brickname))
    else:
        ps = None

    mp = multiproc(nthreads=ncpu)

    # [1] Perform custom sky-subtraction of each CCD.
    newsky = dict()
    for iccd, ccd in enumerate(survey.ccds):
        im = survey.get_image_object(ccd)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
              'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
              'object', getattr(ccd, 'object', None))
        tim = im.get_tractor_image(splinesky=True, subsky=False,
                                   hybridPsf=True, normalizePsf=True)
        #tims.append(tim)

        # Get the (pixel) coordinates of the galaxy on this CCD
        W, H, wcs = tim.imobj.width, tim.imobj.height, tim.subwcs
        _, x0, y0 = wcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
        xcen, ycen = np.round(x0 - 1), np.round(y0 - 1)
        pxscale = im.pixscale

        # Get the image and the splinesky model generated by the pipeline.
        image = tim.getImage()
        weight = tim.getInvvar()
        sky = tim.getSky()
        splinesky = np.zeros_like(image)
        sky.addTo(splinesky)

        mask = (invvar <= 0) * 1 # 1=bad, 0=good
        #mask = np.logical_or( mask, image > (5 * tim.sig1) )
        #mask = np.logical_or( mask, mod > (nsig * tim.sig1) )
        #mask = np.logical_or( mask, (snr > 25) * 1 )
        #mask = binary_dilation(mask, iterations=2)
        mask = (image - splinesky) > 2 * tim.sig1
        
        
            

        pdb.set_trace()

        # Use the algorithm in legacypipe.image.LegacySurveyImage.run_sky to
        # create the image mask, but be more aggressive.
        med = np.median(image[weight > 0])

        good = (wt > 0)
        if np.sum(good) == 0:
            raise RuntimeError('No pixels with weight > 0 in: ' + str(self))
        med = np.median(img[good])

        # For DECam chips where we drop half the chip, spline becomes underconstrained
        if min(img.shape) / boxsize < 4:
            boxsize /= 2

        # Compute initial model...
        skyobj = SplineSky.BlantonMethod(img - med, good, boxsize)
        skymod = np.zeros_like(img)
        skyobj.addTo(skymod)

        # Now mask bright objects in a boxcar-smoothed (image - initial sky model)
        sig1 = 1./np.sqrt(np.median(wt[good]))
        # Smooth by a boxcar filter before cutting pixels above threshold --
        boxcar = 5
        # Sigma of boxcar-smoothed image
        bsig1 = sig1 / boxcar
        masked = np.abs(uniform_filter(img-med-skymod, size=boxcar, mode='constant')
                        > (3.*bsig1))
        masked = binary_dilation(masked, iterations=3)
        good[masked] = False
        sig1b = 1./np.sqrt(np.median(wt[good]))

        
        skyobj = SplineSky.BlantonMethod(image - med, weight>0, 512)
        skymod = np.zeros_like(image)
        skyobj.addTo(skymod)
        sig1 = 1.0 / np.sqrt(np.median(weight[weight > 0]))
        mask = ((image - med - skymod) > (5.0*sig1))*1.0
        mask = binary_dilation(mask, iterations=3)
        mask[weight == 0] = 1 # 0=good, 1=bad
        pipeskypix = np.flatnonzero((mask == 0) * 1)

        
        
        

        if False:
            # Make a more aggressive object mask but reset the badpix and
            # edge bits.
            print('Iteratively building a more aggressive object mask.')
            #newmask = ((image - med - skymod) > (3.0*sig1))*1.0
            newmask = mask.copy()
            #newmask = (weight == 0)*1 
            for bit in ('edge', 'edge2'):
                ww = np.flatnonzero((tim.dq & CP_DQ_BITS[bit]) == CP_DQ_BITS[bit])
                if len(ww) > 0:
                    newmask.flat[ww] = 0
            for jj in range(2):
                gauss = Gaussian2DKernel(stddev=1)
                newmask = convolve(newmask, gauss)
                newmask[newmask > 0] = 1 # 0=good, 1=bad

        #http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
        ysize, xsize = image.shape
        ymask, xmask = np.ogrid[-ycen:ysize-ycen, -xcen:xsize-xcen]
        newmask = xmask**2 + ymask**2 <= radius**2

        newmask += mask
        newmask[newmask > 0] = 1

        # Build a new sky model.
        newskypix = np.flatnonzero((newmask == 0)*1)
        newmed = np.median(image.flat[newskypix]) # need mode with rejection
        newsky = np.zeros_like(image) + newmed
        #newsky = skymodel.copy()

        ## Now do a lower-order polynomial sky subtraction.
        #xall, yall = np.mgrid[:H, :W]
        #xx = xall.flat[newskypix]
        #yy = yall.flat[newskypix]
        #sky = image.flat[newskypix]
        #if False:
        #    plt.clf() ; plt.scatter(xx[:5000], sky[:5000]) ; plt.show()
        #    pdb.set_trace()
        #    pinit = models.Polynomial2D(degree=1)
        #    pfit = fitting.LevMarLSQFitter()
        #    coeff = pfit(pinit, xx, yy, sky)
        #    # evaluate the model back on xall, yall

        # Perform aperture photometry on the sky-subtracted images.
        image_nopipesky = image - skymodel
        image_nonewsky = image - newsky

    pdb.set_trace()

    # [2] Initialize the first step of the pipeline, returning a dictionary with
    # the following keys:
    # 
    #   ['brickid', 'target_extent', 'version_header', 'targetrd',
    #    'brickname', 'pixscale', 'bands', 'survey', 'brick', 'ps',
    #    'H', 'ccds', 'W', 'targetwcs', 'tims']

    unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)    

    def call_stage_tims():
        return stage_tims(ra=onegal['RA'], dec=onegal['DEC'], brickname=brickname,
                          survey=survey, W=2*radius, H=2*radius, pixscale=pixscale,
                          mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                          subsky=False,
                          depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                          unwise_dir=unwise_dir, plots=plots, ps=ps)

    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P = call_stage_tims()
    else:
        P = call_stage_tims()

    tims = P['tims']

    # [2] Read the Tractor catalog and render the model image of each CCD, with
    # and without the central large galaxy.
    tractorfile = os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0
    
    cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(cat), tractorfile), flush=True, file=log)

    # Find and remove the central.  For some reason, match_radec
    # occassionally returns two matches, even though nearest=True.
    m1, m2, d12 = match_radec(cat.ra, cat.dec, onegal['RA'], onegal['DEC'],
                              1/3600.0, nearest=True)
    if len(d12) == 0:
        print('No matching central found -- definitely a problem.')
        raise ValueError
    elif len(d12) > 1:
        m1 = m1[np.argmin(d12)]

    print('Removing central galaxy with index = {}, objid = {}'.format(
        m1, cat[m1].objid), flush=True, file=log)

    keep = ~np.in1d(cat.objid, cat[m1].objid)

    print('Creating tractor sources...', flush=True, file=log)
    srcs = read_fits_catalog(cat, fluxPrefix='')
    srcs_nocentral = np.array(srcs)[keep].tolist()
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    print('Rendering model images...', flush=True, file=log)
    mods = [_get_mod((tim, srcs)) for tim in tims]
    mods_nocentral = [_get_mod((tim, srcs_nocentral)) for tim in tims]

    # [3] Obtain a better estimate of the sky by aggressively masking sources,
    # including the large central galaxy we're interested in.

    # Mask pixels with ivar<=0. Also build an object mask from the model
    # image, to handle systematic residuals.  However, don't mask too
    # aggressively near the center of the galaxy.
    hx = fits.HDUList()

    newtims, newmods = [], []
    for ii, (tim, mod) in enumerate( zip(tims, mods_nocentral) ):
        image = tim.getImage()
        invvar = tim.getInvvar()

        splinesky = tim.getSky()
        skymodel = np.zeros_like(image)
        splinesky.addTo(skymodel)

        # Build a more aggressive object mask, also masking out the full extent
        # of the large galaxy.
        snr = mod * np.sqrt(invvar)

        mask = (invvar <= 0) * 1 # 1=bad, 0=good
        #mask = np.logical_or( mask, image > (5 * tim.sig1) )
        #mask = np.logical_or( mask, mod > (nsig * tim.sig1) )
        #mask = np.logical_or( mask, (snr > 25) * 1 )
        #mask = binary_dilation(mask, iterations=2)

        #http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
        _, x0, y0 = tim.subwcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
        xcen, ycen = np.round(x0 - 1), np.round(y0 - 1)
        ysize, xsize = image.shape
        ymask, xmask = np.ogrid[-ycen:ysize-ycen, -xcen:xsize-xcen]
        objmask = (xmask**2 + ymask**2) <= radius**2
        mask = np.logical_or( mask, objmask )

        # Finally get a new (constant) estimate of the sky and subtract it from
        # the data.
        skypix = np.flatnonzero(mask == 0)
        newsky = np.median(image.flat[skypix]) # need mode with rejection
        #newsky = np.zeros_like(image) + skymed
        print(newsky)

        tim.data = image + skymodel - newsky
        newmod = mod + skymodel - newsky

        hdu = fits.ImageHDU(mask * 1, name='CCD{:03d}'.format(ii))
        hdu.header['SKY'] = newsky
        hx.append(hdu)

        newtims.append(tim)
        newmods.append(newmod)

    #hx = fits.HDUList()
    #hdu = fits.PrimaryHDU()
    #for ii, (tim, sky) in enumerate( zip(newtims, newskies) ):
    #    hdu.header['CCD{:03d}'.format(ii)] = tim.name
    #    hdu.header['SKY{:03d}'.format(ii)] = sky
    
    skyfile = os.path.join(survey.output_dir, '{}-sky.fits'.format(galaxy))
    print('Writing {}'.format(skyfile))
    hx.writeto(skyfile, overwrite=True)

    # [3] Finally, build the custom coadds, with and without the central galaxy.
    print('Producing coadds...', flush=True, file=log)
    def call_make_coadds():
        return make_coadds(newtims, P['bands'], P['targetwcs'], mods=newmods, mp=mp,
                           callback=write_coadd_images,
                           callback_args=(survey, brickname, P['version_header'], 
                                          newtims, P['targetwcs']))
        
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C = call_make_coadds()
    else:
        C = call_make_coadds()

    # Move (rename) the coadds into the desired output directory.
    for suffix in ('image', 'model', 'resid'):
        for band in P['bands']:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-{}-nocentral-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok
            
    if cleanup:
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
        rgb = get_rgb(ims, P['bands'], **rgbkw)
        kwa = {}
        outfn = os.path.join(survey.output_dir, '{}-{}.jpg'.format(galaxy, name))
        print('Writing {}'.format(outfn), flush=True, file=log)
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb

    pdb.set_trace()

    return 1

def coadds_stage_tims(sample, survey=None, mp=None, radius=100,
                      brickname=None, pixscale=0.262, splinesky=True,
                      log=None, plots=False):
    """Initialize the first step of the pipeline, returning
    a dictionary with the following keys:
    
    ['brickid', 'target_extent', 'version_header', 
     'targetrd', 'brickname', 'pixscale', 'bands', 
     'survey', 'brick', 'ps', 'H', 'ccds', 'W', 
     'targetwcs', 'tims']

    """
    from legacypipe.runbrick import stage_tims

    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('qa-{}'.format(brickname))
    else:
        ps = None

    unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)    

    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P = stage_tims(ra=sample['RA'], dec=sample['DEC'], brickname=brickname,
                           survey=survey, W=2*radius, H=2*radius, pixscale=pixscale,
                           mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                           depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                           splinesky=splinesky, unwise_dir=unwise_dir, plots=plots, ps=ps)
    else:
        P = stage_tims(ra=sample['RA'], dec=sample['DEC'], brickname=brickname,
                       survey=survey, W=2*radius, H=2*radius, pixscale=pixscale,
                       mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                       depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                       splinesky=splinesky, unwise_dir=unwise_dir, plots=plots, ps=ps)

    return P

def read_tractor(sample, prefix=None, targetwcs=None,
                 survey=None, log=None):
    """Read the full Tractor catalog for a given brick 
    and remove the BCG.
    
    """
    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec

    # Read the newly-generated Tractor catalog
    fn = os.path.join(survey.output_dir, '{}-tractor.fits'.format(prefix))
    cat = fits_table(fn)
    print('Read {} sources from {}'.format(len(cat), fn), flush=True, file=log)

    # Find and remove the central.  For some reason, match_radec
    # occassionally returns two matches, even though nearest=True.
    m1, m2, d12 = match_radec(cat.ra, cat.dec, sample['RA'], sample['DEC'],
                              3/3600.0, nearest=True)
    if len(d12) == 0:
        print('No matching central found -- definitely a problem.')
        raise ValueError
    elif len(d12) > 1:
        m1 = m1[np.argmin(d12)]

    print('Removed central galaxy with objid = {}'.format(cat[m1].objid),
          flush=True, file=log)

    # To prevent excessive masking, leave the central galaxy and any source
    # who's center is within a half-light radius intact.
    if False:
        fracdev = cat[m1].fracdev[0]
        radius = fracdev * cat[m1].shapedev_r[0] + (1-fracdev) * cat[m1].shapeexp_r[0] # [arcsec]
        if radius > 0:
            n1, n2, nd12 = match_radec(cat.ra, cat.dec, sample['RA'], sample['DEC'],
                                       radius/3600.0, nearest=False)
            m1 = np.hstack( (m1, n1) )
            m1 = np.unique(m1)

    cat.cut( ~np.in1d(cat.objid, m1) )
        
    return cat

def build_model_image(cat, tims, survey=None, log=None):
    """Generate a model image by rendering each source.
    
    """
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.runbrick import _get_mod
    
    print('Creating tractor sources...', flush=True, file=log)
    srcs = read_fits_catalog(cat, fluxPrefix='')
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    print('Rendering model images...', flush=True, file=log)
    mods = [_get_mod((tim, srcs)) for tim in tims]

    return mods

def tractor_coadds(sample, targetwcs, tims, mods, version_header, prefix=None,
                   brickname=None, survey=None, mp=None, log=None, bands=['g','r','z']):
    """Generate individual-band FITS and color coadds for each central using
    Tractor.

    """
    from legacypipe.coadds import make_coadds, write_coadd_images
    #from legacypipe.runbrick import rgbkwargs, rgbkwargs_resid
    from legacypipe.survey import get_rgb, imsave_jpeg

    if brickname is None:
        brickname = sample['BRICKNAME']

    print('Producing coadds...', flush=True, file=log)
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C = make_coadds(tims, bands, targetwcs, mods=mods, mp=mp,
                            callback=write_coadd_images,
                            callback_args=(survey, brickname, version_header, 
                                           tims, targetwcs))
    else:
        C = make_coadds(tims, bands, targetwcs, mods=mods, mp=mp,
                        callback=write_coadd_images,
                        callback_args=(survey, brickname, version_header, 
                                       tims, targetwcs))

    # Move (rename) the coadds into the desired output directory.
    for suffix in np.atleast_1d('model'):
        for band in bands:
            shutil.copy(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-{}-nocentral-{}.fits.fz'.format(prefix, suffix, band))
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
        outfn = os.path.join(survey.output_dir, '{}-{}.jpg'.format(prefix, name))
        print('Writing {}'.format(outfn), flush=True, file=log)
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb

def legacyhalos_custom_coadds(sample, survey=None, prefix=None, objdir=None,
                              ncpu=1, pixscale=0.262, log=None, force=False,
                              splinesky=True, cluster_radius=False):
    """Top-level wrapper script to generate coadds for a single galaxy.

    """ 
    from astrometry.util.multiproc import multiproc
    mp = multiproc(nthreads=ncpu)
    
    #if prefix is None and objdir is None:
    #    objid, objdir = get_objid(sample)
    brickname = custom_brickname(sample['RA'], sample['DEC'])

    survey.output_dir = objdir
    archivedir = objdir.replace('analysis', 'analysis-archive') # hack!

    # Step 0 - Get the cutout radius.
    if cluster_radius:
        from legacyhalos.misc import cutout_radius_cluster
        radius = cutout_radius_cluster(redshift=sample['Z'], pixscale=pixscale,
                                       cluster_radius=sample['R_LAMBDA'])
    else:
        from legacyhalos.misc import cutout_radius_150kpc
        radius = cutout_radius_150kpc(redshift=sample['Z'], pixscale=pixscale)

    # Step 1 - Run legacypipe to generate a custom "brick" and tractor catalog
    # centered on the central.
    success = runbrick(sample, prefix=prefix, survey=survey, radius=radius,
                       ncpu=ncpu, pixscale=pixscale, log=log, force=force,
                       archivedir=archivedir, splinesky=splinesky)
    if success:

        # Step 2 - Read the Tractor catalog for this brick and remove the central.
        cat = read_tractor(sample, prefix=prefix, survey=survey, log=log)

        # Step 3 - Set up the first stage of the pipeline.
        P = coadds_stage_tims(sample, survey=survey, mp=mp, radius=radius,
                              brickname=brickname, pixscale=pixscale, log=log,
                              splinesky=splinesky)

        # Step 4 - Render the model images without the central.
        mods = build_model_image(cat, tims=P['tims'], survey=survey, log=log)

        # Step 3 - Generate and write out the coadds.
        tractor_coadds(sample, P['targetwcs'], P['tims'], mods, P['version_header'],
                       prefix=prefix, brickname=brickname, survey=survey, mp=mp, log=log)
        return 1

    else:
        return 0

def decals_vs_hsc_custom_coadds(onegal, survey=None, ncpu=1, pixscale=0.262,
                                log=None, force=False):
    """Top-level wrapper script to generate custom coadds for a single galaxy.

    """ 
    from astrometry.util.multiproc import multiproc
    mp = multiproc(nthreads=ncpu)
    
    #if prefix is None and objdir is None:
    #    objid, objdir = get_objid(onegal)
    brickname = custom_brickname(onegal['RA'], onegal['DEC'])

    survey.output_dir = objdir
    archivedir = objdir.replace('analysis', 'analysis-archive') # hack!

    # Step 0 - Get the cutout radius.
    from legacyhalos.misc import cutout_radius_150kpc
    radius = cutout_radius_150kpc(redshift=onegal['Z'], pixscale=pixscale)

    # Step 1 - Run legacypipe to generate a custom "brick" and tractor catalog
    # centered on the central.
    success = custom_brick(onegal, prefix=prefix, survey=survey, radius=radius,
                           ncpu=ncpu, pixscale=pixscale, log=log, force=force,
                           archivedir=archivedir, splinesky=splinesky)
    if success:

        # Step 2 - Read the Tractor catalog for this brick and remove the central.
        cat = read_tractor(onegal, prefix=prefix, survey=survey, log=log)

        # Step 3 - Set up the first stage of the pipeline.
        P = coadds_stage_tims(onegal, survey=survey, mp=mp, radius=radius,
                              brickname=brickname, pixscale=pixscale, log=log,
                              splinesky=splinesky)

        # Step 4 - Render the model images without the central.
        mods = build_model_image(cat, tims=P['tims'], survey=survey, log=log)

        # Step 3 - Generate and write out the coadds.
        tractor_coadds(onegal, P['targetwcs'], P['tims'], mods, P['version_header'],
                       prefix=prefix, brickname=brickname, survey=survey, mp=mp, log=log)
        return 1

    else:
        return 0
