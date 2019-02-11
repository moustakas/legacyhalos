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
import fitsio

from astrometry.util.multiproc import multiproc

import legacyhalos.misc
from legacyhalos.misc import custom_brickname
from legacyhalos.io import get_galaxy_galaxydir

def _copyfile(infile, outfile):
    if os.path.isfile(infile):
        shutil.copy2(infile, outfile)
        return 1
    else:
        print('Missing file {}; please check the logfile.'.format(infile))
        return 0

def pipeline_coadds(onegal, galaxy=None, survey=None, radius=None, nproc=1,
                    pixscale=0.262, splinesky=True, log=None, force=False,
                    no_large_galaxies=True, no_gaia=True, no_tycho=True,
                    unwise_coadds=True, cleanup=True):
    """Run legacypipe.runbrick on a custom "brick" centered on the galaxy.

    radius in arcsec

    """
    import subprocess

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()
        
    galaxydir = survey.output_dir

    if galaxy is None:
        galaxy = 'galaxy'

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} '
    cmd += '--survey-dir {survey_dir} '
    #cmd += '--stage image_coadds --early-coadds '
    cmd += '--write-stage tims --write-stage srcs --skip-calibs --no-wise-ceres '
    cmd += '--checkpoint {galaxydir}/{galaxy}-runbrick-checkpoint.p --checkpoint-period 300 '
    cmd += '--pickle {galaxydir}/{galaxy}-runbrick-%%(stage)s.p '
    if unwise_coadds:
        cmd += '--unwise-coadds '
    if no_gaia:
        cmd += '--no-gaia '
    if no_tycho:
        cmd += '--no-tycho '
    if no_large_galaxies:
        cmd += '--no-large-galaxies '
        
    if force:
        cmd += '--force-all '
        checkpointfile = '{galaxydir}/{galaxy}-runbrick-checkpoint.p'.format(galaxydir=galaxydir, galaxy=galaxy)
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
    if not splinesky:
        cmd += '--no-splinesky '
    
    width = np.ceil(2 * radius / pixscale).astype('int') # [pixels]

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), galaxy=galaxy,
                     ra=onegal['RA'], dec=onegal['DEC'], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     galaxydir=galaxydir, survey_dir=survey.survey_dir)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)
    if err != 0:
        print('Something went wrong; please check the logfile.')
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
                         'legacysurvey-{}-maskbits.fits.fz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-maskbits.fits.fz'.format(galaxy)) )
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
                             'legacysurvey-{}-invvar-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-invvar-{}.fits.fz'.format(galaxy, band)) )
            if not ok:
                return ok

        # JPG images

        # Look for WISE stuff in the unwise module--
        if unwise_coadds:
            for band in ('W1', 'W2', 'W3', 'W4'):
                for imtype in ('image', 'model', 'invvar'):
                    ok = _copyfile(
                        os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                     'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                        os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                    if not ok:
                        return ok

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
                os.path.join(survey.output_dir, '{}-pipeline-{}-grz.jpg'.format(galaxy, imtype)) )
            if not ok:
                return ok

        if cleanup:
            shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
            shutil.rmtree(os.path.join(survey.output_dir, 'metrics'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

        return 1

def _custom_sky(skyargs):
    """Perform custom sky-subtraction on a single CCD (with multiprocessing).

    """
    from scipy.stats import sigmaclip
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import uniform_filter

    from tractor.splinesky import SplineSky
    from astrometry.util.miscutils import estimate_mode
    from astrometry.util.fits import fits_table
    from astropy.table import Table

    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    survey, brickname, onegal, radius_arcsec, ccd = skyargs

    im = survey.get_image_object(ccd)
    hdr = im.read_image_header()
    hdr.delete('INHERIT')
    hdr.delete('EXTVER')

    print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
          'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
          'object', getattr(ccd, 'object', None))

    radius = np.round(radius_arcsec / im.pixscale).astype('int') # [pixels]
    
    tim = im.get_tractor_image(splinesky=True, subsky=False,
                               hybridPsf=True, normalizePsf=True)

    targetwcs, bands = tim.subwcs, tim.band
    H, W = targetwcs.shape
    H, W = np.int(H), np.int(W)

    img = tim.getImage()
    ivar = tim.getInvvar()

    # Read the splinesky model (for comparison purposes).  Code snippet taken
    # from image.LegacySurveyImage.read_sky_model.
    #T = Table.read(im.merged_splineskyfn)
    #T.remove_column('skyclass') # causes problems when writing out with fitsio
    #T = fitsio.read(im.merged_splineskyfn)
    #I, = np.nonzero((T['expnum'] == im.expnum) * np.array([c.strip() == im.ccdname for c in T['ccdname']]))

    T = fits_table(im.merged_splineskyfn)
    I, = np.nonzero((T.expnum == im.expnum) * np.array([c.strip() == im.ccdname for c in T.ccdname]))
    if len(I) != 1:
        print('Multiple splinesky models!')
        return 0
    splineskytable = T[I]

    #splineskytable = T[I].as_array() # convert to ndarray
    #splineskytable = tim.getSky()

    #pipesky = np.zeros_like(img)
    #tim.getSky().addTo(pipesky)

    ## Old method
    #if False:
    #    mp = multiproc()
    #    
    #    S = stage_srcs(pixscale=im.pixscale, targetwcs=targetwcs, W=W, H=H,
    #                   bands=bands, tims=[tim], mp=mp, nsigma=5, survey=survey,
    #                   brick=brickname, brickname=brickname, gaia_stars=False,
    #                   star_clusters=False, large_galaxies=False)
    #    
    #    mask = S['blobs'] != -1 # 1 = bad
    #    mask = np.logical_or( mask, ivar <= 0 )
    #    mask = binary_dilation(mask, iterations=2)

    # Masked pixels in the inverse variance map.
    ivarmask = ivar <= 0

    # Mask known stars and large galaxies.
    refs, _ = get_reference_sources(survey, targetwcs, im.pixscale, ['r'],
                                    tycho_stars=True, gaia_stars=True,
                                    large_galaxies=False, star_clusters=False)
    refmask = (get_inblob_map(targetwcs, refs) != 0)

    # Mask the object of interest.
    #http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    _, x0, y0 = targetwcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
    xcen, ycen = np.round(x0 - 1).astype('int'), np.round(y0 - 1).astype('int')
    ymask, xmask = np.ogrid[-ycen:H-ycen, -xcen:W-xcen]
    galmask = (xmask**2 + ymask**2) <= radius**2

    # Define an annulus of sky pixels centered on the object of interest.
    inmask = (xmask**2 + ymask**2) <= 2*radius**2
    outmask = (xmask**2 + ymask**2) <= 5*radius**2
    skymask = (outmask*1 - inmask*1 - galmask*1) == 1

    # Get an initial guess of the sky using the mode, otherwise the median.
    skypix = (ivarmask*1 + refmask*1 + galmask*1) == 0
    skysig1 = 1.0 / np.sqrt(np.median(ivar[skypix]))
    try:
        skyval = estimate_mode(img[skypix], raiseOnWarn=True)
    except:
        skyval = np.median(img[skypix])
   
    # Mask objects in a boxcar-smoothed (image - initial sky model), smoothed by
    # a boxcar filter before cutting pixels above the n-sigma threshold.
    boxcar = 5
    boxsize = 1024
    if min(img.shape) / boxsize < 4: # handle half-DECam chips
        boxsize /= 2

    # Compute initial model...
    skyobj = SplineSky.BlantonMethod(img - skyval, skypix, boxsize)
    skymod = np.zeros_like(img)
    skyobj.addTo(skymod)

    bskysig1 = skysig1 / boxcar # sigma of boxcar-smoothed image.
    objmask = np.abs(uniform_filter(img-skyval-skymod, size=boxcar, mode='constant') > (3 * bskysig1))
    objmask = binary_dilation(objmask, iterations=3)

    skypix = ( (ivarmask*1 + refmask*1 + galmask*1 + objmask*1) == 0 ) * skymask
    skysig = 1.0 / np.sqrt(np.median(ivar[skypix]))
    skymed = np.median(img[skypix])
    try:
        skymode = estimate_mode(img[skypix], raiseOnWarn=True)
    except:
        skymode = 0.0

    # Build the final bit-mask image.
    #   0    = 
    #   2**0 = ivarmask - CP-masked pixels
    #   2**1 = refmask  - reference stars and galaxies
    #   2**2 = galmask  - central galaxy & system
    #   2**3 = objmask  - threshold detected objects
    mask = np.zeros_like(img).astype(np.int16)
    mask[ivarmask] += 2**0
    mask[refmask]  += 2**1
    mask[galmask]  += 2**2
    mask[objmask]  += 2**3

    # Add the sky values and also the central pixel coordinates of the object of
    # interest (so we won't need the WCS object downstream, in QA).
    for card, value in zip(('SKYMODE', 'SKYMED', 'SKYSIG'), (skymode, skymed, skysig)):
        hdr.add_record(dict(name=card, value=value))
    hdr.add_record(dict(name='XCEN', value=x0-1)) # 0-indexed
    hdr.add_record(dict(name='YCEN', value=y0-1))

    out = dict()
    key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
    out['{}-mask'.format(key)] = mask
    out['{}-image'.format(key)] = img
    out['{}-splinesky'.format(key)] = splineskytable
    out['{}-header'.format(key)] = hdr
    #out['{}-skymode'.format(key)] = skymode
    #out['{}-skymed'.format(key)] = skymed
    #out['{}-skysig'.format(key)] = skysig
    
    return out

def custom_coadds(onegal, galaxy=None, survey=None, radius=None, nproc=1,
                  pixscale=0.262, log=None, plots=False, verbose=False,
                  cleanup=True):
    """Build a custom set of coadds for a single galaxy, with a custom mask and sky
    model.

    radius in arcsec

    """
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from astrometry.libkd.spherematch import match_radec
    from tractor.sky import ConstantSky
    
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

    mp = multiproc(nthreads=nproc)

    width = np.ceil(2 * radius / pixscale).astype('int') # [pixels]

    unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)    

    # [1] Initialize the "tims" stage of the pipeline, returning a
    # dictionary with the following keys:
    
    #   ['brickid', 'target_extent', 'version_header', 'targetrd',
    #    'brickname', 'pixscale', 'bands', 'survey', 'brick', 'ps',
    #    'H', 'ccds', 'W', 'targetwcs', 'tims']

    def call_stage_tims():
        """Note that we return just the portion of the CCDs centered on the galaxy, and
        that we turn off sky-subtraction.

        """
        return stage_tims(ra=onegal['RA'], dec=onegal['DEC'], brickname=brickname,
                          survey=survey, W=width, H=width, pixscale=pixscale,
                          mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                          splinesky=True, subsky=False, # note!
                          depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                          unwise_dir=unwise_dir, plots=plots, ps=ps)

    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P = call_stage_tims()
    else:
        P = call_stage_tims()

    tims = P['tims']

    # [2] Derive the custom mask and sky background for each (full) CCD and
    # write out a MEF -custom-mask.fits.gz file.
    skyargs = [(survey, brickname, onegal, radius, _ccd) for _ccd in survey.ccds]
    result = mp.map(_custom_sky, skyargs)
    #result = list( zip( *mp.map(_custom_sky, args) ) )
    sky = dict()
    [sky.update(res) for res in result]
    del result

    H, W = P['targetwcs'].shape

    _comask = np.zeros((len(tims), H, W), np.int16)
    for ii, ccd in enumerate(survey.ccds):
        im = survey.get_image_object(ccd)
        key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
        mask = sky['{}-mask'.format(key)]#.astype('uint8')
        try:
            Yo, Xo, Yi, Xi, _ = resample_with_wcs(P['targetwcs'], im.get_wcs())
            _comask[ii, Yo, Xo] = mask[Yi, Xi]
        except:
            continue

    comask = np.sum(_comask, axis=0)
    hdr = fitsio.FITSHDR()
    P['targetwcs'].add_to_header(hdr)
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    
    maskfile = os.path.join(survey.output_dir, '{}-custom-mask.fits.gz'.format(galaxy))
    print('Writing {}'.format(maskfile))
    fitsio.write(maskfile, comask, header=hdr, clobber=True)

    # Write out separate CCD-level files with the images/data, individual masks
    # (converted to unsigned integer), and the pipeline/splinesky binary FITS
    # tables.
    ccdfile = os.path.join(survey.output_dir, '{}-custom-ccdmask.fits.gz'.format(galaxy))
    print('Writing {}'.format(ccdfile))
    if os.path.isfile(ccdfile):
        os.remove(ccdfile)
    with fitsio.FITS(ccdfile, 'rw') as ff:
        for ii, ccd in enumerate(survey.ccds):
            im = survey.get_image_object(ccd)
            key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
            hdr = sky['{}-header'.format(key)]
            ff.write(sky['{}-mask'.format(key)].astype('uint8'), extname=key, header=hdr)

    ccdfile = os.path.join(survey.output_dir, '{}-ccddata.fits.fz'.format(galaxy))
    print('Writing {}'.format(ccdfile))
    if os.path.isfile(ccdfile):
        os.remove(ccdfile)
    with fitsio.FITS(ccdfile, 'rw') as ff:
        for ii, ccd in enumerate(survey.ccds):
            im = survey.get_image_object(ccd)
            key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
            hdr = sky['{}-header'.format(key)]
            ff.write(sky['{}-image'.format(key)].astype('f4'), extname=key, header=hdr)

    skyfile = os.path.join(survey.output_dir, '{}-pipeline-sky.fits'.format(galaxy))
    print('Writing {}'.format(skyfile))
    if os.path.isfile(skyfile):
        os.remove(skyfile)
    for ii, ccd in enumerate(survey.ccds):
        im = survey.get_image_object(ccd)
        key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
        sky['{}-splinesky'.format(key)].write_to(skyfile, append=ii>0, extname=key)
        
    # [3] Modify each tim by subtracting our new estimate of the sky.
    newtims = []
    for tim in tims:
        image = tim.getImage()
        key = '{}-{:02d}-{}'.format(tim.imobj.name, tim.imobj.hdu, tim.imobj.band)
        newsky = sky['{}-header'.format(key)]['SKYMODE']
        tim.setImage(image - newsky)
        tim.sky = ConstantSky(0)
        newtims.append(tim)
    del sky

    # [4] Read the Tractor catalog and render the model image of each CCD, with
    # and without the central large galaxy.
    tractorfile = os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0
    
    cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(cat), tractorfile), flush=True, file=log)

    # Find and remove all the objects within XX arcsec of the target
    # coordinates.
    m1, m2, d12 = match_radec(cat.ra, cat.dec, onegal['RA'], onegal['DEC'], 3/3600.0, nearest=False)
    if len(d12) == 0:
        print('No matching galaxies found -- probably not what you wanted.', flush=True, file=log)
        #raise ValueError
        keep = np.ones(len(T)).astype(bool)
    else:
        keep = ~np.isin(cat.objid, cat[m1].objid)        

    #print('Removing central galaxy with index = {}, objid = {}'.format(
    #    m1, cat[m1].objid), flush=True, file=log)

    print('Creating tractor sources...', flush=True, file=log)
    srcs = read_fits_catalog(cat, fluxPrefix='')
    srcs_nocentral = np.array(srcs)[keep].tolist()
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    print('Rendering model images with and without surrounding galaxies...', flush=True, file=log)
    modargs = [(tim, srcs) for tim in newtims]
    mods = mp.map(_get_mod, modargs)

    modargs = [(tim, srcs_nocentral) for tim in newtims]
    mods_nocentral = mp.map(_get_mod, modargs)

    # [5] Build the custom coadds, with and without the surrounding galaxies.
    print('Producing coadds...', flush=True, file=log)
    def call_make_coadds(usemods):
        return make_coadds(newtims, P['bands'], P['targetwcs'], mods=usemods, mp=mp,
                           callback=write_coadd_images,
                           callback_args=(survey, brickname, P['version_header'], 
                                          newtims, P['targetwcs']))

    # Custom coadds (all galaxies).
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C = call_make_coadds(mods)
    else:
        C = call_make_coadds(mods)

    for suffix in ('image', 'model'):
        for band in P['bands']:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-custom-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
                #os.path.join(survey.output_dir, '{}-custom-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok

    # Custom coadds (without the central).
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C_nocentral = call_make_coadds(mods_nocentral)
    else:
        C_nocentral = call_make_coadds(mods_nocentral)

    # Move (rename) the coadds into the desired output directory - no central.
    for suffix in np.atleast_1d('model'):
    #for suffix in ('image', 'model'):
        for band in P['bands']:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-custom-{}-nocentral-{}.fits.fz'.format(galaxy, suffix, band)) )
                #os.path.join(survey.output_dir, '{}-custom-{}-nocentral-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok
            
    if cleanup:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))

    # [6] Finally, build png images.
    def call_make_png(C, nocentral=False):
        rgbkwargs = dict(mnmx=(-1, 100), arcsinh=1)
        #rgbkwargs_resid = dict(mnmx=(0.1, 2), arcsinh=1)
        rgbkwargs_resid = dict(mnmx=(-1, 100), arcsinh=1)

        if nocentral:
            coadd_list = [('custom-model-nocentral', C.comods, rgbkwargs),
                          ('custom-image-central', C.coresids, rgbkwargs_resid)]
        else:
            coadd_list = [('custom-image', C.coimgs,   rgbkwargs),
                          ('custom-model', C.comods,   rgbkwargs),
                          ('custom-resid', C.coresids, rgbkwargs_resid)]

        for suffix, ims, rgbkw in coadd_list:
            rgb = get_rgb(ims, P['bands'], **rgbkw)
            kwa = {}
            outfn = os.path.join(survey.output_dir, '{}-{}-grz.jpg'.format(galaxy, suffix))
            print('Writing {}'.format(outfn), flush=True, file=log)
            imsave_jpeg(outfn, rgb, origin='lower', **kwa)
            del rgb

    call_make_png(C, nocentral=False)
    call_make_png(C_nocentral, nocentral=True)

    return 1
