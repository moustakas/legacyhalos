"""
legacyhalos.coadds
==================

Code to generate grzW1W2 custom coadds / mosaics.

python -u legacyanalysis/extract-calibs.py --drdir /project/projectdirs/cosmo/data/legacysurvey/dr5 --radec 342.4942 -0.6706 --width 300 --height 300

"""
import os, sys, time, pdb
import shutil
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import fitsio

import tractor
from astrometry.util.fits import fits_table
from astrometry.util.multiproc import multiproc
from legacypipe.catalog import read_fits_catalog

import legacyhalos.misc
from legacyhalos.misc import custom_brickname

def _forced_phot(args):
    """Wrapper function for the multiprocessing."""
    return forced_phot(*args)

def forced_phot(newtims, custom_srcs, band):
    """Perform forced photometry, returning the bandpass, the (newly optimized)
    flux, and the (new) inverse variance flux (all in nanomaggies).

    """
    bandtims = [tim for tim in newtims if tim.band == band]
    tr = tractor.Tractor(bandtims, custom_srcs)
    tr.freezeParamsRecursive('*')
    tr.thawPathsTo(band)
    R = tr.optimize_forced_photometry(
        minsb=0, mindlnp=1.0, sky=False, fitstats=True,
        variance=True, shared_params=False, wantims=False)
    return (band, np.array(tr.getParams()), R.IV)

def _mosaic_width(radius_mosaic, pixscale):
    """Ensure the mosaic is an odd number of pixels so the central can land on a
    whole pixel (important for ellipse-fitting).

    radius_mosaic in arcsec

    """
    #width = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]
    width = 2 * radius_mosaic / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width

def _copyfile(infile, outfile):
    if os.path.isfile(infile):
        os.rename(infile, outfile)
        #shutil.copy2(infile, outfile)
        return 1
    else:
        print('Missing file {}; please check the logfile.'.format(infile))
        return 0

def isolate_central(cat, wcs, psf_sigma=1.1, radius_search=5.0, centrals=True):
    """Isolate the central galaxy.

    radius_mosaic in arcsec
    """
    from astrometry.libkd.spherematch import match_radec
    from legacyhalos.mge import find_galaxy

    if type(wcs) is not tractor.wcs.ConstantFitsWcs:
        wcs = tractor.wcs.ConstantFitsWcs(wcs)

    racen, deccen = wcs.wcs.crval
    _, width = wcs.wcs.shape
    radius_mosaic = width * wcs.wcs.pixel_scale() / 2 # [arcsec]
        
    keep = np.ones(len(cat)).astype(bool)
    if centrals:
        # Build a model image with all the sources whose centroids are within
        # the inner XX% of the mosaic and then "find" the central galaxy.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, racen, deccen,
                                  0.5*radius_mosaic/3600.0, nearest=False)
        srcs = read_fits_catalog(cat[m1], fluxPrefix='')
        mod = legacyhalos.misc.srcs2image(srcs, wcs, psf_sigma=psf_sigma)
        if np.sum(np.isnan(mod)) > 0:
            print('Problem rendering model image of galaxy {}'.format(galaxy),
                  flush=True, file=log)

        mgegalaxy = find_galaxy(mod, nblob=1, binning=3, quiet=True)

        # Now use the ellipse parameters to get a better list of the model
        # sources in and around the central, and remove the largest ones.
        majoraxis = mgegalaxy.majoraxis
        these = legacyhalos.misc.ellipse_mask(width/2, width/2, majoraxis, majoraxis*(1-mgegalaxy.eps),
                                              np.radians(mgegalaxy.theta-90), cat.bx, cat.by)

        galrad = np.max(np.array((cat.shapedev_r, cat.shapeexp_r)), axis=0)
        #galrad = (cat.fracdev * cat.shapedev_r + (1-cat.fracdev) * cat.shapeexp_r) # type-weighted radius
        these *= galrad > 3

        # Also add the sources nearest to the central coordinates.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, racen, deccen,
                                  radius_search/3600.0, nearest=False)
        if len(m1) > 0:
            these[m1] = True

        if np.sum(these) > 0:
            keep[these] = False
        else:
            m1, m2, d12 = match_radec(cat.ra, cat.dec, racen, deccen,
                                      radius_search/3600.0, nearest=False)
            if len(d12) > 0:
                keep = ~np.isin(cat.objid, cat[m1].objid)
    else:
        # Find and remove all the objects within XX arcsec of the target
        # coordinates.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, racen, deccen,
                                  radius_search/3600.0, nearest=False)
        if len(d12) > 0:
            keep = ~np.isin(cat.objid, cat[m1].objid)
        else:
            print('No matching galaxies found -- probably not what you wanted.', flush=True, file=log)
            #raise ValueError

    return keep

def pipeline_coadds(onegal, galaxy=None, survey=None, radius_mosaic=None,
                    nproc=1, pixscale=0.262, run='decam', splinesky=True,
                    log=None, force=False, no_large_galaxies=True, no_gaia=True,
                    no_tycho=True, unwise=True, apodize=False, cleanup=True):
    """Run legacypipe.runbrick on a custom "brick" centered on the galaxy.

    radius_mosaic in arcsec

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
    cmd += '--survey-dir {survey_dir} --run {run} '
    #cmd += '--stage image_coadds --early-coadds '
    #cmd += '--write-stage tims '
    cmd += '--write-stage srcs '
    #cmd += '--min-mjd 0 ' # obsolete
    cmd += '--skip-calibs '
    #cmd += '--no-wise-ceres '
    cmd += '--checkpoint {galaxydir}/{galaxy}-runbrick-checkpoint.p '
    cmd += '--pickle {galaxydir}/{galaxy}-runbrick-%%(stage)s.p '
    if unwise:
        cmd += '--unwise-coadds '
    else:
        cmd += '--no-wise '
    if apodize:
        cmd += '--apodize '
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

    width = _mosaic_width(radius_mosaic, pixscale)

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), galaxy=galaxy,
                     ra=onegal['RA'], dec=onegal['DEC'], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     galaxydir=galaxydir, survey_dir=survey.survey_dir, run=run)
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
            os.path.join(survey.output_dir, '{}-pipeline-tractor.fits'.format(galaxy)) )
        if not ok:
            return ok

        # CCDs, maskbits, blob images, outlier masks, and depth images
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-ccds.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-ccds-{}.fits'.format(galaxy, run)) )
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

        ok = _copyfile(
            os.path.join(survey.output_dir, 'metrics', 'cus', 'outlier-mask-{}.fits.fz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-outlier-mask.fits.fz'.format(galaxy)) )
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
        if unwise:
            for band in ('W1', 'W2', 'W3', 'W4'):
                for imtype in ('image', 'model', 'invvar'):
                    ok = _copyfile(
                        os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                     'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                        os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                    if not ok:
                        return ok

            for imtype, suffix in zip(('wise', 'wisemodel'),
                                      ('pipeline-image', 'pipeline-model')):
                ok = _copyfile(
                    os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                    os.path.join(survey.output_dir, '{}-{}-W1W2.jpg'.format(galaxy, suffix)) )
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
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import uniform_filter
    from astropy.stats import sigma_clipped_stats

    from tractor.splinesky import SplineSky
    from astrometry.util.miscutils import estimate_mode
    from astrometry.util.resample import resample_with_wcs

    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    survey, brickname, brickwcs, onegal, radius_mask_arcsec, apodize, sky_annulus, ccd = skyargs

    im = survey.get_image_object(ccd)
    hdr = im.read_image_header()
    hdr.delete('INHERIT')
    hdr.delete('EXTVER')

    print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
          'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
          'object', getattr(ccd, 'object', None))

    radius_mask = np.round(radius_mask_arcsec / im.pixscale).astype('int') # [pixels]
    
    tim = im.get_tractor_image(splinesky=True, subsky=False, hybridPsf=True,
                               normalizePsf=True, apodize=apodize)

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
    galmask = (xmask**2 + ymask**2) <= radius_mask**2

    # Optionally define an annulus of sky pixels centered on the object of interest.
    if sky_annulus:
        inmask = (xmask**2 + ymask**2) <= 2*radius_mask**2
        outmask = (xmask**2 + ymask**2) <= 5*radius_mask**2
        skymask = (outmask*1 - inmask*1 - galmask*1) == 1
    else:
        skymask = np.ones_like(img).astype(bool)

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

    #print('ahack!!', im.expnum, im.ccdname)
    #if im.expnum == 625736 and im.ccdname == 'S22':
    try:
        skymean, skymed, skysig = sigma_clipped_stats(img, mask=~skypix, sigma=3.0)
    except:
        skymean, skymed, skysig = 0.0, 0.0, 0.0
    #skysig = 1.0 / np.sqrt(np.median(ivar[skypix]))
    #skymed = np.median(img[skypix])
    try:
        skymode = estimate_mode(img[skypix], raiseOnWarn=True).astype('f4')
    except:
        skymode = np.array(0.0).astype('f4')

    # Build the final bit-mask image.
    #   0    = 
    #   2**0 = refmask  - reference stars and galaxies
    #   2**1 = objmask  - threshold-detected objects
    #   2**2 = galmask  - central galaxy & system
    mask = np.zeros_like(img).astype(np.int16)
    #mask[ivarmask] += 2**0
    mask[refmask]  += 2**0
    mask[objmask]  += 2**1
    mask[galmask]  += 2**2

    # Resample the mask onto the final mosaic image.
    HH, WW = brickwcs.shape
    comask = np.zeros((HH, WW), np.int16)
    try:
        Yo, Xo, Yi, Xi, _ = resample_with_wcs(brickwcs, targetwcs)
        comask[Yo, Xo] = mask[Yi, Xi]
    except:
        pass

    # Add the sky values and also the central pixel coordinates of the object of
    # interest (so we won't need the WCS object downstream, in QA).
    for card, value in zip(('SKYMODE', 'SKYMED', 'SKYMEAN', 'SKYSIG'),
                           (skymode, skymed, skymean, skysig)):
        hdr.add_record(dict(name=card, value=value))
    hdr.add_record(dict(name='XCEN', value=x0-1)) # 0-indexed
    hdr.add_record(dict(name='YCEN', value=y0-1))

    customsky = fits_table()
    customsky.camera = [im.camera]
    customsky.expnum = [im.expnum]
    customsky.ccdname = [im.ccdname]
    customsky.skymode = [skymode]
    customsky.skymed = [skymed]
    customsky.skymean = [skymean]
    customsky.skysig = [skysig]
    customsky.xcen = [x0 - 1] # 0-indexed
    customsky.ycen = [y0 - 1]
    customsky.to_np_arrays()

    out = dict()
    ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
    #ext = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
    out['{}-mask'.format(ext)] = mask
    out['{}-image'.format(ext)] = img
    out['{}-splinesky'.format(ext)] = splineskytable
    out['{}-header'.format(ext)] = hdr
    out['{}-customsky'.format(ext)] = customsky
    out['{}-comask'.format(ext)] = comask
    
    return out

def custom_coadds(onegal, galaxy=None, survey=None, radius_mosaic=None,
                  radius_mask=None, nproc=1, pixscale=0.262, log=None,
                  apodize=False, plots=False, verbose=False, cleanup=True,
                  write_ccddata=True, sky_annulus=True, centrals=True):
    """Build a custom set of coadds for a single galaxy, with a custom mask and sky
    model.

    radius_mosaic and radius_mask in arcsec

    centrals - if this is the centrals project (legacyhalos or HSC) then deal
      with the central with dedicated code.

    """
    import copy
    import tractor
    
    from legacypipe.runbrick import stage_tims
    from legacypipe.runbrick import _get_mod
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.survey import get_rgb, imsave_jpeg
    from legacypipe.image import DQ_BITS
            
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

    if radius_mask is None:
        radius_mask = radius_mosaic
        radius_search = 5.0 # [arcsec]
    else:
        radius_search = radius_mask
        
    width = _mosaic_width(radius_mosaic, pixscale)

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
                          depth_cut=False, apodize=apodize, do_calibs=False, rex=True, 
                          unwise_dir=unwise_dir, plots=plots, ps=ps)

    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P = call_stage_tims()
    else:
        P = call_stage_tims()

    tims, brickwcs = P['tims'], P['targetwcs']
    bands, version_header = P['bands'], P['version_header']
    del P

    # Read and apply the outlier masks.
    outliersfile = os.path.join(survey.output_dir, '{}-outlier-mask.fits.fz'.format(galaxy))
    if not os.path.isfile(outliersfile):
        print('Missing outliers masks {}'.format(outliersfile), flush=True, file=log)
        return 0

    outliers = fitsio.FITS(outliersfile)
    for tim in tims:
        ext = '{}-{}-{}'.format(tim.imobj.camera, tim.imobj.expnum,
                                tim.imobj.ccdname.lower().strip())
        if ext in outliers:
            mask = outliers[ext].read()
            maskhdr = outliers[ext].read_header()
            tim.dq |= (mask > 0) * DQ_BITS['outlier']
            tim.inverr[mask > 0] = 0.0
        else:
            print('Warning: extension {} not found in image {}'.format(ext, outliersfile), flush=True, file=log)

    # [2] Derive the custom mask and sky background for each (full) CCD and
    # write out a MEF -custom-mask.fits.gz file.
    skyargs = [(survey, brickname, brickwcs, onegal, radius_mask, apodize, sky_annulus, _ccd)
               for _ccd in survey.ccds]
    result = mp.map(_custom_sky, skyargs)
    #result = list( zip( *mp.map(_custom_sky, args) ) )
    sky = dict()
    [sky.update(res) for res in result]
    del result

    if write_ccddata:
        # Write out the "coadd" mask.
        cokeys = [key for key in sky.keys() if 'comask' in key]
        _comask = np.array([sky[cokey] for cokey in cokeys])

        comask = np.bitwise_or.reduce(_comask, axis=0)
        hdr = fitsio.FITSHDR()
        brickwcs.add_to_header(hdr)
        hdr.delete('IMAGEW')
        hdr.delete('IMAGEH')

        maskfile = os.path.join(survey.output_dir, '{}-custom-mask-grz.fits.gz'.format(galaxy))
        fitsio.write(maskfile, comask, header=hdr, clobber=True)
        print('Writing {}'.format(maskfile), flush=True, file=log)
        del comask

        skyfile = os.path.join(survey.output_dir, '{}-pipeline-sky.fits'.format(galaxy))
        print('Writing {}'.format(skyfile), flush=True, file=log)
        if os.path.isfile(skyfile):
            os.remove(skyfile)
        for ii, ccd in enumerate(survey.ccds):
            im = survey.get_image_object(ccd)
            ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
            sky['{}-splinesky'.format(ext)].write_to(skyfile, append=ii>0, extname=ext)
        
        skyfile = os.path.join(survey.output_dir, '{}-custom-sky.fits'.format(galaxy))
        print('Writing {}'.format(skyfile), flush=True, file=log)
        if os.path.isfile(skyfile):
            os.remove(skyfile)
        for ii, ccd in enumerate(survey.ccds):
            im = survey.get_image_object(ccd)
            ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
            sky['{}-customsky'.format(ext)].write_to(skyfile, append=ii>0, extname=ext)
        
        # Write out separate CCD-level files with the images/data and individual
        # masks (converted to unsigned integer).  These are still pretty big and
        # I'm not sure we will ever need them.  Keep the code here for legacy
        # value but don't write out.
        if False:
            ccdfile = os.path.join(survey.output_dir, '{}-custom-ccdmask-grz.fits.gz'.format(galaxy))
            print('Writing {}'.format(ccdfile), flush=True, file=log)
            if os.path.isfile(ccdfile):
                os.remove(ccdfile)
            with fitsio.FITS(ccdfile, 'rw') as ff:
                for ii, ccd in enumerate(survey.ccds):
                    im = survey.get_image_object(ccd)
                    ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
                    hdr = sky['{}-header'.format(ext)]
                    ff.write(sky['{}-mask'.format(ext)], extname=ext, header=hdr)

        # These are the actual images, which results in a giant file.  Keeping
        # the code here for legacy purposes but I'm not sure we should ever
        # write it out.
        if False:
            ccdfile = os.path.join(survey.output_dir, '{}-ccddata-grz.fits.fz'.format(galaxy))
            print('Writing {}'.format(ccdfile), flush=True, file=log)
            if os.path.isfile(ccdfile):
                os.remove(ccdfile)
            with fitsio.FITS(ccdfile, 'rw') as ff:
                for ii, ccd in enumerate(survey.ccds):
                    im = survey.get_image_object(ccd)
                    ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
                    hdr = sky['{}-header'.format(ext)]
                    ff.write(sky['{}-image'.format(ext)].astype('f4'), extname=ext, header=hdr)

    # [3] Modify each tim by subtracting our new estimate of the sky. Then
    # sky-subtract the pipeline_tims so we can build coadds without the central,
    # below.
    custom_tims, pipeline_tims = [], []
    for tim in tims:
        custom_tim = copy.deepcopy(tim)
        image = custom_tim.getImage()
        ext = '{}-{}-{}'.format(custom_tim.imobj.camera, custom_tim.imobj.expnum,
                                custom_tim.imobj.ccdname.lower().strip())
        newsky = sky['{}-header'.format(ext)]['SKYMED']
        custom_tim.setImage(image - newsky)
        custom_tim.sky = tractor.sky.ConstantSky(0)
        custom_tims.append(custom_tim)
        del custom_tim

        pipeline_tim = copy.deepcopy(tim)
        pipesky = np.zeros_like(tim.getImage())
        pipeline_tim.sky.addTo(pipesky)
        pipeline_tim.setImage(image - pipesky)
        pipeline_tim.sky = tractor.sky.ConstantSky(0)
        pipeline_tims.append(pipeline_tim)
        del pipeline_tim
    del sky, tims

    # [4] Read the pipeline Tractor catalog and update the individual-object
    # photometry measured from the custom sky-subtracted CCDs.
    tractorfile = os.path.join(survey.output_dir, '{}-pipeline-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile), flush=True, file=log)
        return 0
    pipeline_cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(pipeline_cat), tractorfile), flush=True, file=log)

    pipeline_srcs = read_fits_catalog(pipeline_cat, fluxPrefix='')
    custom_srcs = [src.copy() for src in pipeline_srcs]

    print('Performing forced photometry on the custom sky-subtracted CCDs.', flush=True, file=log)
    forcedflx = mp.map(_forced_phot, [(custom_tims, custom_srcs, band) for band in bands])
    if False:
        forcedflx = []
        for band in bands:
            t0 = time.time()
            forcedflx.append(forced_phot(custom_tims, custom_srcs, band))
            print('  band {} took {:.3f} sec'.format(band, time.time()-t0), flush=True, file=log)

    # Populate the new custom catalog and write out.
    custom_cat = pipeline_cat.copy()
    for band, flux, ivar in forcedflx:
        custom_cat.set('flux_{}'.format(band), flux.astype('f4'))
        custom_cat.set('flux_ivar_{}'.format(band), ivar.astype('f4'))
        
    tractorfile = os.path.join(survey.output_dir, '{}-custom-tractor.fits'.format(galaxy))
    if os.path.isfile(tractorfile):
        os.remove(tractorfile)
    custom_cat.writeto(tractorfile)
    print('Wrote {} sources to {}'.format(len(custom_cat), tractorfile), flush=True, file=log)
        
    custom_srcs = read_fits_catalog(custom_cat, fluxPrefix='')

    # [5] Next, render the model image of each CCD, with and without the central
    # large galaxy (custom and pipeline).

    # Custom code for dealing with centrals.
    keep = isolate_central(custom_cat, brickwcs, centrals=centrals)

    custom_srcs_nocentral = np.array(custom_srcs)[keep].tolist()
    pipeline_srcs_nocentral = np.array(pipeline_srcs)[keep].tolist()
    
    print('Rendering model images with and without surrounding galaxies...', flush=True, file=log)
    custom_mods = mp.map(_get_mod, [(tim, custom_srcs) for tim in custom_tims])
    pipeline_mods = mp.map(_get_mod, [(tim, pipeline_srcs) for tim in pipeline_tims])

    custom_mods_nocentral = mp.map(_get_mod, [(tim, custom_srcs_nocentral) for tim in custom_tims])
    pipeline_mods_nocentral = mp.map(_get_mod, [(tim, pipeline_srcs_nocentral) for tim in pipeline_tims])
    
    #import matplotlib.pyplot as plt ; plt.imshow(np.log10(mod), origin='lower') ; plt.savefig('junk.png')    
    #pdb.set_trace()

    # [6] Build the custom coadds, with and without the surrounding galaxies.
    print('Producing coadds...', flush=True, file=log)
    def call_make_coadds(usemods, usetims):
        return make_coadds(usetims, bands, brickwcs, mods=usemods, mp=mp,
                           callback=write_coadd_images,
                           callback_args=(survey, brickname, version_header, 
                                          usetims, brickwcs))

    # Custom coadds--all galaxies. (Pipeline coadds already exist.)
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C = call_make_coadds(custom_mods, custom_tims)
            #P = call_make_coadds(pipeline_mods, pipeline_tims)
    else:
        C = call_make_coadds(custom_mods, custom_tims)
        #P = call_make_coadds(pipeline_mods, pipeline_tims)

    for suffix in ('image', 'model'):
        for band in bands:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                os.path.join(survey.output_dir, '{}-custom-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
                #os.path.join(survey.output_dir, '{}-custom-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok

    # Custom coadds--without the central. 
    imtype = 'custom'
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C_nocentral = call_make_coadds(custom_mods_nocentral, custom_tims)
    else:
        C_nocentral = call_make_coadds(custom_mods_nocentral, custom_tims)
    for band in bands:
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                               brickname, 'legacysurvey-{}-model-{}.fits.fz'.format(
                brickname, band)),
            os.path.join(survey.output_dir, '{}-{}-model-nocentral-{}.fits.fz'.format(galaxy, imtype, band)) )
        if not ok:
            return ok
            
    # Pipeline coadds--without the central. 
    imtype = 'pipeline'
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P_nocentral = call_make_coadds(pipeline_mods_nocentral, pipeline_tims)
    else:
        P_nocentral = call_make_coadds(pipeline_mods_nocentral, pipeline_tims)
    for band in bands:
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                               brickname, 'legacysurvey-{}-model-{}.fits.fz'.format(
                brickname, band)),
            os.path.join(survey.output_dir, '{}-{}-model-nocentral-{}.fits.fz'.format(galaxy, imtype, band)) )
        if not ok:
            return ok
            
    # [7] Finally, build png images.
    def call_make_png(R, imtype, nocentral=False):
        rgbkwargs = dict(mnmx=(-1, 100), arcsinh=1)
        #rgbkwargs_resid = dict(mnmx=(0.1, 2), arcsinh=1)
        rgbkwargs_resid = dict(mnmx=(-1, 100), arcsinh=1)

        if nocentral:
            coadd_list = [('{}-model-nocentral'.format(imtype), R.comods, rgbkwargs),
                          ('{}-image-central'.format(imtype), R.coresids, rgbkwargs_resid)]
        else:
            coadd_list = [('custom-image', R.coimgs,   rgbkwargs),
                          ('custom-model', R.comods,   rgbkwargs),
                          ('custom-resid', R.coresids, rgbkwargs_resid)]

        for suffix, ims, rgbkw in coadd_list:
            rgb = get_rgb(ims, bands, **rgbkw)
            kwa = {}
            outfn = os.path.join(survey.output_dir, '{}-{}-grz.jpg'.format(galaxy, suffix))
            print('Writing {}'.format(outfn), flush=True, file=log)
            imsave_jpeg(outfn, rgb, origin='lower', **kwa)
            del rgb

    call_make_png(C, 'custom', nocentral=False)
    call_make_png(C_nocentral, 'custom', nocentral=True)
    call_make_png(P_nocentral, 'pipeline', nocentral=True)

    if cleanup:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
        for stage in ('srcs', 'checkpoint'):
            picklefile = os.path.join(survey.output_dir, '{}-runbrick-{}.p'.format(galaxy, stage))
            if os.path.isfile(picklefile):
                os.remove(picklefile)

    return 1
