"""
legacyhalos.coadds
==================

"""
import os, pdb
import numpy as np

from legacyhalos.misc import custom_brickname

def _mosaic_width(radius_mosaic, pixscale):
    """Ensure the mosaic is an odd number of pixels so the central can land on a
    whole pixel (important for ellipse-fitting).

    radius_mosaic in arcsec

    """
    #width = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]
    width = 2 * radius_mosaic / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width

def _rearrange_files(galaxy, output_dir, brickname, stagesuffix, run,
                     unwise=True, galex=False, cleanup=False, just_coadds=False,
                     clobber=False, require_grz=True):
    """Move (rename) files into the desired output directory and clean up.

    """
    import fitsio

    def _copyfile(infile, outfile, clobber=False, update_header=False):
        if os.path.isfile(outfile) and not clobber:
            return 1
        if os.path.isfile(infile):
            os.rename(infile, outfile)
            if update_header:
                pass
            return 1
        else:
            print('Missing file {}; please check the logfile.'.format(infile))
            return 0

    def _do_cleanup():
        import shutil
        from glob import glob
        shutil.rmtree(os.path.join(output_dir, 'coadd'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'metrics'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor-i'), ignore_errors=True)
        picklefiles = glob(os.path.join(output_dir, '{}-{}-*.p'.format(galaxy, stagesuffix)))
        for picklefile in picklefiles:
            if os.path.isfile(picklefile):
                os.remove(picklefile)

    # If we made it here and there is no CCDs file it's because legacypipe
    # exited cleanly with "No photometric CCDs touching brick."
    ccdsfile = os.path.join(output_dir, 'coadd', 'cus', brickname,
                            'legacysurvey-{}-ccds.fits'.format(brickname))
    if not os.path.isfile(ccdsfile):
        print('No photometric CCDs touching brick.')
        if cleanup:
            _do_cleanup()
        return 1
    
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-ccds.fits'.format(brickname)),
                     os.path.join(output_dir, '{}-ccds-{}.fits'.format(galaxy, run)),
        clobber=clobber)
    if not ok:
        return ok

    ccdsfile = os.path.join(output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
    allbands = fitsio.read(ccdsfile, columns='filter')
    bands = list(sorted(set(allbands)))

    # For objects on the edge of the footprint we can sometimes lose 3-band
    # coverage if one of the bands is fully masked. Check here and write out all
    # the files except a 
    if require_grz and ('g' not in bands or 'r' not in bands or 'z' not in bands):
        print('Lost grz coverage and require_grz=True.')
        if cleanup:
            _do_cleanup()
        return 1

    # image coadds (FITS + JPG)
    for band in bands:
        for imtype, outtype in zip(('image', 'invvar'), ('image', 'invvar')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                             os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, outtype, band)),
                clobber=clobber, update_header=True)
            if not ok:
                return ok

    # JPG images
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-image.jpg'.format(brickname)),
        os.path.join(output_dir, '{}-{}-image-grz.jpg'.format(galaxy, stagesuffix)),
        clobber=clobber)
    if not ok:
        return ok

    if just_coadds:
        if cleanup:
            _do_cleanup()
        return 1

    # PSFs (none for stagesuffix=='pipeline')
    if stagesuffix != 'pipeline':
        for band in bands:
            for imtype, outtype in zip(['copsf'], ['psf']):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                                 os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, outtype, band)),
                    clobber=clobber)
                if not ok:
                    return ok

    # tractor catalog
    ok = _copyfile(
        os.path.join(output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
        os.path.join(output_dir, '{}-{}-tractor.fits'.format(galaxy, stagesuffix)),
        clobber=clobber)
    if not ok:
        return ok

    # Maskbits, blob images, outlier masks, and depth images.
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-maskbits.fits.fz'.format(brickname)),
        os.path.join(output_dir, '{}-{}-maskbits.fits.fz'.format(galaxy, stagesuffix)),
        clobber=clobber)
    if not ok:
        return ok

    ok = _copyfile(
        os.path.join(output_dir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)),
        os.path.join(output_dir, '{}-{}-blobs.fits.gz'.format(galaxy, stagesuffix)),
        clobber=clobber)
    if not ok:
        return ok

    ok = _copyfile(
        os.path.join(output_dir, 'metrics', 'cus', 'outlier-mask-{}.fits.fz'.format(brickname)),
        os.path.join(output_dir, '{}-{}-outlier-mask.fits.fz'.format(galaxy, stagesuffix)),
        clobber=clobber)
    if not ok:
        return ok

    for band in ['g', 'r', 'z']:
        ok = _copyfile(
            os.path.join(output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
            os.path.join(output_dir, '{}-depth-{}.fits.fz'.format(galaxy, band)),
            clobber=clobber)
        if not ok:
            return ok

    # model coadds
    for band in bands:
        for imtype in ['model']:
        #for imtype in ('model', 'blobmodel'):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, imtype, band)),
                clobber=clobber)
            if not ok:
                return ok

    # JPG images
    for imtype in ('model', 'resid'):
        ok = _copyfile(
            os.path.join(output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
            os.path.join(output_dir, '{}-{}-{}-grz.jpg'.format(galaxy, stagesuffix, imtype)),
            clobber=clobber)
        if not ok:
            return ok

    # Note that the WISE images can get generated by one or all of the
    # custom_coadds() stages (custom, pipeline, largegalaxy). In this case, the
    # images will be the same, but the *model* images can be different (because
    # of the forced photometry), so include the stagesuffix.
    if unwise:
        for band in ('W1', 'W2', 'W3', 'W4'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)),
                    clobber=clobber)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-model-{}.fits.fz'.format(brickname, band)),
                os.path.join(output_dir, '{}-{}-model-{}.fits.fz'.format(galaxy, stagesuffix, band)),
                    clobber=clobber)
            if not ok:
                return ok

        for imtype, suffix in zip(('wise', 'wisemodel'), ('image', 'model')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(output_dir, '{}-{}-W1W2.jpg'.format(galaxy, suffix)),
                    clobber=clobber)
            if not ok:
                return ok

    if galex:
        for band in ('FUV', 'NUV'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)),
                    clobber=clobber)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-model-{}.fits.fz'.format(brickname, band)),
                os.path.join(output_dir, '{}-{}-model-{}.fits.fz'.format(galaxy, stagesuffix, band)),
                    clobber=clobber)
            if not ok:
                return ok

        for imtype, suffix in zip(('galex', 'galexmodel'), ('image', 'model')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(output_dir, '{}-{}-FUVNUV.jpg'.format(galaxy, suffix)),
                    clobber=clobber)
            if not ok:
                return ok

    if cleanup:
        _do_cleanup()

    return 1

def get_ccds(survey, ra, dec, pixscale, width):
    """Quickly get the CCDs touching this custom brick.  This code is mostly taken
    from legacypipe.runbrick.stage_tims.

    """
    from legacypipe.survey import wcs_for_brick, BrickDuck
    brickname = 'custom-{}'.format(custom_brickname(ra, dec))
    brick = BrickDuck(ra, dec, brickname)

    targetwcs = wcs_for_brick(brick, W=width, H=width, pixscale=pixscale)
    ccds = survey.ccds_touching_wcs(targetwcs)
    if ccds is None or np.sum(ccds.ccd_cuts == 0) == 0:
        return []
    ccds.cut(ccds.ccd_cuts == 0)
    ccds.cut(np.array([b in ['g', 'r', 'z'] for b in ccds.filter]))

    return ccds

def custom_coadds(onegal, galaxy=None, survey=None, radius_mosaic=None,
                  nproc=1, pixscale=0.262, run='south', racolumn='RA', deccolumn='DEC', 
                  largegalaxy=False, pipeline=False, custom=True,
                  log=None, apodize=False, unwise=True, galex=False, force=False,
                  plots=False, verbose=False, cleanup=True,
                  write_all_pickles=False, no_subsky=False, subsky_radii=None,
                  customsky=False, just_coadds=False, require_grz=True, no_gaia=False,
                  no_tycho=False):
    """Build a custom set of large-galaxy coadds

    radius_mosaic in arcsec

    You must specify *one* of the following:
      * pipeline - standard call to runbrick
      * custom - for the cluster centrals project; calls stage_largegalaxies but
        with custom sky-subtraction
      * largegalaxy - for the LSLGA project; calls stage_largegalaxies

    """
    import subprocess
    
    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()
        
    if galaxy is None:
        galaxy = 'galaxy'

    if custom:
        stagesuffix = 'custom'
    elif largegalaxy:
        stagesuffix = 'largegalaxy'
    elif pipeline:
        stagesuffix = 'pipeline'
    else:
        print('You must specify at least one of largegalaxy, pipeline, or custom!')
        return 0, ''

    width = _mosaic_width(radius_mosaic, pixscale)
    brickname = 'custom-{}'.format(custom_brickname(onegal[racolumn], onegal[deccolumn]))

    # Quickly read the input CCDs and check that we have all the colors we need.
    bands = ['g', 'r', 'z']
    ccds = get_ccds(survey, onegal[racolumn], onegal[deccolumn], pixscale, width)
    if len(ccds) == 0:
        print('No CCDs touching this brick; nothing to do.')
        return 1, stagesuffix
    
    usebands = list(sorted(set(ccds.filter)))
    these = [filt in usebands for filt in bands]
    print('Bands touching this brick, {}'.format(' '.join([filt for filt in usebands])))
    if np.sum(these) != 3 and require_grz:
        print('Missing imaging in grz and require_grz=True; nothing to do.')
        ccdsfile = os.path.join(survey.output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
        # should we write out the CCDs file?
        print('Writing {} CCDs to {}'.format(len(ccds), ccdsfile))
        ccds.writeto(ccdsfile, overwrite=True)
        return 1, stagesuffix

    # Run the pipeline!
    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} '
    cmd += '--survey-dir {survey_dir} --run {run} '
    if write_all_pickles:
        pass
        #cmd += '--write-stage tims --write-stage srcs '
    else:
        cmd += '--write-stage srcs '
    cmd += '--skip-calibs '
    cmd += '--checkpoint {galaxydir}/{galaxy}-{stagesuffix}-checkpoint.p '
    cmd += '--pickle {galaxydir}/{galaxy}-{stagesuffix}-%%(stage)s.p '
    if just_coadds:
        unwise = False
        cmd += '--stage image_coadds --early-coadds '
    if not unwise:
        cmd += '--no-unwise-coadds --no-wise '
    if galex:
        cmd += '--galex '
    if apodize:
        cmd += '--apodize '
    if no_gaia:
        cmd += '--no-gaia '
    if no_tycho:
        cmd += '--no-tycho '
    if force:
        cmd += '--force-all '
        checkpointfile = '{galaxydir}/{galaxy}-{stagesuffix}-checkpoint.p'.format(
            galaxydir=survey.output_dir, galaxy=galaxy, stagesuffix=stagesuffix)
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
    if no_subsky and subsky_radii:
        if len(subsky_radii) != 3:
            print('subsky_radii must be a 3-element vector')
        cmd += '--no-subsky --subsky-radii {} {} {} '.format(subsky_radii[0], subsky_radii[1], subsky_radii[2]) # [arcsec]
    if customsky:
        print('Skipping custom sky')
        #cmd += '--largegalaxy-skysub '
        #print('HACK!!!!!!!!!!!!!!!!! doing just largegalaxies stage in legacyhalos.coadds')
        #cmd += '--stage largegalaxies '

    # stage-specific options here--
    if largegalaxy:
        cmd += '--fit-on-coadds --saddle-fraction 0.2 --saddle-min 4.0 '
        #cmd += '--nsigma 10 '
    if custom:
        cmd += '--fit-on-coadds '

    #cmd += '--stage srcs ' ; cleanup = False
    #cmd += '--stage fitblobs ' ; cleanup = False
    #cmd += '--stage coadds ' ; cleanup = False
    #cmd += '--stage wise_forced ' ; cleanup = False

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_CODE_DIR'), galaxy=galaxy,
                     ra=onegal[racolumn], dec=onegal[deccolumn], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     galaxydir=survey.output_dir, survey_dir=survey.survey_dir, run=run,
                     stagesuffix=stagesuffix)
    print(cmd, flush=True, file=log)

    err = subprocess.call(cmd.split(), stdout=log, stderr=log)

    if err != 0:
        print('Something went wrong; please check the logfile.')
        return 0, stagesuffix
    else:
        # Move (rename) files into the desired output directory and clean up.
        ok = _rearrange_files(galaxy, survey.output_dir, brickname, stagesuffix,
                              run, unwise=unwise, galex=galex, cleanup=cleanup,
                              just_coadds=just_coadds, clobber=force,
                              require_grz=require_grz)
        return ok, stagesuffix
