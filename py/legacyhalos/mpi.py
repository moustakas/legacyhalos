"""
legacyhalos.mpi
===============

Code to deal with the MPI portion of the pipeline.

"""
import os, time, subprocess, pdb
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import legacyhalos.io
import legacyhalos.html

def _start(galaxy, log=None, seed=None):
    if seed:
        print('Random seed = {}'.format(seed), flush=True)        
    print('Started working on galaxy {} at {}'.format(
        galaxy, time.asctime()), flush=True, file=log)

def _done(galaxy, galaxydir, err, t0, stage, filesuffix=None, log=None):
    if filesuffix is None:
        suffix = ''
    else:
        suffix = '-{}'.format(filesuffix)
    if err == 0:
        print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
        donefile = os.path.join(galaxydir, '{}{}-{}.isfail'.format(galaxy, suffix, stage))
    else:
        donefile = os.path.join(galaxydir, '{}{}-{}.isdone'.format(galaxy, suffix, stage))
        
    cmd = 'touch {}'.format(donefile)
    subprocess.call(cmd.split())
        
    print('Finished galaxy {} in {:.3f} minutes.'.format(
          galaxy, (time.time() - t0)/60), flush=True, file=log)
    
def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1, verbose=False,
                 debug=False, logfile=None, input_ellipse=None, zcolumn=None,
                 sdss=False, sdss_pixscale=0.396, unwise=False, unwise_pixscale=2.75,
                 galex=False, galex_pixscale=1.5, largegalaxy=False, pipeline=True):
    """Wrapper script to do ellipse-fitting.

    """
    import legacyhalos.ellipse

    # Do not force zcolumn here; it's not always wanted or needed in ellipse.
    #if zcolumn is None:
    #    zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err, filesuffix = legacyhalos.ellipse.legacyhalos_ellipse(
            onegal, galaxy=galaxy, galaxydir=galaxydir,
            pixscale=pixscale, nproc=nproc,
            zcolumn=zcolumn, input_ellipse=input_ellipse,
            verbose=verbose, debug=debug,
            sdss=sdss, sdss_pixscale=sdss_pixscale,
            unwise=unwise, unwise_pixscale=unwise_pixscale,
            galex=galex, galex_pixscale=galex_pixscale,
            largegalaxy=largegalaxy, pipeline=pipeline)
        _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err, filesuffix = legacyhalos.ellipse.legacyhalos_ellipse(
                    onegal, galaxy=galaxy, galaxydir=galaxydir,
                    pixscale=pixscale, nproc=nproc,
                    zcolumn=zcolumn, input_ellipse=input_ellipse,
                    verbose=verbose, debug=debug,
                    sdss=sdss, sdss_pixscale=sdss_pixscale,
                    unwise=unwise, unwise_pixscale=unwise_pixscale,
                    galex=galex, galex_pixscale=galex_pixscale,
                    largegalaxy=largegalaxy, pipeline=pipeline)
                _done(galaxy, galaxydir, err, t0, 'ellipse', filesuffix, log=log)

def call_sersic(onegal, galaxy, galaxydir, seed, verbose, debug, logfile):
    """Wrapper script to do Sersic-fitting.

    """
    import legacyhalos.sersic

    t0 = time.time()
    if debug:
        _start(galaxy, seed=seed)
        err = legacyhalos.sersic.legacyhalos_sersic(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                    debug=debug, verbose=verbose, seed=seed)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log, seed=seed)
                err = legacyhalos.sersic.legacyhalos_sersic(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                            debug=debug, verbose=verbose, seed=seed)
                _done(galaxy, err, t0, log=log)

def call_sky(onegal, galaxy, galaxydir, survey, seed, nproc, pixscale,
              verbose, debug, logfile):
    """Wrapper script to do Sersic-fitting.

    """
    import legacyhalos.sky

    t0 = time.time()
    if debug:
        _start(galaxy, seed=seed)
        err = legacyhalos.sky.legacyhalos_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                              nproc=nproc, pixscale=pixscale, seed=seed,
                                              debug=debug, verbose=verbose, force=force)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log, seed=seed)
                err = legacyhalos.sky.legacyhalos_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                                      nproc=nproc, pixscale=pixscale, seed=seed,
                                                      debug=debug, verbose=verbose, force=force)
                _done(galaxy, err, t0, log=log)
                
def call_htmlplots(onegal, galaxy, survey, pixscale=0.262, nproc=1, 
                   verbose=False, debug=False, clobber=False, ccdqa=False,
                   logfile=None, zcolumn='Z', datadir=None, htmldir=None, 
                   largegalaxy=False, 
                   barlen=None, barlabel=None, radius_mosaic_arcsec=None,
                   get_galaxy_galaxydir=None):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if debug:
        _start(galaxy)
        err = legacyhalos.html.make_plots(
            onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
            pixscale=pixscale, zcolumn=zcolumn, nproc=nproc,
            barlen=barlen, barlabel=barlabel,
            radius_mosaic_arcsec=radius_mosaic_arcsec,
            maketrends=False, ccdqa=ccdqa,
            clobber=clobber, verbose=verbose, 
            largegalaxy=largegalaxy, 
            get_galaxy_galaxydir=get_galaxy_galaxydir)
        _done(galaxy, survey.output_dir, err, t0, 'html')
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.html.make_plots(
                    onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
                    pixscale=pixscale, zcolumn=zcolumn, nproc=nproc,
                    barlen=barlen, barlabel=barlabel,
                    radius_mosaic_arcsec=radius_mosaic_arcsec,
                    maketrends=False, ccdqa=ccdqa,
                    clobber=clobber, verbose=verbose,
                    largegalaxy=largegalaxy, 
                    get_galaxy_galaxydir=get_galaxy_galaxydir)
                _done(galaxy, survey.output_dir, err, t0, 'html')

def call_custom_coadds(onegal, galaxy, survey, run, radius_mosaic, nproc=1,
                       pixscale=0.262, racolumn='RA', deccolumn='DEC',
                       largegalaxy=False, pipeline=False, custom=True,
                       apodize=False, unwise=True, galex=False, force=False, plots=False,
                       verbose=False, cleanup=True, write_all_pickles=False,
                       no_splinesky=False, customsky=False,
                       just_coadds=False, require_grz=True, 
                       no_gaia=False, no_tycho=False,
                       debug=False, logfile=None):
    """Wrapper script to build custom coadds.

    radius_mosaic in arcsec

    """
    import legacyhalos.coadds
    
    t0 = time.time()
    if debug:
        _start(galaxy)
        err, filesuffix = legacyhalos.coadds.custom_coadds(
            onegal, galaxy=galaxy, survey=survey, 
            radius_mosaic=radius_mosaic, nproc=nproc, 
            pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn,
            largegalaxy=largegalaxy, pipeline=pipeline, custom=custom,
            run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
            verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
            no_splinesky=no_splinesky, customsky=customsky, just_coadds=just_coadds,
            require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho)
        _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err, filesuffix = legacyhalos.coadds.custom_coadds(
                    onegal, galaxy=galaxy, survey=survey, 
                    radius_mosaic=radius_mosaic, nproc=nproc, 
                    pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn, 
                    largegalaxy=largegalaxy, pipeline=pipeline, custom=custom,
                    run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
                    verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                    no_splinesky=no_splinesky, customsky=customsky, just_coadds=just_coadds,
                    require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho,
                    log=log)
                _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix, log=log)
