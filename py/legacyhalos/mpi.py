"""
legacyhalos.mpi
===============

Code to deal with the MPI portion of the pipeline.

"""
import os, time, pdb
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import legacyhalos.io
import legacyhalos.html

def _start(galaxy, log=None, seed=None):
    if seed:
        print('Random seed = {}'.format(seed), flush=True)        
    print('Started working on galaxy {} at {}'.format(
        galaxy, time.asctime()), flush=True, file=log)

def _done(galaxy, err, t0, log=None):
    if err == 0:
        print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
    print('Finished galaxy {} in {:.3f} minutes.'.format(
          galaxy, (time.time() - t0)/60), flush=True, file=log)

def call_pipeline_coadds(onegal, galaxy, survey, radius_mosaic, nproc=1,
                         pixscale=0.262, racolumn='RA', deccolumn='DEC',
                         apodize=False, unwise=True, force=False, plots=False,
                         verbose=False, cleanup=True, write_all_pickles=False,
                         no_splinesky=False, just_coadds=False,
                         no_large_galaxies=False, no_gaia=False, no_tycho=False,
                         debug=False, logfile=None):
    """Wrapper script to build the pipeline coadds.

    radius_mosaic in arcsec

    """
    run = legacyhalos.io.get_run(onegal)
    
    t0 = time.time()
    if debug:
        _start(galaxy)
        err = legacyhalos.coadds.pipeline_coadds(onegal, galaxy=galaxy, survey=survey, 
                                                 radius_mosaic=radius_mosaic, nproc=nproc, 
                                                 pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn, run=run,
                                                 apodize=apodize, unwise=unwise, force=force, plots=plots,
                                                 verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                                                 no_splinesky=no_splinesky, just_coadds=just_coadds,
                                                 no_large_galaxies=no_large_galaxies, no_gaia=no_gaia, no_tycho=no_tycho)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.coadds.pipeline_coadds(onegal, galaxy=galaxy, survey=survey, 
                                                         radius_mosaic=radius_mosaic, nproc=nproc, 
                                                         pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn, run=run,
                                                         apodize=apodize, unwise=unwise, force=force, plots=plots,
                                                         verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                                                         no_splinesky=no_splinesky, just_coadds=just_coadds,
                                                         no_large_galaxies=no_large_galaxies, no_gaia=no_gaia, no_tycho=no_tycho,
                                                         log=log)
                _done(galaxy, err, t0, log=log)

def call_custom_coadds(onegal, galaxy, radius_mosaic, survey, pixscale=0.262,
                        nproc=1, debug=False, logfile=None, radius_mask=None,
                        sdss=False, sdss_pixscale=0.396, write_ccddata=False,
                        doforced_phot=True, apodize=False):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()
    if debug:
        _start(galaxy)
        if sdss:
            err = legacyhalos.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                 survey=survey, radius_mask=radius_mask, pixscale=sdss_pixscale,
                                                 nproc=nproc)
        else:
            err = legacyhalos.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                   survey=survey, radius_mask=radius_mask, pixscale=pixscale,
                                                   nproc=nproc, write_ccddata=write_ccddata,
                                                   doforced_phot=doforced_phot, apodize=apodize)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                if sdss:
                    err = legacyhalos.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                         survey=survey, radius_mask=radius_mask, pixscale=sdss_pixscale,
                                                         nproc=nproc, log=log)
                else:
                    err = legacyhalos.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                           survey=survey, radius_mask=radius_mask, pixscale=pixscale,
                                                           nproc=nproc, log=log, write_ccddata=write_ccddata,
                                                           doforced_phot=doforced_phot, apodize=apodize)
                _done(galaxy, err, t0, log=log)
                
def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1, verbose=False,
                 debug=False, logfile=None, input_ellipse=None, zcolumn=None,
                 sdss=False, sdss_pixscale=0.396,
                 unwise=False, unwise_pixscale=2.75): #, custom_tractor=True):
    """Wrapper script to do ellipse-fitting.

    """
    import legacyhalos.ellipse

    if zcolumn is None:
        zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = legacyhalos.ellipse.legacyhalos_ellipse(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                      pixscale=pixscale, nproc=nproc,
                                                      zcolumn=zcolumn, input_ellipse=input_ellipse,
                                                      verbose=verbose, debug=debug,
                                                      sdss=sdss, sdss_pixscale=sdss_pixscale,
                                                      unwise=unwise, unwise_pixscale=unwise_pixscale)
                                                      #pipeline=True, custom_tractor=custom_tractor)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.ellipse.legacyhalos_ellipse(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                              pixscale=pixscale, nproc=nproc,
                                                              zcolumn=zcolumn, input_ellipse=input_ellipse,
                                                              verbose=verbose, debug=debug,
                                                              sdss=sdss, sdss_pixscale=sdss_pixscale,
                                                              unwise=unwise, unwise_pixscale=unwise_pixscale)
                _done(galaxy, err, t0, log=log)

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
                   #largegalaxy_montage=False, pipeline_montage=False,
                   barlen=None, barlabel=None, radius_mosaic_arcsec=None,
                   get_galaxy_galaxydir=None):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if debug:
        _start(galaxy)
        err = legacyhalos.html.make_plots(onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
                                          pixscale=pixscale, zcolumn=zcolumn, nproc=nproc,
                                          barlen=barlen, barlabel=barlabel,
                                          radius_mosaic_arcsec=radius_mosaic_arcsec,
                                          maketrends=False, ccdqa=ccdqa,
                                          clobber=clobber, verbose=verbose, 
                                          #pipeline_montage=pipeline_montage, largegalaxy_montage=largegalaxy_montage,
                                          get_galaxy_galaxydir=get_galaxy_galaxydir)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.html.make_plots(onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
                                                  pixscale=pixscale, zcolumn=zcolumn, nproc=nproc,
                                                  barlen=barlen, barlabel=barlabel,
                                                  radius_mosaic_arcsec=radius_mosaic_arcsec,
                                                  maketrends=False, ccdqa=ccdqa,
                                                  clobber=clobber, verbose=verbose, 
                                                  #pipeline_montage=pipeline_montage, largegalaxy_montage=largegalaxy_montage,
                                                  get_galaxy_galaxydir=get_galaxy_galaxydir)
                _done(galaxy, err, t0, log=log)

def call_largegalaxy_coadds(onegal, galaxy, survey, radius_mosaic, nproc=1,
                            pixscale=0.262, racolumn='RA', deccolumn='DEC',
                            apodize=False, unwise=True, force=False, plots=False,
                            verbose=False, cleanup=True, write_all_pickles=False,
                            no_splinesky=False, just_coadds=False, require_grz=True, 
                            no_gaia=False, no_tycho=False,
                            debug=False, logfile=None):
    """Wrapper script to build the pipeline coadds for large galaxies.

    radius_mosaic in arcsec

    """
    run = legacyhalos.io.get_run(onegal)
    
    t0 = time.time()
    if debug:
        _start(galaxy)
        err = legacyhalos.coadds.largegalaxy_coadds(onegal, galaxy=galaxy, survey=survey,
                                                    radius_mosaic=radius_mosaic, nproc=nproc, 
                                                    pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn, run=run,
                                                    apodize=apodize, unwise=unwise, force=force, plots=plots,
                                                    verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                                                    no_splinesky=no_splinesky, just_coadds=just_coadds,
                                                    require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.coadds.largegalaxy_coadds(onegal, galaxy=galaxy, survey=survey,
                                                            radius_mosaic=radius_mosaic, nproc=nproc, 
                                                            pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn, run=run,
                                                            apodize=apodize, unwise=unwise, force=force, plots=plots,
                                                            verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                                                            no_splinesky=no_splinesky, just_coadds=just_coadds,
                                                            require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho,
                                                            log=log)
                _done(galaxy, err, t0, log=log)

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--sdss', action='store_true', help='Analyze the SDSS galaxies.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (used with --sky and --sersic).')

    parser.add_argument('--coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py.')
    parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--sersic', action='store_true', help='Perform Sersic fitting.')
    parser.add_argument('--integrate', action='store_true', help='Integrate the surface brightness profiles.')
    parser.add_argument('--sky', action='store_true', help='Estimate the sky variance.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--sdss-pixscale', default=0.396, type=float, help='SDSS pixel scale (arcsec/pix).')
    
    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--nomakeplots', action='store_true', help='Do not remake the QA plots for the HTML pages.')

    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                
    args = parser.parse_args()

    return args
