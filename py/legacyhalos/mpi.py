"""
legacyhalos.mpi
===============

Code to deal with the MPI portion of the pipeline.

"""
import os, time, pdb
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import legacyhalos.io
import legacyhalos.hsc

def missing_files(sample, filetype='coadds', size=1, htmldir=None,
                  hsc=False, sdss=False, clobber=False):
    """Find missing data of a given filetype."""    

    if filetype == 'coadds':
        filesuffix = '-pipeline-resid-grz.jpg'
    elif filetype == 'custom-coadds':
        filesuffix = '-custom-resid-grz.jpg'
    elif filetype == 'ellipse':
        filesuffix = '-ellipsefit.p'
    elif filetype == 'sersic':
        filesuffix = '-sersic-single.p'
    elif filetype == 'html':
        filesuffix = '-ccdpos.png'
        #filesuffix = '-sersic-exponential-nowavepower.png'
    elif filetype == 'sdss-coadds':
        filesuffix = '-sdss-image-gri.jpg'
    elif filetype == 'sdss-custom-coadds':
        filesuffix = '-sdss-resid-gri.jpg'
    elif filetype == 'sdss-ellipse':
        filesuffix = '-sdss-ellipsefit.p'
    else:
        print('Unrecognized file type!')
        raise ValueError

    ngal = len(sample)
    indices = np.arange(ngal)
    todo = np.ones(ngal, dtype=bool)

    if filetype == 'html':
        if hsc:
            galaxy, _, galaxydir = legacyhalos.hsc.get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
        else:
            galaxy, _, galaxydir = legacyhalos.io.get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    else:
        if hsc:
            galaxy, galaxydir = legacyhalos.hsc.get_galaxy_galaxydir(sample, htmldir=htmldir)
        else:
            galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(sample, htmldir=htmldir)

    for ii, (gal, gdir) in enumerate( zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)) ):
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if os.path.exists(checkfile) and clobber is False:
            todo[ii] = False

    if np.sum(todo) == 0:
        return list()
    else:
        indices = indices[todo]
        
    return np.array_split(indices, size)

def _missing_files(args, sample, size, htmldir=None):
    """Simple task-specific wrapper on missing_files.

    """
    if args.coadds:
        if args.sdss:
            suffix = 'sdss-coadds'
        else:
            suffix = 'coadds'
    elif args.custom_coadds:
        if args.sdss:
            suffix = 'sdss-custom-coadds'
        else:
            suffix = 'custom-coadds'
    elif args.ellipse:
        if args.sdss:
            suffix = 'sdss-ellipse'
        else:
            suffix = 'ellipse'
    elif args.sersic:
        suffix = 'sersic'
    elif args.sky:
        suffix = 'sky'
    elif args.htmlplots:
        suffix = 'html'
    else:
        suffix = ''        

    if suffix != '':
        groups = missing_files(sample, filetype=suffix, size=size, hsc=args.hsc,
                               sdss=args.sdss, clobber=args.clobber, htmldir=htmldir)
    else:
        groups = []        

    return suffix, groups

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

def _call_pipeline_coadds(onegal, galaxy, radius_mosaic, survey, kdccds_north,
                          kdccds_south, pixscale, nproc, force, debug, hsc,
                          logfile):
    """Wrapper script to build the pipeline coadds.

    radius_mosaic in arcsec

    """
    cleanup = True

    t0 = time.time()
    if debug:
        _start(galaxy)
        if hsc:
            run = 'decam'
        else:
            run = legacyhalos.io.get_run(onegal, radius_mosaic, pixscale, kdccds_north, kdccds_south)
        err = legacyhalos.coadds.pipeline_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                 survey=survey, pixscale=pixscale, run=run,
                                                 nproc=nproc, force=force, cleanup=cleanup)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                if hsc:
                    run = 'decam'
                else:
                    run = legacyhalos.io.get_run(onegal, radius_mosaic, pixscale, kdccds_north, kdccds_south, log=log)
                err = legacyhalos.coadds.pipeline_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                         survey=survey, pixscale=pixscale, run=run,
                                                         nproc=nproc, force=force, log=log, cleanup=cleanup)
                _done(galaxy, err, t0, log=log)

def _call_custom_coadds(onegal, galaxy, radius_mosaic, survey, pixscale,
                        nproc, debug, logfile, sdss, sdss_pixscale):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()
    if debug:
        _start(galaxy)
        if sdss:
            err = legacyhalos.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                 survey=survey, pixscale=sdss_pixscale, nproc=nproc)
        else:
            err = legacyhalos.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                   survey=survey, pixscale=pixscale, nproc=nproc)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                if sdss:
                    err = legacyhalos.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                         survey=survey, pixscale=sdss_pixscale,
                                                         nproc=nproc, log=log)
                else:
                    err = legacyhalos.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                           survey=survey, pixscale=pixscale,
                                                           nproc=nproc, log=log)
                _done(galaxy, err, t0, log=log)
                
def _call_ellipse(onegal, galaxy, pixscale=0.262, nproc=1, verbose=False,
                  debug=False, logfile=None, hsc=False, input_ellipse=False,
                  sdss=False, sdss_pixscale=0.396):
    """Wrapper script to do ellipse-fitting.

    """
    import legacyhalos.ellipse

    if hsc:
        zcolumn = 'Z_BEST'
    else:
        zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = legacyhalos.ellipse.legacyhalos_ellipse(onegal, pixscale=pixscale, nproc=nproc,
                                                      zcolumn=zcolumn,
                                                      input_ellipse=input_ellipse,
                                                      verbose=verbose, debug=debug,
                                                      hsc=hsc, sdss=sdss, sdss_pixscale=sdss_pixscale,
                                                      pipeline=True, unwise=False)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.ellipse.legacyhalos_ellipse(onegal, pixscale=pixscale, nproc=nproc,
                                                              zcolumn=zcolumn,
                                                              input_ellipse=input_ellipse,
                                                              verbose=verbose, debug=debug,
                                                              hsc=hsc, sdss=sdss, sdss_pixscale=sdss_pixscale,
                                                              pipeline=True, unwise=False)
                _done(galaxy, err, t0, log=log)

def _call_sersic(onegal, galaxy, galaxydir, seed, verbose, debug, logfile, hsc):
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

def _call_sky(onegal, galaxy, galaxydir, survey, seed, nproc, pixscale,
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
                
def _call_htmlplots(onegal, galaxy, survey, pixscale, nproc, debug, clobber,
                    verbose, ccdqa, logfile, hsc, htmldir):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if hsc:
        zcolumn = 'Z_BEST'
    else:
        zcolumn = 'Z_LAMBDA'

    if debug:
        _start(galaxy)
        err = legacyhalos.html.make_plots(onegal, datadir=None, htmldir=htmldir,
                                          pixscale=pixscale, survey=survey, clobber=clobber,
                                          verbose=verbose, nproc=nproc, zcolumn=zcolumn, 
                                          ccdqa=ccdqa, maketrends=False, hsc=hsc)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.html.make_plots(onegal, datadir=None, htmldir=htmldir,
                                                  pixscale=pixscale, survey=survey, clobber=clobber,
                                                  verbose=verbose, nproc=nproc, zcolumn=zcolumn, 
                                                  ccdqa=ccdqa, maketrends=False, hsc=hsc)
                _done(galaxy, err, t0, log=log)
