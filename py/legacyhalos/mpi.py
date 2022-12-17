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
    
def call_ellipse(galaxy, galaxydir, data, galaxyinfo=None,
                 pixscale=0.262, nproc=1, bands=['g', 'r', 'z'], refband='r',
                 delta_logsma=5, maxsma=None, logsma=True,
                 copy_mw_transmission=False,
                 verbose=False, debug=False, write_donefile=True,
                 logfile=None, input_ellipse=None, sbthresh=None,
                 apertures=None, clobber=False):
    """Wrapper script to do ellipse-fitting.

    """
    import legacyhalos.ellipse

    # Do not force zcolumn here; it's not always wanted or needed in ellipse.
    #if zcolumn is None:
    #    zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = legacyhalos.ellipse.legacyhalos_ellipse(
            galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
            bands=bands, refband=refband,
            pixscale=pixscale, nproc=nproc,
            sbthresh=sbthresh, apertures=apertures, input_ellipse=input_ellipse,
            delta_logsma=delta_logsma, maxsma=maxsma, logsma=logsma,
            copy_mw_transmission=copy_mw_transmission,
            verbose=verbose, debug=debug, clobber=clobber)
        if write_donefile:
            _done(galaxy, galaxydir, err, t0, 'ellipse', data['filesuffix'])
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.ellipse.legacyhalos_ellipse(
                    galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                    bands=bands, refband=refband,
                    pixscale=pixscale, nproc=nproc,
                    sbthresh=sbthresh, apertures=apertures, input_ellipse=input_ellipse,
                    delta_logsma=delta_logsma, maxsma=maxsma, logsma=logsma,
                    copy_mw_transmission=copy_mw_transmission,
                    verbose=verbose, clobber=clobber)
                if write_donefile:
                    _done(galaxy, galaxydir, err, t0, 'ellipse', data['filesuffix'], log=log)

    return err

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
                   logfile=None, zcolumn='Z', galaxy_id=None,
                   datadir=None, htmldir=None, cosmo=None,
                   galex=False, unwise=False, just_coadds=False, write_donefile=True,
                   barlen=None, barlabel=None, radius_mosaic_arcsec=None,
                   get_galaxy_galaxydir=None, read_multiband=None,
                   qa_multiwavelength_sed=None):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if debug:
        _start(galaxy)
        err = legacyhalos.html.make_plots(
            onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
            pixscale=pixscale, zcolumn=zcolumn, galaxy_id=galaxy_id,
            nproc=nproc, barlen=barlen, barlabel=barlabel,
            radius_mosaic_arcsec=radius_mosaic_arcsec,
            maketrends=False, ccdqa=ccdqa,
            clobber=clobber, verbose=verbose, 
            cosmo=cosmo, galex=galex, unwise=unwise, just_coadds=just_coadds,
            get_galaxy_galaxydir=get_galaxy_galaxydir,
            read_multiband=read_multiband,
            qa_multiwavelength_sed=qa_multiwavelength_sed)
        if write_donefile:
            _done(galaxy, survey.output_dir, err, t0, 'html')
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = legacyhalos.html.make_plots(
                    onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
                    pixscale=pixscale, zcolumn=zcolumn, galaxy_id=galaxy_id,
                    nproc=nproc, barlen=barlen, barlabel=barlabel,
                    radius_mosaic_arcsec=radius_mosaic_arcsec,
                    maketrends=False, ccdqa=ccdqa,
                    clobber=clobber, verbose=verbose,
                    cosmo=cosmo, galex=galex, unwise=unwise, just_coadds=just_coadds,
                    get_galaxy_galaxydir=get_galaxy_galaxydir,
                    read_multiband=read_multiband,
                    qa_multiwavelength_sed=qa_multiwavelength_sed)
                if write_donefile:
                    _done(galaxy, survey.output_dir, err, t0, 'html')

def call_custom_coadds(onegal, galaxy, survey, run, radius_mosaic, nproc=1,
                       pixscale=0.262, racolumn='RA', deccolumn='DEC', nsigma=None,
                       custom=True,
                       bands=['g', 'r', 'z'], 
                       apodize=False, unwise=True, galex=False, force=False, plots=False,
                       verbose=False, cleanup=True, write_all_pickles=False,
                       #no_subsky=False,
                       subsky_radii=None,
                       #ubercal_sky=False,
                       write_wise_psf=False,
                       just_coadds=False, require_grz=True, 
                       no_gaia=False, no_tycho=False,
                       no_galex_ceres=False,
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
            nsigma=nsigma, custom=custom,
            bands=bands,
            run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
            verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
            write_wise_psf=write_wise_psf,
            #no_subsky=no_subsky,
            subsky_radii=subsky_radii, #ubercal_sky=ubercal_sky,
            just_coadds=just_coadds,
            require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho,
            no_galex_ceres=no_galex_ceres)
        _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err, filesuffix = legacyhalos.coadds.custom_coadds(
                    onegal, galaxy=galaxy, survey=survey, 
                    radius_mosaic=radius_mosaic, nproc=nproc, 
                    pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn,
                    nsigma=nsigma, custom=custom,
                    bands=bands,
                    run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
                    verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                    write_wise_psf=write_wise_psf,
                    #no_subsky=no_subsky,
                    subsky_radii=subsky_radii, #ubercal_sky=ubercal_sky,
                    just_coadds=just_coadds,
                    require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho,
                    no_galex_ceres=no_galex_ceres,
                    log=log)
                _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix, log=log)
