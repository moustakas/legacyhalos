#!/usr/bin/env python

"""Generate LegacySurvey (*grzW1W2*) coadds of all the central galaxies (and
their surrounding groups and clusters) in the legacyhalos parent sample
(legacyhalos-upenn-parent.fits).

Please note:

  * We assume that the $LEGACYHALOS_DIR directory exists (in $CSCRATCH) with all
    the latest parent catalogs.

  * We do not yet build unWISE coadds.

  * We assume the DECam pixel scale (since we use just DR5 data).

  * The code will not handle clusters that span one or more bricks -- a ToDo
    would be to define custom bricks and generate custom Tractor catalogs.

"""

# Parse args first to enable --help on login nodes where MPI crashes
from __future__ import absolute_import, division, print_function

import os
import numpy as np

def coadds_stage_tims(bcg, survey=None, radius=100):
    """Initialize the first step of the pipeline, returning
    a dictionary with the following keys:
    
    ['brickid', 'target_extent', 'version_header', 
     'targetrd', 'brickname', 'pixscale', 'bands', 
     'survey', 'brick', 'ps', 'H', 'ccds', 'W', 
     'targetwcs', 'tims']

    """
    from legacypipe.runbrick import stage_tims
    
    bbox = [bcg.bx-radius, bcg.bx+radius, bcg.by-radius, bcg.by+radius]
    P = stage_tims(brickname=bcg.brickname, survey=survey, target_extent=bbox,
                   pixPsf=True, hybridPsf=True, depth_cut=False)#, mp=mp)
    return P

def read_tractor(bcg, targetwcs, survey=None, verbose=False):
    """Read the full Tractor catalog for a given brick 
    and remove the BCG.
    
    """
    H, W = targetwcs.shape

    # Read the full Tractor catalog.
    fn = survey.find_file('tractor', brick=bcg.brickname)
    cat = fits_table(fn)
    if verbose:
        print('Read {} sources from {}'.format(len(cat), fn))
    
    # Restrict to just the sources in our little box. 
    ok, xx, yy = targetwcs.radec2pixelxy(cat.ra, cat.dec)
    cat.cut(np.flatnonzero((xx > 0) * (xx < W) * (yy > 0) * (yy < H)))
    if verbose:
        print('Cut to {} sources within our box.'.format(len(cat)))
    
    # Remove the central galaxy.
    cat.cut(np.flatnonzero(cat.objid != bcg.objid))
    if verbose:
        print('Removed central galaxy with objid = {}'.format(bcg.objid))
        
    return cat

def build_model_image(cat, tims, survey=None, verbose=False):
    """Generate a model image by rendering each source.
    
    """
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.runbrick import _get_mod
    
    if verbose:
        print('Creating tractor sources...')
    srcs = read_fits_catalog(cat, fluxPrefix='')
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    if verbose:
        print('Rendering model images...')
    mods = [_get_mod((tim, srcs)) for tim in tims]

    return mods

def tractor_coadds(bcg, targetwcs, tims, mods, version_header,
                   survey=None, verbose=False, bands=['g','r','z']):
    """Generate individual-band FITS and color coadds for each central using
    Tractor.

    """
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.runbrick import rgbkwargs, rgbkwargs_resid
    from legacypipe.survey import get_rgb, imsave_jpeg
    
    if verbose:
        print('Producing coadds...')
    C = make_coadds(tims, bands, targetwcs, mods=mods, #mp=mp,
                    callback=write_coadd_images,
                    callback_args=(survey, bcg.brickname, version_header, 
                                   tims, targetwcs)
                    )
    
    # Move (rename) the coadds into the desired output directory.
    for suffix in ('chi2', 'image', 'invvar', 'model'):
        for band in bands:
            oldfile = os.path.join(survey.output_dir, 'coadd', bcg.brickname[:3], 
                                   bcg.brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    bcg.brickname, suffix, band))
            newfile = os.path.join(survey.output_dir, '{:05d}-{}-{}.fits.fz'.format(bcg.objid, suffix, band))
            shutil.move(oldfile, newfile)
    shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
    
    # Build png postage stamps of the coadds.
    coadd_list = [('image', C.coimgs,   rgbkwargs),
                  ('model', C.comods,   rgbkwargs),
                  ('resid', C.coresids, rgbkwargs_resid)]

    for name, ims, rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        outfn = os.path.join(survey.output_dir, '{:05d}-{}.jpg'.format(bcg.objid, name))
        if verbose:
            print('Writing {}'.format(outfn))
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb

def legacyhalos_coadds(survey, bcgphot, radius, coaddsdir, nproc=1, verbose=True):
    """Generate the coadds for a list of BCGs.""" 

    for ii in range(len(bcgphot)):
        survey.output_dir = os.path.join(coaddsdir, '{:05d}'.format(bcgphot[ii].objid))

        # Step 1 - Set up the first stage of the pipeline.
        P = coadds_stage_tims(bcgphot[ii], survey=survey, radius=radius[ii])

        import pdb ; pdb.set_trace()
        # Step 2 - Read the Tractor catalog for this brick and remove the central.
        cat = read_tractor(bcgphot[ii], P['targetwcs'], survey=survey, verbose=verbose)
           
        # Step 3 - Render the model images without the central.
        mods = build_model_image(cat, tims=P['tims'], survey=survey, verbose=verbose)

        # Step 4 - Generate and write out the coadds.
        tractor_coadds(bcgphot[ii], P['targetwcs'], P['tims'], mods,
                       P['version_header'], survey=survey, verbose=verbose)
