"""
legacyhalos.html
================

Code to generate HTML content.

"""
import os, subprocess, shutil, pdb
import numpy as np
import astropy.table

import legacyhalos.io
import legacyhalos.misc

def _get_cutouts_one(args):
    """Wrapper function for the multiprocessing."""
    return get_cutouts_one(*args)

def get_cutouts_one(group, clobber=False):
    """Get viewer cutouts for a single galaxy."""

    layer = get_layer(group)
    groupname = get_groupname(group)
        
    diam = group_diameter(group) # [arcmin]
    size = np.ceil(diam * 60 / PIXSCALE).astype('int') # [pixels]

    imageurl = '{}/?ra={:.8f}&dec={:.8f}&pixscale={:.3f}&size={:g}&layer={}'.format(
        cutouturl, group['ra'], group['dec'], PIXSCALE, size, layer)
        
    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    cmd = 'wget --continue -O {:s} "{:s}"' .format(jpgfile, imageurl)
    if os.path.isfile(jpgfile) and not clobber:
        print('File {} exists...skipping.'.format(jpgfile))
    else:
        if os.path.isfile(jpgfile):
            os.remove(jpgfile)
        print(cmd)
        os.system(cmd)

def get_cutouts(groupsample, use_nproc=1, clobber=False):
    """Get viewer cutouts of the whole sample."""

    cutoutargs = list()
    for gg in groupsample:
        cutoutargs.append( (gg, clobber) )

    if use_nproc > 1:
        p = multiprocessing.Pool(nproc)
        p.map(_get_cutouts_one, cutoutargs)
        p.close()
    else:
        for args in cutoutargs:
            _get_cutouts_one(args)
    return

def html_javadate():
    """Return a string that embeds a date in a webpage using Javascript.

    """
    import textwrap

    js = textwrap.dedent("""
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    document.write(" " + fyear + " " + lmonth + " " + date)
    </SCRIPT>
    """)

    return js
        
def make_ccd_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=0.262, ccds=None,
                zcolumn='Z', radius_pixel=None, survey=None, mp=None, clobber=False,
                verbose=False):
    """Build CCD-level QA.

    [This script may be obsolete, since we no longer write out the individual
    CCDs.]

    """
    from glob import glob
    from astrometry.util.fits import fits_table
    from legacyhalos.qa import display_ccdpos, _display_ccdmask_and_sky
    from astrometry.util.multiproc import multiproc

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if radius_pixel is None:
        radius_pixel = legacyhalos.misc.cutout_radius_kpc(
            redshift=onegal[zcolumn], pixscale=pixscale,
            radius_kpc=radius_mosaic_kpc) # [pixels]

    qarootfile = os.path.join(htmlgalaxydir, '{}-2d'.format(galaxy))
    #maskfile = os.path.join(galaxydir, '{}-custom-ccdmasks.fits.fz'.format(galaxy))

    if ccds is None:
        ccdsfile = glob(os.path.join(galaxydir, '{}-ccds-*.fits'.format(galaxy))) # north, south
        if os.path.isfile(ccdsfile):
            ccds = survey.cleanup_ccds_table(fits_table(ccdsfile))
            print('Read {} CCDs from {}'.format(len(ccds), ccdsfile))
    
    okfiles = True
    for iccd in range(len(ccds)):
        qafile = '{}-ccd{:02d}.png'.format(qarootfile, iccd)
        okfiles *= os.path.isfile(qafile)

    if not okfiles or clobber:
        if mp is None:
            mp = multiproc(nthreads=1)
            
        ccdargs = [(galaxy, galaxydir, qarootfile, radius_pixel, _ccd, iccd, survey)
                   for iccd, _ccd in enumerate(ccds)]
        mp.map(_display_ccdmask_and_sky, ccdargs)

    grzfile = glob(os.path.join(galaxydir, '{}-*-image-grz.jpg'.format(galaxy)))[0]
    if os.path.isfile(grzfile):
        ccdposfile = os.path.join(htmlgalaxydir, '{}-ccdpos.png'.format(galaxy))
        if not os.path.isfile(ccdposfile) or clobber:
            display_ccdpos(onegal, ccds, png=ccdposfile, zcolumn=zcolumn)
    else:
        print('Unable to make ccdpos QA; montage file {} not found.'.format(grzfile))
        
def make_ccdpos_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=0.262,
                   zcolumn='Z', radius=None, survey=None, clobber=False,
                   verbose=False):
    """Build CCD positions QA.

    radius in pixels

    """
    from glob import glob
    from astrometry.util.fits import fits_table
    from legacyhalos.qa import display_ccdpos

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    #for stage in ('largegalaxy', 'pipeline'):
    #    ccdsfile = glob(os.path.join(galaxydir, '{}-ccds-*.fits'.format(galaxy))) # north, south
    #    #ccdsfile = glob(os.path.join(galaxydir, '{}-{}-ccds-*.fits'.format(galaxy, stage))) # north, south
    #    if len(ccdsfile) == 0:
    #        print('Missing CCDs file for stage {}'.format(stage))
    #        continue
    #    ccdsfile = ccdsfile[0]
    #    ccds = survey.cleanup_ccds_table(fits_table(ccdsfile))
    #    print('Read {} CCDs from {}'.format(len(ccds), ccdsfile))
    #
    #    ccdposfile = os.path.join(htmlgalaxydir, '{}-{}-ccdpos.png'.format(galaxy, stage))
    #    if not os.path.isfile(ccdposfile) or clobber:
    #        display_ccdpos(onegal, ccds, radius=radius, png=ccdposfile, verbose=verbose)

    ccdsfile = glob(os.path.join(galaxydir, '{}-ccds-*.fits'.format(galaxy))) # north, south
    if len(ccdsfile) == 0:
        print('CCDs file not found!')
        return

    ccdsfile = ccdsfile[0]
    ccds = survey.cleanup_ccds_table(fits_table(ccdsfile))
    #if verbose:
    print('Read {} CCDs from {}'.format(len(ccds), ccdsfile))

    grzfile = glob(os.path.join(galaxydir, '{}-*-image-grz.jpg'.format(galaxy)))[0]
    if os.path.isfile(grzfile):
        ccdposfile = os.path.join(htmlgalaxydir, '{}-ccdpos.png'.format(galaxy))
        if not os.path.isfile(ccdposfile) or clobber:
            display_ccdpos(onegal, ccds, radius, grzfile, png=ccdposfile)
    else:
        print('Unable to make ccdpos QA; montage file {} not found.'.format(grzfile))

def make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=None, 
                        barlabel=None, just_coadds=False, clobber=False,
                        verbose=False):
    """Montage the coadds into a nice QAplot.

    barlen - pixels

    """
    from legacyhalos.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    for filesuffix in ['custom', 'pipeline']:
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-montage-grz.png'.format(galaxy, filesuffix))
        if not os.path.isfile(montagefile) or clobber:
            if filesuffix == 'custom':
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-grz'.format(filesuffix),
                              '{}-resid-grz'.format(filesuffix))
            else:
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-grz'.format(filesuffix),
                              '{}-resid-grz'.format(filesuffix))

            # Image coadd with the scale bar label--
            barpngfile = os.path.join(htmlgalaxydir, '{}-{}.png'.format(galaxy, coaddfiles[0]))
                
            # Make sure all the files exist.
            check, _just_coadds = True, just_coadds
            jpgfile = []
            for suffix in coaddfiles:
                _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
                jpgfile.append(_jpgfile)
                if not os.path.isfile(_jpgfile):
                    if verbose:
                        print('File {} not found!'.format(_jpgfile))
                    check = False
                #print(check, _jpgfile)

            # Check for just the image coadd.
            if check is False:
                if os.path.isfile(np.atleast_1d(jpgfile)[0]):
                    _just_coadds = True
                else:
                    continue

            if check or _just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096 and sz[0] < 8192:
                    resize = '1024x1024'
                    #resize = '-resize 2048x2048 '
                elif sz[0] > 8192:
                    resize = '1024x1024'
                    #resize = '-resize 4096x4096 '
                else:
                    resize = None

                # Make a quick thumbnail of just the data.
                cmd = 'convert -thumbnail {0}x{0} {1} {2}'.format(96, np.atleast_1d(jpgfile)[0], thumb2file)
                if os.path.isfile(thumb2file):
                    os.remove(thumb2file)
                print('Writing {}'.format(thumb2file))
                subprocess.call(cmd.split())
                    
                # Add a bar and label to the first image.
                if _just_coadds:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile
                    else:
                        cmd = cmd+' '+jpgfile
                    if sz[0] > 512:
                        thumbsz = 512
                    else:
                        thumbsz = sz[0]
                else:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile+' '
                        cmd = cmd+' '.join(ff for ff in jpgfile[1:])
                    else:
                        cmd = cmd+' '.join(ff for ff in jpgfile)
                    if sz[0] > 512:
                        thumbsz = 512*3
                    else:
                        thumbsz = sz[0]*3
                cmd = cmd+' {}'.format(montagefile)
                print(cmd)

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
                    print(cmd)
                    continue

                # Create a couple smaller thumbnail images
                cmd = 'convert -thumbnail {0} {1} {2}'.format(thumbsz, montagefile, thumbfile)
                #print(cmd)
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)                
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())
                    
                ## Create a couple smaller thumbnail images
                #for tf, sz in zip((thumbfile, thumb2file), (512, 96)):
                #    cmd = 'convert -thumbnail {}x{} {} {}'.format(sz, sz, montagefile, tf)
                #    #if verbose:
                #    print('Writing {}'.format(tf))
                #    subprocess.call(cmd.split())

def make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir, refpixscale=0.262,
                                barlen=None, barlabel=None, just_coadds=False,
                                clobber=False, verbose=False):
    """Montage the GALEX and WISE coadds into a nice QAplot.

    barlen - pixels

    """
    from legacyhalos.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    filesuffix = 'custom'

    for bandsuffix, pixscale in zip(('FUVNUV', 'W1W2'), (1.5, 2.75)):
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-montage-{}.png'.format(galaxy, filesuffix, bandsuffix))
        if not os.path.isfile(montagefile) or clobber:
            coaddfiles = ('{}-image-{}'.format(filesuffix, bandsuffix),
                          '{}-model-{}'.format(filesuffix, bandsuffix),
                          '{}-resid-{}'.format(filesuffix, bandsuffix))

            # Image coadd with the scale bar label--
            barpngfile = os.path.join(htmlgalaxydir, '{}-{}.png'.format(galaxy, coaddfiles[0]))

            # Make sure all the files exist.
            check, _just_coadds = True, just_coadds
            jpgfile = []
            for suffix in coaddfiles:
                _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
                jpgfile.append(_jpgfile)
                if not os.path.isfile(_jpgfile):
                    if verbose:
                        print('File {} not found!'.format(_jpgfile))
                    check = False
                #print(check, _jpgfile)

            # Check for just the image coadd.
            if check is False:
                if os.path.isfile(np.atleast_1d(jpgfile)[0]):
                    _just_coadds = True
                else:
                    continue

            if check or _just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096 and sz[0] < 8192:
                    resize = '1024x1024'
                    #resize = '-resize 2048x2048 '
                elif sz[0] > 8192:
                    resize = '1024x1024'
                    #resize = '-resize 4096x4096 '
                else:
                    resize = None

                # Make a quick thumbnail of just the data.
                cmd = 'convert -thumbnail {0}x{0} {1} {2}'.format(96, np.atleast_1d(jpgfile)[0], thumb2file)
                if os.path.isfile(thumb2file):
                    os.remove(thumb2file)
                print('Writing {}'.format(thumb2file))
                subprocess.call(cmd.split())

                # Add a bar and label to the first image.
                if _just_coadds:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 '
                    if barlen:
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile, scaledfont=True)
                        cmd = cmd+' '+barpngfile
                    else:
                        cmd = cmd+' '+jpgfile
                    if sz[0] > 512:
                        thumbsz = 512
                    else:
                        thumbsz = sz[0]
                else:
                    if resize:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -resize {} -geometry +0+0 '.format(resize)
                    else:
                        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
                    if barlen:
                        pixscalefactor = pixscale / refpixscale
                        #barlen2 = barlen / pixscalefactor
                        #pixscalefactor = 1.0
                        addbar_to_png(jpgfile[0], barlen, barlabel, None, barpngfile,
                                      scaledfont=True, pixscalefactor=pixscalefactor)
                        cmd = cmd+' '+barpngfile+' '
                        cmd = cmd+' '.join(ff for ff in jpgfile[1:])
                    else:
                        cmd = cmd+' '.join(ff for ff in jpgfile)
                    if sz[0] > 512:
                        thumbsz = 512*3
                    else:
                        thumbsz = sz[0]*3
                cmd = cmd+' {}'.format(montagefile)
                print(cmd)

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
                    print(cmd)
                    continue

                # Create a couple smaller thumbnail images
                cmd = 'convert -thumbnail {0} {1} {2}'.format(thumbsz, montagefile, thumbfile)
                #print(cmd)
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)                
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())
        
def make_maskbits_qa(galaxy, galaxydir, htmlgalaxydir, clobber=False, verbose=False):
    """Visualize the maskbits image.

    """
    import fitsio
    from legacyhalos.qa import qa_maskbits

    filesuffix = 'largegalaxy'

    maskbitsfile = os.path.join(htmlgalaxydir, '{}-{}-maskbits.png'.format(galaxy, filesuffix))
    if not os.path.isfile(maskbitsfile) or clobber:
        fitsfile = os.path.join(galaxydir, '{}-{}-maskbits.fits.fz'.format(galaxy, filesuffix))
        tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, filesuffix))
        if not os.path.isfile(fitsfile):
            if verbose:
                print('File {} not found!'.format(fitsfile))
            return
        if not os.path.isfile(tractorfile):
            if verbose:
                print('File {} not found!'.format(tractorfile))
            return
        
        mask = fitsio.read(fitsfile)
        tractor = fitsio.read(tractorfile)

        qa_maskbits(mask, tractor, png=maskbitsfile)

def make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=['g', 'r', 'z'],
                    refband='r', pixscale=0.262, read_multiband=None,
                    qa_multiwavelength_sed=None,
                    galaxy_id=None, barlen=None, barlabel=None, clobber=False, verbose=False,
                    cosmo=None, galex=False, unwise=False, scaledfont=False):
    """Generate QAplots from the ellipse-fitting.

    """
    import fitsio
    from PIL import Image
    from astropy.table import Table
    from legacyhalos.io import read_ellipsefit
    from legacyhalos.qa import (display_multiband, display_ellipsefit,
                                display_ellipse_sbprofile, qa_curveofgrowth,
                                qa_maskbits)
    if qa_multiwavelength_sed is None:
        from legacyhalos.qa import qa_multiwavelength_sed

    Image.MAX_IMAGE_PIXELS = None
    
    # Read the data.
    if read_multiband is None:
        print('Unable to build ellipse QA without specifying read_multiband method.')
        return

    data, galaxyinfo = read_multiband(galaxy, galaxydir, galaxy_id=galaxy_id, bands=bands,
                                      refband=refband, pixscale=pixscale,
                                      verbose=verbose, galex=galex, unwise=unwise)
    
    if not bool(data) or data['missingdata']:
        return

    if data['failed']: # all galaxies dropped
        return

    # optionally read the Tractor catalog
    tractor = None
    if galex or unwise:
        tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, data['filesuffix']))        
        if os.path.isfile(tractorfile):
            tractor = Table(fitsio.read(tractorfile, lower=True))

    ellipsefitall = []
    for igal, galid in enumerate(data['galaxy_id']):
        galid = str(galid)
        ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix=data['filesuffix'],
                                     galaxy_id=galid, verbose=verbose)
        if bool(ellipsefit):
            ellipsefitall.append(ellipsefit)

            if galid.strip() != '':
                galid = '{}-'.format(galid)

            if galex or unwise:
                sedfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}sed.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(sedfile) or clobber:
                    _tractor = None
                    if tractor is not None:
                        _tractor = tractor[(tractor['ref_cat'] != '  ')*np.isin(tractor['ref_id'], data['galaxy_id'][igal])] # fragile...
                    qa_multiwavelength_sed(ellipsefit, tractor=_tractor, png=sedfile, verbose=verbose)
                    
            sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}sbprofile.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(sbprofilefile) or clobber:
                display_ellipse_sbprofile(ellipsefit, plot_radius=False, plot_sbradii=False,
                                          png=sbprofilefile, verbose=verbose, minerr=0.0,
                                          cosmo=cosmo)
                
            cogfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}cog.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(cogfile) or clobber:
                qa_curveofgrowth(ellipsefit, pipeline_ellipsefit={}, plot_sbradii=False,
                                 png=cogfile, verbose=verbose, cosmo=cosmo)
            
            #print('hack!')
            #continue
        
            if unwise:
                multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband-W1W2.png'.format(galaxy, data['filesuffix'], galid))
                thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband-W1W2.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(multibandfile) or clobber:
                    with Image.open(os.path.join(galaxydir, '{}-{}-image-W1W2.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
                                          igal=igal, barlen=barlen, barlabel=barlabel,
                                          png=multibandfile, verbose=verbose, scaledfont=scaledfont,
                                          galex=False, unwise=True)
                    # Create a thumbnail.
                    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                    if os.path.isfile(thumbfile):
                        os.remove(thumbfile)
                    print('Writing {}'.format(thumbfile))
                    subprocess.call(cmd.split())

            if galex:
                multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband-FUVNUV.png'.format(galaxy, data['filesuffix'], galid))
                thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband-FUVNUV.png'.format(galaxy, data['filesuffix'], galid))
                if not os.path.isfile(multibandfile) or clobber:
                    with Image.open(os.path.join(galaxydir, '{}-{}-image-FUVNUV.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
                                          igal=igal, barlen=barlen, barlabel=barlabel,
                                          png=multibandfile, verbose=verbose, scaledfont=scaledfont,
                                          galex=True, unwise=False)
                    # Create a thumbnail.
                    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                    if os.path.isfile(thumbfile):
                        os.remove(thumbfile)
                    print('Writing {}'.format(thumbfile))
                    subprocess.call(cmd.split())

            multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-{}multiband.png'.format(galaxy, data['filesuffix'], galid))
            thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-ellipse-{}multiband.png'.format(galaxy, data['filesuffix'], galid))
            if not os.path.isfile(multibandfile) or clobber:
                with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
                    display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg, bands=bands,
                                      igal=igal, barlen=barlen, barlabel=barlabel,
                                      png=multibandfile, verbose=verbose, scaledfont=scaledfont)
            
                # Create a thumbnail.
                cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                if os.path.isfile(thumbfile):
                    os.remove(thumbfile)
                print('Writing {}'.format(thumbfile))
                subprocess.call(cmd.split())

            ## hack!
            #print('HACK!!!')
            #continue

    ## maskbits QA
    #maskbitsfile = os.path.join(htmlgalaxydir, '{}-{}-maskbits.png'.format(galaxy, data['filesuffix']))
    #if not os.path.isfile(maskbitsfile) or clobber:
    #    fitsfile = os.path.join(galaxydir, '{}-{}-maskbits.fits.fz'.format(galaxy, data['filesuffix']))
    #    tractorfile = os.path.join(galaxydir, '{}-{}-tractor.fits'.format(galaxy, data['filesuffix']))
    #    if not os.path.isfile(fitsfile):
    #        if verbose:
    #            print('File {} not found!'.format(fitsfile))
    #        return
    #    if not os.path.isfile(tractorfile):
    #        if verbose:
    #            print('File {} not found!'.format(tractorfile))
    #        return
    #
    #    mask = fitsio.read(fitsfile)
    #    tractor = Table(fitsio.read(tractorfile, upper=True))
    #
    #    with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, data['filesuffix']))) as colorimg:
    #        qa_maskbits(mask, tractor, ellipsefitall, colorimg, largegalaxy=True, png=maskbitsfile)

def make_sersic_qa(galaxy, galaxydir, htmlgalaxydir, bands=['g', 'r', 'z'],
                   cosmo=None, clobber=False, verbose=False):
    """Generate QAplots from the Sersic modeling.

    """
    from legacyhalos.io import read_sersic
    from legacyhalos.qa import display_sersic

    # Double Sersic
    double = read_sersic(galaxy, galaxydir, modeltype='double')
    if bool(double):
        doublefile = os.path.join(htmlgalaxydir, '{}-sersic-double.png'.format(galaxy))
        if not os.path.isfile(doublefile) or clobber:
            display_sersic(double, cosmo=cosmo, png=doublefile, verbose=verbose)

    # Double Sersic, no wavelength dependence
    double = read_sersic(galaxy, galaxydir, modeltype='double-nowavepower')
    if bool(double):
        doublefile = os.path.join(htmlgalaxydir, '{}-sersic-double-nowavepower.png'.format(galaxy))
        if not os.path.isfile(doublefile) or clobber:
            display_sersic(double, cosmo=cosmo, png=doublefile, verbose=verbose)

    # Single Sersic, no wavelength dependence
    single = read_sersic(galaxy, galaxydir, modeltype='single-nowavepower')
    if bool(single):
        singlefile = os.path.join(htmlgalaxydir, '{}-sersic-single-nowavepower.png'.format(galaxy))
        if not os.path.isfile(singlefile) or clobber:
            display_sersic(single, cosmo=cosmo, png=singlefile, verbose=verbose)

    # Single Sersic
    single = read_sersic(galaxy, galaxydir, modeltype='single')
    if bool(single):
        singlefile = os.path.join(htmlgalaxydir, '{}-sersic-single.png'.format(galaxy))
        if not os.path.isfile(singlefile) or clobber:
            display_sersic(single, cosmo=cosmo, png=singlefile, verbose=verbose)

    # Sersic-exponential
    serexp = read_sersic(galaxy, galaxydir, modeltype='exponential')
    if bool(serexp):
        serexpfile = os.path.join(htmlgalaxydir, '{}-sersic-exponential.png'.format(galaxy))
        if not os.path.isfile(serexpfile) or clobber:
            display_sersic(serexp, cosmo=cosmo, png=serexpfile, verbose=verbose)

    # Sersic-exponential, no wavelength dependence
    serexp = read_sersic(galaxy, galaxydir, modeltype='exponential-nowavepower')
    if bool(serexp):
        serexpfile = os.path.join(htmlgalaxydir, '{}-sersic-exponential-nowavepower.png'.format(galaxy))
        if not os.path.isfile(serexpfile) or clobber:
            display_sersic(serexp, cosmo=cosmo, png=serexpfile, verbose=verbose)

def make_plots(sample, datadir=None, htmldir=None, survey=None, refband='r',
               bands=['g', 'r', 'z'], pixscale=0.262, zcolumn='Z', galaxy_id=None,
               nproc=1, barlen=None, barlabel=None,
               radius_mosaic_arcsec=None, maketrends=False, ccdqa=False,
               clobber=False, verbose=True, get_galaxy_galaxydir=None,
               read_multiband=None, qa_multiwavelength_sed=None,
               cosmo=None, galex=False, unwise=False,
               just_coadds=False, scaledfont=False):
    """Make QA plots.

    """
    from legacyhalos.qa import display_ccdpos
    from legacyhalos.coadds import _mosaic_width
    
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if maketrends:
        from legacyhalos.qa import sample_trends
        sample_trends(sample, htmldir, datadir=datadir, verbose=verbose)

    #from legacyhalos.misc import RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    barlen_kpc = 100
    if barlabel is None:
        barlabel = '100 kpc'

    if get_galaxy_galaxydir is None:
        get_galaxy_galaxydir = legacyhalos.io.get_galaxy_galaxydir

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)
        
    # Loop on each galaxy.
    for ii, onegal in enumerate(sample):
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(onegal, html=True)
            
        if not os.path.isdir(htmlgalaxydir):
            os.makedirs(htmlgalaxydir, exist_ok=True, mode=0o775)
            #shutil.chown(htmlgalaxydir, group='cosmo')

        if barlen is None and zcolumn in onegal.colnames:
            barlen = np.round(barlen_kpc / legacyhalos.misc.arcsec2kpc(
                onegal[zcolumn], cosmo=cosmo) / pixscale).astype('int') # [kpc]

        #if radius_mosaic_arcsec is None:
        #    radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
        #        redshift=onegal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

        # Build the ellipse and photometry plots.
        if not just_coadds:
            make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=bands, refband=refband,
                            pixscale=pixscale, barlen=barlen, barlabel=barlabel,
                            galaxy_id=galaxy_id, 
                            clobber=clobber, verbose=verbose, galex=galex, unwise=unwise,
                            cosmo=cosmo, scaledfont=scaledfont, read_multiband=read_multiband,
                            qa_multiwavelength_sed=qa_multiwavelength_sed)
            #continue # here!
            #pdb.set_trace()

        # Multiwavelength coadds (does not support just_coadds=True)--
        if galex:
            make_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
                                        refpixscale=pixscale,
                                        #barlen=barlen, barlabel=barlabel,
                                        clobber=clobber, verbose=verbose)

        # Build the montage coadds.
        make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=barlen,
                            barlabel=barlabel, clobber=clobber, verbose=verbose,
                            just_coadds=just_coadds)
            
        # CCD positions
        #make_ccdpos_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=pixscale,
        #               radius=radius_mosaic_pixels, zcolumn=zcolumn, survey=survey,
        #               clobber=clobber, verbose=verbose)

        # Build the maskbits figure.
        #make_maskbits_qa(galaxy, galaxydir, htmlgalaxydir, clobber=clobber, verbose=verbose)

        # Sersic fiting results
        if False:
            make_sersic_qa(galaxy, galaxydir, htmlgalaxydir, bands=bands,
                           clobber=clobber, cosmo=cosmo, verbose=verbose)

        # Build the CCD-level QA.  This QA script needs to be last, because we
        # check the completeness of the HTML portion of legacyhalos-mpi based on
        # the ccdpos file.
        if ccdqa:
            from astrometry.util.multiproc import multiproc
            mp = multiproc(nthreads=nproc)
            make_ccd_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=pixscale,
                        zcolumn=zcolumn, mp=mp, survey=survey, clobber=clobber,
                        verbose=verbose)

        # Build the MGE plots.
        #qa_mge_results(galaxy, galaxydir, htmlgalaxydir, refband='r', band=band,
        #               clobber=clobber, verbose=verbose)

    return 1

def skyserver_link(sdss_objid):
    return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(sdss_objid)

# Get the viewer link
def viewer_link(ra, dec, width, sga=False, manga=False, dr10=False):
    baseurl = 'http://legacysurvey.org/viewer-dev/'
    if width > 1200:
        zoom = 13
    elif (width > 400) * (width < 1200):
        zoom = 14
    else:
        zoom = 15

    if dr10:
        drlayer = 'ls-dr10'
    else:
        drlayer = 'ls-dr9'
        
    layer1 = ''
    if sga:
        layer1 = '&sga&sga-parent'
    if manga:
        layer1 = layer1+'&manga'

    viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer={}{}'.format(
        baseurl, ra, dec, zoom, drlayer, layer1)

    return viewer

