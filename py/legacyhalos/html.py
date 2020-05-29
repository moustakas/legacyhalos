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

def get_cutouts(groupsample, use_nproc=nproc, clobber=False):
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

    ccdposfile = os.path.join(htmlgalaxydir, '{}-ccdpos.png'.format(galaxy))
    if not os.path.isfile(ccdposfile) or clobber:
        display_ccdpos(onegal, ccds, png=ccdposfile, zcolumn=zcolumn)

def make_ccdpos_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=0.262,
                   radius=None, survey=None, clobber=False, verbose=False):
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

    ccdposfile = os.path.join(htmlgalaxydir, '{}-ccdpos.png'.format(galaxy))
    if not os.path.isfile(ccdposfile) or clobber:
        display_ccdpos(onegal, ccds, radius=radius, png=ccdposfile, verbose=verbose)

def make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=None,
                        barlabel=None, clobber=False, verbose=False):
    """Montage the coadds into a nice QAplot.

    barlen - pixels

    """
    from legacyhalos.qa import addbar_to_png, fonttype
    from PIL import Image, ImageDraw, ImageFont
    
    Image.MAX_IMAGE_PIXELS = None

    for filesuffix in ['largegalaxy']:
    #for filesuffix in ['largegalaxy', 'pipeline', 'custom']:
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        if not os.path.isfile(montagefile) or clobber:
            if filesuffix == 'custom':
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-nocentral-grz'.format(filesuffix),
                              '{}-image-central-grz'.format(filesuffix))
            else:
                coaddfiles = ('{}-image-grz'.format(filesuffix),
                              '{}-model-grz'.format(filesuffix),
                              '{}-resid-grz'.format(filesuffix))

            # Image coadd with the scale bar label--
            barpngfile = os.path.join(htmlgalaxydir, '{}-{}.png'.format(galaxy, coaddfiles[0]))
                
            # Make sure all the files exist.
            check, just_coadds = True, False
            jpgfile = []
            for suffix in coaddfiles:
                _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
                jpgfile.append(_jpgfile)
                if not os.path.isfile(_jpgfile):
                    if verbose:
                        print('File {} not found!'.format(_jpgfile))
                    check = False

            # Check for just the image coadd.
            if check is False:
                if os.path.isfile(np.atleast_1d(jpgfile)[0]):
                    just_coadds = True

            if check or just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096 and sz[0] < 8192:
                    resize = '-resize 2048x2048 '
                elif sz[0] > 8192:
                    resize = '-resize 1024x1024 '
                    #resize = '-resize 4096x4096 '
                else:
                    resize = ''

                # Make a quick thumbnail of just the data.
                cmd = 'convert -thumbnail {0}x{0} {1} {2}'.format(96, np.atleast_1d(jpgfile)[0], thumb2file)
                if os.path.isfile(thumb2file):
                    os.remove(thumb2file)
                print('Writing {}'.format(thumb2file))
                subprocess.call(cmd.split())
                    
                # Add a bar and label to the first image.
                if just_coadds:
                    cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 {} -geometry +0+0 '.format(resize)
                    #cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 -resize 4096x4096\> '
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
                    cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 {} -geometry +0+0 '.format(resize)
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

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
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

def make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=('g', 'r', 'z'),
                    refband='r', pixscale=0.262, barlen=None, barlabel=None,
                    clobber=False, verbose=False, largegalaxy=False,
                    scaledfont=False):
    """Generate QAplots from the ellipse-fitting.

    """
    import fitsio
    from PIL import Image
    from astropy.table import Table
    from legacyhalos.io import read_multiband, read_ellipsefit
    from legacyhalos.qa import (display_multiband, display_ellipsefit,
                                display_ellipse_sbprofile, qa_curveofgrowth,
                                qa_maskbits)

    # Read the data--
    data = read_multiband(galaxy, galaxydir, bands=bands,
                          refband=refband, pixscale=pixscale,
                          verbose=verbose,
                          largegalaxy=largegalaxy)
    if not bool(data):
        return

    if data['failed']: # all galaxies dropped
        return
    
    # One set of QA plots per galaxy.
    if largegalaxy:
        ellipsefitall = []
        for igal in np.arange(len(data['central_galaxy_id'])):
            central_galaxy_id = data['central_galaxy_id'][igal]
            galaxyid = str(central_galaxy_id)
            filesuffix = 'largegalaxy'

            ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix,
                                         galaxyid=galaxyid, verbose=verbose)
            if bool(ellipsefit):
                ellipsefitall.append(ellipsefit)

                sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-{}-ellipse-sbprofile.png'.format(galaxy, filesuffix, galaxyid))
                if not os.path.isfile(sbprofilefile) or clobber:
                    display_ellipse_sbprofile(ellipsefit, plot_radius=False, plot_sbradii=True, # note, False!
                                              png=sbprofilefile, verbose=verbose, minerr=0.0)

                cogfile = os.path.join(htmlgalaxydir, '{}-{}-{}-ellipse-cog.png'.format(galaxy, filesuffix, galaxyid))
                if not os.path.isfile(cogfile) or clobber:
                    qa_curveofgrowth(ellipsefit, pipeline_ellipsefit={}, plot_sbradii=True,
                                     png=cogfile, verbose=verbose)

                multibandfile = os.path.join(htmlgalaxydir, '{}-{}-{}-ellipse-multiband.png'.format(galaxy, filesuffix, galaxyid))
                thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-{}-ellipse-multiband.png'.format(galaxy, filesuffix, galaxyid))
                if not os.path.isfile(multibandfile) or clobber:
                    with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, filesuffix))) as colorimg:
                        display_multiband(data, ellipsefit=ellipsefit, colorimg=colorimg,
                                          centralindx=igal, barlen=barlen, barlabel=barlabel,
                                          png=multibandfile, verbose=verbose, scaledfont=scaledfont)

                    # Create a thumbnail.
                    cmd = 'convert -thumbnail 1024x1024 {} {}'.format(multibandfile, thumbfile)#.replace('>', '\>')
                    if os.path.isfile(thumbfile):
                        os.remove(thumbfile)
                    print('Writing {}'.format(thumbfile))
                    subprocess.call(cmd.split())

        # maskbits QA
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
            tractor = Table(fitsio.read(tractorfile, upper=True))

            with Image.open(os.path.join(galaxydir, '{}-{}-image-grz.jpg'.format(galaxy, filesuffix))) as colorimg:
                qa_maskbits(mask, tractor, ellipsefitall, colorimg, png=maskbitsfile)

    #for filesuffix in ('largegalaxy', 'custom'):
    #    ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix, verbose=verbose)
    #
    #    #sky_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='sky')
    #    #sdss_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='sdss')
    #    #pipeline_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='pipeline')
    #    sky_ellipsefit, sdss_ellipsefit, pipeline_ellipsefit = {}, {}, {}
    #
    #    if len(ellipsefit) > 0:
    #        # Toss out bad fits.
    #        indx = None
    #        #indx = (isophotfit[refband].stop_code < 4) * (isophotfit[refband].intens > 0)
    #        #indx = (isophotfit[refband].stop_code <= 4) * (isophotfit[refband].intens > 0)
    #
    #        cogfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-cog.png'.format(galaxy, filesuffix))
    #        if not os.path.isfile(cogfile) or clobber:
    #            qa_curveofgrowth(ellipsefit, pipeline_ellipsefit=pipeline_ellipsefit,
    #                             png=cogfile, verbose=verbose)
    #
    #        sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-sbprofile.png'.format(galaxy, filesuffix))
    #        if not os.path.isfile(sbprofilefile) or clobber:
    #            display_ellipse_sbprofile(ellipsefit, sky_ellipsefit=sky_ellipsefit,
    #                                      pipeline_ellipsefit=pipeline_ellipsefit,
    #                                      png=sbprofilefile, verbose=verbose, minerr=0.0,
    #                                      sdss_ellipsefit=sdss_ellipsefit)
    #
    #        multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-multiband.png'.format(galaxy, filesuffix))
    #        if not os.path.isfile(multibandfile) or clobber:
    #            data = read_multiband(galaxy, galaxydir, bands=bands,
    #                                  largegalaxy=filesuffix=='largegalaxy')
    #            
    #            display_multiband(data, ellipsefit=ellipsefit, indx=indx, barlen=barlen,
    #                              barlabel=barlabel, png=multibandfile, verbose=verbose)
    #
    #        ellipsefitfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-ellipsefit.png'.format(galaxy, filesuffix))
    #        if not os.path.isfile(ellipsefitfile) or clobber:
    #            display_ellipsefit(ellipsefit, png=ellipsefitfile, xlog=False, verbose=verbose)

def make_sersic_qa(galaxy, galaxydir, htmlgalaxydir, bands=('g', 'r', 'z'),
                      clobber=False, verbose=False):
    """Generate QAplots from the Sersic modeling.

    """
    from legacyhalos.io import read_sersic
    from legacyhalos.qa import display_sersic

    # Double Sersic
    double = read_sersic(galaxy, galaxydir, modeltype='double')
    if bool(double):
        doublefile = os.path.join(htmlgalaxydir, '{}-sersic-double.png'.format(galaxy))
        if not os.path.isfile(doublefile) or clobber:
            display_sersic(double, png=doublefile, verbose=verbose)

    # Double Sersic, no wavelength dependence
    double = read_sersic(galaxy, galaxydir, modeltype='double-nowavepower')
    if bool(double):
        doublefile = os.path.join(htmlgalaxydir, '{}-sersic-double-nowavepower.png'.format(galaxy))
        if not os.path.isfile(doublefile) or clobber:
            display_sersic(double, png=doublefile, verbose=verbose)

    # Single Sersic, no wavelength dependence
    single = read_sersic(galaxy, galaxydir, modeltype='single-nowavepower')
    if bool(single):
        singlefile = os.path.join(htmlgalaxydir, '{}-sersic-single-nowavepower.png'.format(galaxy))
        if not os.path.isfile(singlefile) or clobber:
            display_sersic(single, png=singlefile, verbose=verbose)

    # Single Sersic
    single = read_sersic(galaxy, galaxydir, modeltype='single')
    if bool(single):
        singlefile = os.path.join(htmlgalaxydir, '{}-sersic-single.png'.format(galaxy))
        if not os.path.isfile(singlefile) or clobber:
            display_sersic(single, png=singlefile, verbose=verbose)

    # Sersic-exponential
    serexp = read_sersic(galaxy, galaxydir, modeltype='exponential')
    if bool(serexp):
        serexpfile = os.path.join(htmlgalaxydir, '{}-sersic-exponential.png'.format(galaxy))
        if not os.path.isfile(serexpfile) or clobber:
            display_sersic(serexp, png=serexpfile, verbose=verbose)

    # Sersic-exponential, no wavelength dependence
    serexp = read_sersic(galaxy, galaxydir, modeltype='exponential-nowavepower')
    if bool(serexp):
        serexpfile = os.path.join(htmlgalaxydir, '{}-sersic-exponential-nowavepower.png'.format(galaxy))
        if not os.path.isfile(serexpfile) or clobber:
            display_sersic(serexp, png=serexpfile, verbose=verbose)

def make_plots(sample, datadir=None, htmldir=None, survey=None, refband='r',
               bands=('g', 'r', 'z'), pixscale=0.262, zcolumn='Z', 
               nproc=1, barlen=None, barlabel=None,
               radius_mosaic_arcsec=None, maketrends=False, ccdqa=False,
               clobber=False, verbose=True, get_galaxy_galaxydir=None,
               largegalaxy=False, scaledfont=False):
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
            barlen = np.round(barlen_kpc / legacyhalos.misc.arcsec2kpc(onegal[zcolumn]) / pixscale).astype('int') # [kpc]

        if radius_mosaic_arcsec is None:
            radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
                redshift=onegal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

        # Build the maskbits figure.
        #make_maskbits_qa(galaxy, galaxydir, htmlgalaxydir, clobber=clobber, verbose=verbose)

        # Build the ellipse plots.
        make_ellipse_qa(galaxy, galaxydir, htmlgalaxydir, bands=bands, refband=refband,
                        pixscale=pixscale, barlen=barlen, barlabel=barlabel, clobber=clobber,
                        verbose=verbose, largegalaxy=largegalaxy, scaledfont=scaledfont)

        # Build the montage coadds.
        make_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=barlen,
                            barlabel=barlabel, clobber=clobber, verbose=verbose)

        # CCD positions
        make_ccdpos_qa(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=pixscale,
                       radius=radius_mosaic_pixels, survey=survey, clobber=clobber,
                       verbose=verbose)

        # Sersic fiting results
        if False:
            make_sersic_qa(galaxy, galaxydir, htmlgalaxydir, bands=bands,
                           clobber=clobber, verbose=verbose)

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

# Get the viewer link
def viewer_link(ra, dec, width, lslga=False):
    baseurl = 'http://legacysurvey.org/viewer-dev/'
    if width > 1200:
        zoom = 13
    elif (width > 400) * (width < 1200):
        zoom = 14
    else:
        zoom = 15
    if lslga:
        layer1 = '&lslga'
    else:
        layer1 = ''
    viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8{}'.format(
        baseurl, ra, dec, zoom, layer1)

    return viewer

def skyserver_link(gal):
    if 'SDSS_OBJID' in gal.colnames:
        return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['SDSS_OBJID'])
    else:
        return ''

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              first=None, last=None, nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
    import subprocess
    import fitsio
    from legacyhalos.misc import RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if sample is None:
        sample = legacyhalos.io.read_sample(first=first, last=last)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    if zcolumn is None:
        zcolumn = ZCOLUMN
        
    galaxy, galaxydir, htmlgalaxydir = legacyhalos.io.get_galaxy_galaxydir(sample, html=True)

    # Write the last-updated date to a webpage.
    js = html_javadate()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * legacyhalos.misc.cutout_radius_kpc(
            redshift=gal[zcolumn], pixscale=0.262,
            radius_kpc=radius_mosaic_kpc) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8'.format(
            baseurl, gal['RA'], gal['DEC'], zoom)
        
        return viewer

    def _skyserver_link(gal):
        if 'SDSS_OBJID' in gal.colnames:
            return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['SDSS_OBJID'])
        else:
            return ''

    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
        #shutil.chown(htmldir, group='cosmo')
    homehtmlfile = os.path.join(htmldir, homehtml)

    #if verbose:
    print('Writing {}'.format(homehtmlfile))
    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>LegacyHalos: Central Galaxies</h1>\n')
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Richness</th>\n')
        html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')
        for ii, (gal, galaxy1, htmlgalaxydir1) in enumerate(zip(
            sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir) )):

            htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        if verbose:
            print('Writing {}'.format(trendshtmlfile))
        with open(trendshtmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>LegacyHalos: Sample Trends</h1>\n')
            html.write('<p><a href="https://github.com/moustakas/legacyhalos">Code and documentation</a></p>\n')
            html.write('<a href="trends/ellipticity_vs_sma.png"><img src="trends/ellipticity_vs_sma.png" alt="Missing file ellipticity_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/gr_vs_sma.png"><img src="trends/gr_vs_sma.png" alt="Missing file gr_vs_sma.png" height="auto" width="50%"></a>')
            html.write('<a href="trends/rz_vs_sma.png"><img src="trends/rz_vs_sma.png" alt="Missing file rz_vs_sma.png" height="auto" width="50%"></a>')

            html.write('<br /><br />\n')
            html.write('<b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')
            html.close()

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    # Make a separate HTML page for each object.
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate( zip(
        sample, np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir) ) ):

        radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
            redshift=gal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        diam_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale)

        ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
        
        if not os.path.exists(htmlgalaxydir1):
            os.makedirs(htmlgalaxydir1, mode=0o775)
            #shutil.chown(htmlgalaxydir1, group='cosmo')

        ccdsfile = os.path.join(galaxydir1, '{}-ccds.fits'.format(galaxy1))
        if os.path.isfile(ccdsfile):
            nccds = fitsio.FITS(ccdsfile)[1].get_nrows()
        else:
            nccds = None

        nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
        prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

        htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>Central Galaxy {}</h1>\n'.format(galaxy1))

            html.write('<a href="../../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../../{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../../{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Richness</th>\n')
            html.write('<th>Pcen</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('<th>SkyServer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Image Mosaics</h2>\n')
            html.write('<p>Each mosaic (left to right: data, model of all but the central galaxy, and residual image containing just the central galaxy) is {:.0f} kpc = {:.3f} arcsec = {:.0f} pixels in diameter.</p>\n'.format(2*radius_mosaic_kpc, 2*radius_mosaic_arcsec, diam_mosaic_pixels))
            #html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr><td><a href="{}-grz-montage.png"><img src="{}-grz-montage.png" alt="Missing file {}-grz-montage.png" height="auto" width="100%"></a></td></tr>\n'.format(galaxy1, galaxy1, galaxy1))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            
            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-multiband.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            #html.write('<td><a href="{}-ellipse-ellipsefit.png"><img src="{}-ellipse-ellipsefit.png" alt="Missing file {}-ellipse-ellipsefit.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
            pngfile = '{}-ellipse-sbprofile.png'.format(galaxy1)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-ellipse-cog.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')

            if False:
                html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
                html.write('<table width="90%">\n')

                # single-sersic
                html.write('<tr>\n')
                html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-single-nowavepower.png"><img src="{}-sersic-single-nowavepower.png" alt="Missing file {}-sersic-single-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-single.png"><img src="{}-sersic-single.png" alt="Missing file {}-sersic-single.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                # Sersic+exponential
                html.write('<tr>\n')
                html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-exponential-nowavepower.png"><img src="{}-sersic-exponential-nowavepower.png" alt="Missing file {}-sersic-exponential-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-exponential.png"><img src="{}-sersic-exponential.png" alt="Missing file {}-sersic-exponential.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                # double-sersic
                html.write('<tr>\n')
                html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
                html.write('</tr>\n')
                html.write('<tr>\n')
                html.write('<td><a href="{}-sersic-double-nowavepower.png"><img src="{}-sersic-double-nowavepower.png" alt="Missing file {}-sersic-double-nowavepower.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('<td><a href="{}-sersic-double.png"><img src="{}-sersic-double.png" alt="Missing file {}-sersic-double.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
                html.write('</tr>\n')

                html.write('</table>\n')

                html.write('<br />\n')

            if nccds and ccdqa:
                html.write('<h2>CCD Diagnostics</h2>\n')
                html.write('<table width="90%">\n')
                html.write('<tr>\n')
                pngfile = '{}-ccdpos.png'.format(galaxy1)
                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                    pngfile))
                html.write('</tr>\n')

                for iccd in range(nccds):
                    html.write('<tr>\n')
                    pngfile = '{}-2d-ccd{:02d}.png'.format(galaxy1, iccd)
                    html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                        pngfile))
                    html.write('</tr>\n')
                html.write('</table>\n')
                html.write('<br />\n')

            html.write('<a href="../../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    # Make the plots.
    if makeplots:
        err = make_plots(sample, datadir=datadir, htmldir=htmldir, refband=refband,
                         bands=bands, pixscale=pixscale, zcolumn=zcolumn, survey=survey,
                         clobber=clobber, verbose=verbose, nproc=nproc, ccdqa=ccdqa,
                         maketrends=maketrends)

    try:
        cmd = '/usr/bin/chgrp -R cosmo {}'.format(htmldir)
        print(cmd)
        err1 = subprocess.call(cmd.split())

        cmd = 'find {} -type d -exec chmod 775 {{}} +'.format(htmldir)
        print(cmd)
        err2 = subprocess.call(cmd.split())

        cmd = 'find {} -type f -exec chmod 664 {{}} +'.format(htmldir)
        print(cmd)
        err3 = subprocess.call(cmd.split())
    except:
        pass

    #if err1 != 0 or err2 != 0 or err3 != 0:
    #    print('Something went wrong updating permissions; please check the logfile.')
    #    return 0
    
    return 1
