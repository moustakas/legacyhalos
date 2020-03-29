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
import legacyhalos.hsc

def qa_ccd(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=0.262, ccds=None,
           zcolumn='Z', radius_pixel=None, survey=None, mp=None, clobber=False,
           verbose=False):
    """Build CCD-level QA.

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

def qa_ccdpos(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=0.262,
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

def qa_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=None,
                      barlabel=None, clobber=False, verbose=False):
    """Montage the coadds into a nice QAplot.

    barlen - pixels

    """
    #from pkg_resources import resource_filename
    from PIL import Image, ImageDraw, ImageFont
    Image.MAX_IMAGE_PIXELS = None
    fonttype = os.path.join(os.getenv('LEGACYHALOS_CODE_DIR'), 'py', 'legacyhalos', 'data', 'Georgia-Italic.ttf')
    #fonttype = resource_filename('legacyhalos', 'data/Georgia.ttf')

    def addbar(jpgfile, barlen, barlabel, imtype, scaledfont=False):
        pngfile = os.path.join(htmlgalaxydir, os.path.basename(jpgfile).replace('.jpg', '.png'))
        with Image.open(jpgfile) as im:
            draw = ImageDraw.Draw(im)
            sz = im.size
            width = np.round(sz[0]/150).astype('int')

            # Bar and label
            if barlen:
                if scaledfont:
                    fntsize = np.round(sz[0]/50).astype('int')
                else:
                    fntsize = 20 # np.round(sz[0]/20).astype('int')
                font = ImageFont.truetype(fonttype, size=fntsize)
                # Add a scale bar and label--
                x0, x1, y0, y1 = 0+fntsize*2, 0+fntsize*2+barlen, sz[1]-fntsize*2, sz[1]-fntsize*2.5#4
                draw.line((x0, y1, x1, y1), fill='white', width=width)
                ww, hh = draw.textsize(barlabel, font=font)
                dx = ((x1-x0) - ww)//2
                #print(x0, x1, y0, y1, ww, x0+dx, sz)
                draw.text((x0+dx, y0), barlabel, font=font)
                #print('Writing {}'.format(pngfile))
            # Image type
            if imtype:
                fntsize = 20 # np.round(sz[0]/20).astype('int')
                font = ImageFont.truetype(fonttype, size=fntsize)
                ww, hh = draw.textsize(imtype, font=font)
                x0, y0, y1 = sz[0]-ww-fntsize*2, sz[1]-fntsize*2, sz[1]-fntsize*2.5#4
                draw.text((x0, y1), imtype, font=font)
            im.save(pngfile)
        return pngfile

    for filesuffix in ('largegalaxy', 'pipeline', 'custom'):
        montagefile = os.path.join(htmlgalaxydir, '{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        thumbfile = os.path.join(htmlgalaxydir, 'thumb-{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        thumb2file = os.path.join(htmlgalaxydir, 'thumb2-{}-{}-grz-montage.png'.format(galaxy, filesuffix))
        if not os.path.isfile(montagefile) or clobber:
            if filesuffix == 'custom':
                coaddfiles = ('{}-image-grz'.format(filesuffix), '{}-model-nocentral-grz'.format(filesuffix), '{}-image-central-grz'.format(filesuffix))
            else:
                coaddfiles = ('{}-image-grz'.format(filesuffix), '{}-model-grz'.format(filesuffix), '{}-resid-grz'.format(filesuffix))
                
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
                    
            # Check for just the image coadd..
            if check is False:
                jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, coaddfiles[0]))
                if os.path.isfile(jpgfile):
                    just_coadds = True
                    
            if check or just_coadds:
                with Image.open(np.atleast_1d(jpgfile)[0]) as im:
                    sz = im.size
                if sz[0] > 4096:
                    resize = '-resize 4096x4096 '
                else:
                    resize = ''
                # Add a bar and label
                if just_coadds:
                    cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 {} -geometry +0+0 '.format(resize)
                    #cmd = 'montage -bordercolor white -borderwidth 1 -tile 1x1 -geometry +0+0 -resize 4096x4096\> '
                    if barlen:
                        pngfile = addbar(jpgfile, barlen, barlabel, None, scaledfont=True)
                        cmd = cmd+' '+pngfile
                    else:
                        cmd = cmd+' '+jpgfile
                else:
                    cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 {} -geometry +0+0 '.format(resize)
                    if barlen:
                        pngfile = [addbar(ff, barlen, barlabel, None) for ff in jpgfile]
                        cmd = cmd+' '+pngfile[0]+' '
                        cmd = cmd+' '.join(ff for ff in jpgfile[1:])
                    else:
                        cmd = cmd+' '.join(ff for ff in jpgfile)
                cmd = cmd+' {}'.format(montagefile)

                #if verbose:
                print('Writing {}'.format(montagefile))
                subprocess.call(cmd.split())
                if not os.path.isfile(montagefile):
                    print('There was a problem writing {}'.format(montagefile))
                    continue

                # Create a couple smaller thumbnail images
                for tf, sz in zip((thumbfile, thumb2file), (512, 96)):
                    cmd = 'convert -thumbnail {}x{} {} {}'.format(sz, sz, montagefile, tf)
                    #if verbose:
                    print('Writing {}'.format(tf))
                    subprocess.call(cmd.split())

def qa_maskbits(galaxy, galaxydir, htmlgalaxydir, clobber=False, verbose=False):
    """Visualize the maskbits image.

    """
    import fitsio
    import matplotlib.pyplot as plt

    for filesuffix in ('largegalaxy', 'pipeline', 'custom'):
        maskbitsfile = os.path.join(htmlgalaxydir, '{}-{}-maskbits.png'.format(galaxy, filesuffix))
        if not os.path.isfile(maskbitsfile) or clobber:
            fitsfile = os.path.join(galaxydir, '{}-{}-maskbits.fits.fz'.format(galaxy, filesuffix))
            if not os.path.isfile(fitsfile):
                if verbose:
                    print('File {} not found!'.format(fitsfile))
                continue

            img = fitsio.read(fitsfile)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img, origin='lower', cmap='gray_r')#, interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.axis('off')
            ax.autoscale(False)

            #if verbose:
            print('Writing {}'.format(maskbitsfile))
            fig.savefig(maskbitsfile, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

def qa_ellipse_results(galaxy, galaxydir, htmlgalaxydir, bands=('g', 'r', 'z'),
                       barlen=None, barlabel=None, clobber=False, verbose=False):
    """Generate QAplots from the ellipse-fitting.

    """
    from legacyhalos.io import read_multiband, read_ellipsefit
    from legacyhalos.qa import (display_multiband, display_ellipsefit,
                                display_ellipse_sbprofile, qa_curveofgrowth)

    for filesuffix in ('largegalaxy', 'custom'):
        ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix=filesuffix, verbose=verbose)

        #sky_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='sky')
        #sdss_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='sdss')
        #pipeline_ellipsefit = read_ellipsefit(galaxy, galaxydir, filesuffix='pipeline')
        sky_ellipsefit, sdss_ellipsefit, pipeline_ellipsefit = {}, {}, {}

        if len(ellipsefit) > 0:
            # Toss out bad fits.
            indx = None
            #indx = (isophotfit[refband].stop_code < 4) * (isophotfit[refband].intens > 0)
            #indx = (isophotfit[refband].stop_code <= 4) * (isophotfit[refband].intens > 0)

            cogfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-cog.png'.format(galaxy, filesuffix))
            if not os.path.isfile(cogfile) or clobber:
                qa_curveofgrowth(ellipsefit, pipeline_ellipsefit=pipeline_ellipsefit,
                                 png=cogfile, verbose=verbose)

            sbprofilefile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-sbprofile.png'.format(galaxy, filesuffix))
            if not os.path.isfile(sbprofilefile) or clobber:
                display_ellipse_sbprofile(ellipsefit, sky_ellipsefit=sky_ellipsefit,
                                          pipeline_ellipsefit=pipeline_ellipsefit,
                                          png=sbprofilefile, verbose=verbose, minerr=0.0,
                                          sdss_ellipsefit=sdss_ellipsefit)

            multibandfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-multiband.png'.format(galaxy, filesuffix))
            if not os.path.isfile(multibandfile) or clobber:
                data = read_multiband(galaxy, galaxydir, bands=bands,
                                      largegalaxy=filesuffix=='largegalaxy')
                
                display_multiband(data, ellipsefit=ellipsefit, indx=indx, barlen=barlen,
                                  barlabel=barlabel, png=multibandfile, verbose=verbose)

            ellipsefitfile = os.path.join(htmlgalaxydir, '{}-{}-ellipse-ellipsefit.png'.format(galaxy, filesuffix))
            if not os.path.isfile(ellipsefitfile) or clobber:
                display_ellipsefit(ellipsefit, png=ellipsefitfile, xlog=False, verbose=verbose)

def qa_mge_results(galaxy, galaxydir, htmlgalaxydir, refband='r', bands=('g', 'r', 'z'),
                   pixscale=0.262, clobber=False, verbose=False):
    """Generate QAplots from the MGE fitting.

    """
    from legacyhalos.io import read_mgefit, read_multiband
    from legacyhalos.qa import display_mge_sbprofile, display_multiband
    
    mgefit = read_mgefit(galaxy, galaxydir)

    if len(mgefit) > 0:

        ## Toss out bad fits.
        #indx = (mgefit[refband].stop_code <= 4) * (mgefit[refband].intens > 0)
        #
        multibandfile = os.path.join(htmlgalaxydir, '{}-mge-multiband.png'.format(galaxy))
        if not os.path.isfile(multibandfile) or clobber:
            data = read_multiband(galaxy, galaxydir, band=band)
            display_multiband(data, mgefit=mgefit, bands=bands, refband=refband,
                              png=multibandfile, contours=True, verbose=verbose)
        
        #isophotfile = os.path.join(htmlgalaxydir, '{}-mge-mgefit.png'.format(galaxy))
        #if not os.path.isfile(isophotfile) or clobber:
        #    # Just display the reference band.
        #    display_mgefit(mgefit, band=refband, indx=indx, pixscale=pixscale,
        #                   png=isophotfile, verbose=verbose)

        sbprofilefile = os.path.join(htmlgalaxydir, '{}-mge-sbprofile.png'.format(galaxy))
        if not os.path.isfile(sbprofilefile) or clobber:
            display_mge_sbprofile(mgefit, band=band, refband=refband, pixscale=pixscale,
                                  png=sbprofilefile, verbose=verbose)
        
def qa_sersic_results(galaxy, galaxydir, htmlgalaxydir, bands=('g', 'r', 'z'),
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
               nproc=1, barlen=None, barlabel=None, radius_mosaic_arcsec=None,
               maketrends=False, ccdqa=False, clobber=False, verbose=True,
               #pipeline_montage=False, largegalaxy_montage=False,
               get_galaxy_galaxydir=None):
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

    if ccdqa:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(nthreads=nproc)

    #from legacyhalos.misc import RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    barlen_kpc = 100
    if barlabel is None:
        barlabel = '100 kpc'

    if get_galaxy_galaxydir is None:
        get_galaxy_galaxydir = legacyhalos.io.get_galaxy_galaxydir

    # Loop on each galaxy.
    for ii, onegal in enumerate(sample):
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(onegal, html=True)
        #if galaxylist is None:
        #    galaxy, galaxydir, htmlgalaxydir = legacyhalos.io.get_galaxy_galaxydir(onegal, html=True)
        #else:
        #    galaxy = galaxylist[ii]
        #    galaxydir = os.path.join(datadir, galaxy)
        #    htmlgalaxydir = os.path.join(htmldir, galaxy)
            
        if not os.path.isdir(htmlgalaxydir):
            os.makedirs(htmlgalaxydir, exist_ok=True, mode=0o775)
            #shutil.chown(htmlgalaxydir, group='cosmo')

        if barlen is None:
            barlen = np.round(barlen_kpc / legacyhalos.misc.arcsec2kpc(onegal[zcolumn]) / pixscale).astype('int') # [kpc]

        if radius_mosaic_arcsec is None:
            radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
                redshift=onegal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
        radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

        # Build the ellipse plots.
        qa_ellipse_results(galaxy, galaxydir, htmlgalaxydir, bands=bands, barlen=barlen,
                           barlabel=barlabel, clobber=clobber, verbose=verbose)

        # CCD positions
        qa_ccdpos(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=pixscale,
                  radius=radius_mosaic_pixels, survey=survey, clobber=clobber,
                  verbose=verbose)

        # Build the montage coadds.
        qa_montage_coadds(galaxy, galaxydir, htmlgalaxydir, barlen=barlen,
                          barlabel=barlabel, clobber=clobber, verbose=verbose)
                          #pipeline_montage=pipeline_montage, largegalaxy_montage=largegalaxy_montage)

        # Build the maskbits figure.
        qa_maskbits(galaxy, galaxydir, htmlgalaxydir, clobber=clobber, verbose=verbose)

        # Sersic fiting results
        if False:
            qa_sersic_results(galaxy, galaxydir, htmlgalaxydir, bands=bands,
                              clobber=clobber, verbose=verbose)

        # Build the CCD-level QA.  This QA script needs to be last, because we
        # check the completeness of the HTML portion of legacyhalos-mpi based on
        # the ccdpos file.
        if ccdqa:
            qa_ccd(onegal, galaxy, galaxydir, htmlgalaxydir, pixscale=pixscale,
                   zcolumn=zcolumn, mp=mp, survey=survey, clobber=clobber,
                   verbose=verbose)

        # Build the MGE plots.
        #qa_mge_results(galaxy, galaxydir, htmlgalaxydir, refband='r', band=band,
        #               clobber=clobber, verbose=verbose)

    return 1

def _javastring():
    """Return a string that embeds a date in a webpage."""
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
    js = _javastring()       

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
