from __future__ import (absolute_import, division)

import os, subprocess
import numpy as np

import seaborn as sns
sns.set(style='ticks', font_scale=1.2, palette='Set2')

def qa_montage_coadds(objid, objdir, htmlobjdir, clobber=False):
    """Montage the coadds into a nice QAplot."""

    montagefile = os.path.join(htmlobjdir, '{}-coadd-montage.png'.format(objid))

    if not os.path.isfile(montagefile) or clobber:
        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
        cmd = cmd+' '.join([os.path.join(objdir, '{}-{}.jpg'.format(objid, suffix)) for
                            suffix in ('image', 'model', 'resid')])
        cmd = cmd+' {}'.format(montagefile)
        print('Writing {}'.format(montagefile))

        err = subprocess.call(cmd.split())

    else:
        err = 0

    return err

def qa_ellipse_results(objid, objdir, htmlobjdir, redshift=None, refband='r',
                       band=('g', 'r', 'z'), pixscale=0.262, clobber=False):
    """Generate QAplots from the ellipse-fitting.

    """
    from legacyhalos.io import read_isophotfit
    from legacyhalos.ellipse import (read_multiband, display_multiband,
                                     display_isophotfit, display_ellipse_sbprofile)

    isophotfit = read_isophotfit(objid, objdir, band=band)
    if len(isophotfit[refband]) > 0:

        # Toss out bad fits.
        #indx = (isophotfit[refband].stop_code < 4) * (isophotfit[refband].intens > 0)
        indx = (isophotfit[refband].stop_code <= 4) * (isophotfit[refband].intens > 0)

        multibandfile = os.path.join(htmlobjdir, '{}-ellipse-multiband.png'.format(objid))
        if not os.path.isfile(multibandfile) or clobber:
            data = read_multiband(objid, objdir, band=band)
            print('Writing {}'.format(multibandfile))
            display_multiband(data, isophotfit=isophotfit, band=band,
                              indx=indx, png=multibandfile)

        isophotfile = os.path.join(htmlobjdir, '{}-ellipse-isophotfit.png'.format(objid))
        if not os.path.isfile(isophotfile) or clobber:
            # Just display the reference band.
            print('Writing {}'.format(isophotfile))
            display_isophotfit(isophotfit, band=refband, redshift=redshift,
                               indx=indx, pixscale=pixscale, png=isophotfile)

        sbprofilefile = os.path.join(htmlobjdir, '{}-ellipse-sbprofile.png'.format(objid))
        if not os.path.isfile(sbprofilefile) or clobber:
            print('Writing {}'.format(sbprofilefile))
            display_ellipse_sbprofile(isophotfit, band=band, redshift=redshift,
                                      indx=indx, pixscale=pixscale, png=sbprofilefile)
        
def qa_mge_results(objid, objdir, htmlobjdir, redshift=None, refband='r',
                   band=('g', 'r', 'z'), pixscale=0.262, clobber=False):
    """Generate QAplots from the MGE fitting.

    """
    from legacyhalos.io import read_mgefit
    from legacyhalos.ellipse import (display_mge_sbprofile, read_multiband,
                                     display_multiband)

    mgefit = read_mgefit(objid, objdir)

    if len(mgefit) > 0:

        ## Toss out bad fits.
        #indx = (mgefit[refband].stop_code <= 4) * (mgefit[refband].intens > 0)
        #
        multibandfile = os.path.join(htmlobjdir, '{}-mge-multiband.png'.format(objid))
        if not os.path.isfile(multibandfile) or clobber:
            data = read_multiband(objid, objdir, band=band)
            print('Writing {}'.format(multibandfile))
            display_multiband(data, mgefit=mgefit, band=band, png=multibandfile,
                              contours=True)
        
        #isophotfile = os.path.join(htmlobjdir, '{}-mge-mgefit.png'.format(objid))
        #if not os.path.isfile(isophotfile) or clobber:
        #    # Just display the reference band.
        #    print('Writing {}'.format(isophotfile))
        #    display_mgefit(mgefit, band=refband, redshift=redshift,
        #                       indx=indx, pixscale=pixscale, png=isophotfile)

        sbprofilefile = os.path.join(htmlobjdir, '{}-mge-sbprofile.png'.format(objid))
        if not os.path.isfile(sbprofilefile) or clobber:
            print('Writing {}'.format(sbprofilefile))
            display_mge_sbprofile(mgefit, band=band, refband=refband, redshift=redshift,
                                  pixscale=pixscale, png=sbprofilefile)
        
def make_plots(sample, analysis_dir=None, htmldir='.', refband='r',
               band=('g', 'r', 'z'), clobber=False):
    """Make DESI targeting QA plots given a passed set of targets

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read fron the file with the passed name (supply the full directory path)
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    targdens : :class:`dictionary`, optional, set automatically by the code if not passed
        A dictionary of DESI target classes and the goal density for that class. Used to
        label the goal density on histogram plots
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    weight : :class:`boolean`, optional, defaults to True
        If this is set, weight pixels using the ``DESIMODEL`` HEALPix footprint file to
        ameliorate under dense pixels at the footprint edges

    Returns
    -------
    Nothing
        But a set of .png plots for target QA are written to qadir

    """
    from legacyhalos.io import get_objid

    objid, objdir = get_objid(sample, analysis_dir=analysis_dir)

    for objid1, objdir1, redshift in zip(np.atleast_1d(objid),
                                         np.atleast_1d(objdir),
                                         sample.z):

        htmlobjdir = os.path.join(htmldir, '{}'.format(objid1))
        
        if not os.path.isdir(htmlobjdir):
            os.makedirs(htmlobjdir, exist_ok=True)

        # Build the montage coadds.
        print('HACK!!!  do not remake the coadds!')
        #err = qa_montage_coadds(objid1, objdir1, htmlobjdir, clobber=clobber)

        #if err == 0:
            # Build the ellipse QAplots.
        #qa_ellipse_results(objid1, objdir1, htmlobjdir, redshift=redshift,
        #                   refband='r', band=band, clobber=clobber)
        
        qa_mge_results(objid1, objdir1, htmlobjdir, redshift=redshift,
                       refband='r', band=band, clobber=clobber)

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
        
def make_html(analysis_dir=None, htmldir=None, band=('g', 'r', 'z'), refband='r', 
              makeplots=True, clobber=False):
    """Create a directory containing a webpage structure in which to embed QA plots

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read fron the file with the passed name (supply the full directory path)
    makeplots : :class:`boolean`, optional, default=True
        If ``True``, then create the plots as well as the webpage
    htmldir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    Returns
    -------
    Nothing
        But the page `index.html` and associated pages and plots are written to ``htmldir``

    Notes
    -----
    If making plots, then the ``DESIMODEL`` environment variable must be set to find 
    the file of HEALPixels that overlap the DESI footprint

    """
    import legacyhalos.io

    if htmldir is None:
        htmldir = legacyhalos.io.html_dir()

    sample = legacyhalos.io.read_catalog(extname='LSPHOT', upenn=True,
                                         columns=('ra', 'dec', 'bx', 'by', 'brickname', 'objid'))
    rm = legacyhalos.io.read_catalog(extname='REDMAPPER', upenn=True,
                                     columns=('mem_match_id', 'z', 'r_lambda'))
    sample.add_columns_from(rm)

    print('Hack -- first 5 galaxies!')
    sample = sample[:5]
    #sample = sample[1050:1055]
    print('Read {} galaxies.'.format(len(sample)))

    objid, objdir = legacyhalos.io.get_objid(sample)

    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    htmlfile = os.path.join(htmldir, 'index.html')

    #Write the last-updated date to a webpage.
    js = _javastring()       

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<h1>LegacyHalos</h1>\n')

        html.write('<b><i>Jump to an object:</i></b>\n')
        html.write('<ul>\n')
        for objid1 in np.atleast_1d(objid):
            htmlfile = os.path.join('{}'.format(objid1), '{}.html'.format(objid1))
            html.write('<li><a href="{}">{}</a>\n'.format(htmlfile, objid1))
        html.write('</ul>\n')

        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Make a separate page for each object.
    for gal, objid1, objdir1 in zip(sample, np.atleast_1d(objid), np.atleast_1d(objdir)):
        htmlobjdir = os.path.join(htmldir, '{}'.format(objid1))
        if not os.path.exists(htmlobjdir):
            os.makedirs(htmlobjdir)
            
        htmlfile = os.path.join(htmlobjdir, '{}.html'.format(objid1))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<h1>Central Galaxy {}</h1>\n'.format(objid1))
            
            html.write('<h2>Coadds</h2>\n')
            html.write('<table cols=1 width="90%">\n')
            html.write('<tr>\n')
            html.write('<td width="100%" align=center><a href="{}-coadd-montage.png"><img src="{}-coadd-montage.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')
            
            html.write('<h2>MGE Fitting Results</h2>\n')

            html.write('<table cols=1 width="90%">\n')
            
            html.write('<tr>\n')
            html.write('<td width="100%" align="center"><a href="{}-mge-multiband.png"><img src="{}-mge-multiband.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('<tr>\n')
            html.write('<td align="left"><a href="{}-mge-sbprofile.png"><img src="{}-mge-sbprofile.png" height="auto" width="75%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')

            html.write('</table>\n')
            html.write('<br />\n')

            if False:
                #html.write('<h2>Ellipse Fitting Results</h2>\n')
                html.write('<table cols=1 width="90%">\n')
                html.write('<tr>\n')
                html.write('<td width="100%" align="center"><a href="{}-ellipse-multiband.png"><img src="{}-ellipse-multiband.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
                html.write('</tr>\n')

                html.write('<tr>\n')
                html.write('<td width="30%" align="center"><a href="{}-ellipse-isophotfit.png"><img src="{}-ellipse-isophotfit.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
                html.write('</tr>\n')

                html.write('<tr>\n')
                html.write('<td width="30%" align="center"><a href="{}-ellipse-sbprofile.png"><img src="{}-ellipse-sbprofile.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
                html.write('</tr>\n')

                html.write('</table>\n')
                html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')
            html.close()

    if makeplots:
        make_plots(sample, analysis_dir=analysis_dir, htmldir=htmldir, refband=refband,
                   band=band, clobber=clobber)
