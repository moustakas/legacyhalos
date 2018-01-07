from __future__ import (absolute_import, division)

import os, subprocess

def _qa_montage_coadds(objid, objdir, htmlobjdir, clobber=False):
    """Montage the coadds into a nice QAplot."""

    montagefile = os.path.join(htmlobjdir, '{}-coadd-montage.png'.format(objid))

    if not os.path.isfile(montagefile) or clobber:
        cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 '
        cmd = cmd+' '.join([os.path.join(objdir, '{}-{}.jpg'.format(objid, suffix)) for
                            suffix in ('image', 'model', 'resid')])
        cmd = cmd+' {}'.format(montagefile)
        print('Writing {}'.format(montagefile))

        err = subprocess.call(cmd.split())

def _qa_ellipse_results(objid, objdir, htmlobjdir, clobber=False):
    """Generate QAplots from the ellipse-fitting."""

    montagefile = os.path.join(htmlobjdir, '{}-coadd-montage.png'.format(objid))

    plt.figure(figsize=(8, 4))

    plt.scatter(isolist.sma**0.25, 22.5-2.5*np.log10(isolist.intens))
    plt.xlabel('sma**1/4')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()
    #plt.show()

    fig, ax1 = plt.subplots()
    ax1.imshow(resid, origin='lower')
    smas = np.linspace(1, 100, 10)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        print(x, y)
        ax1.plot(x, y, color='white')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    plt.subplot(2, 2, 1)
    plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                 fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')

    plt.subplot(2, 2, 2)
    plt.errorbar(isolist.sma, isolist.pa/np.pi*180.,
                 yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')

    plt.subplot(2, 2, 3)
    plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')

    plt.subplot(2, 2, 4)
    plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')

    plt.show()

        
def make_plots(sample, analysis_dir=None, htmldir='.', clobber=False):
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

    Notes
    -----
    The ``DESIMODEL`` environment variable must be set to find the file of HEALPixels 
    that overlap the DESI footprint

    """
    from legacyhalos.io import get_objid

    objid, objdir = get_objid(sample, analysis_dir=analysis_dir)

    for objid1, objdir1 in zip(objid, objdir):

        htmlobjdir = os.path.join(htmldir, '{}'.format(objid1))
        
        if not os.path.isdir(htmlobjdir):
            os.makedirs(htmlobjdir, exist_ok=True)

        # Build the montage coadds.
        _qa_montage_coadds(objid1, objdir1, htmlobjdir, clobber=clobber)

        # Build the ellipse QAplots.
        #_qa_ellipse_results(objid1, htmlobjdir, clobber=clobber)

        # Build the surface brightness profile fitting results.
        #_qa_sbprofile_results(objid1, htmlobjdir, clobber=clobber)
        
        
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
    if (date == 1 || date == 21 || date == 31)
    document.write(" " + lmonth + " " + date + "st, " + fyear)
    else if (date == 2 || date == 22)
    document.write(" " + lmonth + " " + date + "nd, " + fyear)
    else if (date == 3 || date == 23)
    document.write(" " + lmonth + " " + date + "rd, " + fyear)
    else
    document.write(" " + lmonth + " " + date + "th, " + fyear)
    </SCRIPT>
    """)

    return js
        
def make_html(analysis_dir=None, htmldir=None, makeplots=True, clobber=False):
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

    print('Hack -- 5 galaxies!')
    sample = sample[1050:1055]
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
        for objid1 in objid:
            html.write('<li><A HREF="{:}.html">{:}</A>\n'.format(objid1, objid1))
        html.write('</ul>\n')

        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Make a separate page for each object.
    for gal, objid1, objdir1 in zip(sample, objid, objdir):
        htmlfile = os.path.join(htmldir, '{}.html'.format(objid1))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<h1>Central Galaxy {}</h1>\n'.format(objid1))

            html.write('<h2>Coadds</h2>\n')
            html.write('<table COLS=2 WIDTH="100%">\n')
            html.write('<tr>\n')
            html.write('<td WIDTH="50%" align=left><A HREF="{}/{}-coadd-montage.png"><img SRC="{}/{}-coadd-montage.png" height=300 width=900></A></left></td>\n'.format(objid1, objid1, objid1, objid1))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')
            html.close()

    if makeplots:
        make_plots(sample, analysis_dir=analysis_dir, htmldir=htmldir, clobber=clobber)
