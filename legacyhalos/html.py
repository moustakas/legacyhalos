from __future__ import (absolute_import, division)

import os, subprocess, pdb
import numpy as np

import seaborn as sns
sns.set(style='ticks', font_scale=1.4, palette='Set2')

PIXSCALE = 0.262

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
    from legacyhalos.io import read_multiband, read_ellipsefit
    from legacyhalos.qa import (display_multiband, display_ellipsefit,
                                display_ellipse_sbprofile)

    ellipsefit = read_ellipsefit(objid, objdir)

    if len(ellipsefit) > 0:

        # Toss out bad fits.
        indx = None
        #indx = (isophotfit[refband].stop_code < 4) * (isophotfit[refband].intens > 0)
        #indx = (isophotfit[refband].stop_code <= 4) * (isophotfit[refband].intens > 0)

        multibandfile = os.path.join(htmlobjdir, '{}-ellipse-multiband.png'.format(objid))
        if not os.path.isfile(multibandfile) or clobber:
            data = read_multiband(objid, objdir, band=band)
            display_multiband(data, ellipsefit=ellipsefit, band=band, refband=refband,
                              indx=indx, png=multibandfile)

        ellipsefitfile = os.path.join(htmlobjdir, '{}-ellipse-ellipsefit.png'.format(objid))
        if not os.path.isfile(ellipsefitfile) or clobber:
            display_ellipsefit(ellipsefit, band=band, refband=refband, redshift=redshift,
                               pixscale=pixscale, png=ellipsefitfile, xlog=True)
        
        sbprofilefile = os.path.join(htmlobjdir, '{}-ellipse-sbprofile.png'.format(objid))
        if not os.path.isfile(sbprofilefile) or clobber:
            display_ellipse_sbprofile(ellipsefit, band=band, refband=refband, redshift=redshift,
                                      pixscale=pixscale, png=sbprofilefile)
        
def qa_mge_results(objid, objdir, htmlobjdir, redshift=None, refband='r',
                   band=('g', 'r', 'z'), pixscale=0.262, clobber=False):
    """Generate QAplots from the MGE fitting.

    """
    from legacyhalos.io import read_mgefit, read_multiband
    from legacyhalos.qa import display_mge_sbprofile, display_multiband
    
    mgefit = read_mgefit(objid, objdir)

    if len(mgefit) > 0:

        ## Toss out bad fits.
        #indx = (mgefit[refband].stop_code <= 4) * (mgefit[refband].intens > 0)
        #
        multibandfile = os.path.join(htmlobjdir, '{}-mge-multiband.png'.format(objid))
        if not os.path.isfile(multibandfile) or clobber:
            data = read_multiband(objid, objdir, band=band)
            display_multiband(data, mgefit=mgefit, band=band, refband=refband,
                              png=multibandfile, contours=True)
        
        #isophotfile = os.path.join(htmlobjdir, '{}-mge-mgefit.png'.format(objid))
        #if not os.path.isfile(isophotfile) or clobber:
        #    # Just display the reference band.
        #    display_mgefit(mgefit, band=refband, redshift=redshift,
        #                       indx=indx, pixscale=pixscale, png=isophotfile)

        sbprofilefile = os.path.join(htmlobjdir, '{}-mge-sbprofile.png'.format(objid))
        if not os.path.isfile(sbprofilefile) or clobber:
            display_mge_sbprofile(mgefit, band=band, refband=refband, redshift=redshift,
                                  pixscale=pixscale, png=sbprofilefile)
        
def make_plots(sample, analysis_dir=None, htmldir='.', refband='r',
               band=('g', 'r', 'z'), clobber=False):
    """Make QA plots.

    """
    from legacyhalos.io import get_objid
    from legacyhalos.qa import sample_trends

    sample_trends(sample, htmldir, analysis_dir=analysis_dir, refband=refband)

    for gal in sample:
        objid, objdir = get_objid(gal, analysis_dir=analysis_dir)

        htmlobjdir = os.path.join(htmldir, '{}'.format(objid))
        if not os.path.isdir(htmlobjdir):
            os.makedirs(htmlobjdir, exist_ok=True)

        # Build the montage coadds.
        qa_montage_coadds(objid, objdir, htmlobjdir, clobber=clobber)

        # Build the MGE plots.
        qa_mge_results(objid, objdir, htmlobjdir, redshift=gal['z'],
                       refband='r', band=band, clobber=clobber)

        # Build the ellipse plots.
        qa_ellipse_results(objid, objdir, htmlobjdir, redshift=gal['z'],
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
              dr='dr5', first=None, last=None, makeplots=True, clobber=False):
    """Make the HTML pages.

    """
    import legacyhalos.io
    from legacyhalos.util import cutout_radius_100kpc

    if htmldir is None:
        htmldir = legacyhalos.io.html_dir()

    sample = legacyhalos.io.read_sample(first=first, last=last)
    objid, objdir = legacyhalos.io.get_objid(sample)

    # Write the last-updated date to a webpage.
    js = _javastring()       

    # Get the viewer link
    def _viewer_link(gal, dr):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * cutout_radius_100kpc(redshift=gal['z'], pixscale=PIXSCALE) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=decals-{}'.format(
            baseurl, gal['ra'], gal['dec'], zoom, dr)
        return viewer

    def _skyserver_link(gal):
        return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(gal['sdss_objid'])

    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    htmlfile = os.path.join(htmldir, homehtml)

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>LegacyHalos: Central Galaxies</h1>\n')
        html.write('<p>\n')
        html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
        html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
        html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>redMaPPer ID</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Richness</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')
        for ii, (gal, objid1) in enumerate(zip( sample, np.atleast_1d(objid) )):
            htmlfile = os.path.join('{}'.format(objid1), '{}.html'.format(objid1))

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii + 1))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile, objid1))
            html.write('<td>{:.7f}</td>\n'.format(gal['ra']))
            html.write('<td>{:.7f}</td>\n'.format(gal['dec']))
            html.write('<td>{:.5f}</td>\n'.format(gal['z']))
            html.write('<td>{:.4f}</td>\n'.format(gal['lambda_chisq']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal, dr)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Build the trends (trends.html) page--
    htmlfile = os.path.join(htmldir, trendshtml)
    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>LegacyHalos: Sample Trends</h1>\n')
        html.write('<p><a href="https://github.com/moustakas/legacyhalos">Code and documentation</a></p>\n')
        html.write('<a href="trends/sma_vs_ellipticity.png"><img src="trends/sma_vs_ellipticity.png" height="auto" width="50%"></a>')

        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Set up the object iterators
    iterobjid = iter(objid)
    if len(objid) > 1:
        next(iterobjid)
        nextobjid = next(iterobjid) # advance by one
    else:
        nextobjid = objid[0]
    prevobjid = objid[-1]

    # Make a separate HTML page for each object.
    for ii, (gal, objid1, objdir1) in enumerate( zip(sample, np.atleast_1d(objid),
                                                     np.atleast_1d(objdir)) ):
        htmlobjdir = os.path.join(htmldir, '{}'.format(objid1))
        if not os.path.exists(htmlobjdir):
            os.makedirs(htmlobjdir)

        nexthtmlobjdir = os.path.join('../', '{}'.format(nextobjid), '{}.html'.format(nextobjid))
        prevhtmlobjdir = os.path.join('../', '{}'.format(prevobjid), '{}.html'.format(prevobjid))

        htmlfile = os.path.join(htmlobjdir, '{}.html'.format(objid1))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>Central Galaxy {}</h1>\n'.format(objid1))

            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlobjdir, nextobjid))
            html.write('<br />\n')
            html.write('<a href="{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlobjdir, prevobjid))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>redMaPPer ID</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Richness</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('<th>SkyServer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii + 1))
            html.write('<td>{}</td>\n'.format(objid1))
            html.write('<td>{:.7f}</td>\n'.format(gal['ra']))
            html.write('<td>{:.7f}</td>\n'.format(gal['dec']))
            html.write('<td>{:.5f}</td>\n'.format(gal['z']))
            html.write('<td>{:.4f}</td>\n'.format(gal['lambda_chisq']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal, dr)))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Coadds</h2>\n')
            html.write('<p>Each coadd (left to right: data, model, residuals) is 200 kpc by 200 kpc.</p>\n')
            html.write('<table width="90%">\n')
            html.write('<tr><td><a href="{}-coadd-montage.png"><img src="{}-coadd-montage.png" height="auto" width="100%"></a></td></tr>\n'.format(objid1, objid1))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            
            html.write('<h2>Ellipse-Fitting</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            html.write('<td><a href="{}-ellipse-multiband.png"><img src="{}-ellipse-multiband.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            html.write('<td><a href="{}-ellipse-ellipsefit.png"><img src="{}-ellipse-ellipsefit.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('<td><a href="{}-ellipse-sbprofile.png"><img src="{}-ellipse-sbprofile.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('</table>\n')
            
            html.write('<br />\n')

            html.write('<h2>Multi-Gaussian Expansion Fitting</h2>\n')
            html.write('<p>The figures below are a work in progress.</p>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            html.write('<td><a href="{}-mge-multiband.png"><img src="{}-mge-multiband.png" height="auto" width="100%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('<tr>\n')
            html.write('<td><a href="{}-mge-sbprofile.png"><img src="{}-mge-sbprofile.png" height="auto" width="50%"></a></td>\n'.format(objid1, objid1))
            html.write('</tr>\n')
            html.write('</table>\n')
            
            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('</html></body>\n')
            html.close()

        # Update the iterator.
        prevobjid = objid1
        try:
            nextobjid = next(iterobjid)
        except:
            nextobjid = objid[0] # wrap around

    if makeplots:
        make_plots(sample, analysis_dir=analysis_dir, htmldir=htmldir, refband=refband,
                   band=band, clobber=clobber)
