"""
legacyhalos.hsc
===============

Miscellaneous code pertaining to the project comparing HSC and DECaLS surface
brightness profiles.

"""
import os, pdb
import numpy as np

from astrometry.util.fits import fits_table

from legacyhalos.html import make_plots, _javastring
from legacyhalos.misc import plot_style
sns = plot_style()

def make_html(sample, analysisdir, htmldir, band=('g', 'r', 'z'),
              refband='r', pixscale=0.262, nproc=1, dr='dr7', 
              makeplots=True, survey=None, clobber=False, verbose=True):
    """Make the HTML pages.

    """
    import legacyhalos.io
    from legacyhalos.misc import cutout_radius_150kpc

    if analysisdir is None:
        analysisdir = legacyhalos.io.analysis_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.html_dir()

    # Write the last-updated date to a webpage.
    js = _javastring()       

    # Get the viewer link
    def _viewer_link(onegal, dr):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * cutout_radius_150kpc(redshift=onegal['Z'], pixscale=0.262) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=decals-{}'.format(
            baseurl, onegal['RA'], onegal['DEC'], zoom, dr)
        return viewer

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

        html.write('<h1>Central Galaxies: HSC vs DECaLS</h1>\n')
        html.write('<p>\n')
        html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
        html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('</tr>\n')
        for ii, onegal in enumerate( np.atleast_1d(sample) ):
            galaxy = onegal['GALAXY'].decode('utf-8')
            htmlfile = os.path.join(galaxy.lower(), '{}.html'.format(galaxy.lower()))

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile, galaxy.upper()))
            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Make a separate HTML page for each object.
    for ii, onegal in enumerate( np.atleast_1d(sample) ):
        galaxy = onegal['GALAXY'].decode('utf-8').upper()
        gal = galaxy.lower()

        survey.output_dir = os.path.join(analysisdir, gal)
        survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(gal)))
        
        htmlgalaxydir = os.path.join(htmldir, '{}'.format(gal))
        if not os.path.exists(htmlgalaxydir):
            os.makedirs(htmlgalaxydir)

        htmlfile = os.path.join(htmlgalaxydir, '{}.html'.format(gal))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>Galaxy {}</h1>\n'.format(galaxy))

            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
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
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy))
            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Image mosaics</h2>\n')
            html.write('<p>Each mosaic (left to right: data, model of all but the central galaxy, residual image containing just the central galaxy) is 300 kpc by 300 kpc.</p>\n')
            html.write('<table width="90%">\n')
            pngfile = '{}-coadd-montage.png'.format(gal)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            
            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-multiband.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-sbprofile.png'.format(gal)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')
            
            html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
            html.write('<table width="90%">\n')

            # single-sersic
            html.write('<tr>\n')
            html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-single-nowavepower.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-single.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')

            # Sersic+exponential
            html.write('<tr>\n')
            html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-exponential-nowavepower.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-exponential.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')

            # double-sersic
            html.write('<tr>\n')
            html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-double-nowavepower.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-double.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<h2>CCD Diagnostics</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ccdpos.png'.format(gal)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            
            for iccd in range(len(survey.ccds)):
                html.write('<tr>\n')
                pngfile = '{}-2d-ccd{:02d}.png'.format(gal, iccd)
                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                    pngfile))
                html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')
            
            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    if makeplots:
        for ii, onegal in enumerate( np.atleast_1d(sample) ):
            galaxy = onegal['GALAXY'].decode('utf-8').upper()
            gal = galaxy.lower()

            survey.output_dir = os.path.join(analysisdir, gal)
            survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(gal)))
            
            make_plots([onegal], galaxylist=[gal], analysisdir=analysisdir,
                       htmldir=htmldir, clobber=clobber, verbose=verbose,
                       survey=survey, refband=refband, pixscale=pixscale,
                       band=band, nproc=nproc, ccdqa=True, trends=False)
