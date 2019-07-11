"""
legacyhalos.hsc
===============

Miscellaneous code pertaining to the project comparing HSC and DECaLS surface
brightness profiles.

"""
import os
import pdb
import numpy as np

import fitsio
import astropy.table

import legacyhalos.html

def hsc_dir():
    if 'HSC_DIR' not in os.environ:
        print('Required ${HSC_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('HSC_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def hsc_data_dir():
    if 'HSC_DATA_DIR' not in os.environ:
        print('Required ${HSC_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('HSC_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def hsc_html_dir():
    if 'HSC_HTML_DIR' not in os.environ:
        print('Required ${HSC_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('HSC_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from legacyhalos.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = hsc_data_dir()
    if htmldir is None:
        htmldir = hsc_html_dir()

    if 'ID_S16A' in cat.colnames and 'NAME' in cat.colnames:
        galid = cat['ID_S16A']
        name = cat['NAME']
    else:
        print('Missing ID_S16A and NAME in catalog!')
        raise ValuerError

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        if galid == -1:
            galaxy = [name]
        else:
            galaxy = ['{:017d}'.format(galid)]
        pixnum = [radec2pix(nside, cat['RA'], cat['DEC'])]
    else:
        ngal = len(cat)
        #galaxy = np.array(['{:017d}'.format(gid) for gid in galid])
        galaxy = []
        for gid, nm in zip(galid, name):
            if gid == -1:
                galaxy.append(nm.strip())
            else:
                galaxy.append('{:017d}'.format(gid))
        galaxy = np.array(galaxy)
        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(datadir, '{}'.format(nside), '{}'.format(pix), gal)
                          for pix, gal in zip(pixnum, galaxy)])
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, '{}'.format(nside), '{}'.format(pix), gal)
                                  for pix, gal in zip(pixnum, galaxy)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def read_parent(first=None, last=None, verbose=False):
    """Read/generate the parent HSC catalog.

    import fitsio
    from astropy.table import Table, Column, vstack
    s1 = Table(fitsio.read('low-z-shape-for-john.fits', upper=True))
    s2 = Table(fitsio.read('s16a_massive_z_0.5_logm_11.4_decals_full_fdfc_bsm_ell.fits', upper=True))

    s1out = s1['NAME', 'RA', 'DEC', 'Z', 'MEAN_E', 'MEAN_PA']
    s1out.rename_column('Z', 'Z_BEST')
    s1out.add_column(Column(name='ID_S16A', dtype=s2['ID_S16A'].dtype, length=len(s1out)), index=1)
    s1out['ID_S16A'] = -1
    s2out = s2['ID_S16A', 'RA', 'DEC', 'Z_BEST', 'MEAN_E', 'MEAN_PA']
    s2out.add_column(Column(name='NAME', dtype=s1['NAME'].dtype, length=len(s2out)), index=0)
    sout = vstack((s1out, s2out))
    sout.write('hsc-sample-s16a-lowz.fits', overwrite=True)
    
    """
    hdir = hsc_dir()
    # Hack for MUSE proposal
    #samplefile = os.path.join(hdir, 's18a_z0.07_0.12_rcmod_18.5_etg_muse_massive_0313.fits')
    
    # intermediate-z sample only
    #samplefile = os.path.join(hdir, 's16a_massive_z_0.5_logm_11.4_decals_full_fdfc_bsm_ell.fits')
    #samplefile = os.path.join(hdir, 's16a_massive_z_0.5_logm_11.4_dec_30_for_john.fits')

    # low-z sample only
    #samplefile = os.path.join(hdir, 'low-z-shape-for-john.fits')

    # combined sample (see comment block above)
    samplefile = os.path.join(hdir, 'hsc-sample-s16a-lowz.fits')
    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
        rows = np.arange(first, last)
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        rows = np.arange(first, last + 1)

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True))
    #if 'Z_BEST' in sample.colnames:
    #    sample.rename_column('Z_BEST', 'Z')
    #if 'Z_SPEC' in sample.colnames:
    #    sample.rename_column('Z_SPEC', 'Z')
    sample.add_column(astropy.table.Column(name='RELEASE', data=np.repeat(7000, len(sample)).astype(np.int32)))
    
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))
            
    return sample

def make_html(sample=None, datadir=None, htmldir=None, band=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              first=None, last=None, nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
    import subprocess
    import fitsio

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width
    from legacyhalos.misc import cutout_radius_kpc
    from legacyhalos.misc import HSC_RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    if datadir is None:
        datadir = hsc_data_dir()
    if htmldir is None:
        htmldir = hsc_html_dir()

    if sample is None:
        sample = read_parent(first=first, last=last)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    # Write the last-updated date to a webpage.
    js = legacyhalos.html._javastring()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 2 * cutout_radius_kpc(radius_kpc=radius_mosaic_kpc, redshift=gal[zcolumn],
                                      pixscale=pixscale) # [pixels]
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

    def _get_mags(cat, rad='10'):
        res = []
        for band in ('g', 'r', 'z'):
            iv = intflux['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
            ff = intflux['FLUX{}_{}'.format(rad, band.upper())][0]
            if iv > 0:
                ee = 1 / np.sqrt(iv)
                mag = 22.5-2.5*np.log10(ff)
                magerr = 2.5 * ee / ff / np.log(10)
                res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
            else:
                res.append('...')
        return res
            
    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    homehtmlfile = os.path.join(htmldir, homehtml)

    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>HSC Massive Galaxies</h1>\n')
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
        #html.write('<th>Richness</th>\n')
        #html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
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
            #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        with open(trendshtmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>HSC Massive Galaxies: Sample Trends</h1>\n')
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
        radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2

        ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
        #if 'psfdepth_g' not in ellipse.keys():
        #    pdb.set_trace()
        pipeline_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose,
                                                          filesuffix='pipeline')
        
        if not os.path.exists(htmlgalaxydir1):
            os.makedirs(htmlgalaxydir1)

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
            html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
            html.write('</style>\n')

            html.write('<h1>HSC Galaxy {}</h1>\n'.format(galaxy1))

            html.write('<a href="../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
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
            #html.write('<th>Richness</th>\n')
            #html.write('<th>Pcen</th>\n')
            html.write('<th>Viewer</th>\n')
            #html.write('<th>SkyServer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy1))
            html.write('<td>{:.7f}</td>\n'.format(gal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(gal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
            #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
            #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
            #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<h2>Image Mosaics</h2>\n')

            html.write('<table>\n')
            html.write('<tr><th colspan="3">Mosaic radius</th><th colspan="3">Point-source depth<br />(5-sigma, mag)</th><th colspan="3">Image quality<br />(FWHM, arcsec)</th></tr>\n')
            html.write('<tr><th>kpc</th><th>arcsec</th><th>grz pixels</th><th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th></tr>\n')
            html.write('<tr><td>{:.0f}</td><td>{:.3f}</td><td>{:.1f}</td>'.format(
                radius_mosaic_kpc, radius_mosaic_arcsec, radius_mosaic_pixels))
            if bool(ellipse):
                html.write('<td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td>'.format(
                    ellipse['psfdepth_g'], ellipse['psfdepth_min_g'], ellipse['psfdepth_max_g'],
                    ellipse['psfdepth_r'], ellipse['psfdepth_min_r'], ellipse['psfdepth_max_r'],
                    ellipse['psfdepth_z'], ellipse['psfdepth_min_z'], ellipse['psfdepth_max_z']))
                html.write('<td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td></tr>\n'.format(
                    ellipse['psfsize_g'], ellipse['psfsize_min_g'], ellipse['psfsize_max_g'],
                    ellipse['psfsize_r'], ellipse['psfsize_min_r'], ellipse['psfsize_max_r'],
                    ellipse['psfsize_z'], ellipse['psfsize_min_z'], ellipse['psfsize_max_z']))
            html.write('</table>\n')
            #html.write('<br />\n')

            html.write('<p>(Left) data, (middle) model of every object in the field except the central galaxy, (right) residual image containing just the central galaxy.</p>\n')
            #html.write('<br />\n')
            
            html.write('<table width="90%">\n')
            pngfile = '{}-grz-montage.png'.format(galaxy1)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')
            html.write('<p>Spatial distribution of CCDs.</p>\n')

            html.write('<table width="90%">\n')
            pngfile = '{}-ccdpos.png'.format(galaxy1)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            html.write('</table>\n')
            #html.write('<br />\n')

            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
            if bool(ellipse):
                html.write('<table>\n')
                html.write('<tr><th colspan="5">Mean Geometry</th>')

                html.write('<th colspan="4">Ellipse-fitted Geometry</th>')
                if ellipse['input_ellipse']:
                    html.write('<th colspan="2">Input Geometry</th></tr>\n')
                else:
                    html.write('</tr>\n')

                html.write('<tr><th>Integer center<br />(x,y, grz pixels)</th><th>Flux-weighted center<br />(x,y grz pixels)</th><th>Flux-weighted size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>')
                html.write('<th>Semi-major axis<br />(fitting range, arcsec)</th><th>Center<br />(x,y grz pixels)</th><th>PA<br />(deg)</th><th>e</th>')
                if ellipse['input_ellipse']:
                    html.write('<th>PA<br />(deg)</th><th>e</th></tr>\n')
                else:
                    html.write('</tr>\n')

                html.write('<tr><td>({:.0f}, {:.0f})</td><td>({:.3f}, {:.3f})</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(
                    ellipse['x0'], ellipse['y0'], ellipse['mge_xmed'], ellipse['mge_ymed'], ellipse['mge_majoraxis']*pixscale,
                    ellipse['mge_pa'], ellipse['mge_eps']))

                if 'init_smamin' in ellipse.keys():
                    html.write('<td>{:.3f}-{:.3f}</td><td>({:.3f}, {:.3f})<br />+/-({:.3f}, {:.3f})</td><td>{:.1f}+/-{:.1f}</td><td>{:.3f}+/-{:.3f}</td>'.format(
                        ellipse['init_smamin']*pixscale, ellipse['init_smamax']*pixscale, ellipse['x0_median'],
                        ellipse['y0_median'], ellipse['x0_err'], ellipse['y0_err'], ellipse['pa'], ellipse['pa_err'],
                        ellipse['eps'], ellipse['eps_err']))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td><td>...</td>')
                if ellipse['input_ellipse']:
                    html.write('<td>{:.1f}</td><td>{:.3f}</td></tr>\n'.format(
                        np.degrees(ellipse['geometry'].pa)+90, ellipse['geometry'].eps))
                else:
                    html.write('</tr>\n')
                html.write('</table>\n')
                html.write('<br />\n')

                html.write('<table>\n')
                html.write('<tr><th>Fitting range<br />(arcsec)</th><th>Integration<br />mode</th><th>Clipping<br />iterations</th><th>Clipping<br />sigma</th></tr>')
                html.write('<tr><td>{:.3f}-{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                    ellipse[refband]['sma'].min()*pixscale, ellipse[refband]['sma'].max()*pixscale,
                    ellipse['integrmode'], ellipse['nclip'], ellipse['sclip']))
                html.write('</table>\n')
                html.write('<br />\n')
            else:
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            html.write('<td><a href="{}-ellipse-multiband.png"><img src="{}-ellipse-multiband.png" alt="Missing file {}-ellipse-multiband.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            #html.write('<td><a href="{}-ellipse-ellipsefit.png"><img src="{}-ellipse-ellipsefit.png" alt="Missing file {}-ellipse-ellipsefit.png" height="auto" width="100%"></a></td>\n'.format(galaxy1, galaxy1, galaxy1))
            pngfile = '{}-ellipse-sbprofile.png'.format(galaxy1)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-ellipse-cog.png'.format(galaxy1)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            #html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<h2>Observed & rest-frame photometry</h2>\n')

            html.write('<h4>Integrated photometry</h4>\n')
            html.write('<table>\n')
            html.write('<tr>')
            html.write('<th colspan="3">Curve of growth<br />(custom sky, mag)</th><th colspan="3">Curve of growth<br />(pipeline sky, mag)</th>')
            html.write('</tr>')

            html.write('<tr>')
            html.write('<th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th>')
            html.write('</tr>')

            html.write('<tr>')
            if bool(ellipse):
                g, r, z = (ellipse['cog_params_g']['mtot'], ellipse['cog_params_r']['mtot'],
                           ellipse['cog_params_z']['mtot'])
                html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>')

            if bool(pipeline_ellipse):
                g, r, z = (pipeline_ellipse['cog_params_g']['mtot'], pipeline_ellipse['cog_params_r']['mtot'],
                           pipeline_ellipse['cog_params_z']['mtot'])
                html.write('<td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>')
                
            html.write('</tr>')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<h4>Aperture photometry</h4>\n')
            html.write('<table>\n')
            html.write('<tr>')
            html.write('<th colspan="3"><10 kpc (mag)</th>')
            html.write('<th colspan="3"><30 kpc (mag)</th>')
            html.write('<th colspan="3"><100 kpc (mag)</th>')
            html.write('</tr>')

            html.write('<tr>')
            html.write('<th>g</th><th>r</th><th>z</th>')
            html.write('<th>g</th><th>r</th><th>z</th>')
            html.write('<th>g</th><th>r</th><th>z</th>')
            html.write('</tr>')

            if intflux:
                html.write('<tr>')
                g, r, z = _get_mags(intflux[ii], rad='10')
                html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                g, r, z = _get_mags(intflux[ii], rad='30')
                html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                g, r, z = _get_mags(intflux[ii], rad='100')
                html.write('<td>{}</td><td>{}</td><td>{}</td>'.format(g, r, z))
                html.write('</tr>')

            html.write('</table>\n')
            html.write('<br />\n')

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

            html.write('<a href="../../../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Next Central Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
            html.write('<br />\n')
            html.write('<a href="../../../{}">Previous Central Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    # Make the plots.
    if makeplots:
        err = legacyhalos.html.make_plots(sample, datadir=datadir, htmldir=htmldir, refband=refband,
                                          band=band, pixscale=pixscale, survey=survey, clobber=clobber,
                                          verbose=verbose, nproc=nproc, ccdqa=ccdqa, maketrends=maketrends,
                                          zcolumn=zcolumn, hsc=True)

    cmd = 'chgrp -R cosmo {}'.format(htmldir)
    print(cmd)
    err1 = subprocess.call(cmd.split())

    cmd = 'find {} -type d -exec chmod 775 {{}} +'.format(htmldir)
    print(cmd)
    err2 = subprocess.call(cmd.split())

    cmd = 'find {} -type f -exec chmod 664 {{}} +'.format(htmldir)
    print(cmd)
    err3 = subprocess.call(cmd.split())

    if err1 != 0 or err2 != 0 or err3 != 0:
        print('Something went wrong updating permissions; please check the logfile.')
        return 0
    
    return 1
