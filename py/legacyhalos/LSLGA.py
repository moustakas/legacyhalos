"""
legacyhalos.LSLGA
=================

Code to deal with the LSLGA sample and project.

"""
import os, time, pdb
import numpy as np
import astropy

import legacyhalos.io

RADIUS_CLUSTER_KPC = 100.0     # default cluster radius
ZCOLUMN = 'Z'

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the pipeline coadds and return (using --early-coadds in runbrick.py.')
    #parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')
    #parser.add_argument('--LSLGA', action='store_true', help='Special code for large galaxies.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')

    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--nomakeplots', action='store_true', help='Do not remake the QA plots for the HTML pages.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                

    parser.add_argument('--build-LSLGA', action='store_true', help='Build the LSLGA reference catalog.')
    args = parser.parse_args()

    return args

def missing_files_groups(args, sample, size, htmldir=None):
    """Simple task-specific wrapper on missing_files.

    """
    if args.coadds:
        suffix = 'coadds'
    #elif args.custom_coadds:
    #    suffix = 'custom-coadds'
    #elif args.LSLGA:
    #    suffix = 'pipeline-coadds'
    elif args.htmlplots:
        suffix = 'html'
    else:
        suffix = ''        

    if suffix != '':
        groups = missing_files(sample, filetype=suffix, size=size,
                               clobber=args.clobber, htmldir=htmldir)
    else:
        groups = []        

    return suffix, groups

def missing_files(sample, filetype='coadds', size=1, htmldir=None,
                  clobber=False):
    """Find missing data of a given filetype."""    

    if filetype == 'coadds':
        filesuffix = '-pipeline-resid-grz.jpg'
    #elif filetype == 'custom-coadds':
    #    filesuffix = '-custom-resid-grz.jpg'
    #elif filetype == 'LSLGA':
    #    filesuffix = '-custom-resid-grz.jpg'
    elif filetype == 'html':
        filesuffix = '-ccdpos.png'
        #filesuffix = '-sersic-exponential-nowavepower.png'
    else:
        print('Unrecognized file type!')
        raise ValueError

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)
    todo = np.ones(ngal, dtype=bool)

    if filetype == 'html':
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    else:
        galaxy, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir)

    for ii, (gal, gdir) in enumerate( zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)) ):
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if os.path.exists(checkfile) and clobber is False:
            todo[ii] = False

    if np.sum(todo) == 0:
        return list()
    else:
        indices = indices[todo]
        
    return np.array_split(indices, size)

def LSLGA_dir():
    if 'LSLGA_DIR' not in os.environ:
        print('Required ${LSLGA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def LSLGA_data_dir():
    if 'LSLGA_DATA_DIR' not in os.environ:
        print('Required ${LSLGA_DATA_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def LSLGA_html_dir():
    if 'LSLGA_HTML_DIR' not in os.environ:
        print('Required ${LSLGA_HTML_DIR environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('LSLGA_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False,
                         candidates=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = LSLGA_data_dir()
    if htmldir is None:
        htmldir = LSLGA_html_dir()

    # Handle groups.
    if 'GROUP_NAME' in cat.colnames:
        galcolumn = 'GROUP_NAME'
        racolumn = 'GROUP_RA'
    else:
        galcolumn = 'GALAXY'
        racolumn = 'RA'

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat[galcolumn]]
        ra = [cat[racolumn]]
    else:
        ngal = len(cat)
        galaxy = cat[galcolumn]
        ra = cat[racolumn]

    def raslice(ra):
        return '{:06d}'.format(int(ra*1000))[:3]

    galaxydir = np.array([os.path.join(datadir, raslice(ra), gal) for gal, ra in zip(galaxy, ra)])
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, raslice(ra), gal) for gal, ra in zip(galaxy, ra)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def LSLGA_version():
    version = 'v5.0'
    return version

def read_sample(first=None, last=None, galaxylist=None, verbose=False, preselect_sample=True):
    """Read/generate the parent LSLGA catalog.

    """
    import fitsio
    version = LSLGA_version()
    samplefile = os.path.join(LSLGA_dir(), 'sample', version, 'LSLGA-{}.fits'.format(version))

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # Choose the parent sample here.
    if preselect_sample:
        from legacyhalos.brick import brickname as get_brickname

        d25min = 1.0 # [arcmin]
        sample = fitsio.read(samplefile, columns=['GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_PRIMARY', 'IN_DESI'])
        bigcut = np.where((sample['GROUP_DIAMETER'] > 1) * (sample['GROUP_PRIMARY'] == 1) * (sample['IN_DESI']))[0]

        brickname = get_brickname(sample['GROUP_RA'][bigcut], sample['GROUP_DEC'][bigcut])
        nbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9e-north-bricklist.txt'), dtype='str')
        sbricklist = np.loadtxt(os.path.join(LSLGA_dir(), 'sample', 'dr9e-south-bricklist.txt'), dtype='str')
        bricklist = np.union1d(nbricklist, sbricklist)
        #rows = np.where([brick in bricklist for brick in brickname])[0]
        brickcut = np.where(np.isin(brickname, bricklist))[0]

        rows = np.arange(len(sample))
        rows = rows[bigcut][brickcut]
        nrows = len(rows)
        print('Selecting {} galaxies in the dr9e footprint.'.format(nrows))
    else:
        rows = None

    if first is None:
        first = 0
    if last is None:
        last = nrows
        if rows is None:
            rows = np.arange(first, last)
        else:
            rows = rows[np.arange(first, last)]
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        if rows is None:
            rows = np.arange(first, last+1)
        else:
            rows = rows[np.arange(first, last+1)]

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))

    # Add an (internal) index number:
    sample['INDEX'] = rows
    
    #print('Hack the sample!')
    #tt = astropy.table.Table.read('/global/projecta/projectdirs/desi/users/ioannis/dr9d-lslga/dr9d-lslga-south.fits')
    #tt = tt[tt['D25'] > 1]
    #galaxylist = tt['GALAXY'].data

    # strip whitespace
    sample['GALAXY'] = [gg.strip() for gg in sample['GALAXY']]
    if 'GROUP_NAME' in sample.colnames:
        galcolumn = 'GROUP_NAME'
        sample['GROUP_NAME'] = [gg.strip() for gg in sample['GROUP_NAME']]
    
    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        sample = sample[np.isin(sample[galcolumn], galaxylist)]

    return sample

def build_model_LSLGA(sample, clobber=False):
    """Gather all the fitting results and build the final model-based LSLGA catalog. 

    """
    from astrometry.libkd.spherematch import match_radec

    # This is a little fragile.
    ver = legacyhalos.LSLGA.LSLGA_version()
    outdir = os.path.dirname(os.getenv('LARGEGALAXIES_CAT'))
    outfile = os.path.join(outdir, 'LSLGA-model-{}.fits'.format(ver))
    if not os.path.isfile(outfile) or clobber:
        import fitsio
        import astropy.table

        def get_d25_ba_pa(r50, e1, e2):
            d25 = 3.0 * r50 # hack!
            ee = np.hypot(e1, e2)
            ba = (1 - ee) / (1 + ee)
            #pa = -np.rad2deg(np.arctan2(e2, e1) / 2)
            pa = 180 - (-np.rad2deg(np.arctan2(e2, e1) / 2))
            return d25, ba, pa

        out = []
        for onegal in sample:
            onegal = astropy.table.Table(onegal)
            galaxy, galaxydir = legacyhalos.LSLGA.get_galaxy_galaxydir(onegal)
            catfile = os.path.join(galaxydir, '{}-pipeline-tractor.fits'.format(galaxy))
            if not os.path.isfile(catfile):
                print('Skipping missing file {}'.format(catfile))
                continue
            cat = fitsio.read(catfile)
            # Need to be smarter here; maybe include all galaxies larger than 10ish arcsec??              
            #this = cat['ref_cat'] == 'L4'
            pdb.set_trace()
            this, m2, d12 = match_radec(cat['ra'], cat['dec'], onegal['RA'], onegal['DEC'],
                                        1.0/3600.0, nearest=True)
            if len(this) == 0:
                print('Fix me!')
                continue
            out1 = astropy.table.Table(cat[this])

            # Convert the Tractor results to LSLGA format.
            d25, ba, pa = get_d25_ba_pa(out1['shape_r'], out1['shape_e1'], out1['shape_e2'])
            out1['d25_model'] = d25.astype('f4')
            out1['pa_model'] = pa.astype('f4')
            out1['ba_model'] = ba.astype('f4')

            # Merge the catalogs together.
            onegal.rename_column('RA', 'RA_LSLGA')
            onegal.rename_column('DEC', 'DEC_LSLGA')
            onegal.rename_column('TYPE', 'MORPHTYPE')
            out.append(astropy.table.hstack((out1, onegal)))

        out = astropy.table.vstack(out)
        [out.rename_column(col, col.upper()) for col in out.colnames]

        print('Writing {} galaxies to {}'.format(len(out), outfile))
        pdb.set_trace()
        out.write(outfile, overwrite=True)
    else:
        print('Use --clobber to overwrite existing catalog {}'.format(outfile))

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=True,
              clobber=False, verbose=True, maketrends=False, ccdqa=False):
    """Make the HTML pages.

    """
    import subprocess
    import fitsio

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width
    #from legacyhalos.misc import cutout_radius_kpc
    #from legacyhalos.misc import HSC_RADIUS_CLUSTER_KPC as radius_mosaic_kpc

    datadir = LSLGA_data_dir()
    htmldir = LSLGA_html_dir()

    if sample is None:
        sample = read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    # group by RA slices
    raslices = np.array([str(ra)[:3] for ra in sample[racolumn]])
    rasorted = np.argsort(raslices)

    # Write the last-updated date to a webpage.
    js = legacyhalos.html._javastring()       

    # Get the viewer link
    def _viewer_link(gal):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = gal['D25'] * 60 * 2 / pixscale
        #width = 2 * cutout_radius_kpc(radius_kpc=radius_mosaic_kpc, redshift=gal[zcolumn],
        #                              pixscale=pixscale) # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=dr8'.format(
            baseurl, gal[racolumn], gal[deccolumn], zoom)
        
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
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>Legacy Survey Large Galaxy Atlas (LSLGA)</h1>\n')
        if maketrends:
            html.write('<p>\n')
            html.write('<a href="{}">Sample Trends</a><br />\n'.format(trendshtml))
            html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
            html.write('</p>\n')

        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[inslice], html=True)

            html.write('<h3>RA Slice {}</h3>\n'.format(raslice))

            html.write('<table>\n')

            #html.write('<tr>\n')
            #html.write('<th></th>\n')
            #html.write('<th></th>\n')
            #html.write('<th>RA</th>\n')
            #html.write('<th>Dec</th>\n')
            #html.write('<th>D(25)</th>\n')
            #html.write('<th></th>\n')
            #html.write('</tr>\n')
            #
            #html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            #html.write('<th>Galaxy</th>\n')
            #html.write('<th>(deg)</th>\n')
            #html.write('<th>(deg)</th>\n')
            #html.write('<th>(arcsec)</th>\n')
            #html.write('<th>Viewer</th>\n')

            html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            html.write('<th>Index</th>\n')
            html.write('<th>LSLGA ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')

            html.write('</tr>\n')
            for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['LSLGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.2f}</td>\n'.format(gal[diamcolumn]))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
            html.write('</table>\n')
            #count += 1

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

        # Make a separate HTML page for each object.
        galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)
        
        nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
        prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
        nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
        prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)
        #pdb.set_trace()

        for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate( zip(
            sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir) ) ):

            #radius_mosaic_arcsec = legacyhalos.misc.cutout_radius_kpc(
            #    redshift=gal[zcolumn], radius_kpc=radius_mosaic_kpc) # [arcsec]
            #radius_mosaic_pixels = _mosaic_width(radius_mosaic_arcsec, pixscale) / 2
            #
            #ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose)
            ##if 'psfdepth_g' not in ellipse.keys():
            ##    pdb.set_trace()
            #pipeline_ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, verbose=verbose,
            #                                                  filesuffix='pipeline')

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

                html.write('<h1>Galaxy {}</h1>\n'.format(galaxy1))

                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
                html.write('<br />\n')
                html.write('<br />\n')

                # Table of properties
                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th>Index</th>\n')
                html.write('<th>LSLGA ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
                #html.write('<th>Richness</th>\n')
                #html.write('<th>Pcen</th>\n')
                html.write('<th>Viewer</th>\n')
                #html.write('<th>SkyServer</th>\n')
                html.write('</tr>\n')

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(ii))
                #print(gal['INDEX'], gal['LSLGA_ID'], gal['GALAXY'])
                html.write('<td>{:g}</td>\n'.format(gal['INDEX']))
                html.write('<td>{:g}</td>\n'.format(gal['LSLGA_ID']))
                html.write('<td>{}</td>\n'.format(galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(gal[racolumn]))
                html.write('<td>{:.7f}</td>\n'.format(gal[deccolumn]))
                html.write('<td>{:.2f}</td>\n'.format(gal[diamcolumn]))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(gal)))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
                html.write('</table>\n')

                html.write('<h2>Image Mosaics</h2>\n')

                if False:
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

                html.write('<p>(Left) data, (middle) model, and (right) residual image mosaic.</p>\n')
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

                if False:
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

                html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
                html.write('<br />\n')
                html.write('<a href="../../{}">Next Galaxy ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
                html.write('<br />\n')
                html.write('<a href="../../{}">Previous Galaxy ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
                html.write('<br />\n')

                html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
                html.write('<br />\n')
                html.write('</html></body>\n')
                html.close()

    # Make the plots.
    if makeplots:
        err = legacyhalos.html.make_plots(sample, datadir=datadir, htmldir=htmldir, refband=refband,
                                          bands=bands, pixscale=pixscale, survey=survey, clobber=clobber,
                                          verbose=verbose, nproc=nproc, ccdqa=ccdqa, maketrends=maketrends,
                                          zcolumn=zcolumn)

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

