"""
legacyhalos.manga
=================

Code to deal with the MaNGA-NSA sample and project.

"""
import os, shutil, pdb
import numpy as np
import astropy

import legacyhalos.io

ZCOLUMN = 'Z'
RACOLUMN = 'OBJRA'
DECCOLUMN = 'OBJDEC'
GALAXYCOLUMN = 'PLATEIFU'

RADIUSFACTOR = 1.5 # 10
MANGA_RADIUS = 36.75 # / 2 # [arcsec]

ELLIPSEBITS = dict(
    largeshift = 2**0, # >10-pixel shift in the flux-weighted center
    )

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--pipeline-coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')
    #parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')

    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
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

    parser.add_argument('--build-SGA', action='store_true', help='Build the SGA reference catalog.')
    args = parser.parse_args()

    return args

def missing_files(args, sample, size=1, clobber_overwrite=None):
    from astrometry.util.multiproc import multiproc
    from legacyhalos.io import _missing_files_one

    dependson = None
    galaxy, galaxydir = get_galaxy_galaxydir(sample)        
    if args.coadds:
        suffix = 'coadds'
        filesuffix = '-custom-coadds.isdone'
    elif args.pipeline_coadds:
        suffix = 'pipeline-coadds'
        if args.just_coadds:
            filesuffix = '-pipeline-image-grz.jpg'
        else:
            filesuffix = '-pipeline-coadds.isdone'
    elif args.ellipse:
        suffix = 'ellipse'
        filesuffix = '-custom-ellipse.isdone'
        dependson = '-custom-coadds.isdone'
    elif args.build_SGA:
        suffix = 'build-SGA'
        filesuffix = '-custom-ellipse.isdone'
    elif args.htmlplots:
        suffix = 'html'
        if args.just_coadds:
            filesuffix = '-custom-grz-montage.png'
        else:
            filesuffix = '-ccdpos.png'
            #filesuffix = '-custom-maskbits.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    elif args.htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-custom-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=args.htmldir, html=True)
    else:
        print('Nothing to do.')
        return

    # Make clobber=False for build_SGA and htmlindex because we're not making
    # the files here, we're just looking for them. The argument args.clobber
    # gets used downstream.
    if args.htmlindex or args.build_SGA:
        clobber = False
    else:
        clobber = args.clobber

    if clobber_overwrite is not None:
        clobber = clobber_overwrite

    if type(sample) is astropy.table.row.Row:
        ngal = 1
    else:
        ngal = len(sample)
    indices = np.arange(ngal)

    mp = multiproc(nthreads=args.nproc)
    missargs = []
    for gal, gdir in zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir)):
        #missargs.append([gal, gdir, filesuffix, dependson, clobber])
        checkfile = os.path.join(gdir, '{}{}'.format(gal, filesuffix))
        if dependson:
            missargs.append([checkfile, os.path.join(gdir, '{}{}'.format(gal, dependson)), clobber])
        else:
            missargs.append([checkfile, None, clobber])
        
    todo = np.array(mp.map(_missing_files_one, missargs))

    itodo = np.where(todo == 'todo')[0]
    idone = np.where(todo == 'done')[0]
    ifail = np.where(todo == 'fail')[0]

    if len(ifail) > 0:
        fail_indices = [indices[ifail]]
    else:
        fail_indices = [np.array([])]

    if len(idone) > 0:
        done_indices = [indices[idone]]
    else:
        done_indices = [np.array([])]

    if len(itodo) > 0:
        _todo_indices = indices[itodo]
        todo_indices = np.array_split(_todo_indices, size) # unweighted

        ## Assign the sample to ranks to make the D25 distribution per rank ~flat.
        #
        ## https://stackoverflow.com/questions/33555496/split-array-into-equally-weighted-chunks-based-on-order
        #weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        #cumuweight = weight.cumsum() / weight.sum()
        #idx = np.searchsorted(cumuweight, np.linspace(0, 1, size, endpoint=False)[1:])
        #if len(idx) < size: # can happen in corner cases
        #    todo_indices = np.array_split(_todo_indices, size) # unweighted
        #else:
        #    todo_indices = np.array_split(_todo_indices, idx) # weighted
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices
    
def get_raslice(ra):
    return '{:06d}'.format(int(ra*1000))[:3]

def get_plate(plate):
    return '{:05d}'.format(plate)

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = legacyhalos.io.legacyhalos_data_dir()
    if htmldir is None:
        htmldir = legacyhalos.io.legacyhalos_html_dir()

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat[GALAXYCOLUMN]]
        plate = [cat['PLATE']]
    else:
        ngal = len(cat)
        galaxy = cat[GALAXYCOLUMN]
        plate = cat['PLATE']

    galaxydir = np.array([os.path.join(datadir, get_plate(plt), gal) for gal, plt in zip(galaxy, plate)])
    if html:
        htmlgalaxydir = np.array([os.path.join(htmldir, get_plate(plt), gal) for gal, plt in zip(galaxy, plate)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None):
    """Read/generate the parent SGA catalog.

    """
    import fitsio
    from legacyhalos.desiutil import brickname as get_brickname
            
    samplefile = os.path.join(legacyhalos.io.legacyhalos_dir(), 'drpall-v2_4_3.fits')

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(samplefile)
    nrows = info[ext].get_nrows()

    # Select unique galaxies--
    # https://www.sdss.org/dr16/manga/manga-tutorials/drpall/#py-uniq-gals

    tbdata = fitsio.read(samplefile, lower=True, columns='mangaid')
    pdb.set_trace()
    
    # Find all datacubes 
    cube_bools = (tbdata['mngtarg1'] != 0) | (tbdata['mngtarg3'] != 0)
    cubes = tbdata[cube_bools]

    # Find galaxies excluding those from the Coma, IC342, M31 ancillary programs (bits 19,20,21)
    targ3 = tbdata['mngtarg3']
    galaxies = tbdata[cube_bools & ((targ3 & 1<<19) == 0) & ((targ3 & 1<<20) == 0) & ((targ3 & 1<<21) == 0)]
    print('Number of galaxies', len(galaxies))

    # Get unique galaxies
    uniq_vals, uniq_idx=np.unique(galaxies['mangaid'], return_index=True)
    uniq_galaxies = galaxies[uniq_idx]
    print('Unique galaxies', len(uniq_galaxies))

    # Immediately toss out Apogee--
    rows = np.where(nsaid != -9999)[0]
    nrows = len(rows)
    #rows = None

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

    sample = astropy.table.Table(info[ext].read(rows=rows, upper=True, columns=columns))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(sample), samplefile))

    # Add an (internal) index number:
    sample.add_column(astropy.table.Column(name='INDEX', data=rows), index=0)
    
    ## strip whitespace
    #if 'GALAXY' in sample.colnames:
    #    sample['GALAXY'] = [gg.strip() for gg in sample['GALAXY']]
    #if 'GROUP_NAME' in sample.colnames:
    #    galcolumn = 'GROUP_NAME'
    #    sample['GROUP_NAME'] = [gg.strip() for gg in sample['GROUP_NAME']]

    if galaxylist is not None:
        if verbose:
            print('Selecting specific galaxies.')
        these = np.isin(sample[GALAXYCOLUMN], galaxylist)
        if np.count_nonzero(these) == 0:
            print('No matching galaxies!')
            return astropy.table.Table()
        else:
            sample = sample[these]

    return sample

def _get_diameter(ellipse):
    """Wrapper to get the mean D(26) diameter.

    ellipse - legacyhalos.ellipse dictionary

    diam in arcmin

    """
    if ellipse['radius_sb26'] > 0:
        diam, diamref = 2 * ellipse['radius_sb26'] / 60, 'SB26' # [arcmin]
    elif ellipse['radius_sb25'] > 0:
        diam, diamref = 1.2 * 2 * ellipse['radius_sb25'] / 60, 'SB25' # [arcmin]
    #elif ellipse['radius_sb24'] > 0:
    #    diam, diamref = 1.5 * ellipse['radius_sb24'] * 2 / 60, 'SB24' # [arcmin]
    else:
        diam, diamref = 1.2 * ellipse['d25_leda'], 'LEDA' # [arcmin]
        #diam, diamref = 2 * ellipse['majoraxis'] * ellipse['refpixscale'] / 60, 'WGHT' # [arcmin]

    if diam <= 0:
        raise ValueError('Doom has befallen you.')

    return diam, diamref

def _get_mags(cat, rad='10', kpc=False, pipeline=False, cog=False, R24=False, R25=False, R26=False):
    res = []
    for band in ('g', 'r', 'z'):
        mag = None
        if kpc:
            iv = cat['FLUX{}_IVAR_{}'.format(rad, band.upper())][0]
            ff = cat['FLUX{}_{}'.format(rad, band.upper())][0]
        elif pipeline:
            iv = cat['flux_ivar_{}'.format(band)]
            ff = cat['flux_{}'.format(band)]
        elif R24:
            mag = cat['{}_mag_sb24'.format(band)]
        elif R25:
            mag = cat['{}_mag_sb25'.format(band)]
        elif R26:
            mag = cat['{}_mag_sb26'.format(band)]
        elif cog:
            mag = cat['{}_cog_params_mtot'.format(band)]
        else:
            print('Thar be rocks ahead!')
        if mag:
            res.append('{:.3f}'.format(mag))
        else:
            if ff > 0:
                mag = 22.5-2.5*np.log10(ff)
                if iv > 0:
                    ee = 1 / np.sqrt(iv)
                    magerr = 2.5 * ee / ff / np.log(10)
                res.append('{:.3f}'.format(mag))
                #res.append('{:.3f}+/-{:.3f}'.format(mag, magerr))
            else:
                res.append('...')
    return res

def build_homehtml(sample, htmldir, homehtml='index.html', pixscale=0.262,
                   racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
                   maketrends=False, fix_permissions=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    import legacyhalos.html
    
    homehtmlfile = os.path.join(htmldir, homehtml)
    print('Building {}'.format(homehtmlfile))

    js = legacyhalos.html.html_javadate()       

    # group by RA slices
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(homehtmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>Siena Galaxy Atlas 2020 (SGA 2020)</h1>\n')
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

            html.write('<tr>\n')
            #html.write('<th>Number</th>\n')
            html.write('<th> </th>\n')
            html.write('<th>Index</th>\n')
            html.write('<th>ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')

            html.write('</tr>\n')
            for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-largegalaxy-grz-montage.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-largegalaxy-grz-montage.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, lslga=True)

                html.write('<tr>\n')
                #html.write('<td>{:g}</td>\n'.format(count))
                #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.4f}</td>\n'.format(diam1))
                #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
                #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                html.write('</tr>\n')
            html.write('</table>\n')
            #count += 1

        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    if fix_permissions:
        shutil.chown(homehtmlfile, group='cosmo')

    # Optionally build the trends (trends.html) page--
    if maketrends:
        trendshtmlfile = os.path.join(htmldir, trendshtml)
        print('Building {}'.format(trendshtmlfile))
        with open(trendshtmlfile, 'w') as html:
        #with open(os.open(trendshtmlfile, os.O_CREAT | os.O_WRONLY, 0o664), 'w') as html:
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

        if fix_permissions:
            shutil.chown(trendshtmlfile, group='cosmo')

def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)

def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, homehtml, htmldir,
                       racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy, prevgalaxy,
                       nexthtmlgalaxydir, prevhtmlgalaxydir, verbose, clobber, fix_permissions):
    """Build the web page for a single galaxy.

    """
    import fitsio
    from glob import glob
    import legacyhalos.io
    import legacyhalos.html
    
    if not os.path.exists(htmlgalaxydir1):
        os.makedirs(htmlgalaxydir1)
        if fix_permissions:
            for topdir, dirs, files in os.walk(htmlgalaxydir1):
                for dd in dirs:
                    shutil.chown(os.path.join(topdir, dd), group='cosmo')

    htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
    if os.path.isfile(htmlfile) and not clobber:
        print('File {} exists and clobber=False'.format(htmlfile))
        return
    
    nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
    prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))
    
    js = legacyhalos.html.html_javadate()

    # Support routines--

    def _read_ccds_tractor_sample(prefix):
        nccds, tractor, sample = None, None, None
        
        ccdsfile = glob(os.path.join(galaxydir1, '{}-{}-ccds-*.fits'.format(galaxy1, prefix))) # north or south
        if len(ccdsfile) > 0:
            nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()

        # samplefile can exist without tractorfile when using --just-coadds
        samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))
                
        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # We just care about the galaxies in our sample
            if prefix == 'largegalaxy':
                wt, ws = [], []
                for ii, sid in enumerate(sample['SGA_ID']):
                    ww = np.where(tractor['ref_id'] == sid)[0]
                    if len(ww) > 0:
                        wt.append(ww)
                        ws.append(ii)
                if len(wt) == 0:
                    print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxy1))
                    tractor = None
                else:
                    wt = np.hstack(wt)
                    ws = np.hstack(ws)
                    tractor = tractor[wt]
                    sample = sample[ws]
                    srt = np.argsort(tractor['flux_r'])[::-1]
                    tractor = tractor[srt]
                    sample = sample[srt]
                    assert(np.all(tractor['ref_id'] == sample['SGA_ID']))

        return nccds, tractor, sample

    def _html_group_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        viewer_link = legacyhalos.html.viewer_link(ra1, dec1, diam1*2*60/pixscale, lslga=True)

        html.write('<h2>Group Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        html.write('<th>Index<br />(Primary)</th>\n')
        html.write('<th>ID<br />(Primary)</th>\n')
        html.write('<th>Group Name</th>\n')
        html.write('<th>Group RA</th>\n')
        html.write('<th>Group Dec</th>\n')
        html.write('<th>Group Diameter<br />(arcmin)</th>\n')
        #html.write('<th>Richness</th>\n')
        #html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
        #html.write('<td>{:g}</td>\n'.format(ii))
        #print(gal['INDEX'], gal['SGA_ID'], gal['GALAXY'])
        html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal['SGA_ID']))
        html.write('<td>{}</td>\n'.format(gal['GROUP_NAME']))
        html.write('<td>{:.7f}</td>\n'.format(ra1))
        html.write('<td>{:.7f}</td>\n'.format(dec1))
        html.write('<td>{:.4f}</td>\n'.format(diam1))
        #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
        #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
        #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(viewer_link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
        html.write('</tr>\n')
        html.write('</table>\n')

        # Add the properties of each galaxy.
        html.write('<h3>Group Members</h3>\n')
        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>ID</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>Morphology</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>D(25)<br />(arcmin)</th>\n')
        #html.write('<th>PA<br />(deg)</th>\n')
        #html.write('<th>e</th>\n')
        html.write('</tr>\n')
        for groupgal in sample:
            #if '031705' in gal['GALAXY']:
            #    print(groupgal['GALAXY'])
            html.write('<tr>\n')
            html.write('<td>{}</td>\n'.format(groupgal['SGA_ID']))
            html.write('<td>{}</td>\n'.format(groupgal['GALAXY']))
            typ = groupgal['MORPHTYPE'].strip()
            if typ == '' or typ == 'nan':
                typ = '...'
            html.write('<td>{}</td>\n'.format(typ))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
            html.write('<td>{:.4f}</td>\n'.format(groupgal['D25_LEDA']))
            #if np.isnan(groupgal['PA']):
            #    pa = 0.0
            #else:
            #    pa = groupgal['PA']
            #html.write('<td>{:.2f}</td>\n'.format(pa))
            #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
            html.write('</tr>\n')
        html.write('</table>\n')

    def _html_image_mosaics(html):
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

        pngfile, thumbfile = '{}-largegalaxy-grz-montage.png'.format(galaxy1), 'thumb-{}-largegalaxy-grz-montage.png'.format(galaxy1)
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
        html.write('<table width="90%">\n')
        html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile, thumbfile))
        html.write('</table>\n')

        pngfile, thumbfile = '{}-pipeline-grz-montage.png'.format(galaxy1), 'thumb-{}-pipeline-grz-montage.png'.format(galaxy1)
        if os.path.isfile(os.path.join(htmlgalaxydir1, pngfile)):
            html.write('<p>Pipeline (left) data, (middle) model, and (right) residual image mosaic.</p>\n')
            html.write('<table width="90%">\n')
            html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile, thumbfile))
            html.write('</table>\n')

    def _html_ellipsefit_and_photometry(html, tractor, sample):
        html.write('<h2>Elliptical Isophote Analysis</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return
            
        html.write('<h3>Geometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="5">Tractor</th>\n')
        html.write('<th colspan="3">ID</th>\n')
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="5">Ellipse Fitting</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>ID</th>\n')
        html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(25)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(24)<br />(arcsec)</th><th>R(25)<br />(arcsec)</th><th>R(26)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('</tr>\n')

        for tt in tractor:
            ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
            pa = pa % 180

            html.write('<tr><td>{}</td>\n'.format(tt['ref_id']))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                     galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['d25_leda']*60/2, ellipse['pa_leda'], 1-ellipse['ba_leda']))
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa'], ellipse['eps']))

                rr = []
                for rad in [ellipse['radius_sb24'], ellipse['radius_sb25'], ellipse['radius_sb26']]:
                    if rad < 0:
                        rr.append('...')
                    else:
                        rr.append('{:.3f}'.format(rad))
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    rr[0], rr[1], rr[2], ellipse['pa'], ellipse['eps']))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th><th></th>\n')
        html.write('<th colspan="3"></th>\n')
        html.write('<th colspan="12">Curve of Growth</th>\n')
        html.write('</tr>\n')
        html.write('<tr><th></th><th></th>\n')
        html.write('<th colspan="3">Tractor</th>\n')
        html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>ID</th><th>Galaxy</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('<th>g</th><th>r</th><th>z</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            g, r, z = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td><td>{}</td>\n'.format(tt['ref_id'], ss['GALAXY']))
            html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))

            galaxyid = str(tt['ref_id'])
            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                        galaxyid=galaxyid, verbose=False)
            if bool(ellipse):
                g, r, z = _get_mags(ellipse, R24=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R25=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, R26=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                g, r, z = _get_mags(ellipse, cog=True)
                html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            html.write('<h4>{} - {}</h4>\n'.format(galaxyid, sample['GALAXY'][igal]))

            ellipse = legacyhalos.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='largegalaxy',
                                                     galaxyid=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue
            #if False:
            #    html.write('<table>\n')
            #    html.write('<tr><th>Fitting range<br />(arcsec)</th><th>Integration<br />mode</th><th>Clipping<br />iterations</th><th>Clipping<br />sigma</th></tr>')
            #    html.write('<tr><td>{:.3f}-{:.3f}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            #        ellipse[refband]['sma'].min()*pixscale, ellipse[refband]['sma'].max()*pixscale,
            #        ellipse['integrmode'], ellipse['nclip'], ellipse['sclip']))
            #    html.write('</table>\n')
            #    html.write('<br />\n')
            #
            #    html.write('<table>\n')
            #    html.write('<tr><th colspan="5">Mean Geometry</th>')
            #
            #    html.write('<th colspan="4">Ellipse-fitted Geometry</th>')
            #    if ellipse['input_ellipse']:
            #        html.write('<th colspan="2">Input Geometry</th></tr>\n')
            #    else:
            #        html.write('</tr>\n')
            #
            #    html.write('<tr><th>Integer center<br />(x,y, grz pixels)</th><th>Flux-weighted center<br />(x,y grz pixels)</th><th>Flux-weighted size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>')
            #    html.write('<th>Semi-major axis<br />(fitting range, arcsec)</th><th>Center<br />(x,y grz pixels)</th><th>PA<br />(deg)</th><th>e</th>')
            #    if ellipse['input_ellipse']:
            #        html.write('<th>PA<br />(deg)</th><th>e</th></tr>\n')
            #    else:
            #        html.write('</tr>\n')
            #
            #   html.write('<tr><td>({:.0f}, {:.0f})</td><td>({:.3f}, {:.3f})</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>'.format(
            #       ellipse['x0'], ellipse['y0'], ellipse['mge_xmed'], ellipse['mge_ymed'], ellipse['mge_majoraxis']*pixscale,
            #       ellipse['mge_pa'], ellipse['mge_eps']))
            #
            #   if 'init_smamin' in ellipse.keys():
            #       html.write('<td>{:.3f}-{:.3f}</td><td>({:.3f}, {:.3f})<br />+/-({:.3f}, {:.3f})</td><td>{:.1f}+/-{:.1f}</td><td>{:.3f}+/-{:.3f}</td>'.format(
            #           ellipse['init_smamin']*pixscale, ellipse['init_smamax']*pixscale, ellipse['x0_median'],
            #           ellipse['y0_median'], ellipse['x0_err'], ellipse['y0_err'], ellipse['pa'], ellipse['pa_err'],
            #           ellipse['eps'], ellipse['eps_err']))
            #   else:
            #       html.write('<td>...</td><td>...</td><td>...</td><td>...</td>')
            #   if ellipse['input_ellipse']:
            #       html.write('<td>{:.1f}</td><td>{:.3f}</td></tr>\n'.format(
            #           np.degrees(ellipse['geometry'].pa)+90, ellipse['geometry'].eps))
            #   else:
            #       html.write('</tr>\n')
            #   html.write('</table>\n')
            #   html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')

            pngfile = '{}-largegalaxy-{}-ellipse-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-largegalaxy-{}-ellipse-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" width="100%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-largegalaxy-{}-ellipse-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-largegalaxy-{}-ellipse-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_maskbits(html):
        html.write('<h2>Masking Geometry</h2>\n')
        pngfile = '{}-largegalaxy-maskbits.png'.format(galaxy1)
        html.write('<p>Left panel: color mosaic with the original and final ellipse geometry shown. Middle panel: <i>original</i> maskbits image based on the Hyperleda geometry. Right panel: distribution of all sources and frozen sources (the size of the orange square markers is proportional to the r-band flux).</p>\n')
        html.write('<table width="90%">\n')
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(pngfile))
        html.write('</table>\n')

    def _html_ccd_diagnostics(html):
        html.write('<h2>CCD Diagnostics</h2>\n')

        html.write('<table width="90%">\n')
        pngfile = '{}-ccdpos.png'.format(galaxy1)
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile))
        html.write('</table>\n')
        #html.write('<br />\n')
        
    # Read the catalogs and then build the page--
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='largegalaxy')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>{}</h1>\n'.format(galaxy1))
        raslice = get_raslice(gal[racolumn])
        html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_group_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        _html_maskbits(html)
        _html_ccd_diagnostics(html)

        html.write('<br /><br />\n')
        html.write('<a href="../../{}">Home</a>\n'.format(homehtml))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
        html.write('<br />\n')

        html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
        html.write('<br />\n')
        html.write('</html></body>\n')

    if fix_permissions:
        #print('Fixing permissions.')
        shutil.chown(htmlfile, group='cosmo')

def make_html(sample=None, datadir=None, htmldir=None, bands=('g', 'r', 'z'),
              refband='r', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              nproc=1, survey=None, makeplots=False,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None, fix_permissions=True):
    """Make the HTML pages.

    """
    import subprocess
    from astrometry.util.multiproc import multiproc

    import legacyhalos.io
    from legacyhalos.coadds import _mosaic_width

    datadir = legacyhalos.io.legacyhalos_data_dir()
    htmldir = legacyhalos.io.legacyhalos_html_dir()
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)

    if sample is None:
        sample = read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)
        
    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    _, _, done, _ = missing_files(args, sample)
    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return
    
    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    trendshtml = 'trends.html'
    homehtml = 'index.html'

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_homehtml(sample, htmldir, homehtml=homehtml, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   maketrends=maketrends, fix_permissions=fix_permissions)

    # Now build the individual pages in parallel.
    raslices = np.array([get_raslice(ra) for ra in sample[racolumn]])
    rasorted = np.argsort(raslices)
    galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample[rasorted], html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=nproc)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, homehtml, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber, fix_permissions])
    ok = mp.map(_build_htmlpage_one, args)
    
    return 1

#def read_manga_parent(verbose=False):
#    """Read the parent MaNGA-NSA catalog.
#    
#    """
#    sampledir = sample_dir()
#    mangafile = os.path.join(sampledir, 'drpall-v2_1_2.fits')
#    nsafile = os.path.join(sampledir, 'nsa_v1_0_1.fits')
#
#    allmanga = Table(fitsio.read(mangafile, upper=True))
#    _, uindx = np.unique(allmanga['MANGAID'], return_index=True)
#    manga = allmanga[uindx]
#    if verbose:
#        print('Read {}/{} unique galaxies from {}'.format(len(manga), len(allmanga), mangafile), flush=True)
#    #plateifu = [pfu.strip() for pfu in manga['PLATEIFU']]
#
#    catid, rowid = [], []
#    for mid in manga['MANGAID']:
#        cid, rid = mid.split('-')
#        catid.append(cid.strip())
#        rowid.append(rid.strip())
#    catid, rowid = np.hstack(catid), np.hstack(rowid)
#    keep = np.where(catid == '1')[0] # NSA
#    rows = rowid[keep].astype(np.int32)
#
#    print('Selected {} MaNGA galaxies from the NSA'.format(len(rows)))
#    #ww = [np.argwhere(rr[0]==rows) for rr in np.array(np.unique(rows, return_counts=True)).T if rr[1]>=2]
#
#    srt = np.argsort(rows)
#    manga = manga[keep][srt]
#    nsa = Table(fitsio.read(nsafile, rows=rows[srt], upper=True))
#    if verbose:
#        print('Read {} galaxies from {}'.format(len(nsa), nsafile), flush=True)
#    nsa.rename_column('PLATE', 'PLATE_NSA')
#    
#    return hstack( (manga, nsa) )

#def make_html(sample, analysisdir=None, htmldir=None, band=('g', 'r', 'z'),
#              refband='r', pixscale=0.262, nproc=1, dr='dr7', ccdqa=False,
#              makeplots=True, survey=None, clobber=False, verbose=True):
#    """Make the HTML pages.
#
#    """
#    #import legacyhalos.io
#    #from legacyhalos.misc import cutout_radius_150kpc
#
#    if analysisdir is None:
#        analysisdir = analysis_dir()
#    if htmldir is None:
#        htmldir = html_dir()
#
#    # Write the last-updated date to a webpage.
#    js = _javastring()       
#
#    # Get the viewer link
#    def _viewer_link(onegal, dr):
#        baseurl = 'http://legacysurvey.org/viewer/'
#        width = 3 * onegal['NSA_PETRO_TH50'] / pixscale # [pixels]
#        if width > 400:
#            zoom = 14
#        else:
#            zoom = 15
#        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=decals-{}'.format(
#            baseurl, onegal['RA'], onegal['DEC'], zoom, dr)
#        return viewer
#
#    homehtml = 'index.html'
#
#    # Build the home (index.html) page--
#    if not os.path.exists(htmldir):
#        os.makedirs(htmldir)
#    htmlfile = os.path.join(htmldir, homehtml)
#
#    with open(htmlfile, 'w') as html:
#        html.write('<html><body>\n')
#        html.write('<style type="text/css">\n')
#        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
#        html.write('</style>\n')
#
#        html.write('<h1>MaNGA-NSA</h1>\n')
#        html.write('<p>\n')
#        html.write('<a href="https://github.com/moustakas/LSLGA">Code and documentation</a>\n')
#        html.write('</p>\n')
#
#        html.write('<table>\n')
#        html.write('<tr>\n')
#        html.write('<th>Number</th>\n')
#        html.write('<th>Galaxy</th>\n')
#        html.write('<th>RA</th>\n')
#        html.write('<th>Dec</th>\n')
#        html.write('<th>Redshift</th>\n')
#        html.write('<th>Viewer</th>\n')
#        html.write('</tr>\n')
#        for ii, onegal in enumerate( np.atleast_1d(sample) ):
#            galaxy = onegal['MANGAID']
#            if type(galaxy) is np.bytes_:
#                galaxy = galaxy.decode('utf-8')
#                
#            htmlfile = os.path.join(galaxy, '{}.html'.format(galaxy))
#            html.write('<tr>\n')
#            html.write('<td>{:g}</td>\n'.format(ii))
#            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile, galaxy))
#            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
#            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
#            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
#            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
#            html.write('</tr>\n')
#        html.write('</table>\n')
#        
#        html.write('<br /><br />\n')
#        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
#        html.write('</html></body>\n')
#        html.close()
#
#    # Make a separate HTML page for each object.
#    for ii, onegal in enumerate( np.atleast_1d(sample) ):
#        galaxy = onegal['MANGAID']
#        if type(galaxy) is np.bytes_:
#            galaxy = galaxy.decode('utf-8')
#        plateifu = onegal['PLATEIFU']
#        if type(plateifu) is np.bytes_:
#            plateifu = plateifu.decode('utf-8')
#
#        width_arcsec = 2 * MANGA_RADIUS
#        #width_arcsec = RADIUSFACTOR * onegal['NSA_PETRO_TH50']
#        width_kpc = width_arcsec / LSLGA.misc.arcsec2kpc(onegal['Z'])
#
#        survey.output_dir = os.path.join(analysisdir, galaxy)
#        survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)))
#        
#        htmlgalaxydir = os.path.join(htmldir, '{}'.format(galaxy))
#        if not os.path.exists(htmlgalaxydir):
#            os.makedirs(htmlgalaxydir)
#
#        htmlfile = os.path.join(htmlgalaxydir, '{}.html'.format(galaxy))
#        with open(htmlfile, 'w') as html:
#            html.write('<html><body>\n')
#            html.write('<style type="text/css">\n')
#            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
#            html.write('</style>\n')
#
#            html.write('<h1>MaNGA ID {}</h1>\n'.format(galaxy))
#
#            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
#            html.write('<br />\n')
#            html.write('<br />\n')
#
#            # Table of properties
#            html.write('<table>\n')
#            html.write('<tr>\n')
#            html.write('<th>Number</th>\n')
#            html.write('<th>MaNGA ID</th>\n')
#            html.write('<th>PLATEIFU</th>\n')
#            html.write('<th>RA</th>\n')
#            html.write('<th>Dec</th>\n')
#            html.write('<th>Redshift</th>\n')
#            html.write('<th>Viewer</th>\n')
#            html.write('</tr>\n')
#
#            html.write('<tr>\n')
#            html.write('<td>{:g}</td>\n'.format(ii))
#            html.write('<td>{}</td>\n'.format(galaxy))
#            html.write('<td>{}</td>\n'.format(plateifu))
#            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
#            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
#            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
#            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
#            html.write('</tr>\n')
#            html.write('</table>\n')
#
#            html.write('<h2>Multiwavelength mosaics</h2>\n')
#            html.write("""<p>From left to right: GALEX (FUV/NUV), DESI Legacy Surveys (grz), and unWISE (W1/W2)
#            mosaics ({:.2f} arcsec or {:.0f} kpc on a side).</p>\n""".format(width_arcsec, width_kpc))
#            #html.write("""<p>From left to right: GALEX (FUV/NUV), DESI Legacy Surveys (grz), and unWISE (W1/W2)
#            #mosaic ({0:.0f} kpc on a side).</p>\n""".format(width_kpc))
#            html.write('<table width="90%">\n')
#            pngfile = '{}-multiwavelength-data.png'.format(galaxy)
#            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
#                pngfile))
#            html.write('</table>\n')
#            #html.write('<br />\n')
#            
#            ###########################################################################
#            html.write('<h2>Multiwavelength image modeling</h2>\n')
#            html.write("""<p>From left to right: data; model image of all sources except the central, resolved galaxy;
#            residual image containing just the central galaxy.</p><p>From top to bottom: GALEX (FUV/NUV), DESI Legacy
#            Surveys (grz), and unWISE (W1/W2) mosaic ({:.2f} arcsec or {:.0f} kpc on a side).</p>\n""".format(
#                width_arcsec, width_kpc))
#
#            html.write('<table width="90%">\n')
#            pngfile = '{}-multiwavelength-models.png'.format(galaxy)
#            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
#                pngfile))
#            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
#            html.write('</table>\n')
#            html.write('<br />\n')
#
#            ###########################################################################
#            
#            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
#
#            if False:
#                html.write('<table width="90%">\n')
#                html.write('<tr>\n')
#                pngfile = '{}-ellipse-multiband.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                html.write('</tr>\n')
#                html.write('</table>\n')
#
#            html.write('<table width="90%">\n')
#            html.write('<tr>\n')
#            pngfile = '{}-ellipse-sbprofile.png'.format(galaxy)
#            html.write('<td width="100%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                pngfile))
#            html.write('<td></td>\n')
#            html.write('</tr>\n')
#            html.write('</table>\n')
#
#            if False:
#                html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
#                html.write('<table width="90%">\n')
#
#                # single-sersic
#                html.write('<tr>\n')
#                html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
#                html.write('</tr>\n')
#                html.write('<tr>\n')
#                pngfile = '{}-sersic-single-nowavepower.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                pngfile = '{}-sersic-single.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                html.write('</tr>\n')
#
#                # Sersic+exponential
#                html.write('<tr>\n')
#                html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
#                html.write('</tr>\n')
#                html.write('<tr>\n')
#                pngfile = '{}-sersic-exponential-nowavepower.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                pngfile = '{}-sersic-exponential.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                html.write('</tr>\n')
#
#                # double-sersic
#                html.write('<tr>\n')
#                html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
#                html.write('</tr>\n')
#                html.write('<tr>\n')
#                pngfile = '{}-sersic-double-nowavepower.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                pngfile = '{}-sersic-double.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                html.write('</tr>\n')
#                html.write('</table>\n')
#                html.write('<br />\n')
#
#            if False:
#                html.write('<h2>CCD Diagnostics</h2>\n')
#                html.write('<table width="90%">\n')
#                html.write('<tr>\n')
#                pngfile = '{}-ccdpos.png'.format(galaxy)
#                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                    pngfile))
#                html.write('</tr>\n')
#
#                for iccd in range(len(survey.ccds)):
#                    html.write('<tr>\n')
#                    pngfile = '{}-2d-ccd{:02d}.png'.format(galaxy, iccd)
#                    html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
#                        pngfile))
#                    html.write('</tr>\n')
#                html.write('</table>\n')
#                html.write('<br />\n')
#            
#            html.write('<br />\n')
#            html.write('<br />\n')
#            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
#            html.write('<br />\n')
#
#            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
#            html.write('<br />\n')
#            html.write('</html></body>\n')
#            html.close()
#
#    if makeplots:
#        for onegal in sample:
#            galaxy = onegal['MANGAID']
#            if type(galaxy) is np.bytes_:
#                galaxy = galaxy.decode('utf-8')
#            galaxydir = os.path.join(analysisdir, galaxy)
#            htmlgalaxydir = os.path.join(htmldir, galaxy)
#
#            survey.output_dir = os.path.join(analysisdir, galaxy)
#            #survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)))
#
#            # Custom plots
#            ellipsefit = read_ellipsefit(galaxy, galaxydir)
#            
#            cogfile = os.path.join(htmlgalaxydir, '{}-curve-of-growth.png'.format(galaxy))
#            if not os.path.isfile(cogfile) or clobber:
#                LSLGA.qa.qa_curveofgrowth(ellipsefit, png=cogfile, verbose=verbose)
#                
#            #pdb.set_trace()
#                
#            sbprofilefile = os.path.join(htmlgalaxydir, '{}-ellipse-sbprofile.png'.format(galaxy))
#            if not os.path.isfile(sbprofilefile) or clobber:
#                LSLGA.qa.display_ellipse_sbprofile(ellipsefit, png=sbprofilefile,
#                                                   verbose=verbose)
#            
#            LSLGA.qa.qa_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
#                                               clobber=clobber, verbose=verbose)
#
#            # Plots common to legacyhalos
#            #make_plots([onegal], galaxylist=[galaxy], analysisdir=analysisdir,
#            #           htmldir=htmldir, clobber=clobber, verbose=verbose,
#            #           survey=survey, refband=refband, pixscale=pixscale,
#            #           band=band, nproc=nproc, ccdqa=ccdqa, trends=False)
#
#    print('HTML pages written to {}'.format(htmldir))
