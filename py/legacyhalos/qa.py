"""
legacyhalos.qa
==============

Code to do produce various QA (quality assurance) plots. 

https://xkcd.com/color/rgb/

"""
import matplotlib as mpl
mpl.use('Agg')

import os, pdb
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import legacyhalos.io
import legacyhalos.misc

#from legacyhalos.misc import RADIUS_CLUSTER_KPC

sns, _ = legacyhalos.misc.plot_style()
    
fonttype = os.path.join(os.getenv('LEGACYHALOS_CODE_DIR'), 'py', 'legacyhalos', 'data', 'Georgia-Italic.ttf')
prop = mpl.font_manager.FontProperties(fname=fonttype, size=12)

# color-blind friendly color cycle: 
# https://twitter.com/rachel_kurchin/status/1229567059694170115
cb_colors = {'blue': '#377eb8',
             'orange': '#ff7f00',
             'green': '#4daf4a',
             'pink': '#f781bf',
             'brown': '#a65628',
             'purple': '#984ea3',
             'gray': '#999999',
             'red': '#e41a1c',
             'yellow': '#dede00'}

def _sbprofile_colors(makeiter=True, galex=False, unwise=False):
    """Return an iterator of colors good for the surface brightness profile plots.
    https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette

    """
    try:
        _colors = sns.color_palette('Set1', n_colors=14, desat=0.75)
    except:
        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        
    sbcolors = [_colors[1], _colors[2], _colors[0]]
    colorcolors = [_colors[3], _colors[4]]
    if galex or unwise:
        _morecolors = sns.color_palette('Set2', n_colors=8, desat=0.75)
        if galex:
            sbcolors = sbcolors + [_morecolors[0], _morecolors[1]]
        if unwise:
            sbcolors = sbcolors + [_morecolors[2], _morecolors[3], _morecolors[6], _morecolors[7]]
    colors = sbcolors + colorcolors

    if makeiter:
        colors = iter(colors)
        
    return colors

def draw_ellipse_on_png(im, x0, y0, ba, pa, major_axis_diameter_arcsec,
                        pixscale, color='#3388ff'):
    """

    """
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None
    
    minor_axis_diameter_arcsec = major_axis_diameter_arcsec * ba

    overlay_height = int(major_axis_diameter_arcsec / pixscale)
    overlay_width = int(minor_axis_diameter_arcsec / pixscale)
    overlay = Image.new('RGBA', (overlay_width, overlay_height))

    draw = ImageDraw.ImageDraw(overlay)
    box_corners = (0, 0, overlay_width, overlay_height)
    draw.ellipse(box_corners, fill=None, outline=color, width=3)

    rotated = overlay.rotate(pa, expand=True)
    rotated_width, rotated_height = rotated.size
    paste_shift_x = int(x0 - rotated_width / 2)
    paste_shift_y = int(y0 - rotated_height / 2)
    im.paste(rotated, (paste_shift_x, paste_shift_y), rotated)

def addbar_to_png(jpgfile, barlen, barlabel, imtype, pngfile, scaledfont=True,
                  pixscalefactor=1.0, fntsize=20):
    """Support routine for routines in html.

    fntsize - only used if scaledfont=False
    """
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    with Image.open(jpgfile) as im:
        draw = ImageDraw.Draw(im)
        sz = im.size
        width = np.round(pixscalefactor*sz[0]/150).astype('int')
        # Bar and label
        if barlen:
            if scaledfont:
                fntsize = np.round(pixscalefactor*sz[0]/50).astype('int')                
                #fntsize = np.round(0.05*sz[0]).astype('int')
                #fntsize = np.round(sz[0]/50).astype('int')
            #if fntsize < 56:
            #    fntsize = 56
            font = ImageFont.truetype(fonttype, size=fntsize)
            # Add a scale bar and label--
            x0, x1, y0, y1 = 0+fntsize*2, 0+fntsize*2+barlen*pixscalefactor, sz[1]-fntsize*2, sz[1]-fntsize*2.5#4
            #print(sz, fntsize, x0, x1, y0, y1, barlen*pixscalefactor)
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
        print('Writing {}'.format(pngfile))
        im.save(pngfile)
    return pngfile

def qa_maskbits(mask, tractor, ellipsefitall, colorimg, largegalaxy=False, png=None):
    """For the SGA, display the maskbits image with some additional information
    about the catalog.

    colorblind-friendly colors are from
    https://twitter.com/rachel_kurchin/status/1229567059694170115

    """
    from photutils import EllipticalAperture
    from PIL import ImageDraw, Image

    from tractor.ellipses import EllipseE
    from legacypipe.reference import get_large_galaxy_version
    from legacyhalos.misc import is_in_ellipse
    from legacyhalos.SGA import _get_diameter

    Image.MAX_IMAGE_PIXELS = None
    imgsz = colorimg.size
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5*3, 5), sharey=True)

    # original maskbits
    ax2.imshow(mask, origin='lower', cmap='gray_r')#, interpolation='none')
    ax2.set_aspect('equal')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    #ax2.set_title('Original maskbits')
    #ax2.axis('off')
    #ax2.autoscale(False)
    #ax2.scatter(tractor['BX'], tractor['BY'], alpha=0.3, s=10, color='#999999')

    ax3.scatter(tractor['BX'], tractor['BY'], alpha=0.3, s=10, color='#999999',
                label='All Sources')
    ax3.set_aspect('equal')
    sz = mask.shape
    ax3.set_xlim(0, sz[1]-1)
    ax3.set_ylim(0, sz[0]-1)
    #ax3.imshow(mask*0, origin='lower', cmap='gray_r')#, interpolation='none')
    #ax3.plot([0, sz[1]-1], [0, sz[0]-1])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ##ax3.set_title('Original maskbits')
    #ax3.axis('off')

    #refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    #ilslga = np.where(tractor['REF_CAT'] == refcat)[0]
    #ax3.scatter(tractor['BX'][ilslga], tractor['BY'][ilslga], s=50,
    #            edgecolor='k', color='blue')
    
    #ax3.autoscale(False)
    ax3.margins(0, tight=True)

    minmarker, maxmarker = 30, 300
    for igal, ellipsefit in enumerate(ellipsefitall):
        diam, diamref = _get_diameter(ellipsefit)
        ragal, decgal, pa, ba = ellipsefit['ra_moment'], ellipsefit['dec_moment'], ellipsefit['pa_moment'], 1-ellipsefit['eps_moment']

        reff, e1, e2 = EllipseE.fromRAbPhi(diam*60/2, ba, 180-pa) # note the 180 rotation
        inellipse = np.where(is_in_ellipse(tractor['RA'], tractor['DEC'], ragal, decgal, reff, e1, e2))[0]
        if len(inellipse) < 3:
            continue

        # scale the size of the marker by flux
        minflx, maxflx = np.percentile(tractor['FLUX_R'][inellipse], [50, 95])
        if maxflx > minflx:
            ss = maxmarker * (tractor['FLUX_R'][inellipse] - minflx) / (maxflx - minflx)
        else:
            ss = np.repeat(maxmarker, len(tractor))
        ss[ss < minmarker] = minmarker
        ss[ss > maxmarker] = maxmarker

        if igal == 0:
            ax3.scatter(tractor['BX'][inellipse], tractor['BY'][inellipse], s=ss,
                        marker='s', edgecolor='k', color=cb_colors['orange'], label='Frozen Sources')
        else:
            ax3.scatter(tractor['BX'][inellipse], tractor['BY'][inellipse], s=ss,
                        marker='s', edgecolor='k', color=cb_colors['orange'])

        # ellipse geometry
        maxis = diam * 60 / ellipsefit['refpixscale'] / 2 # [pixels]
        ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                     maxis, maxis*(1 - ellipsefit['eps_moment']),
                                     np.radians(ellipsefit['pa_moment']-90))
        if igal == 0:
            ellaper.plot(color=cb_colors['blue'], lw=2, axes=ax2, alpha=0.9, label='R(26)')
        else:
            ellaper.plot(color=cb_colors['blue'], lw=2, axes=ax2, alpha=0.9)
        ellaper.plot(color=cb_colors['blue'], lw=2, ls='-', axes=ax3, alpha=0.9)

        draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], imgsz[1]-ellipsefit['y0_moment'],
                            1-ellipsefit['eps_moment'],
                            ellipsefit['pa_moment'], 2 * maxis * ellipsefit['refpixscale'],
                            ellipsefit['refpixscale'], color=cb_colors['blue']) # '#ffaa33')
        if 'd25_leda' in ellipsefit.keys():
            draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], imgsz[1]-ellipsefit['y0_moment'],
                                ellipsefit['ba_leda'], ellipsefit['pa_leda'],
                                ellipsefit['d25_leda'] * 60.0, ellipsefit['refpixscale'],
                                color=cb_colors['red'])
        
            # Hyperleda geometry
            maxis = ellipsefit['d25_leda'] * 60 / ellipsefit['refpixscale'] / 2 # [pixels]
            ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                         maxis, maxis * ellipsefit['ba_leda'],
                                         np.radians(ellipsefit['pa_leda']-90))
        if igal == 0:
            ellaper.plot(color=cb_colors['red'], lw=2, ls='-', axes=ax2, alpha=1.0, label='Hyperleda')
        else:
            ellaper.plot(color=cb_colors['red'], lw=2, ls='-', axes=ax2, alpha=1.0)
        ellaper.plot(color=cb_colors['red'], lw=2, ls='-', axes=ax3, alpha=1.0)

    # color mosaic
    draw = ImageDraw.Draw(colorimg)
    ax1.imshow(np.flipud(colorimg), interpolation='none') # not sure why I have to flip here...
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    #ax1.axis('off')
    #ax1.autoscale(False)
    #ax1.scatter(tractor['BX'], imgsz[1]-tractor['BY'], alpha=1.0, s=10, color='red')
    #ax1.scatter(tractor['BX'], tractor['BY'], alpha=1.0, s=10, color='#999999')

    hh, ll = ax2.get_legend_handles_labels()
    if len(hh) > 0:
        ax2.legend(loc='lower right', fontsize=12)
        lgnd = ax3.legend(loc='lower right', fontsize=12)
        lgnd.legendHandles[0]._sizes = [40]
        lgnd.legendHandles[1]._sizes = [40]    

    fig.subplots_adjust(wspace=0.05, right=0.9)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight')#, pad_inches=0)
        plt.close(fig)
    else:
        plt.show()
    
def qa_curveofgrowth(ellipsefit, pipeline_ellipsefit=None, png=None,
                     plot_sbradii=False, cosmo=None, verbose=True):
    """Plot up the curve of growth versus semi-major axis.

    """
    import matplotlib.ticker as ticker
    from legacyhalos.ellipse import cog_model
    #from legacyhalos.ellipse import CogModel

    if ellipsefit['success'] is False or np.atleast_1d(ellipsefit['sma_r'])[0] == -1:
        return
    
    fig, ax = plt.subplots(figsize=(9, 7))
    bands, refband = ellipsefit['bands'], ellipsefit['refband']

    galex = 'FUV' in bands
    unwise = 'W1' in bands
    colors = _sbprofile_colors(galex=galex, unwise=unwise)
        
    if 'redshift' in ellipsefit.keys():
        redshift = ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
    else:
        redshift, smascale = None, None
        
    #maxsma = ellipsefit['cog_sma_{}'.format(refband)].max()
    maxsma = 0

    yfaint, ybright = 0, 50
    for filt in bands:
        col = next(colors) # iterate here in case we're missing a bandpass
        
        #flux = ellipsefit['apphot_mag_{}'.format(filt)]
        #good = np.where( np.isfinite(flux) * (flux > 0) )[0]
        #mag = 22.5-2.5*np.log10(flux[good])
        cogsma = ellipsefit['cog_sma_{}'.format(filt.lower())]
        cogflux = ellipsefit['cog_flux_{}'.format(filt.lower())]
        cogivar = ellipsefit['cog_flux_ivar_{}'.format(filt.lower())]
        chi2 = ellipsefit['cog_chi2_{}'.format(filt.lower())]
        if np.atleast_1d(cogsma)[0] == -1.0 or chi2 == -1.0: # no measurement, or failed
            continue

        good = np.where((cogflux > 0) * (cogivar > 0))[0]
        cog = 22.5 - 2.5 * np.log10(cogflux[good])
        cogferr = 1.0 / np.sqrt(cogivar[good])
        cogerr = 2.5 * cogferr / cogflux[good] / np.log(10)
        sma = cogsma[good]

        radius = sma**0.25
        xlim = (0.9, radius.max()*1.01)
        
        magtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        m0 = ellipsefit['cog_m0_{}'.format(filt.lower())]
        alpha1 = ellipsefit['cog_alpha1_{}'.format(filt.lower())]
        alpha2 = ellipsefit['cog_alpha2_{}'.format(filt.lower())]

        #magtot = np.mean(mag[-5:])
        if pipeline_ellipsefit and False:
            pipeline_magtot = pipeline_ellipsefit['cog_mtot_{}'.format(filt.lower())]
            label = '{}={:.3f} ({:.3f})'.format(filt, magtot, pipeline_magtot)
        else:
            #label = r'${}$'.format(filt.lower())
            label = r'${}={:.3f}$'.format(filt, magtot)
            #label = r'{}={:.3f} ($\chi^2_\nu={:.1f}$)'.format(filt, magtot, chi2)
            
        #ax.plot(sma, cog, label=label)
        ax.fill_between(radius, cog-cogerr, cog+cogerr, label=label,
                        color=col, alpha=0.9)
                        #facecolor=col, edgecolor='k', lw=2)
        #if np.any(np.iscomplex(sma)) or np.any(np.iscomplex(cog)) or np.any(np.iscomplex(cogerr)):
        #    pdb.set_trace()

        if pipeline_ellipsefit and False:
            _sma = pipeline_ellipsefit['cog_sma_{}'.format(filt.lower())]
            _cog = pipeline_ellipsefit['cog_flux_{}'.format(filt.lower())]
            _cogerr = pipeline_ellipsefit['cog_flux_ivar_{}'.format(filt.lower())]
            #ax.plot(_sma, _cog, alpha=0.5, color='gray')
            ax.fill_between(_sma, _cog-_cogerr, _cog+_cogerr,
                            facecolor=col, alpha=0.5)#, edgecolor='k', lw=1)

        smamodel = np.linspace(np.min(sma), xlim[1], 50)
        #smamodel = np.linspace(np.min(sma), np.max(sma), 50)
        cogmodel = cog_model(smamodel, magtot, m0, alpha1, alpha2)
        #cogmodel = CogModel().evaluate(sma, magtot, m0, alpha1, alpha2)
        ax.plot(smamodel**0.25, cogmodel, color=col, lw=2, ls='-', alpha=0.5)

        if sma.max() > maxsma:
            maxsma = sma.max()
        
        #print(filt, np.mean(mag[-5:]))
        #print(filt, mag[-5:], np.mean(mag[-5:])
        #print(filt, np.min(mag))

        inrange = np.where((radius >= xlim[0]) * (radius <= xlim[1]))[0]
        if cog[inrange].max() > yfaint:
            yfaint = cog[inrange].max()
        if cog[inrange].min() < ybright:
            ybright = cog[inrange].min()
        #if filt == 'r':
        #    pdb.set_trace()

    #ax.set_xlabel(r'Semi-major axis (arcsec)')
    #ax.set_ylabel('Cumulative brightness (AB mag)')
    ax.set_xlabel(r'(Semi-major axis $r$ / arcsec)$^{1/4}$')
    ax.set_ylabel('$m(<r)$ (mag)')

    xmin = 0.9
    if maxsma > 0:
        ax.set_xlim(xmin, (maxsma**0.25))#*1.01)
        #ax.set_xlim(0, maxsma*1.01)
    else:
        ax.set_xlim(0, 3) # hack!

    # overplot the model curves
    colors = _sbprofile_colors(galex=galex, unwise=unwise)
    for filt in bands:
        col = next(colors) # iterate here in case we're missing a bandpass
        magtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        m0 = ellipsefit['cog_m0_{}'.format(filt.lower())]
        alpha1 = ellipsefit['cog_alpha1_{}'.format(filt.lower())]
        alpha2 = ellipsefit['cog_alpha2_{}'.format(filt.lower())]

        if (magtot > 0) * (m0 > 0) * (alpha1 != 0.0) * (alpha2 != 0.0):
            smamodel = np.linspace(0.7, maxsma, 200)
            cogmodel = cog_model(smamodel, magtot, m0, alpha1, alpha2)
            #if filt == 'W4':
            #    pdb.set_trace()
            if np.min(cogmodel) < ybright:
                ybright = np.min(cogmodel)
            ax.plot(smamodel**0.25, cogmodel, color=col, lw=2, ls='-', alpha=0.6)

    ax.margins()
    xlim = ax.get_xlim()
    if smascale:
        ax_twin = ax.twiny()
        ax_twin.set_xlim(xlim[0]*smascale, xlim[1]*smascale)
        ax_twin.set_xlabel('Semi-major axis (kpc)')
    #ax_twin.margins(x=0)

    yfaint += 0.5
    ybright += -0.5
    
    ax.set_ylim(yfaint, ybright)
    ylim = ax.get_ylim()        
    if np.abs(ylim[1]-ylim[0]) > 15:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    #if np.abs(ylim[1]-ylim[0]) > 8:
    #    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    if False:
        ax_twin = ax.twinx()
        ax_twin.set_ylim(yfaint, ybright)
        ax_twin.set_ylabel('Cumulative Flux (AB mag)')#, rotation=-90)

    hh, ll = ax.get_legend_handles_labels()
    if len(hh) > 0:
        if len(hh) > 3:
            ncol = 2
        else:
            ncol = 1
        leg1 = ax.legend(loc='lower right', fontsize=14, ncol=ncol)

    # Plot some threshold radii for the large-galaxy project--
    if plot_sbradii:
        lline, llabel = [], []
        if ellipsefit['sma_sb24'] > 0: #< xlim[1]:
            ll = ax.axvline(x=ellipsefit['sma_sb24'], lw=2, color='k', ls='-.')
            lline.append(ll), llabel.append('R(24)')
            
        if ellipsefit['sma_sb25'] > 0: #< xlim[1]:
            ll = ax.axvline(x=ellipsefit['sma_sb25'], lw=2, color='k', ls='--')
            lline.append(ll), llabel.append('R(25)')
            
        if ellipsefit['sma_sb26'] > 0: #< xlim[1]:            
            ll = ax.axvline(x=ellipsefit['sma_sb26'], lw=2, color='k', ls='-')
            lline.append(ll), llabel.append('R(26)')

        if False:
            ll = ax.axvline(x=ellipsefit['majoraxis'] * ellipsefit['refpixscale'],
                            lw=2, color='#e41a1c', ls='dotted')
            lline.append(ll), llabel.append('Moment Size')

        if len(lline) > 0:
            leg2 = ax.legend(lline, llabel, loc='lower left', fontsize=14, frameon=False)
            ax.add_artist(leg1)
        
    if smascale:
        fig.subplots_adjust(left=0.15, bottom=0.15, top=0.85, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.85, right=0.88)
    else:
        fig.subplots_adjust(left=0.15, bottom=0.15, top=0.95, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.95, right=0.88)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def qa_multiwavelength_sed(ellipsefit, tractor=None, png=None, verbose=True):
    """Plot up the multiwavelength SED.

    """
    from copy import deepcopy
    import matplotlib.ticker as ticker
    from astropy.table import Table
    from legacyhalos.io import get_run
    
    if ellipsefit['success'] is False or np.atleast_1d(ellipsefit['sma_r'])[0] == -1:
        return
    
    bands, refband = ellipsefit['bands'], ellipsefit['refband']

    galex = 'FUV' in bands
    unwise = 'W1' in bands
    colors = _sbprofile_colors(galex=galex, unwise=unwise)
        
    if 'redshift' in ellipsefit.keys():
        redshift = ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
    else:
        redshift, smascale = None, None

    # see also Morrisey+05
    effwave_north = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4815.95363513, 'r': 6437.79282937, 'z': 9229.65786449}
    effwave_south = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4890.03670428, 'r': 6469.62203811, 'z': 9196.46396394}

    _tt = Table()
    _tt['RA'] = [ellipsefit['ra_moment']]
    _tt['DEC'] = [ellipsefit['dec_moment']]
    run = get_run(_tt)

    if run == 'north':
        effwave = effwave_north
    else:
        effwave = effwave_south

    # build the arrays
    nband = len(bands)
    bandwave = np.array([effwave[filt.lower()] for filt in bands])

    _phot = {'abmag': np.zeros(nband, 'f4')-1,
             'abmagerr': np.zeros(nband, 'f4')+0.5,
             'lower': np.zeros(nband, bool)}
    phot = {'mag_tot': deepcopy(_phot), 'tractor': deepcopy(_phot), 'mag_sb25': deepcopy(_phot)}

    for ifilt, filt in enumerate(bands):
        mtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        if mtot > 0:
            phot['mag_tot']['abmag'][ifilt] = mtot
            phot['mag_tot']['abmagerr'][ifilt] = 0.1
            phot['mag_tot']['lower'][ifilt] = False

        flux = ellipsefit['flux_sb25_{}'.format(filt.lower())]
        ivar = ellipsefit['flux_ivar_sb25_{}'.format(filt.lower())]
        #print(filt, mag)

        if flux > 0 and ivar > 0:
            mag = 22.5 - 2.5 * np.log10(flux)
            ferr = 1.0 / np.sqrt(ivar)
            magerr = 2.5 * ferr / flux / np.log(10)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = magerr
            phot['mag_sb25']['lower'][ifilt] = False
        if flux <=0 and ivar > 0:
            ferr = 1.0 / np.sqrt(ivar)
            mag = 22.5 - 2.5 * np.log10(ferr)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = 0.75
            phot['mag_sb25']['lower'][ifilt] = True

        if tractor is not None:
            flux = tractor['flux_{}'.format(filt.lower())]
            ivar = tractor['flux_ivar_{}'.format(filt.lower())]
            #if filt == 'FUV':
            #    pdb.set_trace()
            if flux > 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(flux)
                phot['tractor']['abmagerr'][ifilt] = 0.1
            if flux <= 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(1/np.sqrt(ivar))
                phot['tractor']['abmagerr'][ifilt] = 0.75
                phot['tractor']['lower'][ifilt] = True

    #print(phot['mag_tot']['abmag'])
    #print(phot['mag_sb25']['abmag'])
    #print(phot['tractor']['abmag'])

    def _addphot(thisphot, color, marker, alpha, label):
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        lolims=True, linestyle='none', alpha=alpha)#, lolims=True)
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        label=label, linestyle='none', alpha=alpha)
    
    # make the plot
    fig, ax = plt.subplots(figsize=(9, 7))

    # get the plot limits
    good = np.where(phot['mag_tot']['abmag'] > 0)[0]
    ymax = np.min(phot['mag_tot']['abmag'][good])
    ymin = np.max(phot['mag_tot']['abmag'][good])

    good = np.where(phot['tractor']['abmag'] > 0)[0]
    if np.min(phot['tractor']['abmag'][good]) < ymax:
        ymax = np.min(phot['tractor']['abmag'][good])
    if np.max(phot['tractor']['abmag']) > ymin:
        ymin = np.max(phot['tractor']['abmag'][good])
    #print(ymin, ymax)

    ymin += 1.5
    ymax -= 1.5

    wavemin, wavemax = 0.1, 30

    # have to set the limits before plotting since the axes are reversed
    if np.abs(ymax-ymin) > 15:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim(ymin, ymax)
    _addphot(phot['mag_tot'], color='red', marker='s', alpha=1.0, label=r'$m_{\mathrm{tot}}$')
    _addphot(phot['mag_sb25'], color='orange', marker='^', alpha=0.9, label=r'$m(r<R_{25})$')
    _addphot(phot['tractor'], color='blue', marker='o', alpha=0.75, label='Tractor')

    #thisphot = phot['tractor']
    #color='blue'
    #marker='o'
    #label='Tractor'

    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                label=label, linestyle='none')
    
    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
    ##ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good],
    ##            marker='o', uplims=thisphot['lower'][good], linestyle='none')
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good][0],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                uplims=thisphot['lower'][good], linestyle='none')#, lolims=True)
                    
    ax.set_xlabel(r'Observed-frame Wavelength ($\mu$m)') 
    ax.set_ylabel(r'Apparent Brightness (AB mag)') 
    ax.set_xlim(wavemin, wavemax)
    ax.set_xscale('log')
    ax.legend(loc='lower right')

    def _frmt(value, _):
        if value < 1:
            return '{:.1f}'.format(value)
        else:
            return '{:.0f}'.format(value)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xticks([0.1, 0.2, 0.4, 1.0, 3.0, 5.0, 10, 20])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_frmt))

    if smascale:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.85, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.85, right=0.88)
    else:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.95, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.95, right=0.88)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def display_sersic(sersic, png=None, cosmo=None, verbose=False):
    """Plot a wavelength-dependent surface brightness profile and model fit.

    """
    markers = iter(['o', 's', 'D'])
    colors = _sbprofile_colors()

    if sersic['success']:
        smascale = legacyhalos.misc.arcsec2kpc(sersic['redshift'], cosmo=cosmo)
        model = sersic['bestfit']
    else:
        smascale = 1
        model = None

    ymnmax = [40, 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for band, lam in zip( sersic['bands'], (sersic['lambda_g'],
                                            sersic['lambda_r'],
                                            sersic['lambda_z']) ):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #good = (lam == sersic['wave']) * np.isfinite(sersic['sb'])
            good = (lam == sersic['wave']) * np.isfinite(sersic['sb']) * (sersic['sb'] / sersic['sberr'] > 1)

        wave = sersic['wave'][good]
        rad = sersic['radius'][good]
        sb = sersic['sb'][good]
        sberr = sersic['sberr'][good]

        srt = np.argsort(rad)
        rad, sb, sberr, wave = rad[srt], sb[srt], sberr[srt], wave[srt]

        if model is not None:
            filt = '${}$:'.format(band)
            if 'single' in sersic['modeltype']:
                n = r'$n={:.2f}$'.format(model.get_sersicn(nref=model.nref, lam=lam, alpha=model.alpha))
                r50 = r'$r_{{50}}={:.2f}\ kpc$'.format(model.get_r50(r50ref=model.r50ref, lam=lam, beta=model.beta) * smascale)
                label = '{} {}, {}'.format(filt, n, r50)
                labelfont = 14
            elif 'exponential' in sersic['modeltype']:
                n1 = r'$n_{{1}}={:.2f}$'.format(model.get_sersicn(nref=model.nref1, lam=lam, alpha=model.alpha1))
                n2 = r'$n_{{2}}={:.2f}$'.format(model.nref2.value)
                r50_1 = r'$r_{{50,1}}={:.2f}$'.format(model.get_r50(r50ref=model.r50ref1, lam=lam, beta=model.beta1) * smascale)
                r50_2 = r'$r_{{50,2}}={:.2f}\ kpc$'.format(model.get_r50(r50ref=model.r50ref2, lam=lam, beta=model.beta2) * smascale)
                label = '{} {}, {}, {}, {}'.format(filt, n1, n2, r50_1, r50_2)
                labelfont = 12
            elif 'double' in sersic['modeltype']:
                n1 = r'$n_{{1}}={:.2f}$'.format(model.get_sersicn(nref=model.nref1, lam=lam, alpha=model.alpha1))
                n2 = r'$n_{{2}}={:.2f}$'.format(model.get_sersicn(nref=model.nref2, lam=lam, alpha=model.alpha2))
                r50_1 = r'$r_{{50,1}}={:.2f}$'.format(model.get_r50(r50ref=model.r50ref1, lam=lam, beta=model.beta1) * smascale)
                r50_2 = r'$r_{{50,2}}={:.2f}\ kpc$'.format(model.get_r50(r50ref=model.r50ref2, lam=lam, beta=model.beta2) * smascale)
                label = '{} {}, {}, {}, {}'.format(filt, n1, n2, r50_1, r50_2)
                labelfont = 12
            elif 'triple' in sersic['modeltype']:
                n1 = r'$n_{{1}}={:.2f}$'.format(model.get_sersicn(nref=model.nref1, lam=lam, alpha=model.alpha1))
                n2 = r'$n_{{2}}={:.2f}$'.format(model.get_sersicn(nref=model.nref2, lam=lam, alpha=model.alpha2))
                n3 = r'$n_{{3}}={:.2f}$'.format(model.get_sersicn(nref=model.nref3, lam=lam, alpha=model.alpha3))
                r50_1 = r'$r_{{50,1}}={:.2f}$'.format(model.get_r50(r50ref=model.r50ref1, lam=lam, beta=model.beta1) * smascale)
                r50_2 = r'$r_{{50,2}}={:.2f}$'.format(model.get_r50(r50ref=model.r50ref2, lam=lam, beta=model.beta2) * smascale)
                r50_3 = r'$r_{{50,3}}={:.2f}\ kpc$'.format(model.get_r50(r50ref=model.r50ref3, lam=lam, beta=model.beta3) * smascale)
                #label = '{}, {}, {}\n{}, {}, {}'.format(n1, n2, n3, r50_1, r50_2, r50_3)
                label = '{} {}, {}, {}\n    {}, {}, {}'.format(filt, n1, n2, n3, r50_1, r50_2, r50_3)
                labelfont = 12
            else:
                raise ValueError('Unrecognized model type {}'.format(sersic['modeltype']))
        else:
            label = band
            labelfont = 12

        col = next(colors)
        #ax.plot(rad, 22.5-2.5*np.log10(sb), label=band)
        #ax.scatter(rad, 22.5-2.5*np.log10(sb), color=col,
        #           alpha=1, s=50, label=label, marker=next(markers))
        mu = 22.5 - 2.5 * np.log10(sb)
        muerr = 2.5 * sberr / np.log(10) / sb
            
        ax.fill_between(rad, mu-muerr, mu+muerr, facecolor=col, label=label, alpha=1)

        if np.nanmin(mu-muerr) < ymnmax[0]:
            ymnmax[0] = np.nanmin(mu-muerr)
        if np.nanmax(mu+muerr) > ymnmax[1]:
            ymnmax[1] = np.nanmax(mu+muerr)
        
        # optionally overplot the model
        if model is not None:
            #ww = sersic['wave_uniform'] == lam
            #sb_model = model(sersic['radius_uniform'][ww], sersic['wave_uniform'][ww])
            #ax.plot(sersic['radius_uniform'][ww], 22.5-2.5*np.log10(sb_model), color='k', ls='--', lw=2, alpha=1)
            sb_model = model(rad, wave)
            ax.plot(rad, 22.5-2.5*np.log10(sb_model), color='k', ls='--', lw=2, alpha=1)

            if False:
                #wave_model = wave ; rad_model = rad
                wave_model = np.zeros_like(rad_model) + lam

                from legacyhalos.sersic import SersicSingleWaveModel
                sb_model2 = SersicSingleWaveModel(seed=model.seed, psfsigma_g=model.psfsigma_g*0,
                                                  psfsigma_r=model.psfsigma_r*0, psfsigma_z=model.psfsigma_z*0,
                                                  pixscale=model.pixscale).evaluate(
                                                  #rad, wave,
                                                  rad_model, wave2,
                                                  nref=model.nref, r50ref=model.r50ref, 
                                                  alpha=model.alpha, beta=model.beta, 
                                                  mu50_g=model.mu50_g, mu50_r=model.mu50_r, mu50_z=model.mu50_z)
                #ax.plot(rad_model, 22.5-2.5*np.log10(sb_model2), ls='-', lw=2, alpha=1, color='orange')
                #ax.plot(rad, 22.5-2.5*np.log10(sb_model2), ls='-', lw=2, alpha=1, color='orange')
                #pdb.set_trace()

            # plot the individual Sersic profiles
            if model.__class__.__name__ == 'SersicDoubleWaveModel' and band == 'r' and 0 == 1:
                from legacyhalos.sersic import SersicSingleWaveModel

                rad_model = np.linspace(0, 200, 150)
                wave_model = np.zeros_like(rad_model) + lam

                model1 = SersicSingleWaveModel(nref=model.nref1.value, r50ref=model.r50ref1.value,
                                               alpha=model.alpha1.value, beta=model.beta1.value,
                                               mu50_g=model.mu50_g1.value, mu50_r=model.mu50_r1.value,
                                               mu50_z=model.mu50_z1.value)
                model2 = SersicSingleWaveModel(nref=model.nref2.value, r50ref=model.r50ref2.value,
                                               alpha=model.alpha2.value, beta=model.beta2.value,
                                               mu50_g=model.mu50_g2.value, mu50_r=model.mu50_r2.value,
                                               mu50_z=model.mu50_z2.value)
                ax.plot(rad_model, 22.5-2.5*np.log10(model1(rad_model, wave_model)),
                        color='gray', alpha=0.5, ls='-.', lw=2)
                ax.plot(rad_model, 22.5-2.5*np.log10(model2(rad_model, wave_model)),
                        color='gray', alpha=0.5, ls='-.', lw=2)
            
    # legend with the best-fitting parameters
    if model is not None:
        chi2 = r'$\chi^2_\nu={:.2f}$'.format(sersic['chi2'])
        lambdaref = '{}'.format(sersic['lambda_ref'])
        if sersic['modeltype'] == 'single':
            if sersic['converged']:
                alpha = '{:.2f}\pm{:.2f}'.format(sersic['alpha'], sersic['alpha_err'])
                beta = '{:.2f}\pm{:.2f}'.format(sersic['beta'], sersic['beta_err'])
                nref = '{:.2f}\pm{:.2f}'.format(sersic['nref'], sersic['nref_err'])
                r50ref = '{:.2f}\pm{:.2f}'.format(sersic['r50ref'], sersic['r50ref_err'])
                n = r'$n(\lambda) = ({nref})(\lambda/{lambdaref})^{{{alpha}}}$'.format(
                    nref=nref, lambdaref=lambdaref, alpha=alpha)
                r50 = r'$r_{{50}}(\lambda) = ({r50ref})(\lambda/{lambdaref})^{{{beta}}}\ arcsec$'.format(
                    r50ref=r50ref, lambdaref=lambdaref, beta=beta)
            else:
                alpha = '{:.2f}'.format(sersic['alpha'])
                beta = '{:.2f}'.format(sersic['beta'])
                nref = '{:.2f}'.format(sersic['nref'])
                r50ref = '{:.2f}'.format(sersic['r50ref'])
                n = r'$n(\lambda) = {nref}\ (\lambda/{lambdaref})^{{{alpha}}}$'.format(
                    nref=nref, lambdaref=lambdaref, alpha=alpha)
                r50 = r'$r_{{50}}(\lambda) = {r50ref}\ (\lambda/{lambdaref})^{{{beta}}}\ arcsec$'.format(
                    r50ref=r50ref, lambdaref=lambdaref, beta=beta)
            txt = chi2+'\n'+n+'\n'+r50
        elif sersic['modeltype'] == 'single-nowavepower':
            alphabeta = r'$\alpha={:.2f},\ \beta={:.2f}$'.format(sersic['alpha'], sersic['beta'])
            if sersic['converged']:
                nref = r'{:.2f}\pm{:.2f}'.format(sersic['nref'], sersic['nref_err'])
                r50ref = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref'], sersic['r50ref_err'])
                n = r'$n = {nref}$'.format(nref=nref)
                r50 = r'$r_{{50}} = {r50ref}\ arcsec$'.format(r50ref=r50ref)
            else:
                nref = r'{:.2f}'.format(sersic['nref'])
                r50ref = r'{:.2f}'.format(sersic['r50ref'])
                n = r'$n = {nref}$'.format(nref=nref)
                r50 = r'$r_{{50}} = {r50ref}\ arcsec$'.format(r50ref=r50ref)
            txt = chi2+'\n'+alphabeta+'\n'+n+'\n'+r50
        elif sersic['modeltype'] == 'exponential':
            if sersic['converged']:
                alpha1 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha1'], sersic['alpha1_err'])
                beta1 = r'{:.2f}\pm{:.2f}'.format(sersic['beta1'], sersic['beta1_err'])
                beta2 = r'{:.2f}\pm{:.2f}'.format(sersic['beta2'], sersic['beta2_err'])
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
                n1 = r'$n_1(\lambda) = ({nref1})(\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2 = {nref2}$'.format(nref2=nref2)
                r50_1 = r'$r_{{50,1}}(\lambda) = ({r50ref1})(\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = ({r50ref2})(\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
            else:
                alpha1 = r'{:.2f}'.format(sersic['alpha1'])
                beta1 = r'{:.2f}'.format(sersic['beta1'])
                beta2 = r'{:.2f}'.format(sersic['beta2'])
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
                n1 = r'$n_1(\lambda) = {nref1}\ (\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2 = {nref2}$'.format(nref2=nref2)
                r50_1 = r'$r_{{50,1}}(\lambda) = {r50ref1}\ (\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = {r50ref2}\ (\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
            txt = chi2+'\n'+n1+'\n'+n2+'\n'+r50_1+'\n'+r50_2
        elif sersic['modeltype'] == 'exponential-nowavepower':
            alpha = r'$\alpha_1={:.2f}$'.format(sersic['alpha1'])
            beta = r'$\beta_1=\beta_2={:.2f}$'.format(sersic['beta1'])
            if sersic['converged']:
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
            else:
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
            n = r'$n_1 = {nref1},\ n_2 = {nref2}$'.format(nref1=nref1, nref2=nref2)
            r50 = r'$r_{{50,1}} = {r50ref1}\ r_{{50,2}} = {r50ref2}\ arcsec$'.format(r50ref1=r50ref1, r50ref2=r50ref2)
            txt = chi2+'\n'+alpha+'\n'+beta+'\n'+n+'\n'+r50
        elif sersic['modeltype'] == 'double':
            if sersic['converged']:
                alpha1 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha1'], sersic['alpha1_err'])
                alpha2 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha2'], sersic['alpha2_err'])
                beta1 = r'{:.2f}\pm{:.2f}'.format(sersic['beta1'], sersic['beta1_err'])
                beta2 = r'{:.2f}\pm{:.2f}'.format(sersic['beta2'], sersic['beta2_err'])
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}\pm{:.2f}'.format(sersic['nref2'], sersic['nref2_err'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
                n1 = r'$n_1(\lambda) = ({nref1})(\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2(\lambda) = ({nref2})(\lambda/{lambdaref})^{{{alpha2}}}$'.format(
                    nref2=nref2, lambdaref=lambdaref, alpha2=alpha2)
                r50_1 = r'$r_{{50,1}}(\lambda) = ({r50ref1})(\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = ({r50ref2})(\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
            else:
                alpha1 = r'{:.2f}'.format(sersic['alpha1'])
                alpha2 = r'{:.2f}'.format(sersic['alpha2'])
                beta1 = r'{:.2f}'.format(sersic['beta1'])
                beta2 = r'{:.2f}'.format(sersic['beta2'])
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
                n1 = r'$n_1(\lambda) = {nref1}\ (\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2(\lambda) = {nref2}\ (\lambda/{lambdaref})^{{{alpha2}}}$'.format(
                    nref2=nref2, lambdaref=lambdaref, alpha2=alpha2)
                r50_1 = r'$r_{{50,1}}(\lambda) = {r50ref1}\ (\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = {r50ref2}\ (\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
            txt = chi2+'\n'+n1+'\n'+n2+'\n'+r50_1+'\n'+r50_2
        elif sersic['modeltype'] == 'double-nowavepower':
            alpha = r'$\alpha_1=\alpha_2={:.2f}$'.format(sersic['alpha1'])
            beta = r'$\beta_1=\beta_2={:.2f}$'.format(sersic['beta1'])
            if sersic['converged']:
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}\pm{:.2f}'.format(sersic['nref2'], sersic['nref2_err'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
            else:
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
            n = r'$n_1 = {nref1},\ n_2 = {nref2}$'.format(nref1=nref1, nref2=nref2)
            r50 = r'$r_{{50,1}} = {r50ref1}\ r_{{50,2}} = {r50ref2}\ arcsec$'.format(r50ref1=r50ref1, r50ref2=r50ref2)
            txt = chi2+'\n'+alpha+'\n'+beta+'\n'+n+'\n'+r50
        elif sersic['modeltype'] == 'triple':
            if sersic['converged']:
                alpha1 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha1'], sersic['alpha1_err'])
                alpha2 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha2'], sersic['alpha2_err'])
                alpha3 = r'{:.2f}\pm{:.2f}'.format(sersic['alpha3'], sersic['alpha3_err'])
                beta1 = r'{:.2f}\pm{:.2f}'.format(sersic['beta1'], sersic['beta1_err'])
                beta2 = r'{:.2f}\pm{:.2f}'.format(sersic['beta2'], sersic['beta2_err'])
                beta3 = r'{:.2f}\pm{:.2f}'.format(sersic['beta3'], sersic['beta3_err'])
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}\pm{:.2f}'.format(sersic['nref2'], sersic['nref2_err'])
                nref3 = r'{:.2f}\pm{:.2f}'.format(sersic['nref3'], sersic['nref3_err'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
                r50ref3 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref3'], sersic['r50ref3_err'])
                n1 = r'$n_1(\lambda) = ({nref1})(\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2(\lambda) = ({nref2})(\lambda/{lambdaref})^{{{alpha2}}}$'.format(
                    nref2=nref2, lambdaref=lambdaref, alpha2=alpha2)
                n3 = r'$n_3(\lambda) = ({nref3})(\lambda/{lambdaref})^{{{alpha3}}}$'.format(
                    nref3=nref3, lambdaref=lambdaref, alpha3=alpha3)
                r50_1 = r'$r_{{50,1}}(\lambda) = ({r50ref1})(\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = ({r50ref2})(\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
                r50_3 = r'$r_{{50,3}}(\lambda) = ({r50ref3})(\lambda/{lambdaref})^{{{beta3}}}\ arcsec$'.format(
                    r50ref3=r50ref3, lambdaref=lambdaref, beta3=beta3)
            else:
                alpha1 = r'{:.2f}'.format(sersic['alpha1'])
                alpha2 = r'{:.2f}'.format(sersic['alpha2'])
                alpha3 = r'{:.2f}'.format(sersic['alpha3'])
                beta1 = r'{:.2f}'.format(sersic['beta1'])
                beta2 = r'{:.2f}'.format(sersic['beta2'])
                beta3 = r'{:.2f}'.format(sersic['beta3'])
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                nref3 = r'{:.2f}'.format(sersic['nref3'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
                r50ref3 = r'{:.2f}'.format(sersic['r50ref3'])
                n1 = r'$n_1(\lambda) = {nref1}\ (\lambda/{lambdaref})^{{{alpha1}}}$'.format(
                    nref1=nref1, lambdaref=lambdaref, alpha1=alpha1)
                n2 = r'$n_2(\lambda) = {nref2}\ (\lambda/{lambdaref})^{{{alpha2}}}$'.format(
                    nref2=nref2, lambdaref=lambdaref, alpha2=alpha2)
                n3 = r'$n_3(\lambda) = {nref3}\ (\lambda/{lambdaref})^{{{alpha3}}}$'.format(
                    nref3=nref3, lambdaref=lambdaref, alpha3=alpha3)
                r50_1 = r'$r_{{50,1}}(\lambda) = {r50ref1}\ (\lambda/{lambdaref})^{{{beta1}}}\ arcsec$'.format(
                    r50ref1=r50ref1, lambdaref=lambdaref, beta1=beta1)
                r50_2 = r'$r_{{50,2}}(\lambda) = {r50ref2}\ (\lambda/{lambdaref})^{{{beta2}}}\ arcsec$'.format(
                    r50ref2=r50ref2, lambdaref=lambdaref, beta2=beta2)
                r50_3 = r'$r_{{50,3}}(\lambda) = {r50ref3}\ (\lambda/{lambdaref})^{{{beta3}}}\ arcsec$'.format(
                    r50ref3=r50ref3, lambdaref=lambdaref, beta3=beta3)
            txt = chi2+'\n'+n1+', '+r50_1+'\n'+n2+', '+r50_2+'\n'+n3+', '+r50_3
            #txt = chi2+'\n'+n1+'\n'+n2+'\n'+n3+'\n'+r50_1+'\n'+r50_2+'\n'+r50_3
        elif sersic['modeltype'] == 'triple-nowavepower':
            alpha = r'$\alpha_1=\alpha_2=\alpha_3={:.2f}$'.format(sersic['alpha1'])
            beta = r'$\beta_1=\beta_2=\beta_3={:.2f}$'.format(sersic['beta1'])
            if sersic['converged']:
                nref1 = r'{:.2f}\pm{:.2f}'.format(sersic['nref1'], sersic['nref1_err'])
                nref2 = r'{:.2f}\pm{:.2f}'.format(sersic['nref2'], sersic['nref2_err'])
                nref3 = r'{:.2f}\pm{:.2f}'.format(sersic['nref3'], sersic['nref3_err'])
                r50ref1 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref1'], sersic['r50ref1_err'])
                r50ref2 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref2'], sersic['r50ref2_err'])
                r50ref3 = r'{:.2f}\pm{:.2f}'.format(sersic['r50ref3'], sersic['r50ref3_err'])
            else:
                nref1 = r'{:.2f}'.format(sersic['nref1'])
                nref2 = r'{:.2f}'.format(sersic['nref2'])
                nref3 = r'{:.2f}'.format(sersic['nref3'])
                r50ref1 = r'{:.2f}'.format(sersic['r50ref1'])
                r50ref2 = r'{:.2f}'.format(sersic['r50ref2'])
                r50ref3 = r'{:.2f}'.format(sersic['r50ref3'])
            n = r'$n_1 = {nref1},\ n_2 = {nref2},\ n_3 = {nref3}$'.format(nref1=nref1, nref2=nref2, nref3=nref3)
            r50 = r'$r_{{50,1}} = {r50ref1},\ r_{{50,2}} = {r50ref2},\ r_{{50,3}} = {r50ref3}\ arcsec$'.format(
                r50ref1=r50ref1, r50ref2=r50ref2, r50ref3=r50ref3)
            txt = chi2+'\n'+alpha+'\n'+beta+'\n'+n+'\n'+r50
                
        ax.text(0.07, 0.04, txt, ha='left', va='bottom', linespacing=1.3,
                transform=ax.transAxes, fontsize=10)

    ax.set_xlabel(r'Galactocentric radius $r^{1/4}$ (arcsec)')
    ax.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
    #ax.set_ylabel(r'Surface Brightness $\mu$ (mag arcsec$^{-2}$)')

    ylim = [ymnmax[0]-1.5, ymnmax[1]+0.5]
    if sersic['modeltype'] == 'triple':
        ylim[0] = ylim[0] - 1.0
        ylim[1] = ylim[1] + 2.0
    #if ylim[1] > 33:
    #    ylim[1] = 33

    ax.set_ylim(ylim)
    ax.invert_yaxis()
    #ax.margins()
    ax.margins(ymargins=0)
    #ax.set_yscale('log')

    ax.set_xlim(0, rad.max()*1.05)
    #ax.set_xlim(xmin=0)

    ax2 = ax.twiny()
    xlim = ax.get_xlim()
    ax2.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
    ax2.set_xlabel('Galactocentric radius (kpc)')

    ax.legend(loc='upper right', fontsize=labelfont)

    ylim = ax.get_ylim()
    if sersic['success']:
        ax.fill_between([0, 3*model.psfsize_r], [ylim[0], ylim[0]], # [arcsec]
                        [ylim[1], ylim[1]], facecolor='grey', alpha=0.1)
        ax.text(0.02, 0.07, 'PSF\n(3$\sigma$)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

    fig.subplots_adjust(bottom=0.15, top=0.85, right=0.95, left=0.17)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)#, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def display_multiband(data, ellipsefit=None, colorimg=None, indx=None,
                      igal=0, inchperband=8, contours=False, barlen=None,
                      barlabel=None, png=None, verbose=True, vertical=False,
                      scaledfont=False, galex=False, unwise=False):
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    vertical -- for talks...

    """
    import numpy.ma as ma
    from PIL import Image, ImageDraw, ImageFont
    from photutils import EllipticalAperture
    from photutils.isophote import EllipseSample, Isophote
    import matplotlib.patches as mpatches

    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize
    from astropy.visualization import ZScaleInterval as Interval
    #from astropy.visualization import PercentileInterval as Interval

    Image.MAX_IMAGE_PIXELS = None

    # stupidly fragile
    #bands = data['bands']
    
    # handle GALEX and WISE
    refpixscale = data['refpixscale']
    def _get_pixscalefactor(filt):
        if 'filt2pixscale' in data.keys():
            pixscale = data['filt2pixscale'][filt]            
            if np.isclose(pixscale, refpixscale): # avoid rounding issues
                pixscale = refpixscale
                pixscalefactor = 1.0
            else:
                pixscalefactor = refpixscale / pixscale
        else:
            pixscalefactor = 1.0
        return pixscalefactor
                
    if galex:
        bands = ['FUV', 'NUV']
    elif unwise:
        bands = ['W1', 'W2', 'W3', 'W4']
    else:
        bands = ['g', 'r', 'z']
    nband = len(bands)

    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    #cmap = plt.cm.plasma
    #cmap = plt.cm.cividis
    cmap = plt.cm.inferno
    #cmap = plt.cm.viridis
    stretch = Stretch(a=0.9)
    interval = Interval(contrast=0.5, nsamples=10000)

    #cmap = 'RdBu_r'
    #cmap = {'g': 'winter_r', 'r': 'summer', 'z': 'autumn_r'}
    #cmap = {'g': 'Blues', 'r': 'Greens', 'z': 'Reds'}

    # Remove ahead of time to play nice with Imagemagick thumbnail-making, later.
    if png and os.path.isfile(png):
        os.remove(png)
            
    if vertical:
        fig, ax = plt.subplots(nband+1, 1, figsize=(inchperband, inchperband*(nband+1)))
    else:
        fig, ax = plt.subplots(1, nband+1, figsize=(inchperband*(nband+1), inchperband))

     # First display the color mosaic...
    if ellipsefit and ellipsefit['success']:
        sz = colorimg.size
        if False:
            if 'd25_leda' in ellipsefit.keys():
                draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], sz[1]-ellipsefit['y0_moment'],
                                    ellipsefit['ba_leda'], ellipsefit['pa_leda'], ellipsefit['d25_leda'] * 60.0,
                                    ellipsefit['refpixscale'], color=cb_colors['red']) # '#3388ff')        
            draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], sz[1]-ellipsefit['y0_moment'], 1-ellipsefit['eps_moment'],
                                ellipsefit['pa_moment'], 2 * ellipsefit['majoraxis'] * ellipsefit['refpixscale'],
                                ellipsefit['refpixscale'], color=cb_colors['green']) # '#ffaa33')

        if 'radius_sb26' in ellipsefit.keys(): # old data model
            if ellipsefit['radius_sb26'] > 0:
                sbr = ellipsefit['radius_sb26']
            elif ellipsefit['radius_sb25'] > 0:
                sbr = ellipsefit['radius_sb25'] * 1.2
            else:
                sbr = -1
        else:
            if ellipsefit['sma_sb26'] > 0:
                sbr = ellipsefit['sma_sb26']
            elif ellipsefit['sma_sb25'] > 0:
                sbr = ellipsefit['sma_sb25'] * 1.2
            else:
                sbr = -1
            
        if sbr > 0:
            draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], sz[1]-ellipsefit['y0_moment'],
                                1-ellipsefit['eps_moment'],
                                ellipsefit['pa_moment'], 2 * sbr, ellipsefit['refpixscale'],
                                color=cb_colors['blue'])

        pixscalefactor = _get_pixscalefactor(bands[0]) # not smart!

        draw = ImageDraw.Draw(colorimg)
        if barlen and barlabel and galex is False and unwise is False:
            width = np.round(pixscalefactor*sz[0]/150).astype('int')
            if scaledfont:
                fntsize = np.round(pixscalefactor*sz[0]/50).astype('int')
            else:
                fntsize = 20 # np.round(sz[0]/20).astype('int')
            font = ImageFont.truetype(fonttype, size=fntsize)
            # Add a scale bar and label--
            x0, x1, y0, y1 = 0+fntsize*2, 0+fntsize*2+barlen*pixscalefactor, sz[1]-fntsize*2, sz[1]-fntsize*2.5#4
            #print(sz, fntsize, x0, x1, y0, y1, barlen*pixscalefactor)            
            draw.line((x0, y1, x1, y1), fill='white', width=width)
            ww, hh = draw.textsize(barlabel, font=font)
            dx = ((x1-x0) - ww)//2
            #print(x0, x1, y0, y1, ww, x0+dx, sz)
            draw.text((x0+dx, y0), barlabel, font=font)
            
        ax[0].imshow(colorimg, interpolation='nearest')
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].axis('off')
        ax[0].autoscale(False)

    # ...now the individual bandpasses.        
    for ii, (filt, ax1) in enumerate(zip(bands, ax[1:])):

        pixscalefactor = _get_pixscalefactor(filt)

        #mge = data['mge'][igal]
        dat = data['{}_masked'.format(filt.lower())][igal]

        #if filt == 'W4':
        #    plt.clf() ; plt.imshow(dat.data, origin='lower') ; plt.savefig('junk.png')
        #    pdb.set_trace()

        # There's an annoying bug in matplotlib>2.0.2 which ignores masked
        # pixels (it used to render them in white), so we have to overplot the
        # mask.
        # https://github.com/matplotlib/matplotlib/issues/11039
        # https://stackoverflow.com/questions/22128166/two-different-color-colormaps-in-the-same-imshow-matplotlib
        #cmap.set_bad('white', alpha=1.0) # doesn't work!

        if False:
            img = ma.masked_array(dat.data, dat.mask)
            mask = ma.masked_array(dat.data, ~dat.mask)
            try:
                norm = ImageNormalize(img, interval=interval, stretch=stretch)
            except:
                norm = ImageNormalize(img, interval=interval)
                #pdb.set_trace()

            ax1.imshow(img, origin='lower', norm=norm, cmap=cmap, #cmap=cmap[filt],
                       interpolation='none')
            ax1.imshow(mask, origin='lower', cmap=mpl.colors.ListedColormap(['white']),
                       interpolation='none')
        else:
            if np.sum(dat) == 0:
                ax1.imshow(np.zeros(shape=dat.shape), origin='lower', interpolation='nearest')
            else:
                try:
                    norm = ImageNormalize(dat, interval=interval, stretch=stretch)
                except:
                    norm = ImageNormalize(dat, interval=interval)
                ax1.imshow(dat, origin='lower', norm=norm, cmap=cmap, #cmap=cmap[filt],
                           interpolation='nearest')
        plt.text(0.09, 0.9, filt, transform=ax1.transAxes, fontweight='bold',
                 ha='left', va='center', color='k', fontsize=34)

        # Add a scale bar and label
        if barlen and ii == 0 and False:
            sz = img.shape
            x0, y0 = sz[0]*0.08, sz[0]*0.05
            x1, y1 = x0 + barlen, y0*3
            ax1.plot([x0, x1], [y1, y1], lw=3, color='k')
            ax1.text(x0+barlen/2, y0, barlabel, ha='center', va='center',
                     transform=None, fontproperties=prop)

        #if geometry:
        #    ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
        #                                 geometry.sma*(1 - geometry.eps), geometry.pa)
        #    ellaper.plot(color='k', lw=1, ax=ax1, alpha=0.75)

        if ellipsefit and ellipsefit['success'] and np.atleast_1d(ellipsefit['sma_{}'.format(filt.lower())])[0] != -1:
            #nfit = len(ellipsefit['{}_sma'.format(filt.lower())])
            #nplot = np.rint(0.01*nfit).astype('int')
            nplot = 9
            sma_lw = 3
            sma_alpha = 1.0
            #if nplot < 10:
            #    nplot = 10
            #    sma_lw = 3
            #    sma_alpha = 1.0
            #smas = np.linspace(0, ellipsefit['{}_sma'.format(filt.lower())][indx].max(), nplot)
            if len(ellipsefit['sma_{}'.format(filt.lower())]) > nplot:
                smas = ellipsefit['sma_{}'.format(filt.lower())][::len(ellipsefit['sma_{}'.format(filt.lower())]) // nplot]
            else:
                smas = ellipsefit['sma_{}'.format(filt.lower())]

            # When we used to write out the ellipse pickle files with
            # the Isophote objects we used the snippet of code below to
            # render the fitted ellipses.  Now, just draw the ellipse
            # shapes.
            #for sma in smas:
            #    efit = ellipsefit[filt].get_closest(sma)
            #    x, y, = efit.sampled_coordinates()
            #    ax1.plot(x, y, color='k', lw=1, alpha=0.5)#, label='Fitted isophote')
            #x0, y0, eps, pa = mge['x0'], mge['y0'], mge['eps'], mge['pa']
            for sma in smas:
                this = np.argmin(np.abs(ellipsefit['sma_{}'.format(filt.lower())]-sma))
                ax1.add_patch(mpatches.Ellipse((ellipsefit['x0_{}'.format(filt.lower())][this], ellipsefit['y0_{}'.format(filt.lower())][this]),
                                               2*ellipsefit['sma_{}'.format(filt.lower())][this],
                                               2*ellipsefit['sma_{}'.format(filt.lower())][this]*(1-ellipsefit['eps_{}'.format(filt.lower())][this]),
                                               ellipsefit['pa_{}'.format(filt.lower())][this]-90,
                                               color='k', lw=sma_lw, alpha=sma_alpha, fill=False))#, label='Fitted isophote')

            # Visualize the mean geometry
            maxis = ellipsefit['majoraxis'] # [pixels]
            ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                         maxis, maxis*(1 - ellipsefit['eps_moment']),
                                         np.radians(ellipsefit['pa_moment']-90))
            if False:
                ellaper.plot(lw=5, axes=ax1, alpha=1.0, label='Moment geometry',
                             color=cb_colors['green'])

            # Visualize the ellipse-fitted geometry
            maxis = sbr / ellipsefit['refpixscale'] # [pixels]
            if maxis > 0:# and galex is False and unwise is False:
                ellaper = EllipticalAperture((ellipsefit['x0_moment']*pixscalefactor, ellipsefit['y0_moment']*pixscalefactor),
                                             maxis*pixscalefactor, maxis*(1 - ellipsefit['eps_moment'])*pixscalefactor,
                                             np.radians(ellipsefit['pa_moment']-90))
                #ellaper.plot(color=cb_colors['blue'], lw=5, axes=ax1, alpha=1.0, label='Ellipse geometry')
                ellaper.plot(color=cb_colors['blue'], lw=5, axes=ax1, alpha=1.0, label='R(26)')

            # Visualize the LSLGA geometry, if present.
            if ('pa_leda' in ellipsefit.keys()) * ('ba_leda' in ellipsefit.keys()) * ('d25_leda' in ellipsefit.keys()):
                maxis = ellipsefit['d25_leda'] * 60 / ellipsefit['refpixscale'] / 2 # [pixels]
                ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                             maxis, maxis * ellipsefit['ba_leda'],
                                             np.radians(ellipsefit['pa_leda']-90))
                if False:
                    ellaper.plot(color=cb_colors['red'], lw=5, axes=ax1, alpha=1.0, label='Hyperleda geometry')
            #pdb.set_trace()
            ## Visualize the fitted geometry
            #maxis = mge['majoraxis'] * 1.2
            #ellaper = EllipticalAperture((x0, y0), maxis, maxis*(1 - eps), pa)
            #ellaper.plot(color='k', lw=2, ax=ax1, alpha=1.0, label='Fitted geometry')

            # Visualize the input geometry
            if ellipsefit['input_ellipse']:
                if False:
                    geometry = ellipsefit['geometry']
                    #maxis = geometry.sma
                    maxis = geometry.sma * 0.8
                    ellaper = EllipticalAperture((geometry.x0, geometry.y0), maxis,
                                                 maxis*(1 - geometry.eps), geometry.pa)
                    ellaper.plot(color='navy', lw=2, axes=ax1, alpha=1.0, label='Input geometry')

            if ii == nband-1:
                fntsize = 20
                #fntsize = np.round(0.01*img.shape[0]).astype('int')
                #if fntsize < 20:
                #    fntsize = 20
                #print('Font size {}'.format(fntsize))
                hh, ll = ax1.get_legend_handles_labels()
                if len(hh) > 0:
                    ax1.legend(loc='lower right', fontsize=fntsize, frameon=True,
                               borderaxespad=1.0)
        else:
            pass
            #from photutils import EllipticalAperture
            #geometry = ellipsefit['geometry']
            #ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
            #                             geometry.sma*(1 - geometry.eps), geometry.pa)
            #ellaper.plot(color='k', lw=1, axes=ax1, alpha=0.5)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        #ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    if vertical:
        fig.subplots_adjust(hspace=0.01, top=0.99, bottom=0.01, left=0.01, right=0.99)
    else:
        fig.subplots_adjust(wspace=0.01, top=0.99, bottom=0.01, left=0.01, right=0.99)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight')#, dpi=72)#, pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def display_ellipsefit(ellipsefit, xlog=False, cosmo=None, png=None, verbose=True):
    """Display the isophote fitting results."""

    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    try:
        colors = iter(sns.color_palette())
    except:
        colors = _sbprofile_colors()

    if ellipsefit['success']:
        
        band, refband = ellipsefit['bands'], ellipsefit['refband']
        refpixscale, redshift = ellipsefit['refpixscale'], ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
        
        good = (ellipsefit[refband]['stop_code'] < 4)
        bad = ~good
        ax1.fill_between(ellipsefit[refband]['sma'][good] * refpixscale,
                         ellipsefit[refband]['eps_moment'][good]-ellipsefit[refband]['eps_moment_err'][good],
                         ellipsefit[refband]['eps_moment'][good]+ellipsefit[refband]['eps_moment_err'][good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax1.scatter(ellipsefit[refband]['sma'][bad] * refpixscale,
                        ellipsefit[refband]['eps_moment'][bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)

        #ax1.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             ellipsefit[refband].eps[good],
        #             ellipsefit[refband].ellip_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax1.set_ylim(0, 0.5)
        ax1.xaxis.set_major_formatter(ScalarFormatter())

        ax2.fill_between(ellipsefit[refband]['sma'][good] * refpixscale, 
                         np.degrees(ellipsefit[refband]['pa_moment'][good]-ellipsefit[refband]['pa_moment_err'][good]),
                         np.degrees(ellipsefit[refband]['pa_moment'][good]+ellipsefit[refband]['pa_moment_err'][good]))#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax2.scatter(ellipsefit[refband]['sma'][bad] * refpixscale,
                        np.degrees(ellipsefit[refband]['pa_moment'][bad]),
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax2.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             np.degrees(ellipsefit[refband].pa[good]),
        #             np.degrees(ellipsefit[refband].pa_err[good]), fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax2.set_ylim(0, 180)

        ax3.fill_between(ellipsefit[refband]['sma'][good] * refpixscale,
                         ellipsefit[refband]['x0_moment'][good]-ellipsefit[refband]['x0_moment_err'][good],
                         ellipsefit[refband]['x0_moment'][good]+ellipsefit[refband]['x0_moment_err'][good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax3.scatter(ellipsefit[refband]['sma'][bad] * refpixscale,
                        ellipsefit[refband]['x0_moment'][bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax3.errorbar(ellipsefit[refband].sma[good] * smascale, ellipsefit[refband].x0[good],
        #             ellipsefit[refband].x0_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        ax3.xaxis.set_major_formatter(ScalarFormatter())
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax4.fill_between(ellipsefit[refband]['sma'][good] * refpixscale, 
                         ellipsefit[refband]['y0_moment'][good]-ellipsefit[refband]['y0_moment_err'][good],
                         ellipsefit[refband]['y0_moment'][good]+ellipsefit[refband]['y0_moment_err'][good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax4.scatter(ellipsefit[refband]['sma'][bad] * refpixscale,
                        ellipsefit[refband]['y0_moment'][bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax4.errorbar(ellipsefit[refband].sma[good] * smascale, ellipsefit[refband].y0[good],
        #             ellipsefit[refband].y0_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
            
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right')
        ax4.xaxis.set_major_formatter(ScalarFormatter())
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xlim(left=0)
        
        xlim = ax1.get_xlim()
        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax1_twin.set_xlabel('Galactocentric radius (kpc)')

        ax2_twin = ax2.twiny()
        ax2_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax2_twin.set_xlabel('Galactocentric radius (kpc)')

        ax1.set_ylabel(r'Ellipticity $\epsilon$')
        ax2.set_ylabel('Position Angle (deg)')
        ax3.set_xlabel(r'Galactocentric radius $r^{1/4}$ (arcsec)')
        ax3.set_ylabel(r'$x$ Center')
        ax4.set_xlabel(r'Galactocentric radius $r^{1/4}$ (arcsec)')
        ax4.set_ylabel(r'$y$ Center')

        if xlog:
            for xx in (ax1, ax2, ax3, ax4):
                xx.set_xscale('log')

        fig.subplots_adjust(hspace=0.03, wspace=0.03, bottom=0.15, right=0.85, left=0.15)

        if png:
            #if verbose:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
        
def display_ellipse_sbprofile(ellipsefit, pipeline_ellipsefit={}, sky_ellipsefit={}, 
                              sdss_ellipsefit={}, minerr=0.0, plot_radius=True,
                              plot_sbradii=False, cosmo=None, png=None, use_ylim=None,
                              verbose=True):
    """Display the multi-band surface brightness profile.

    2-panel

    """
    import matplotlib.ticker as ticker
    import astropy.stats
    from legacyhalos.ellipse import ellipse_sbprofile

    isdict = type(ellipsefit) is dict

    if ellipsefit['success'] and np.atleast_1d(ellipsefit['sma_r'])[0] != -1:

        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr, sma_not_radius=~plot_radius)

        if isdict:
            bands = ellipsefit['bands']
            refband = ellipsefit['refband']
            if 'redshift' in ellipsefit.keys():
                redshift = ellipsefit['redshift']
                radscale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
            else:
                redshift = None
        else:
            bands = ellipsefit['bands'].tolist()[0]
            refband = ellipsefit['refband'][0]
            if 'refpixscale' in ellipsefit.colnames:
                pixscale = ellipsefit['refpixscale'][0]
            else:
                pixscale = ellipsefit['pixscale'][0]
            if 'redshift' in ellipsefit.colnames:
                sbprofile['redshift'] = ellipsefit['redshift'][0]
                radscale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
            else:
                redshift = None

        galex = 'FUV' in bands
        unwise = 'W1' in bands
        colors = _sbprofile_colors(galex=galex, unwise=unwise)

        yminmax = [40, 0]
        xminmax = [0.9, 0]

        if galex and unwise:
            fig, ax1 = plt.subplots(figsize=(9, 7))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw = {'height_ratios':[2, 1]})
            
        for filt in bands:
            col = next(colors)
            
            mu = sbprofile['mu_{}'.format(filt.lower())]
            muerr = sbprofile['muerr_{}'.format(filt.lower())]
            if len(mu) == 0 or mu[0] == -1: # no good data
                continue
            radius = sbprofile['radius_{}'.format(filt.lower())]**0.25

            #good = (ellipsefit[filt].stop_code < 4)
            #bad = ~good
            
            #with np.errstate(invalid='ignore'):
            #    good = np.isfinite(mu) * (mu / muerr > 3)
            #good = np.isfinite(mu)
            #if np.sum(good) == 0:
            #    continue
            #sma = sma[good]
            #mu = mu[good]
            #muerr = muerr[good]
            
            ax1.fill_between(radius, mu-muerr, mu+muerr, label=r'${}$'.format(filt),
                             facecolor=col, edgecolor='k', lw=1, alpha=0.9)

            if bool(pipeline_ellipsefit) and False:
                pipeline_sbprofile = ellipse_sbprofile(pipeline_ellipsefit, minerr=minerr,
                                                       sma_not_radius=plot_radius)
                _radius = pipeline_sbprofile['radius_{}'.format(filt.lower())]**0.25
                _mu = pipeline_sbprofile['mu_{}'.format(filt.lower())]
                _muerr = pipeline_sbprofile['mu_{}_err'.format(filt.lower())]
                #ax1.plot(radius, mu, color='k', alpha=0.5)
                ax1.fill_between(_radius, _mu-_muerr, _mu+_muerr, color=col,
                                 alpha=0.2, edgecolor='k', lw=3)
                
            if bool(sky_ellipsefit):
                print('Fix me')
                skyradius = sky_ellipsefit['radius'] * ellipsefit['refpixscale']

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sky = astropy.stats.mad_std(sky_ellipsefit[filt], axis=1, ignore_nan=True)
                    # sky = np.nanstd(skyellipsefit[filt], axis=1) # / np.sqrt(skyellipsefit[
                    
                skygood = np.isfinite(sky)
                skyradius = skyradius[skygood]
                skymu = 22.5 - 2.5 * np.log10(sky[skygood])
                ax1.plot( skyradius, skymu , color=col, ls='--', alpha=0.5)

            if np.nanmin(mu-muerr) < yminmax[0]:
                yminmax[0] = np.nanmin(mu-muerr)
            if np.nanmax(mu+muerr) > yminmax[1]:
                yminmax[1] = np.nanmax(mu+muerr)
            if np.nanmax(radius) > xminmax[1]:
                xminmax[1] = np.nanmax(radius)

            #ax1.axhline(y=ellipsefit['mu_{}_sky'.format(filt.lower())], color=col, ls='--')
            #if filt == refband:
            #    ysky = ellipsefit['mu_{}_sky'.format(filt.lower())] - 2.5 * np.log10(0.1) # 10% of sky
            #    ax1.axhline(y=ysky, color=col, ls='--')

        if bool(sdss_ellipsefit):
            sdss_sbprofile = ellipse_sbprofile(sdss_ellipsefit, minerr=minerr, sma_not_radius=plot_radius)
            for filt in sdss_ellipsefit['bands']:
                radius = sdss_sbprofile['radius_{}'.format(filt.lower())]**0.25
                mu = sdss_sbprofile['mu_{}'.format(filt.lower())]
                muerr = sdss_sbprofile['mu_{}_err'.format(filt.lower())]
                #ax1.plot(radius, mu, color='k', alpha=0.5)
                ax1.fill_between(radius, mu-muerr, mu+muerr, label=r'${}$'.format(filt.lower()),
                                 facecolor='k', alpha=0.2, edgecolor='k', lw=3)
                
        ax1.set_ylabel(r'$\mu(r)$ (mag arcsec$^{-2}$)')
        #ax1.set_ylabel(r'Surface Brightness $\mu(a)$ (mag arcsec$^{-2}$)')

        ylim = [yminmax[0]-0.75, yminmax[1]+0.5]
        if ylim[0] < 13:
            ylim[0] = 13
        if ylim[1] > 34:
            ylim[1] = 34

        if use_ylim is not None:
            ax1.set_ylim(use_ylim)
        else:
            ax1.set_ylim(ylim)
        ax1.invert_yaxis()

        ylim = ax1.get_ylim()        
        if np.abs(ylim[1]-ylim[0]) > 15:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))

        #ylim = ax1.get_ylim()        
        #if np.abs(ylim[1]-ylim[0]) > 15:
        #    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5)) # wavelength spacing of ticks [Angstrom]

        xlim = [xminmax[0], xminmax[1]*1.01]
        ax1.set_xlim(xlim)
        #ax1.set_xlim(xmin=0)
        #ax1.margins(xmargin=0)

        if redshift:
            ax1_twin = ax1.twiny()
            ax1_twin.set_xlim(xlim)
            #xlim_twinx = (radscale*xlim[0]**4, radscale*xlim[1]**4)
            #ax1_twin.set_xlim(xlim_twinx[0], xlim_twinx[1])
            #ax1_twin.set_xticks(ax1.get_xticks()**4*radscale)
            #ax1_twin.set_xticks(np.linspace(xlim_twinx[0], xlim_twinx[1], len(ax1.get_xticks())+1))
            kpc = np.array([1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200])
            #kpc = kpc[kpc < radscale*xlim[1]**4]
            kpc = kpc[(kpc >= radscale*xlim[0]**4) * (kpc <= radscale*xlim[1]**4)]
            ax1_twin.set_xticks((kpc / radscale)**0.25)
            ax1_twin.set_xticklabels(['{:g}'.format(kk) for kk in kpc])
            if plot_radius:
                ax1_twin.set_xlabel(r'Galactocentric radius (kpc)')
            else:
                ax1_twin.set_xlabel(r'Semi-major axis (kpc)')

            #ax1.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
            #ax1.set_ylim(31.99, 18)

        hh, ll = ax1.get_legend_handles_labels()
        if len(hh) > 0:
            if len(hh) > 3:
                ncol = 2
            else:
                ncol = 1
            leg1 = ax1.legend(loc='upper right', fontsize=14, ncol=ncol)

        # Plot some threshold radii for the large-galaxy project--
        if plot_sbradii:
            lline, llabel = [], []
            if ellipsefit['radius_sb24'] > 0:
                rr = (ellipsefit['radius_sb24'])**0.25
                ll, = ax1.plot([rr, rr], [ylim[1], 24], lw=2, color='k', ls='-.')
                lline.append(ll), llabel.append('R(24)')
                
            if ellipsefit['radius_sb25'] > 0:
                rr = (ellipsefit['radius_sb25'])**0.25
                #ax1.plot([xlim[0], rr], [25, 25], lw=2, color='k', ls='-')
                ll, = ax1.plot([rr, rr], [ylim[1], 25], lw=2, color='k', ls='--')
                lline.append(ll), llabel.append('R(25)')
                
            if ellipsefit['radius_sb26'] > 0:
                rr = (ellipsefit['radius_sb26'])**0.25
                #ax1.plot([xlim[0], rr], [26, 26], lw=2, color='k', ls='--')
                ll, = ax1.plot([rr, rr], [ylim[1], 26], lw=2, color='k', ls='-')
                lline.append(ll), llabel.append('R(26)')

            if False:
                rr = (ellipsefit['majoraxis'] * ellipsefit['refpixscale'])**0.25
                ll, = ax1.plot([rr, rr], [ylim[1], ylim[0]], lw=2, color='#e41a1c', ls='dotted')
                #ll, = ax1.plot([rr, rr], [ylim[1], sbprofile['mu_{}'.format(refband)][0]], lw=2, color='red', ls='dotted')
                #ll = ax1.axvline(x=rr, lw=2, color='red', ls='dotted')
                lline.append(ll), llabel.append('Moment Size')

            if len(lline) > 0:
                leg2 = ax1.legend(lline, llabel, loc='lower left', frameon=False, fontsize=14)
                ax1.add_artist(leg1)
            
        # Now the color-radius plot
        if galex and unwise:
            if plot_radius:
                ax1.set_xlabel(r'(Galactocentric radius / arcsec)$^{1/4}$')
            else:
                ax1.set_xlabel(r'(Semi-major axis $r$ / arcsec)$^{1/4}$')
            #ax1.set_xlabel(r'Galactocentric radius $r^{1/4}$ (arcsec)')
            
            #ax1.text(0.07, 0.1, 'PSF\n(2$\sigma$)', ha='center', va='center',
            #         transform=ax1.transAxes, fontsize=10)
            #ax1.fill_between([0, (2*ellipsefit['psfsize_r'])**0.25],
            #                [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='grey', alpha=0.1)
        else:
            if len(sbprofile['radius_gr']) > 0 and sbprofile['radius_gr'][0] != -1:
                ax2.fill_between(sbprofile['radius_gr']**0.25,
                                 sbprofile['gr'] - sbprofile['gr_err'],
                                 sbprofile['gr'] + sbprofile['gr_err'],
                                 label=r'$g - r$', facecolor=next(colors), alpha=0.75,
                                 edgecolor='k', lw=2)

            if len(sbprofile['radius_rz']) > 0 and sbprofile['radius_rz'][0] != -1:
                ax2.fill_between(sbprofile['radius_rz']**0.25,
                                 sbprofile['rz'] - sbprofile['rz_err'],
                                 sbprofile['rz'] + sbprofile['rz_err'],
                                 label=r'$r - z$', facecolor=next(colors), alpha=0.75,
                                 edgecolor='k', lw=2)

            if plot_radius:
                ax2.set_xlabel(r'(Galactocentric radius / arcsec)$^{1/4}$')
            else:
                ax2.set_xlabel(r'(Semi-major axis $r$ / arcsec)$^{1/4}$')
            #ax2.set_xlabel(r'Galactocentric radius $r^{1/4}$ (arcsec)')

            hh, ll = ax2.get_legend_handles_labels()
            if len(hh) > 0:
                ax2.legend(loc='upper right', fontsize=14)
                #ax2.legend(bbox_to_anchor=(0.25, 0.98))

            ax2.set_ylabel('Color (mag)')
            ax2.set_ylim(-0.5, 3)
            ax2.set_xlim(xlim)
            ax2.autoscale(False) # do not scale further

            for xx in (ax1, ax2):
                ylim = xx.get_ylim()
                xx.fill_between([0, (2*ellipsefit['psfsize_r'])**0.25],
                                [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='grey', alpha=0.1)
            
            ax1.text(0.07, 0.1, 'PSF\n(2$\sigma$)', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=10)
            #ax2.text(0.03, 0.1, 'PSF\n(3$\sigma$)', ha='center', va='center',
            #    transform=ax2.transAxes, fontsize=10)

        if redshift:
            fig.subplots_adjust(hspace=0.0, left=0.15, bottom=0.15, top=0.85, right=0.95)
        else:
            fig.subplots_adjust(hspace=0.0, left=0.15, bottom=0.15, top=0.95, right=0.95)

        if png:
            #if verbose:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
        
def sample_trends(sample, htmldir, analysisdir=None, verbose=True, cosmo=None, xlim=(0, 100)):
    """Trends with the whole sample.

    """
    from astropy.cosmology import WMAP9 as cosmo
    from legacyhalos.io import read_ellipsefit
    from legacyhalos.ellipse import ellipse_sbprofile
    from legacyhalos.misc import statsinbins

    trendsdir = os.path.join(htmldir, 'trends')
    if not os.path.isdir(trendsdir):
        os.makedirs(trendsdir, exist_ok=True)

    ngal = len(sample)
    if ngal < 3:
        return

    # color vs semi-major axis
    def __color_vs_sma(color, label):

        # read all the fits / data
        allsma, allgood, allcolor, allcolorerr = [], [], [], []
        smamax, nsma, refindx = 0.0, 0, -1
        
        for ii, gal in enumerate(sample):
            objid, objdir = get_objid(gal, analysisdir=analysisdir)
            ellipsefit = read_ellipsefit(objid, objdir)
            if len(ellipsefit) > 0:
                if ellipsefit['success']:                    
                    refband, redshift = ellipsefit['refband'], ellipsefit['redshift']
                    smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
                    sbprofile = ellipse_sbprofile(ellipsefit, minerr=0.01)

                    sma = sbprofile['sma'] * smascale
                    if sma.max() > smamax:
                        refindx = ii
                        nsma = len(sma)
                        smamax = sma.max()
                    
                    allsma.append( sma )
                    #good.append( (ellipsefit[refband].stop_code < 4) )
                    allgood.append( np.arange( len(ellipsefit[refband].sma) ) )
                    allcolor.append( sbprofile[color] )
                    allcolorerr.append( sbprofile['{}_err'.format(color)] )
                else:
                    allsma.append([]), allgood.append([]), allcolor.append([]), allcolorerr.append([])
            else:
                allsma.append([]), allgood.append([]), allcolor.append([]), allcolorerr.append([])

        # get the median and interquartile trend
        color_stats = statsinbins(np.hstack(allsma), np.hstack(allcolor), 3, minpts=5)

        if False:
            refsma = allsma[refindx] # reference semimajor axis
            allcolor_interp = np.zeros( (ngal, len(refsma)) ) * np.nan
            for ii in range(ngal):
                if len(allsma[ii]) > 0:
                    allcolor_interp[ii, :] = np.interp(refsma, allsma[ii], allcolor[ii],
                                                       left=np.nan, right=np.nan)
            color_trend = np.nanpercentile(allcolor_interp, [25, 50, 75], axis=0)

        # now make the plot
        png = os.path.join(trendsdir, '{}_vs_sma.png'.format(color))
        fig, ax1 = plt.subplots()
        for ii, gal in enumerate(sample):
            if len(allsma[ii]) > 0:
                thisgood = allgood[ii]
                thissma = allsma[ii][thisgood]
                thiscolor = allcolor[ii][thisgood]
                thiscolorerr = allcolorerr[ii][thisgood]
                
                ax1.fill_between(thissma, thiscolor-thiscolorerr, thiscolor+thiscolorerr,
                                 alpha=0.1, color='gray')

        ax1.plot(color_stats['xmedian'], color_stats['ymedian'], color=sns.xkcd_rgb['blood red'], lw=2, ls='-')
        ax1.plot(color_stats['xmedian'], color_stats['y25'], color=sns.xkcd_rgb['blood red'], lw=2, ls='--')
        ax1.plot(color_stats['xmedian'], color_stats['y75'], color=sns.xkcd_rgb['blood red'], lw=2, ls='--')

        ax1.grid()
        ax1.set_xlim(xlim)
        ax1.set_ylim(0, 2.5)
        ax1.set_ylabel(r'{}'.format(label))
        ax1.set_xlabel('Galactocentric radius $r^{1/4}$ (kpc)')

        fig.subplots_adjust(bottom=0.15, right=0.95, left=0.15, top=0.95)

        if png:
            #if verbose:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
            
    def _color_vs_sma():
        __color_vs_sma('gr', '$g - r$')
        __color_vs_sma('rz', '$r - z$')
        
    # Ellipticity vs semi-major axis
    def _ellipticity_vs_sma():
        
        png = os.path.join(trendsdir, 'ellipticity_vs_sma.png')
    
        fig, ax1 = plt.subplots()
        for gal in sample:
            objid, objdir = get_objid(gal, analysisdir=analysisdir)

            ellipsefit = read_ellipsefit(objid, objdir)
            if len(ellipsefit) > 0:
                if ellipsefit['success']:
                    refband, redshift = ellipsefit['refband'], ellipsefit['redshift']
                    smascale = ellipsefit['refpixscale'] * legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/pixel]
                    
                    good = (ellipsefit[refband].stop_code < 4)
                    #good = np.arange( len(ellipsefit[refband].sma) )
                    ax1.fill_between(ellipsefit[refband].sma[good] * smascale, 
                                     ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                                     ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good],
                                     alpha=0.6, color='gray')

        ax1.grid()
        ax1.set_xlim(xlim)
        ax1.set_ylim(0, 0.5)
        ax1.set_ylabel('Ellipticity')
        ax1.set_xlabel('Galactocentric radius $r$ (kpc)')

        fig.subplots_adjust(bottom=0.15, right=0.95, left=0.15, top=0.95)

        if png:
            #if verbose:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()

    # Build all the plots here.
    
    _color_vs_sma()       # color vs semi-major axis
    _ellipticity_vs_sma() # ellipticity vs semi-major axis

def display_ccdpos(onegal, ccds, radius, grzfile, pixscale=0.262, png=None, verbose=False):
    """Visualize the position of all the CCDs contributing to the image stack of a
    single galaxy.

    radius in pixels

    """
    wcs_clust = legacyhalos.misc.simple_wcs(onegal, factor=1.0, radius=radius, pixscale=pixscale)
    wcs = legacyhalos.misc.simple_wcs(onegal, factor=15, radius=radius, pixscale=pixscale)
    width, height = wcs.get_width() * pixscale / 3600, wcs.get_height() * pixscale / 3600 # [degrees]
    bb, bbcc = wcs.radec_bounds(), wcs.radec_center() # [degrees]
    
    radius_deg = radius * pixscale / 3600 # [degrees]
    #pad = 0.2
    pad = 2 * radius_deg # [degrees]
    bb_clust, bbcc_clust = wcs_clust.radec_bounds(), wcs_clust.radec_center() # [degrees]
    
    delta = np.max( (np.diff(bb[0:2]), np.diff(bb[2:4])) ) / 2 + pad / 2
    xlim = bbcc[0] - delta, bbcc[0] + delta
    ylim = bbcc[1] - delta, bbcc[1] + delta
    #print(xlim, ylim, pad)

    #pdb.set_trace()

    col = iter(plt.cm.rainbow(np.linspace(0, 1, len(ccds))))
    
    fig, allax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)

    for ax, band in zip(allax, ('g', 'r', 'z')):
        ax.set_xlabel('RA (deg)')
        ax.text(0.9, 0.05, band, ha='center', va='bottom',
                transform=ax.transAxes, fontsize=18)

        if band == 'g':
            ax.set_ylabel('Dec (deg)')

        these = np.where(ccds.filter == band)[0]
        #col = plt.cm.Set1(np.linspace(0, 1, len(ccds)))
        for ii, ccd in enumerate(ccds[these]):
            #print(ccd.expnum, ccd.ccdname, ccd.filter)
            W, H, ccdwcs = legacyhalos.misc.ccdwcs(ccd)

            cc = ccdwcs.radec_bounds()
            ax.add_patch(patches.Rectangle((cc[0], cc[2]), cc[1]-cc[0],
                                           cc[3]-cc[2], fill=False, lw=2, 
                                           edgecolor=next(col),
                                           label='{}-{}'.format(ccds.expnum[these[ii]], ccds.ccdname[these[ii]])))
                                           #label='ccd{:02d}'.format(these[ii])))
            ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=8)

        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.add_patch(patches.Rectangle((bbcc_clust[0]-radius_deg, bbcc_clust[1]-radius_deg), radius_deg*2, radius_deg*2,
                                       fill=False, edgecolor='black', lw=1, ls='-'))

        if False:
            ax.add_patch(patches.Circle((bbcc_clust[0], bbcc_clust[1]), 1.1*radius_deg, # inner sky annulus
                                        fill=False, edgecolor='gray', lw=1))
            ax.add_patch(patches.Circle((bbcc_clust[0], bbcc_clust[1]), 1.2*radius_deg, # outer sky annulus
                                        fill=False, edgecolor='gray', lw=1))
        
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.invert_xaxis()
        ax.set_aspect('equal')
        #print(ax.get_xlim(), ax.get_ylim())
        
    plt.subplots_adjust(bottom=0.12, wspace=0.05, left=0.12, right=0.97, top=0.95)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

def display_ccd_apphot():
    deltar = 5.0
    rin = np.arange(0.0, radius/2, 1.0)
    nap = len(rin)

    apphot = Table(np.zeros(nap, dtype=[('RCEN', 'f4'), ('RIN', 'f4'),
                                        ('ROUT', 'f4'), ('PIPEFLUX', 'f4'),
                                        ('NEWFLUX', 'f4'), ('PIPESKYFLUX', 'f4'), 
                                        ('NEWSKYFLUX', 'f4'), ('AREA', 'f4'),
                                        ('SKYAREA', 'f4')]))
    apphot['RIN'] = rin
    apphot['ROUT'] = rin + deltar
    apphot['RCEN'] = rin + deltar / 2.0
    for ii in range(nap):
        ap = CircularAperture((xcen, ycen), apphot['RCEN'][ii])
        skyap = CircularAnnulus((xcen, ycen), r_in=apphot['RIN'][ii],
                                r_out=apphot['ROUT'][ii])

        #pdb.set_trace()
        apphot['PIPEFLUX'][ii] = aperture_photometry(image_nopipesky, ap)['aperture_sum'].data
        apphot['NEWFLUX'][ii] = aperture_photometry(image_nonewsky, ap)['aperture_sum'].data
        apphot['PIPESKYFLUX'][ii] = aperture_photometry(image_nopipesky, skyap)['aperture_sum'].data
        apphot['NEWSKYFLUX'][ii] = aperture_photometry(image_nonewsky, skyap)['aperture_sum'].data

        apphot['AREA'][ii] = ap.area()
        apphot['SKYAREA'][ii] = skyap.area()

    # Convert to arcseconds
    apphot['RIN'] *= im.pixscale
    apphot['ROUT'] *= im.pixscale
    apphot['RCEN'] *= im.pixscale
    apphot['AREA'] *= im.pixscale**2
    apphot['SKYAREA'] *= im.pixscale**2
    print(apphot)
    #pdb.set_trace()

    # Now generate some QAplots related to the sky.
    sbinsz = 0.001
    srange = (-5 * sig1, +5 * sig1)
    #sbins = 50
    sbins = np.int( (srange[1]-srange[0]) / sbinsz )

    qaccd = os.path.join('.', 'qa-{}-ccd{:02d}-sky.png'.format(prefix.lower(), iccd))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('{} (ccd{:02d})'.format(tim.name, iccd), y=0.97)
    for data1, label, color in zip((image_nopipesky.flat[pipeskypix],
                                    image_nonewsky.flat[newskypix]),
                                   ('Pipeline Sky', 'Custom Sky'), setcolors):
        nn, bins = np.histogram(data1, bins=sbins, range=srange)
        nn = nn/float(np.max(nn))
        cbins = (bins[:-1] + bins[1:]) / 2.0
        #pdb.set_trace()
        ax[0].step(cbins, nn, color=color, lw=2, label=label)
        ax[0].set_ylim(0, 1.2)
        #(nn, bins, _) = ax[0].hist(data1, range=srange, bins=sbins,
        #                           label=label, normed=True, lw=2, 
        #                           histtype='step', color=color)
    ylim = ax[0].get_ylim()
    ax[0].vlines(0.0, ylim[0], 1.05, colors='k', linestyles='dashed')
    ax[0].set_xlabel('Residuals (nmaggie)')
    ax[0].set_ylabel('Relative Fraction of Pixels')
    ax[0].legend(frameon=False, loc='upper left')

    ax[1].plot(apphot['RCEN'], apphot['PIPESKYFLUX']/apphot['SKYAREA'], 
                  label='Pipeline', color=setcolors[0])
    ax[1].plot(apphot['RCEN'], apphot['NEWSKYFLUX']/apphot['SKYAREA'], 
                  label='Custom', color=setcolors[1])
    #ax[1].scatter(apphot['RCEN'], apphot['PIPESKYFLUX']/apphot['SKYAREA'], 
    #              label='DR2 Pipeline', marker='o', color=setcolors[0])
    #ax[1].scatter(apphot['RCEN']+1.0, apphot['NEWSKYFLUX']/apphot['SKYAREA'], 
    #              label='Large Galaxy Pipeline', marker='s', color=setcolors[1])
    ax[1].set_xlabel('Galactocentric Radius (arcsec)')
    ax[1].set_ylabel('Flux in {:g}" Annulus (nmaggie/arcsec$^2$)'.format(deltar))
    ax[1].set_xlim(-2.0, apphot['ROUT'][-1])
    ax[1].legend(frameon=False, loc='upper right')

    xlim = ax[1].get_xlim()
    ylim = ax[1].get_ylim()
    ax[1].hlines(0.0, xlim[0], xlim[1]*0.99999, colors='k', linestyles='dashed')
    #ax[1].vlines(gal['RADIUS'], ylim[0], ylim[1]*0.5, colors='k', linestyles='dashed')

    plt.tight_layout(w_pad=0.25)
    plt.subplots_adjust(bottom=0.15, top=0.88)
    print('Writing {}'.format(qaccd))
    plt.savefig(qaccd)
    plt.close(fig)

def _display_ccdmask_and_sky(ccdargs):
    """Visualize the image, the custom mask, custom sky, and the pipeline sky (via
    multiprocessing) of a single CCD.

    """
    import matplotlib.patches as patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import uniform_filter

    import fitsio
    from astrometry.util.util import Tan
    from astrometry.util.fits import fits_table
    from tractor.splinesky import SplineSky
    from tractor.basics import NanoMaggies
        
    galaxy, galaxydir, qarootfile, radius_pixel, ccd, iccd, survey = ccdargs

    im = survey.get_image_object(ccd)

    # Read the tim.
    # targetwcs = im.get_wcs()
    #print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
    #      'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
    #      'object', getattr(ccd, 'object', None))
    
    #tim = im.get_tractor_image(splinesky=True, subsky=False,
    #                           hybridPsf=True, normalizePsf=True)
    #
    #targetwcs = tim.subwcs
    #H, W = targetwcs.shape
    #H, W = np.int(H), np.int(W)

    ## Get the image, read and instantiate the pipeline (splinesky) model.
    #image = tim.getImage()
    #weight = tim.getInvvar()
    #pipesky = np.zeros_like(image)
    #tim.getSky().addTo(pipesky)

    # Reproduce the (pipeline) image mask derived in
    # legacypipe.decam.run_calibs.
    #if False:
    #    boxsize, boxcar = 512, 5
    #    if min(image.shape) / boxsize < 4:
    #        boxsize /= 2
    #
    #    good = weight > 0
    #    if np.sum(good) == 0:
    #        raise RuntimeError('No pixels with weight > 0.')
    #    med = np.median(image[good])
    #
    #    skyobj = SplineSky.BlantonMethod(image - med, good, boxsize)
    #    skymod = np.zeros_like(image)
    #    skyobj.addTo(skymod)
    #
    #    bsig1 = ( 1 / np.sqrt( np.median(weight[good]) ) ) / boxcar
    #
    #    mask = np.abs( uniform_filter(image - med - skymod, size=boxcar, mode='constant') > (3 * bsig1) )
    #    mask = binary_dilation(mask, iterations=3)

    # Read the custom mask and (constant) sky value.
    key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
    image, hdr = fitsio.read(os.path.join(galaxydir, '{}-ccddata-grz.fits.fz'.format(galaxy)), header=True, ext=key)
    
    newmask = fitsio.read(os.path.join(galaxydir, '{}-custom-ccdmask-grz.fits.gz'.format(galaxy)), ext=key)
    newsky = np.zeros_like(image).astype('f4') + hdr['SKYMED']

    # Rebuild the pipeline (spline) sky model (see legacypipe.image.LegacySurveyImage.read_sky_model)
    Ti = fits_table(os.path.join(galaxydir, '{}-pipeline-sky.fits'.format(galaxy)), ext=key)[0]
    h, w = Ti.gridh, Ti.gridw
    Ti.gridvals = Ti.gridvals[:h, :w]
    Ti.xgrid = Ti.xgrid[:w]
    Ti.ygrid = Ti.ygrid[:h]
    splinesky = SplineSky.from_fits_row(Ti)

    pipesky = np.zeros_like(image)
    splinesky.addTo(pipesky)
    pipesky /= NanoMaggies.zeropointToScale(im.ccdzpt)

    # Get the (pixel) coordinates of the galaxy on this CCD
    #_, x0, y0 = targetwcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
    #xcen, ycen = np.round(x0 - 1).astype('int'), np.round(y0 - 1).astype('int')
    xcen, ycen = hdr['XCEN'], hdr['YCEN']

    # Visualize the data, the mask, and the sky.
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(12, 4.5))
    #fig, ax = plt.subplots(1, 5, sharey=True, figsize=(14, 4.5))
    fig.suptitle('{} (ccd{:02d})'.format(key.lower(), iccd), y=0.95, fontsize=14)

    vmin_image, vmax_image = np.percentile(image, (1, 99))
    #vmin_weight, vmax_weight = np.percentile(weight, (1, 99))
    vmin_mask, vmax_mask = (0, 1)
    vmin_sky, vmax_sky = np.percentile(pipesky, (0.1, 99.9))

    cmap = 'viridis' # 'inferno'

    for thisax, data, title in zip(ax.flat, (image, newmask, pipesky, newsky), 
                                   ('Image', 'Custom Mask',
                                    'Pipeline Sky', 'Custom Sky')):
    #for thisax, data, title in zip(ax.flat, (image, mask, newmask, pipesky, newsky), 
    #                               ('Image', 'Pipeline Mask', 'Custom Mask',
    #                                'Pipeline Sky', 'Custom Sky')):
        if 'Mask' in title:
            vmin, vmax = vmin_mask, vmax_mask
        elif 'Sky' in title:
            vmin, vmax = vmin_sky, vmax_sky
        elif 'Image' in title:
            vmin, vmax = vmin_image, vmax_image

        thisim = thisax.imshow(data, cmap=cmap, interpolation='nearest',
                               origin='lower', vmin=vmin, vmax=vmax)
        thisax.add_patch(patches.Circle((xcen, ycen), radius_pixel, fill=False, edgecolor='white', lw=2))
        thisax.add_patch(patches.Circle((xcen, ycen), 2*radius_pixel, fill=False, edgecolor='white', lw=1))
        thisax.add_patch(patches.Circle((xcen, ycen), 5*radius_pixel, fill=False, edgecolor='white', lw=1))

        div = make_axes_locatable(thisax)
        cax = div.append_axes('right', size='15%', pad=0.1)
        cbar = fig.colorbar(thisim, cax=cax, format='%.4g')

        thisax.set_title(title, fontsize=10)
        thisax.xaxis.set_visible(False)
        thisax.yaxis.set_visible(False)
        thisax.set_aspect('equal')

    ## Shared colorbar.
    #plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.88, wspace=0.6)

    qafile = '{}-ccd{:02d}.png'.format(qarootfile, iccd)
    print('Writing {}'.format(qafile))
    fig.savefig(qafile)
    plt.close(fig)

def _display_ellipse_sbprofile(ellipsefit, skyellipsefit={}, minerr=0.0,
                              cosmo=None, png=None, verbose=True):
    """Display the multi-band surface brightness profile.

    4-panel including PA and ellipticity

    """
    import astropy.stats
    from legacyhalos.ellipse import ellipse_sbprofile

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)
        
        band, refband = ellipsefit['bands'], ellipsefit['refband']
        redshift, refpixscale = ellipsefit['redshift'], ellipsefit['refpixscale']
        smascale = legacyhalos.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]

        if png:
            sbfile = png.replace('.png', '.txt')
            legacyhalos.io.write_sbprofile(sbprofile, smascale, sbfile)

        yminmax = [40, 0]
        xminmax = [0, 0]
        colors = _sbprofile_colors()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                                                 gridspec_kw = {'height_ratios':[0.8, 0.8, 2, 1.2]})

        # ax1 - ellipticity versus semi-major axis
        good = (ellipsefit[refband].stop_code < 4)
        bad = ~good
        if False:
            ax1.fill_between(ellipsefit[refband].sma[good] * refpixscale,
                             ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                             ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax1.scatter(ellipsefit[refband].sma[bad] * refpixscale, ellipsefit[refband].eps[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        else:
            ax1.plot(ellipsefit[refband].sma * refpixscale, ellipsefit[refband].eps, zorder=1, alpha=0.9, lw=2)
            ax1.scatter(ellipsefit[refband].sma * refpixscale, ellipsefit[refband].eps,
                        marker='s', s=50, edgecolor='k', lw=2, alpha=0.75, zorder=2)
            #ax1.fill_between(ellipsefit[refband].sma * refpixscale,
            #                 ellipsefit[refband].eps-0.02,
            #                 ellipsefit[refband].eps+0.02, color='gray', alpha=0.5)

        # ax2 - position angle versus semi-major axis
        if False:
            ax2.fill_between(ellipsefit[refband].sma[good] * refpixscale, 
                             np.degrees(ellipsefit[refband].pa[good]-ellipsefit[refband].pa_err[good]),
                             np.degrees(ellipsefit[refband].pa[good]+ellipsefit[refband].pa_err[good]))#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax2.scatter(ellipsefit[refband].sma[bad] * refpixscale, np.degrees(ellipsefit[refband].pa[bad]),
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        else:
            ax2.plot(ellipsefit[refband].sma * refpixscale, np.degrees(ellipsefit[refband].pa), zorder=1, alpha=0.9, lw=2)
            ax2.scatter(ellipsefit[refband].sma * refpixscale, np.degrees(ellipsefit[refband].pa),
                        marker='s', s=50, edgecolor='k', lw=2, alpha=0.75, zorder=2)
            #ax2.fill_between(ellipsefit[refband].sma * refpixscale,
            #                 np.degrees(ellipsefit[refband].pa)-5,
            #                 np.degrees(ellipsefit[refband].pa)+5, color='gray', alpha=0.5)

        ax1.set_ylabel('Ellipticity')
        #ax1.set_ylabel(r'Ellipticity $\epsilon$')
        ax1.set_ylim(0, 0.6)

        ax2.set_ylabel('P. A. (deg)')
        #ax2.set_ylabel(r'$\theta$ (deg)')
        ax2.set_ylim(-10, 180)
        #ax2.set_ylabel('Position Angle (deg)')
        
        for filt in band:
            sma = sbprofile['sma']
            mu = sbprofile['mu_{}'.format(filt.lower())]
            muerr = sbprofile['mu_{}_err'.format(filt.lower())]

            #good = (ellipsefit[filt].stop_code < 4)
            #bad = ~good
            
            #with np.errstate(invalid='ignore'):
            #    good = np.isfinite(mu) * (mu / muerr > 3)
            good = np.isfinite(mu)
            sma = sma[good]
            mu = mu[good]
            muerr = muerr[good]
                
            col = next(colors)
            ax3.fill_between(sma, mu-muerr, mu+muerr, label=r'${}$'.format(filt.lower()), color=col,
                             alpha=0.75, edgecolor='k', lw=2)

            if np.nanmin(mu-muerr) < yminmax[0]:
                yminmax[0] = np.nanmin(mu-muerr)
            if np.nanmax(mu+muerr) > yminmax[1]:
                yminmax[1] = np.nanmax(mu+muerr)
            if np.nanmax(sma) > xminmax[1]:
                xminmax[1] = np.nanmax(sma)

            if bool(skyellipsefit):
                skysma = skyellipsefit['sma'] * refpixscale

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sky = astropy.stats.mad_std(skyellipsefit[filt], axis=1, ignore_nan=True)
                    # sky = np.nanstd(skyellipsefit[filt], axis=1) # / np.sqrt(skyellipsefit[
                    
                skygood = np.isfinite(sky)
                skysma = skysma[skygood]
                skymu = 22.5 - 2.5 * np.log10(sky[skygood])
                ax3.plot( skysma, skymu , color=col, ls='--', alpha=0.75)
                if skymu.max() > yminmax[1]:
                    yminmax[1] = skymu.max()

                ax3.text(0.05, 0.04, 'Sky Variance', ha='left', va='center',
                         transform=ax3.transAxes, fontsize=12)

            #ax3.axhline(y=ellipsefit['mu_{}_sky'.format(filt.lower())], color=col, ls='--')
            #if filt == refband:
            #    ysky = ellipsefit['mu_{}_sky'.format(filt.lower())] - 2.5 * np.log10(0.1) # 10% of sky
            #    ax3.axhline(y=ysky, color=col, ls='--')

        ax3.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
        #ax3.set_ylabel(r'Surface Brightness $\mu(a)$ (mag arcsec$^{-2}$)')
        #ax3.set_ylabel(r'Surface Brightness $\mu$ (mag arcsec$^{-2}$)')

        ylim = [yminmax[0]-0.5, yminmax[1]+0.75]
        if ylim[0] < 17:
            ylim[0] = 17
        if ylim[1] > 32.5:
            ylim[1] = 32.5
        ax3.set_ylim(ylim)
        ax3.invert_yaxis()

        xlim = [xminmax[0], xminmax[1]*1.01]
        #ax3.set_xlim(xmin=0)
        #ax3.margins(xmargin=0)
        
        #ax1.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
        #ax1.set_ylim(31.99, 18)

        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax1_twin.set_xlabel('Semi-major axis (kpc)')

        ax3.legend(loc='upper right')

        # color vs semi-major axis
        ax4.fill_between(sbprofile['sma'],
                         sbprofile['gr'] - sbprofile['gr_err'],
                         sbprofile['gr'] + sbprofile['gr_err'],
                         label=r'$g - r$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax4.fill_between(sbprofile['sma'],
                         sbprofile['rz'] - sbprofile['rz_err'],
                         sbprofile['rz'] + sbprofile['rz_err'],
                         label=r'$r - z$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax4.set_xlabel(r'Semi-major axis (arcsec)')
        #ax4.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        #ax4.legend(loc='upper left')
        ax4.legend(bbox_to_anchor=(0.25, 0.99))
        
        ax4.set_ylabel('Color (mag)')
        ax4.set_ylim(-0.5, 2.8)

        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xlim(xlim)
            
            ylim = xx.get_ylim()
            xx.fill_between([0, 3*ellipsefit['psfsize_r']], [ylim[0], ylim[0]],
                            [ylim[1], ylim[1]], color='grey', alpha=0.1)
            
        ax4.text(0.03, 0.09, 'PSF\n(3$\sigma$)', ha='center', va='center',
            transform=ax4.transAxes, fontsize=10)

        fig.subplots_adjust(hspace=0.0)

        if png:
            #if verbose:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
        
