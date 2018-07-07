"""
legacyhalos.qa
==============

Code to do produce various QA (quality assurance) plots. 

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='ticks', font_scale=1.4, palette='Set2')

def display_sersic_single(sersic, png=None, verbose=False):
    """Plot a wavelength-dependent surface brightness profile.

    """
    from legacyhalos.misc import arcsec2kpc

    colors = iter(sns.color_palette())
    markers = iter(['o', 's', 'D'])

    smascale = arcsec2kpc(sersic['redshift'])

    if sersic['success']:
        model = sersic['bestfit']
    else:
        model = None

    fig, ax = plt.subplots(figsize=(8, 5))
    for band, lam in zip( sersic['band'], (sersic['lambda_g'],
                                           sersic['lambda_r'],
                                           sersic['lambda_z']) ):
        good = lam == sersic['wave']
        wave = sersic['wave'][good]
        rad = sersic['radius'][good]
        sb = sersic['sb'][good]

        srt = np.argsort(rad)
        rad, sb, wave = rad[srt], sb[srt], wave[srt]

        if model is not None:
            n = model.get_sersicn(nref=model.nref, lam=lam, alpha=model.alpha)
            r50 = model.get_r50(r50ref=model.nref, lam=lam, beta=model.beta)
            label = r'${}:\ n={:.2f}\ r_{{50}}={:.2f}$ arcsec'.format(band, n, r50)
        else:
            label = band

        col = next(colors)
        #ax.plot(rad, 22.5-2.5*np.log10(sb), label=band)
        ax.scatter(rad, 22.5-2.5*np.log10(sb), color=col,
                   alpha=1, s=50, label=label, marker=next(markers))
        
        # optionally overplot the model
        if model is not None:
            sb_model = model(rad, wave[srt])
            ax.plot(rad, 22.5-2.5*np.log10(sb_model), color='k', #color=col, 
                        ls='--', lw=2, alpha=0.5)

    ax.set_xlabel('Galactocentric radius (arcsec)')
    ax.set_ylabel(r'Surface Brightness $\mu$ (mag arcsec$^{-2}$)')
    ax.invert_yaxis()
    #ax.set_yscale('log')

    ax.set_xlim(xmin=0)

    ax2 = ax.twiny()
    xlim = ax.get_xlim()
    ax2.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
    ax2.set_xlabel('Galactocentric radius (kpc)')

    ax.legend(loc='upper right', markerscale=1.2)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def display_multiband(data, geometry=None, mgefit=None, ellipsefit=None, indx=None,
                      magrange=10, inchperband=3, contours=False, png=None,
                      verbose=True):
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    """
    from astropy.visualization import ZScaleInterval as Interval
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize

    band = data['band']
    nband = len(band)

    fig, ax = plt.subplots(1, 3, figsize=(inchperband*nband, nband))
    for filt, ax1 in zip(band, ax):

        img = data['{}_masked'.format(filt)]
        #img = data[filt]

        norm = ImageNormalize(img, interval=Interval(contrast=0.95),
                              stretch=Stretch(a=0.95))

        im = ax1.imshow(img, origin='lower', norm=norm, cmap='viridis',
                        interpolation='nearest')
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, #fontweight='bold',
                 ha='center', va='center', color='k', fontsize=14)

        if mgefit:
            from mge.mge_print_contours import _multi_gauss, _gauss2d_mge

            sigmapsf = np.atleast_1d(0)
            normpsf = np.atleast_1d(1)
            _magrange = 10**(-0.4*np.arange(0, magrange, 1)[::-1]) # 0.5 mag/arcsec^2 steps
            #_magrange = 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

            model = _multi_gauss(mgefit[filt].sol, img, sigmapsf, normpsf,
                                 mgefit['xpeak'], mgefit['ypeak'],
                                 mgefit['pa'])
            
            peak = data[filt][mgefit['xpeak'], mgefit['ypeak']]
            levels = peak * _magrange
            s = img.shape
            extent = [0, s[1], 0, s[0]]

            ax1.contour(model, levels, colors='k', linestyles='solid',
                        extent=extent, alpha=0.75, lw=1)

        if geometry:
            from photutils import EllipticalAperture
            
            ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                         geometry.sma*(1 - geometry.eps), geometry.pa)
            ellaper.plot(color='k', lw=1, ax=ax1)

        if ellipsefit:
            if ellipsefit['success']:
                if len(ellipsefit[filt]) > 0:
                    if indx is None:
                        indx = np.ones(len(ellipsefit[filt]), dtype=bool)

                    nfit = len(indx) # len(ellipsefit[filt])
                    nplot = np.rint(0.5*nfit).astype('int')

                    smas = np.linspace(0, ellipsefit[filt].sma[indx].max(), nplot)
                    for sma in smas:
                        efit = ellipsefit[filt].get_closest(sma)
                        x, y, = efit.sampled_coordinates()
                        ax1.plot(x, y, color='k', alpha=0.75)
            else:
                from photutils import EllipticalAperture
                geometry = ellipsefit['geometry']
                ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                             geometry.sma*(1 - geometry.eps), geometry.pa)
                ellaper.plot(color='k', lw=1, ax=ax1)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        #ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def display_ellipsefit(ellipsefit, xlog=False, png=None, verbose=True):
    """Display the isophote fitting results."""

    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    colors = iter(sns.color_palette())

    band, refband = ellipsefit['band'], ellipsefit['refband']
    pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for filt in np.atleast_1d(refband):

        if ellipsefit['success']:
            good = (ellipsefit[filt].stop_code < 4)
            bad = ~good

            ax1.fill_between(ellipsefit[filt].sma[good] * pixscale,
                             ellipsefit[filt].eps[good]-ellipsefit[filt].ellip_err[good],
                             ellipsefit[filt].eps[good]+ellipsefit[filt].ellip_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax1.scatter(ellipsefit[filt].sma[bad] * pixscale, ellipsefit[filt].eps[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)

            #ax1.errorbar(ellipsefit[filt].sma[good] * smascale,
            #             ellipsefit[filt].eps[good],
            #             ellipsefit[filt].ellip_err[good], fmt='o',
            #             markersize=4)#, color=color[filt])
            #ax1.set_ylim(0, 0.5)
            ax1.xaxis.set_major_formatter(ScalarFormatter())

            ax2.fill_between(ellipsefit[filt].sma[good] * pixscale, 
                             ellipsefit[filt].pa[good]-ellipsefit[filt].pa_err[good],
                             ellipsefit[filt].pa[good]+ellipsefit[filt].pa_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax2.scatter(ellipsefit[filt].sma[bad] * pixscale, ellipsefit[filt].pa[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
            #ax2.errorbar(ellipsefit[filt].sma[good] * smascale,
            #             np.degrees(ellipsefit[filt].pa[good]),
            #             np.degrees(ellipsefit[filt].pa_err[good]), fmt='o',
            #             markersize=4)#, color=color[filt])
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position('right')
            ax2.xaxis.set_major_formatter(ScalarFormatter())
            #ax2.set_ylim(0, 180)

            ax3.fill_between(ellipsefit[filt].sma[good] * pixscale,
                             ellipsefit[filt].x0[good]-ellipsefit[filt].x0_err[good],
                             ellipsefit[filt].x0[good]+ellipsefit[filt].x0_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax3.scatter(ellipsefit[filt].sma[bad] * pixscale, ellipsefit[filt].x0[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
            #ax3.errorbar(ellipsefit[filt].sma[good] * smascale, ellipsefit[filt].x0[good],
            #             ellipsefit[filt].x0_err[good], fmt='o',
            #             markersize=4)#, color=color[filt])
            ax3.xaxis.set_major_formatter(ScalarFormatter())
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax4.fill_between(ellipsefit[filt].sma[good] * pixscale, 
                             ellipsefit[filt].y0[good]-ellipsefit[filt].y0_err[good],
                             ellipsefit[filt].y0[good]+ellipsefit[filt].y0_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax4.scatter(ellipsefit[filt].sma[bad] * pixscale, ellipsefit[filt].y0[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
            #ax4.errorbar(ellipsefit[filt].sma[good] * smascale, ellipsefit[filt].y0[good],
            #             ellipsefit[filt].y0_err[good], fmt='o',
            #             markersize=4)#, color=color[filt])
            
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right')
        ax4.xaxis.set_major_formatter(ScalarFormatter())
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    ax1.set_ylabel('Ellipticity')
    ax2.set_ylabel('Position Angle (deg)')
    ax3.set_xlabel('Semimajor Axis (arcsec)')
    ax3.set_ylabel(r'$x_{0}$')
    ax4.set_xlabel('Semimajor Axis (arcsec)')
    ax4.set_ylabel(r'$y_{0}$')
    
    if xlog:
        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xscale('log')

    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.15, right=0.85, left=0.15)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_ellipse_sbprofile(ellipsefit, minerr=0.02, sersicfit=None,
                              png=None, verbose=True):
    """Display the multi-band surface brightness profile.

    """
    from legacyhalos.ellipse import ellipse_sbprofile

    band, refband, redshift = ellipsefit['band'], ellipsefit['refband'], ellipsefit['redshift']

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)

    colors = iter(sns.color_palette())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filt in band:

        if ellipsefit['success']:
            good = (ellipsefit[filt].stop_code < 4)
            bad = ~good

            col = next(colors)
            ax1.fill_between(sbprofile['sma'], 
                sbprofile['mu_{}'.format(filt)] - sbprofile['mu_{}_err'.format(filt)],
                sbprofile['mu_{}'.format(filt)] + sbprofile['mu_{}_err'.format(filt)],
                #sbprofile['{}'.format(filt)] - sbprofile['{}_err'.format(filt)],
                #sbprofile['{}'.format(filt)] + sbprofile['{}_err'.format(filt)],
                label=r'${}$'.format(filt), color=col, alpha=0.75, edgecolor='k', lw=2)
            #if np.count_nonzero(bad) > 0:
            #    ax1.scatter(sbprofile['sma'][bad], sbprofile[filt][bad], marker='s',
            #                s=40, edgecolor='k', lw=2, alpha=0.75)

            #ax1.axhline(y=ellipsefit['mu_{}_sky'.format(filt)], color=col, ls='--')
            if filt == refband:
                ysky = ellipsefit['mu_{}_sky'.format(filt)] - 2.5 * np.log10(0.1) # 10% of sky
                ax1.axhline(y=ysky, color=col, ls='--')

            # Overplot the best-fitting model.
            if sersicfit:
                from astropy.modeling.models import Sersic1D
                rad = np.arange(0, sbprofile['sma'].max(), 0.1)
                sbmodel = -2.5 * np.log10( Sersic1D.evaluate(
                    rad, sersicfit[filt].amplitude, sersicfit[filt].r_eff,
                    sersicfit[filt].n) )
                ax1.plot(rad, sbmodel, lw=2, ls='--', alpha=1, color=col)
            
    ax1.set_ylabel(r'Surface Brightness (mag arcsec$^{-2}$)')
    ax1.set_ylim(30, 17)

    #ax1.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
    #ax1.set_ylim(31.99, 18)

    if ellipsefit['success']:
        ax1.legend(loc='upper right')

        ax2.fill_between(sbprofile['sma'],
                         sbprofile['rz'] - sbprofile['rz_err'],
                         sbprofile['rz'] + sbprofile['rz_err'],
                         label=r'$r - z$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax2.fill_between(sbprofile['sma'],
                         sbprofile['gr'] - sbprofile['gr_err'],
                         sbprofile['gr'] + sbprofile['gr_err'],
                         label=r'$g - r$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax2.set_xlabel('Semimajor Axis (arcsec)', alpha=0.75)
        ax2.legend(loc='upper left')
    else:
        ax2.set_xlabel('Semimajor Axis', alpha=0.75)
        
    ax2.set_ylabel('Color (mag)')
    ax2.set_ylim(0, 2.4)

    fig.subplots_adjust(hspace=0.0)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_mge_sbprofile(mgefit, indx=None, png=None, verbose=True):
    """Display the multi-band surface brightness profile."""

    colors = iter(sns.color_palette())

    #if indx is None:
    #    indx = np.ones_like(mgefit[refband].radius, dtype='bool')
        
    def _sbprofile(mgefit, indx, smascale):
        """Convert fluxes to magnitudes and colors."""
        sbprofile = dict()
        
        with np.errstate(invalid='ignore'):
            for filt in band:
                #indx = np.ones_like(mgefit[filt].radius, dtype='bool')
                indx = np.argsort(mgefit[filt].radius)
                
                sbprofile['{}_sma'.format(filt)] = mgefit[filt].radius[indx] * smascale
                sbprofile[filt] = 22.5 - 2.5 * np.log10(mgefit[filt].counts[indx])
                sbprofile['{}_err'.format(filt)] = 2.5 * mgefit[filt].err[indx] / mgefit[filt].counts[indx] / np.log(10)

                sbprofile['{}_model'.format(filt)] = 22.5 - 2.5 * np.log10(mgefit[filt].yfit[indx])

                #sbprofile['{}_err'.format(filt)] = mgefit[filt].int_err[indx] / \
                #  mgefit[filt].intens[indx] / np.log(10)

                # Just for the plot use a minimum uncertainty
                #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

        #sbprofile['gr'] = sbprofile['g'] - sbprofile['r']
        #sbprofile['rz'] = sbprofile['r'] - sbprofile['z']
        #sbprofile['gr_err'] = np.sqrt(sbprofile['g_err']**2 + sbprofile['r_err']**2)
        #sbprofile['rz_err'] = np.sqrt(sbprofile['r_err']**2 + sbprofile['z_err']**2)
        #
        ## Just for the plot use a minimum uncertainty
        #sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
        #sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

        return sbprofile

    sbprofile = _sbprofile(mgefit, indx, smascale)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for filt in band:
        ax1.scatter(sbprofile['{}_sma'.format(filt)], sbprofile[filt], marker='s',
                     label=r'${}$'.format(filt), color=next(colors), alpha=0.75, s=10)
        #ax1.errorbar(sbprofile['{}_sma'.format(filt)], sbprofile[filt], yerr=sbprofile['{}_err'.format(filt)],
        #             label=r'${}$'.format(filt), color=next(colors), alpha=0.75, fmt='s',
        #             markersize=2)
        #ax1.fill_between(sbprofile['{}_sma'.format(filt)], sbprofile[filt]-sbprofile['{}_err'.format(filt)],
        #                 sbprofile[filt]+sbprofile['{}_err'.format(filt)],
        #                 label=r'${}$'.format(filt), color=next(colors), alpha=0.75)
        #ax1.plot(sbprofile['{}_sma'.format(filt)], sbprofile['{}_model'.format(filt)], lw=2, color='k')
    ax1.set_ylabel('AB Magnitude')
    ax1.set_ylim(32.99, 20)
    #ax1.invert_yaxis()
    ax1.legend(loc='upper right')

    #ax2.fill_between(sbprofile['sma'], sbprofile['rz']-sbprofile['rz_err'],
    #                 sbprofile['rz']+sbprofile['rz_err'],
    #                 label=r'$r - z$', color=next(colors), alpha=0.75)
    #
    #ax2.fill_between(sbprofile['sma'], sbprofile['gr']-sbprofile['gr_err'],
    #                 sbprofile['gr']+sbprofile['gr_err'],
    #                 label=r'$g - r$', color=next(colors), alpha=0.75)

    ax2.set_xlabel('Semimajor Axis ({})'.format(smaunit), alpha=0.75)
    ax2.set_ylabel('Color')
    ax2.set_ylim(-0.5, 2.5)
    #ax2.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.0)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def sample_trends(sample, htmldir, analysisdir=None, verbose=True):
    """Trends with the whole sample.

    """
    from astropy.cosmology import WMAP9 as cosmo
    from legacyhalos.io import get_objid, read_ellipsefit
    from legacyhalos.ellipse import ellipse_sbprofile
    from legacyhalos.misc import arcsec2kpc

    trendsdir = os.path.join(htmldir, 'trends')
    if not os.path.isdir(trendsdir):
        os.makedirs(trendsdir, exist_ok=True)

    # color vs semi-major axis
    def _color_vs_sma():
        png = os.path.join(trendsdir, 'color_vs_ellipticity.png')
    
        fig, ax1 = plt.subplots()
        for gal in sample:
            objid, objdir = get_objid(gal, analysisdir=analysisdir)

            ellipsefit = read_ellipsefit(objid, objdir)
            if len(ellipsefit) > 0:
                if ellipsefit['success']:                    
                    refband, redshift = ellipsefit['refband'], ellipsefit['redshift']
                    smascale = arcsec2kpc(redshift) # [kpc/arcsec]
                    
                    sbprofile = ellipse_sbprofile(ellipsefit, minerr=0.01)
                    good = (ellipsefit[refband].stop_code < 4)
                    ax1.fill_between(sbprofile['sma'][good] * smascale,
                                     sbprofile['gr'][good]-sbprofile['gr_err'][good],
                                     sbprofile['gr'][good]+sbprofile['gr_err'][good],
                                     alpha=0.6, color='gray')

        ax1.grid()
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 2.5)
        ax1.set_ylabel(r'$g - r$')
        ax1.set_xlabel('Semimajor Axis (kpc)')

        fig.subplots_adjust(bottom=0.15, right=0.95, left=0.15, top=0.95)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
            
    # Ellipticity vs semi-major axis
    def _ellipticity_vs_sma():
        
        png = os.path.join(trendsdir, 'sma_vs_ellipticity.png')
    
        fig, ax1 = plt.subplots()
        for gal in sample:
            objid, objdir = get_objid(gal, analysisdir=analysisdir)

            ellipsefit = read_ellipsefit(objid, objdir)
            if len(ellipsefit) > 0:
                if ellipsefit['success']:
                    refband, redshift = ellipsefit['refband'], ellipsefit['redshift']
                    smascale = arcsec2kpc(redshift) # [kpc/arcsec]
                    
                    good = (ellipsefit[refband].stop_code < 4)
                    ax1.fill_between(ellipsefit[refband].sma[good] * smascale, 
                                     ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                                     ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good],
                                     alpha=0.6, color='gray')

        ax1.grid()
        ax1.set_ylim(0, 0.5)
        ax1.set_ylabel('Ellipticity')
        ax1.set_xlabel('Semimajor Axis (kpc)')

        fig.subplots_adjust(bottom=0.15, right=0.95, left=0.15, top=0.95)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()

    # Build all the plots here.
    
    _color_vs_sma()       # color vs semi-major axis
    _ellipticity_vs_sma() # ellipticity vs semi-major axis
