"""
legacyhalos.qa
==============

Code to do produce various QA (quality assurance) plots. 

"""
from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='talk', style='ticks')#, palette='Set1')

def display_multiband(data, band=('g', 'r', 'z'), refband='r', geometry=None,
                      mgefit=None, ellipsefit=None, indx=None, magrange=10,
                      inchperband=3, contours=False, png=None):
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    """
    from astropy.visualization import ZScaleInterval as Interval
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize
    
    nband = len(band)

    fig, ax = plt.subplots(1, 3, figsize=(inchperband*nband, nband))
    for filt, ax1 in zip(band, ax):

        img = data['{}_masked'.format(filt)]
        #img = data[filt]

        norm = ImageNormalize(img, interval=Interval(contrast=0.95),
                              stretch=Stretch(a=0.95))

        im = ax1.imshow(img, origin='lower', norm=norm, cmap='viridis')
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, #fontweight='bold',
                 ha='center', va='center', color='k', fontsize=14)

        if mgefit:
            from mge.mge_print_contours import _multi_gauss, _gauss2d_mge

            sigmapsf = np.atleast_1d(0)
            normpsf = np.atleast_1d(1)
            _magrange = 10**(-0.4*np.arange(0, magrange, 1)[::-1]) # 0.5 mag/arcsec^2 steps
            #_magrange = 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

            model = _multi_gauss(mgefit[filt].sol, img, sigmapsf, normpsf,
                                 mgefit[filt].xmed, mgefit[filt].ymed,
                                 mgefit[filt].pa)
            
            peak = data[filt][mgefit[filt].xpeak, mgefit[filt].ypeak]
            levels = peak * _magrange
            s = img.shape
            extent = [0, s[1], 0, s[0]]

            ax1.contour(model, levels, colors='k', linestyles='solid', extent=extent,
                        alpha=0.5, lw=1)

        if geometry:
            from photutils import EllipticalAperture
            
            ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                         geometry.sma*(1 - geometry.eps), geometry.pa)
            ellaper.plot(color='k', lw=1, ax=ax1)

        if ellipsefit:
            if len(ellipsefit[filt]) > 0:
                if indx is None:
                    indx = np.ones(len(ellipsefit[filt]), dtype=bool)

                nfit = len(indx) # len(ellipsefit[filt])
                nplot = np.rint(1*nfit).astype('int')
                
                smas = np.linspace(0, ellipsefit[filt].sma[indx].max(), nplot)
                for sma in smas:
                    efit = ellipsefit[filt].get_closest(sma)
                    x, y, = efit.sampled_coordinates()
                    ax1.plot(x, y, color='k', alpha=0.75)
            else:
                from photutils import EllipticalAperture
                geometry = ellipsefit['{}_geometry'.format(refband)]
                ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                             geometry.sma*(1 - geometry.eps), geometry.pa)
                ellaper.plot(color='k', lw=1, ax=ax1)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    if png:
        print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight', pad_inches=0)
        #import pdb ; pdb.set_trace()
        #plt.close(fig)
    else:
        plt.show()

def display_ellipsefit(ellipsefit, band=('g', 'r', 'z'), refband='r', redshift=None,
                       pixscale=0.262, xlog=False, indx=None, png=None):
    """Display the isophote fitting results."""

    from matplotlib.ticker import FormatStrFormatter

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    if indx is None:
        indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for filt in band:
        ax1.errorbar(ellipsefit[filt].sma[indx] * smascale, ellipsefit[filt].eps[indx],
                     ellipsefit[filt].ellip_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        #ax1.set_ylim(0, 0.5)
        
        ax2.errorbar(ellipsefit[filt].sma[indx] * smascale, np.degrees(ellipsefit[filt].pa[indx]),
                     np.degrees(ellipsefit[filt].pa_err[indx]), fmt='o',
                     markersize=4)#, color=color[filt])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        #ax2.set_ylim(0, 180)

        ax3.errorbar(ellipsefit[filt].sma[indx] * smascale, ellipsefit[filt].x0[indx],
                     ellipsefit[filt].x0_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        ax4.errorbar(ellipsefit[filt].sma[indx] * smascale, ellipsefit[filt].y0[indx],
                     ellipsefit[filt].y0_err[indx], fmt='o',
                     markersize=4)#, color=color[filt])
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right')
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    ax1.set_ylabel('Ellipticity')
    ax2.set_ylabel('Position Angle (deg)')
    ax3.set_xlabel('Semimajor Axis ({})'.format(smaunit))
    ax3.set_ylabel(r'$x_{0}$')
    ax4.set_xlabel('Semimajor Axis ({})'.format(smaunit))
    ax4.set_ylabel(r'$y_{0}$')
    
    if xlog:
        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xscale('log')

    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.15, right=0.85, left=0.15)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_ellipse_sbprofile(ellipsefit, band=('g', 'r', 'z'), refband='r',
                              redshift=None, indx=None, pixscale=0.262,
                              minerr=0.02, png=None):
    """Display the multi-band surface brightness profile."""

    colors = iter(sns.color_palette())

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    if indx is None:
        indx = np.ones(len(ellipsefit[refband]), dtype=bool)
        
    def _sbprofile(ellipsefit, indx, smascale):
        """Convert fluxes to magnitudes and colors."""
        sbprofile = dict()
        sbprofile['sma'] = ellipsefit['r'].sma[indx] * smascale
        
        with np.errstate(invalid='ignore'):
            for filt in band:
                sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
                sbprofile['{}_err'.format(filt)] = ellipsefit[filt].int_err[indx] / \
                  ellipsefit[filt].intens[indx] / np.log(10)

                # Just for the plot use a minimum uncertainty
                sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr
                
        sbprofile['gr'] = sbprofile['g'] - sbprofile['r']
        sbprofile['rz'] = sbprofile['r'] - sbprofile['z']
        sbprofile['gr_err'] = np.sqrt(sbprofile['g_err']**2 + sbprofile['r_err']**2)
        sbprofile['rz_err'] = np.sqrt(sbprofile['r_err']**2 + sbprofile['z_err']**2)

        # Just for the plot use a minimum uncertainty
        sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
        sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

        return sbprofile

    sbprofile = _sbprofile(ellipsefit, indx, smascale)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for filt in band:
        ax1.fill_between(sbprofile['sma'], sbprofile[filt]-sbprofile['{}_err'.format(filt)],
                         sbprofile[filt]+sbprofile['{}_err'.format(filt)],
                         label=r'${}$'.format(filt), color=next(colors), alpha=0.75)
    ax1.set_ylabel('AB Magnitude')
    ax1.set_ylim(32.99, 20)
    #ax1.invert_yaxis()
    ax1.legend(loc='upper right')

    ax2.fill_between(sbprofile['sma'], sbprofile['rz']-sbprofile['rz_err'],
                     sbprofile['rz']+sbprofile['rz_err'],
                     label=r'$r - z$', color=next(colors), alpha=0.75)
    
    ax2.fill_between(sbprofile['sma'], sbprofile['gr']-sbprofile['gr_err'],
                     sbprofile['gr']+sbprofile['gr_err'],
                     label=r'$g - r$', color=next(colors), alpha=0.75)

    ax2.set_xlabel('Semimajor Axis ({})'.format(smaunit), alpha=0.75)
    ax2.set_ylabel('Color')
    ax2.set_ylim(-0.5, 2.5)
    ax2.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.0)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
def display_mge_sbprofile(mgefit, band=('g', 'r', 'z'), refband='r', redshift=None,
                          indx=None, pixscale=0.262, png=None):
    """Display the multi-band surface brightness profile."""

    colors = iter(sns.color_palette())

    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

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
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()
        
