"""
legacyhalos.qa
==============

Code to do produce various QA (quality assurance) plots. 

https://xkcd.com/color/rgb/

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import legacyhalos.misc

sns, _ = legacyhalos.misc.plot_style()
#snscolors = sns.color_palette()

#import matplotlib as mpl 
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['font.family'] = 'serif'

#from matplotlib import rc
#rc('text.usetex'] = True
#rc('font.family'] = 'serif'

def _sbprofile_colors():
    """Return an iterator of colors good for the surface brightness profile plots.
    https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette

    """
    _colors = sns.color_palette('Set1', n_colors=8, desat=0.75)
    colors = iter([_colors[1], _colors[2], _colors[0], _colors[3], _colors[4]])
    return colors

def display_sersic(sersic, png=None, verbose=False):
    """Plot a wavelength-dependent surface brightness profile and model fit.

    """
    markers = iter(['o', 's', 'D'])
    colors = _sbprofile_colors()

    if sersic['success']:
        smascale = legacyhalos.misc.arcsec2kpc(sersic['redshift'])
        model = sersic['bestfit']
    else:
        smascale = 1
        model = None

    ymnmax = [40, 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for band, lam in zip( sersic['band'], (sersic['lambda_g'],
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
            filt = '${}:\ $'.format(band)
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
            else:
                raise ValueError('Unrecognized model type {}'.format(sersic['modeltype']))
        else:
            label = band
            labelfont = 14

        col = next(colors)
        #ax.plot(rad, 22.5-2.5*np.log10(sb), label=band)
        #ax.scatter(rad, 22.5-2.5*np.log10(sb), color=col,
        #           alpha=1, s=50, label=label, marker=next(markers))
        mu = 22.5 - 2.5 * np.log10(sb)
        muerr = 2.5 * sberr / np.log(10) / sb
            
        ax.fill_between(rad, mu-muerr, mu+muerr, color=col, label=label, alpha=0.9)

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
                
        ax.text(0.07, 0.04, txt, ha='left', va='bottom', linespacing=1.3,
                transform=ax.transAxes, fontsize=12)

    ax.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
    ax.set_ylabel(r'Surface Brightness $\mu(r)$ (mag arcsec$^{-2}$)')

    ylim = [ymnmax[0]-0.5, ymnmax[1]+0.5]
    if ylim[1] < 32.5:
        ylim[1] = 32.5

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
        ax.fill_between([0, 3*model.psfsigma_r*sersic['pixscale']], [ylim[0], ylim[0]], # [arcsec]
                        [ylim[1], ylim[1]], color='grey', alpha=0.1)
        ax.text(0.03, 0.07, 'PSF\n(3$\sigma$)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

    fig.subplots_adjust(bottom=0.15, top=0.85, right=0.95, left=0.12)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)#, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def display_multiband(data, geometry=None, mgefit=None, ellipsefit=None, indx=None,
                      magrange=10, inchperband=3, contours=False, png=None,
                      verbose=True, vertical=False):
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    vertical -- for talks...

    """
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize

    band = data['band']
    nband = len(band)

    #cmap = 'RdBu_r'
    #from astropy.visualization import PercentileInterval as Interval
    #interval = Interval(0.9)

    cmap = 'viridis'
    from astropy.visualization import ZScaleInterval as Interval
    interval = Interval(contrast=0.9)

    #cmap = {'g': 'winter_r', 'r': 'summer', 'z': 'autumn_r'}
    #cmap = {'g': 'Blues', 'r': 'Greens', 'z': 'Reds'}

    stretch = Stretch(a=0.95)

    if vertical:
        fig, ax = plt.subplots(3, 1, figsize=(nband, inchperband*nband))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(inchperband*nband, nband))
        
    for filt, ax1 in zip(band, ax):

        img = data['{}_masked'.format(filt)]
        #img = data[filt]

        norm = ImageNormalize(img, interval=interval, stretch=stretch)

        im = ax1.imshow(img, origin='lower', norm=norm, cmap=cmap, #cmap=cmap[filt],
                        interpolation='nearest')
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, #fontweight='bold',
                 ha='center', va='center', color='k', fontsize=16)

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
                        extent=extent, alpha=0.5, lw=1)

        if geometry:
            from photutils import EllipticalAperture
            
            ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                         geometry.sma*(1 - geometry.eps), geometry.pa)
            ellaper.plot(color='k', lw=1, ax=ax1, alpha=0.75)

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
                        ax1.plot(x, y, color='k', lw=1, alpha=0.5)
            else:
                from photutils import EllipticalAperture
                geometry = ellipsefit['geometry']
                ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                             geometry.sma*(1 - geometry.eps), geometry.pa)
                ellaper.plot(color='k', lw=1, ax=ax1, alpha=0.5)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        #ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    if vertical:
        fig.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    else:
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

    if ellipsefit['success']:
        
        band, refband = ellipsefit['band'], ellipsefit['refband']
        pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift) # [kpc/arcsec]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
        
        good = (ellipsefit[refband].stop_code < 4)
        bad = ~good
        ax1.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                         ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax1.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].eps[bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)

        #ax1.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             ellipsefit[refband].eps[good],
        #             ellipsefit[refband].ellip_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax1.set_ylim(0, 0.5)
        ax1.xaxis.set_major_formatter(ScalarFormatter())

        ax2.fill_between(ellipsefit[refband].sma[good] * pixscale, 
                         np.degrees(ellipsefit[refband].pa[good]-ellipsefit[refband].pa_err[good]),
                         np.degrees(ellipsefit[refband].pa[good]+ellipsefit[refband].pa_err[good]))#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax2.scatter(ellipsefit[refband].sma[bad] * pixscale, np.degrees(ellipsefit[refband].pa[bad]),
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax2.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             np.degrees(ellipsefit[refband].pa[good]),
        #             np.degrees(ellipsefit[refband].pa_err[good]), fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax2.set_ylim(0, 180)

        ax3.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         ellipsefit[refband].x0[good]-ellipsefit[refband].x0_err[good],
                         ellipsefit[refband].x0[good]+ellipsefit[refband].x0_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax3.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].x0[bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax3.errorbar(ellipsefit[refband].sma[good] * smascale, ellipsefit[refband].x0[good],
        #             ellipsefit[refband].x0_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        ax3.xaxis.set_major_formatter(ScalarFormatter())
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax4.fill_between(ellipsefit[refband].sma[good] * pixscale, 
                         ellipsefit[refband].y0[good]-ellipsefit[refband].y0_err[good],
                         ellipsefit[refband].y0[good]+ellipsefit[refband].y0_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax4.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].y0[bad],
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
            xx.set_xlim(xmin=0)
        
        xlim = ax1.get_xlim()
        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax1_twin.set_xlabel('Galactocentric radius (kpc)')

        ax2_twin = ax2.twiny()
        ax2_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax2_twin.set_xlabel('Galactocentric radius (kpc)')

        ax1.set_ylabel(r'Ellipticity $\epsilon$')
        ax2.set_ylabel('Position Angle (deg)')
        ax3.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        ax3.set_ylabel(r'$x$ Center')
        ax4.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        ax4.set_ylabel(r'$y$ Center')

        if xlog:
            for xx in (ax1, ax2, ax3, ax4):
                xx.set_xscale('log')

        fig.subplots_adjust(hspace=0.03, wspace=0.03, bottom=0.15, right=0.85, left=0.15)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
        
def display_ellipse_sbprofile(ellipsefit, skyellipsefit={}, minerr=0.0,
                              png=None, verbose=True):
    """Display the multi-band surface brightness profile.

    """
    import astropy.stats
    from legacyhalos.ellipse import ellipse_sbprofile

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)

        colors = _sbprofile_colors()

        band, refband = ellipsefit['band'], ellipsefit['refband']
        redshift, pixscale = ellipsefit['redshift'], ellipsefit['pixscale']
        smascale = legacyhalos.misc.arcsec2kpc(redshift) # [kpc/arcsec]

        yminmax = [40, 0]
        xminmax = [0, 0]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                                                 gridspec_kw = {'height_ratios':[0.8, 0.8, 2, 1.2]})

        # ax1 - ellipticity versus semi-major axis
        good = (ellipsefit[refband].stop_code < 4)
        bad = ~good
        if False:
            ax1.fill_between(ellipsefit[refband].sma[good] * pixscale,
                             ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                             ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good])#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax1.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].eps[bad],
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        else:
            ax1.plot(ellipsefit[refband].sma * pixscale, ellipsefit[refband].eps, zorder=1, alpha=0.9, lw=2)
            ax1.scatter(ellipsefit[refband].sma * pixscale, ellipsefit[refband].eps,
                        marker='s', s=50, edgecolor='k', lw=2, alpha=0.75, zorder=2)
            #ax1.fill_between(ellipsefit[refband].sma * pixscale,
            #                 ellipsefit[refband].eps-0.02,
            #                 ellipsefit[refband].eps+0.02, color='gray', alpha=0.5)

        # ax2 - position angle versus semi-major axis
        if False:
            ax2.fill_between(ellipsefit[refband].sma[good] * pixscale, 
                             np.degrees(ellipsefit[refband].pa[good]-ellipsefit[refband].pa_err[good]),
                             np.degrees(ellipsefit[refband].pa[good]+ellipsefit[refband].pa_err[good]))#,
                             #edgecolor='k', lw=2)
            if np.count_nonzero(bad) > 0:
                ax2.scatter(ellipsefit[refband].sma[bad] * pixscale, np.degrees(ellipsefit[refband].pa[bad]),
                            marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        else:
            ax2.plot(ellipsefit[refband].sma * pixscale, np.degrees(ellipsefit[refband].pa), zorder=1, alpha=0.9, lw=2)
            ax2.scatter(ellipsefit[refband].sma * pixscale, np.degrees(ellipsefit[refband].pa),
                        marker='s', s=50, edgecolor='k', lw=2, alpha=0.75, zorder=2)
            #ax2.fill_between(ellipsefit[refband].sma * pixscale,
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
            mu = sbprofile['mu_{}'.format(filt)]
            muerr = sbprofile['mu_{}_err'.format(filt)]

            #good = (ellipsefit[filt].stop_code < 4)
            #bad = ~good
            
            #with np.errstate(invalid='ignore'):
            #    good = np.isfinite(mu) * (mu / muerr > 3)
            good = np.isfinite(mu)
            sma = sma[good]
            mu = mu[good]
            muerr = muerr[good]
                
            col = next(colors)
            ax3.fill_between(sma, mu-muerr, mu+muerr, label=r'${}$'.format(filt), color=col,
                             alpha=0.75, edgecolor='k', lw=2)

            if np.nanmin(mu-muerr) < yminmax[0]:
                yminmax[0] = np.nanmin(mu-muerr)
            if np.nanmax(mu+muerr) > yminmax[1]:
                yminmax[1] = np.nanmax(mu+muerr)
            if np.nanmax(sma) > xminmax[1]:
                xminmax[1] = np.nanmax(sma)

            if bool(skyellipsefit):
                skysma = skyellipsefit['sma'] * pixscale

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

            #ax3.axhline(y=ellipsefit['mu_{}_sky'.format(filt)], color=col, ls='--')
            #if filt == refband:
            #    ysky = ellipsefit['mu_{}_sky'.format(filt)] - 2.5 * np.log10(0.1) # 10% of sky
            #    ax3.axhline(y=ysky, color=col, ls='--')

        ax3.set_ylabel(r'$\mu(a)$ (mag arcsec$^{-2}$)')
        #ax3.set_ylabel(r'Surface Brightness $\mu(a)$ (mag arcsec$^{-2}$)')
        #ax3.set_ylabel(r'Surface Brightness $\mu(r)$ (mag arcsec$^{-2}$)')

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
        ax1_twin.set_xlabel('Semi-major Axis $a$ (kpc)')

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

        ax4.set_xlabel(r'Semi-major Axis $a$ (arcsec)')
        #ax4.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        #ax4.legend(loc='upper left')
        ax4.legend(bbox_to_anchor=(0.25, 0.99))
        
        ax4.set_ylabel('Color (mag)')
        ax4.set_ylim(-0.5, 2.8)

        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xlim(xlim)
            
            ylim = xx.get_ylim()
            xx.fill_between([0, 3*ellipsefit['psfsigma_r']*ellipsefit['pixscale']], [ylim[0], ylim[0]],
                            [ylim[1], ylim[1]], color='grey', alpha=0.1)
            
        ax4.text(0.03, 0.09, 'PSF\n(3$\sigma$)', ha='center', va='center',
            transform=ax4.transAxes, fontsize=10)

        fig.subplots_adjust(hspace=0.0)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()
        
def _display_ellipse_sbprofile(ellipsefit, skyellipsefit={}, minerr=0.0,
                               png=None, use_ylim=None, verbose=True):
    """Display the multi-band surface brightness profile.

    """
    import astropy.stats
    from legacyhalos.ellipse import ellipse_sbprofile

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)

        colors = _sbprofile_colors()

        band, refband = ellipsefit['band'], ellipsefit['refband']
        redshift = ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift) # [kpc/arcsec]

        yminmax = [40, 0]
        xminmax = [0, 0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw = {'height_ratios':[2, 1]})
        for filt in band:
            sma = sbprofile['sma']
            mu = sbprofile['mu_{}'.format(filt)]
            muerr = sbprofile['mu_{}_err'.format(filt)]

            #good = (ellipsefit[filt].stop_code < 4)
            #bad = ~good
            
            #with np.errstate(invalid='ignore'):
            #    good = np.isfinite(mu) * (mu / muerr > 3)
            good = np.isfinite(mu)
            sma = sma[good]
            mu = mu[good]
            muerr = muerr[good]
                
            col = next(colors)
            ax1.fill_between(sma, mu-muerr, mu+muerr, label=r'${}$'.format(filt), color=col,
                             alpha=0.75, edgecolor='k', lw=2)
            if bool(skyellipsefit):
                skysma = skyellipsefit['sma'] * ellipsefit['pixscale']

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sky = astropy.stats.mad_std(skyellipsefit[filt], axis=1, ignore_nan=True)
                    # sky = np.nanstd(skyellipsefit[filt], axis=1) # / np.sqrt(skyellipsefit[
                    
                skygood = np.isfinite(sky)
                skysma = skysma[skygood]
                skymu = 22.5 - 2.5 * np.log10(sky[skygood])
                ax1.plot( skysma, skymu , color=col, ls='--', alpha=0.5)

            if np.nanmin(mu-muerr) < yminmax[0]:
                yminmax[0] = np.nanmin(mu-muerr)
            if np.nanmax(mu+muerr) > yminmax[1]:
                yminmax[1] = np.nanmax(mu+muerr)
            if np.nanmax(sma) > xminmax[1]:
                xminmax[1] = np.nanmax(sma)

            #ax1.axhline(y=ellipsefit['mu_{}_sky'.format(filt)], color=col, ls='--')
            #if filt == refband:
            #    ysky = ellipsefit['mu_{}_sky'.format(filt)] - 2.5 * np.log10(0.1) # 10% of sky
            #    ax1.axhline(y=ysky, color=col, ls='--')

        ax1.set_ylabel(r'Surface Brightness $\mu(a)$ (mag arcsec$^{-2}$)')

        ylim = [yminmax[0]-0.5, yminmax[1]+0.5]
        if ylim[0] < 17:
            ylim[0] = 17
        if ylim[1] > 33:
            ylim[1] = 33

        if use_ylim is not None:
            ax1.set_ylim(use_ylim)
        else:
            ax1.set_ylim(ylim)
        ax1.invert_yaxis()

        xlim = [xminmax[0], xminmax[1]*1.01]
        ax1.set_xlim(xlim)
        #ax1.set_xlim(xmin=0)
        #ax1.margins(xmargin=0)
        
        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        #ax1_twin.set_xlabel(r'Galactocentric radius $r$ (kpc)')
        ax1_twin.set_xlabel(r'Semi-major Axis $a$ (kpc)')

        #ax1.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
        #ax1.set_ylim(31.99, 18)

        ax1.legend(loc='upper right')

        ax2.fill_between(sbprofile['sma'],
                         sbprofile['gr'] - sbprofile['gr_err'],
                         sbprofile['gr'] + sbprofile['gr_err'],
                         label=r'$g - r$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax2.fill_between(sbprofile['sma'],
                         sbprofile['rz'] - sbprofile['rz_err'],
                         sbprofile['rz'] + sbprofile['rz_err'],
                         label=r'$r - z$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        ax2.set_xlabel(r'Semi-major Axis $a$ (arcsec)')
        #ax2.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        #ax2.legend(loc='upper left')
        ax2.legend(bbox_to_anchor=(0.25, 0.98))
        
        ax2.set_ylabel('Color (mag)')
        ax2.set_ylim(-0.5, 3)

        for xx in (ax1, ax2):
            ylim = xx.get_ylim()
            xx.fill_between([0, 3*ellipsefit['psfsigma_r']*ellipsefit['pixscale']], [ylim[0], ylim[0]],
                            [ylim[1], ylim[1]], color='grey', alpha=0.1)
            
        ax2.text(0.03, 0.07, 'PSF\n(3$\sigma$)', ha='center', va='center',
            transform=ax2.transAxes, fontsize=10)

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

    ax2.set_xlabel('Galactocentric radius $r$ ({})'.format(smaunit), alpha=0.75)
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
        
def sample_trends(sample, htmldir, analysisdir=None, verbose=True, xlim=(0, 100)):
    """Trends with the whole sample.

    """
    from astropy.cosmology import WMAP9 as cosmo
    from legacyhalos.io import get_objid, read_ellipsefit
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
                    smascale = legacyhalos.misc.arcsec2kpc(redshift) # [kpc/arcsec]
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
        ax1.set_xlabel('Galactocentric radius $r$ (kpc)')

        fig.subplots_adjust(bottom=0.15, right=0.95, left=0.15, top=0.95)

        if png:
            if verbose:
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
                    smascale = ellipsefit['pixscale'] * legacyhalos.misc.arcsec2kpc(redshift) # [kpc/pixel]
                    
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
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()

    # Build all the plots here.
    
    _color_vs_sma()       # color vs semi-major axis
    _ellipticity_vs_sma() # ellipticity vs semi-major axis

def display_ccdpos(onegal, ccds, radius=None, pixscale=0.262, png=None, verbose=False):
    """Visualize the position of all the CCDs contributing to the image stack of a
    single galaxy.

    """
    if radius is None:
        radius = legacyhalos.misc.cutout_radius_150kpc(
            redshift=onegal['Z'], pixscale=pixscale) # [pixels]

    wcs = legacyhalos.misc.simple_wcs(onegal, radius=radius, pixscale=pixscale)
    width, height = wcs.get_width() * pixscale / 3600, wcs.get_height() * pixscale / 3600 # [degrees]
    bb, bbcc = wcs.radec_bounds(), wcs.radec_center() # [degrees]
    pad = 0.2 # [degrees]

    fig, allax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)

    for ax, band in zip(allax, ('g', 'r', 'z')):
        ax.set_aspect('equal')
        ax.set_xlim(bb[0]+width+pad, bb[0]-pad)
        ax.set_ylim(bb[2]-pad, bb[2]+height+pad)
        ax.set_xlabel('RA (deg)')
        ax.text(0.9, 0.05, band, ha='center', va='bottom',
                transform=ax.transAxes, fontsize=18)

        if band == 'g':
            ax.set_ylabel('Dec (deg)')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        #ax.add_patch(patches.Rectangle((bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2],
        #                               fill=False, edgecolor='black', lw=3, ls='--'))
        ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), radius * pixscale / 3600,
                                    fill=False, edgecolor='black', lw=2))

        these = np.where(ccds.filter == band)[0]
        col = plt.cm.Set1(np.linspace(0, 1, len(ccds)))
        for ii, ccd in enumerate(ccds[these]):
            print(ccd.expnum, ccd.ccdname, ccd.filter)
            W, H, ccdwcs = legacyhalos.misc.ccdwcs(ccd)

            cc = ccdwcs.radec_bounds()
            ax.add_patch(patches.Rectangle((cc[0], cc[2]), cc[1]-cc[0],
                                           cc[3]-cc[2], fill=False, lw=2, 
                                           edgecolor=col[these[ii]],
                                           label='ccd{:02d}'.format(these[ii])))
            ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=10)

    plt.subplots_adjust(bottom=0.12, wspace=0.05, left=0.1, right=0.97)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

