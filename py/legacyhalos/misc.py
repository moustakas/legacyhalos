"""
legacyhalos.misc
================

Miscellaneous utility code used by various scripts.

"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np

def area():
    """Return the area of the DR6+DR7 sample.  See the
    `legacyhalos-sample-selection.ipynb` notebook for this calculation.

    """
    return 6692.0

def cosmology(WMAP=False, Planck=False):
    """Establish the default cosmology for the project."""

    if WMAP:
        from astropy.cosmology import WMAP9 as cosmo
    elif Planck:
        from astropy.cosmology import Planck15 as cosmo
    else:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)        

    return cosmo

def plot_style(paper=False, talk=False):

    import seaborn as sns
    rc = {'font.family': 'serif'}#, 'text.usetex': True}
    #rc = {'font.family': 'serif', 'text.usetex': True,
    #       'text.latex.preamble': r'\boldmath'})
    palette = 'Set2'
    
    if paper:
        palette = 'deep'
        rc.update({'text.usetex': False})
    
    if talk:
        pass

    sns.set(style='ticks', font_scale=1.6, rc=rc)
    sns.set_palette(palette, 12)

    colors = sns.color_palette()
    #sns.reset_orig()

    return sns, colors

def get_logger(logfile):
    """Instantiate a simple logger.

    """
    import logging
    from contextlib import redirect_stdout

    fmt = "%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s"
    #fmt = '%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s:%(asctime)s: %(message)s']
    datefmt = '%Y-%m-%dT%H:%M:%S'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # logging to logfile
    ch = logging.FileHandler(logfile, mode='w')
    #ch.setLevel(logging.INFO)
    ch.setFormatter( logging.Formatter(fmt, datefmt=datefmt) )
    logger.addHandler(ch)

    ### log stdout
    #ch = logging.StreamHandler()
    #ch.setLevel(logging.DEBUG)
    #ch.setFormatter( logging.Formatter(fmt, datefmt=datefmt) )
    #logger.addHandler(ch)
    #
    #logger.write = lambda msg: logger.info(msg) if msg != '\n' else None

    return logger

def destroy_logger(log):
    allhndl = list(log.handlers)
    for hndl in allhndl:
        log.removeHandler(hndl)
        hndl.flush()
        hndl.close()

def cutout_radius_150kpc(redshift, pixscale=0.262, radius_kpc=150):
    """Get a cutout radius of 150 kpc [in pixels] at the redshift of the cluster.

    """
    cosmo = cosmology()
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
    radius = np.rint(radius_kpc * arcsec_per_kpc / pixscale).astype(int) # [pixels]
    return radius

def cutout_radius_cluster(redshift, cluster_radius, pixscale=0.262, factor=1.0,
                          rmin=50, rmax=500, bound=False):
    """Get a cutout radius which depends on the richness radius (in h^-1 Mpc)
    R_LAMBDA of each cluster (times an optional fudge factor).

    Optionally bound the radius to (rmin, rmax).

    """
    cosmo = cosmology()

    radius_kpc = cluster_radius * 1e3 * cosmo.h # cluster radius in kpc
    radius = np.rint(factor * radius_kpc * cosmo.arcsec_per_kpc_proper(redshift).value / pixscale)

    if bound:
        radius[radius < rmin] = rmin
        radius[radius > rmax] = rmax

    return radius # [pixels]

def arcsec2kpc(redshift):
    """Compute and return the scale factor to convert a physical axis in arcseconds
    to kpc.

    """
    cosmo = cosmology()
    return 1 / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/arcsec]

def statsinbins(xx, yy, binsize=0.1, minpts=10, xmin=None, xmax=None):
    """Compute various statistics in running bins along the x-axis.

    """
    from scipy.stats import binned_statistic

    # Need an exception if there are fewer than three arguments.
    if xmin == None:
        xmin = xx.min()
    if xmax == None:
        xmax = xx.max()

    nbin = int( (np.nanmax(xx) - np.nanmin(xx) ) / binsize )
    stats = np.zeros(nbin, [('xmean', 'f4'), ('xmedian', 'f4'), ('xbin', 'f4'),
                            ('npts', 'i4'), ('ymedian', 'f4'), ('ymean', 'f4'),
                            ('ystd', 'f4'), ('y25', 'f4'), ('y75', 'f4')])

    if False:
        def median(x):
            return np.nanmedian(x)

        def mean(x):
            return np.nanmean(x)

        def std(x):
            return np.nanstd(x)

        def q25(x):
            return np.nanpercentile(x, 25)

        def q75(x):
            return np.nanpercentile(x, 75)

        ystat, bin_edges, _ = binned_statistic(xx, yy, bins=nbin, statistic='median')
        stats['median'] = ystat

        bin_width = (bin_edges[1] - bin_edges[0])
        xmean = bin_edges[1:] - bin_width / 2

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic='mean')
        stats['mean'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=std)
        stats['std'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q25)
        stats['q25'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q75)
        stats['q75'] = ystat

        keep = (np.nonzero( stats['median'] ) * np.isfinite( stats['median'] ))[0]
        xmean = xmean[keep]
        stats = stats[keep]
    else:
        _xbin = np.linspace(xmin, xmax, nbin)
        idx  = np.digitize(xx, _xbin)

        for kk in range(nbin):
            these = idx == kk
            npts = np.count_nonzero( yy[these] )

            stats['xbin'][kk] = _xbin[kk]
            stats['npts'][kk] = npts

            if npts > 0:
                stats['xmedian'][kk] = np.nanmedian( xx[these] )
                stats['xmean'][kk] = np.nanmean( xx[these] )

                stats['ystd'][kk] = np.nanstd( yy[these] )
                stats['ymean'][kk] = np.nanmean( yy[these] )

                qq = np.nanpercentile( yy[these], [25, 50, 75] )
                stats['y25'][kk] = qq[0]
                stats['ymedian'][kk] = qq[1]
                stats['y75'][kk] = qq[2]

        keep = stats['npts'] > minpts
        if np.count_nonzero(keep) == 0:
            return None
        else:
            return stats[keep]

def custom_brickname(ra, dec, prefix='custom-'):
    brickname = 'custom-{:06d}{}{:05d}'.format(
        int(1000*ra), 'm' if dec < 0 else 'p',
        int(1000*np.abs(dec)))
    return brickname

def lambda2mhalo(richness, redshift=0.3, Saro=False):
    """
    Convert cluster richness, lambda, to halo mass, given various 
    calibrations.
    
      * Saro et al. 2015: Equation (7) and Table 2 gives M(500).
      * Melchior et al. 2017: Equation (51) and Table 4 gives M(200).
      * Simet et al. 2017: 
    
    Other SDSS-based calibrations: Li et al. 2016; Miyatake et al. 2016; 
    Farahi et al. 2016; Baxter et al. 2016.

    TODO: Return the variance!

    """
    if Saro:
        pass
    
    # Melchior et al. 2017 (default)
    logM0, Flam, Gz, lam0, z0 = 14.371, 1.12, 0.18, 30.0, 0.5
    Mhalo = 10**logM0 * (richness / lam0)**Flam * ( (1 + redshift) / (1 + z0) )**Gz
    
    return Mhalo

def radec2pix(nside, ra, dec):
    '''Convert `ra`, `dec` to nested pixel number.

    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.

    Returns:
        Array of integer pixel numbers using nested numbering scheme.

    Notes:
        This is syntactic sugar around::

            hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

        but also works with older versions of healpy that didn't have
        `lonlat` yet.
    '''
    import healpy as hp
    theta, phi = np.radians(90-dec), np.radians(ra)
    if np.isnan(np.sum(theta)) :
        raise ValueError("some NaN theta values")

    if np.sum((theta < 0)|(theta > np.pi))>0 :
        raise ValueError("some theta values are outside [0,pi]: {}".format(theta[(theta < 0)|(theta > np.pi)]))

    return hp.ang2pix(nside, theta, phi, nest=True)

def pix2radec(nside, pix):
    '''Convert nested pixel number to `ra`, `dec`.

    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.

    Returns:
        Array of RA, Dec coorindates using nested numbering scheme. 

    Notes:
        This is syntactic sugar around::
            hp.pixelfunc.pix2ang(nside, pix, nest=True)
    
    '''
    import healpy as hp

    theta, phi = hp.pixelfunc.pix2ang(nside, pix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)
    
    return ra, dec
    
def get_lambdabins(verbose=False):
    """Fixed bins of richness.
    
    nn = 7
    ll = 10**np.linspace(np.log10(5), np.log10(500), nn)
    #ll = np.linspace(5, 500, nn)
    mh = np.log10(lambda2mhalo(ll))
    for ii in range(nn):
        print('{:.3f}, {:.3f}'.format(ll[ii], mh[ii]))    
    """
    
    # Roughly 13.5, 13.9, 14.2, 14.6, 15, 15.7 Msun
    lambdabins = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 250.0])
    #lambdabins = np.array([5, 25, 50, 100, 500])
    nlbins = len(lambdabins)
    
    mhalobins = np.log10(lambda2mhalo(lambdabins))
    
    if verbose:
        for ii in range(nlbins - 1):
            print('Bin {}: lambda={:03d}-{:03d}, Mhalo={:.3f}-{:.3f} Msun'.format(
                ii, lambdabins[ii], lambdabins[ii+1], mhalobins[ii], mhalobins[ii+1]))
    return lambdabins

def get_zbins(zmin=0.05, zmax=0.6, dt=1.0, verbose=False):
    """Establish redshift bins which are equal in lookback time."""
    import astropy.units as u
    from astropy.cosmology import z_at_value
    
    cosmo = cosmology()
    tmin, tmax = cosmo.lookback_time([zmin, zmax])
    if verbose:
        print('Cosmic time spanned = {:.3f} Gyr'.format(tmax - tmin))
    
    ntbins = np.round( (tmax.value - tmin.value) / dt + 1 ).astype('int')
    #tbins = np.arange(tmin.value, tmax.value, dt) * u.Gyr
    tbins = np.linspace(tmin.value, tmax.value, ntbins) * u.Gyr
    zbins = np.around([z_at_value(cosmo.lookback_time, tt) for tt in tbins], decimals=3)
    tbins = tbins.value
    
    # Now fix the bins:
    # zbins = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.6])
    zbins = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    tbins = cosmo.lookback_time(zbins).value
    
    if verbose:
        for ii in range(ntbins - 1):
            print('Bin {}: z={:.3f}-{:.3f}, t={:.3f}-{:.3f} Gyr'.format(
                ii, zbins[ii], zbins[ii+1], tbins[ii], tbins[ii+1]))
    return zbins

def get_mstarbins(deltam=0.1, satellites=False):
    """Fixed bins of stellar mass.
    
    nn = 7
    ll = 10**np.linspace(np.log10(5), np.log10(500), nn)
    #ll = np.linspace(5, 500, nn)
    mh = np.log10(lambda2mhalo(ll))
    for ii in range(nn):
        print('{:.3f}, {:.3f}'.format(ll[ii], mh[ii]))    
    """

    if satellites:
        pass # code me
    else:
        mstarmin, mstarmax = 9.0, 14.0

    nmstarbins = np.round( (mstarmax - mstarmin) / deltam ).astype('int') + 1
    mstarbins = np.linspace(mstarmin, mstarmax, nmstarbins)
    
    return mstarbins

