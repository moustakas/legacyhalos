"""
legacyhalos.misc
================

Miscellaneous utility code used by various scripts.

"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np

def legacyhalos_plot_style():
    import seaborn as sns
    rc = {'font.family': 'serif', 'text.usetex': True}
    #rc = {'font.family': 'serif', 'text.usetex': True,
    #       'text.latex.preamble': r'\boldmath'})
    sns.set(style='ticks', font_scale=1.5, palette='Set2', rc=rc)
    #sns.reset_orig()

    return sns

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
    from astropy.cosmology import WMAP9 as cosmo
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
    radius = np.rint(radius_kpc * arcsec_per_kpc / pixscale).astype(int) # [pixels]
    return radius

def cutout_radius_cluster(redshift, cluster_radius, pixscale=0.262, factor=1.0,
                          rmin=50, rmax=500, bound=False):
    """Get a cutout radius which depends on the richness radius (in h^-1 Mpc)
    R_LAMBDA of each cluster (times an optional fudge factor).

    Optionally bound the radius to (rmin, rmax).

    """
    from astropy.cosmology import WMAP9 as cosmo

    radius_kpc = cluster_radius * 1e3 * cosmo.h # cluster radius in kpc
    radius = np.rint(factor * radius_kpc * cosmo.arcsec_per_kpc_proper(redshift).value / pixscale)

    if bound:
        radius[radius < rmin] = rmin
        radius[radius > rmax] = rmax

    return radius

def arcsec2kpc(redshift):
    """Compute and return the scale factor to convert a physical axis in arcseconds
    to kpc.

    """
    from astropy.cosmology import WMAP9 as cosmo
    return 1 / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/arcsec]

def medxbin(xx, yy, binsize, minpts=20, xmin=None, xmax=None):
    """
    Compute the median (and other statistics) in fixed bins along the x-axis.
    
    """
    from scipy import ptp

    # Need an exception if there are fewer than three arguments.
    if xmin == None:
        xmin = xx.min()
    if xmax == None:
        xmax = xx.max()

    nbin = int( ptp(xx) / binsize )
    bins = np.linspace(xmin, xmax, nbin)
    idx  = np.digitize(xx, bins)

    stats = np.zeros(nbin, [('median', 'f4'), ('std', 'f4'), ('q25', 'f4'), ('q75', 'f4')])
    for kk in range(nbin):
        npts = len( yy[idx == kk] )
        if npts > minpts:
            stats['std'][kk] = np.nanstd( yy[idx==kk] )

            qq = np.nanpercentile( yy[idx==kk], [25, 50, 75] )
            stats['q25'][kk] = qq[0]
            stats['median'][kk] = qq[1]
            stats['q75'][kk] = qq[2]

    # Remove bins with too few points.
    good = np.nonzero( stats['median'] )
    stats = stats[good]

    return bins[good], stats

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
