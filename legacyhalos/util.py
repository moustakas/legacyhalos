"""
legacyhalos.util
================

Miscellaneous utility code used by various scripts.

"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np

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

def cutout_radius_100kpc(redshift, pixscale=0.262, radius_kpc=100):
    """Get a cutout radius of 100 kpc [in pixels] at the redshift of the cluster.

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

def ellipse_sbprofile(ellipsefit, band=('g', 'r', 'z'), refband='r',
                      minerr=0.02, redshift=None, pixscale=0.262):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    """
    if redshift:
        from astropy.cosmology import WMAP9 as cosmo
        smascale = pixscale / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/pixel]
        smaunit = 'kpc'
    else:
        smascale = 1.0
        smaunit = 'pixels'

    indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    sbprofile = dict()
    sbprofile['smaunit'] = smaunit
    sbprofile['sma'] = ellipsefit['r'].sma[indx] * smascale

    with np.errstate(invalid='ignore'):
        for filt in band:
            #area = ellipsefit[filt].sarea[indx] * pixscale**2

            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])

            #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
            sbprofile['mu_{}_err'.format(filt)] = ellipsefit[filt].int_err[indx] / \
              ellipsefit[filt].intens[indx] / np.log(10)

            #sbprofile['mu_{}'.format(filt)] = sbprofile[filt] + 2.5 * np.log10(area)

            # Just for the plot use a minimum uncertainty
            #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
    sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
    sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)

    # Just for the plot use a minimum uncertainty
    sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
    sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

    # Add the effective wavelength of each bandpass, although this needs to take
    # into account the DECaLS vs BASS/MzLS filter curves.
    from speclite import filters
    filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')
    for ii, band in enumerate(('g', 'r', 'z', 'W1', 'W2')):
        sbprofile.update({'{}_wave_eff'.format(band): filt.effective_wavelengths[ii].value})

    return sbprofile
