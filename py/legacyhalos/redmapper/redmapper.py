"""
legacyhalos.redmapper.redmapper
===============================

Utility code specific to redMaPPer

"""
import numpy as np

import legacyhalos.io
import legacyhalos.misc
from legacyhalos.redmapper import pzutils

def compute_nofz(cat=None, nboot=None, seed=None, area=None,
               dz=0.2, descale=False, verbose=True):
    '''Calculate n(z) for several thresholds in lambda.

    Outputs the results to a file.  Incorporates P(z) and error estimation
    Setting descale=True 

    Parameters
    ----------
    cat : :class:`astropy.table.Table`, optional
        Input catalog.  Default is to read the full legacyhalos sample.
    nboot : :class:`int`, optional, default None
        Number of bootstrap samples to generate.
    seed : :class:`int`, optional, default None
        Seed for reproducibility; only used with nboot.
    area : :class:`float`, optional, default None
        Area of the input catalog [deg**2].
    dz : :class:`float`, optional, default 0.025
        Redshift binning width.
    descale : :class:`bool`, optional, default False
        Apply scaleval correction to lambda_chisq (see Sec 5.1 in Rykoff+14).
    verbose : :class:`bool`, optional, default True
        Be loquacious! 

    Returns
    -------

    Raises
    ------

    '''
    if cat is None:
        cat = legacyhalos.io.read_sample(verbose=verbose)
    ngal = len(cat)

    if area is None:
        area = legacyhalos.misc.area() # [deg2]

    cosmo = legacyhalos.misc.cosmology()
    
    # Optionally create bootstrap subsamples.
    if nboot is not None:
        bootindx = pzutils.bootstrap_resample_simple(ngal, nboot=nboot, seed=seed)

    mylambda = cat['LAMBDA_CHISQ'].data
    mylambda_err = cat['LAMBDA_CHISQ_E'].data
    if descale:
        mylambda /= cat['SCALEVAL'].data
        mylambda_err /= cat['SCALEVAL'].data

    # Bins of redshift and lambda.
    lambins = legacyhalos.misc.get_lambdabins()
    lmin, lmax = lambins[:-1], lambins[1:]
    #lmin = [5, 10, 20, 40, 60]
    #lmax = [10, 20, 200, 200, 200]

    zbins = legacyhalos.misc.get_zbins()
    #zmin, zmax = zbins[:-1], zbins[1:]

    nz = int( np.ceil(zbins.max() / dz) )
    zmin = np.array( range(nz) ) * dz
    zmax = zmin + (zmin[1] - zmin[0])
    zmid = (zmin + zmax) / 2.0
    
    nlambda, nz = len(lmin), len(zmin)

    # Get the comoving volume in each redshift slice.
    vol = np.zeros(nz)
    for jj in range(nz):
        vol[jj] = ( ( cosmo.comoving_volume(zmax[jj]) - cosmo.comoving_volume(zmin[jj]) ).value *
                    area / (4*np.pi*180**2/np.pi**2) )

    nofz = np.zeros([nlambda, nz])     # output probability map
    p_zbin = np.zeros([nz, ngal])      # P(z|lambda) summed over all galaxies
    p_lbin = np.zeros([nlambda, ngal]) # P(lambda|z) summed over all galaxies

    for ii in range(nlambda):
        if verbose:
            print('Get probabilities for being in lambda bin {:02d}/{:02d}, {:.1f}-{:.1f}'.format(
                ii, nlambda, lmin[ii], lmax[ii]))
        p_lbin[ii] = pzutils.p_in_lambdabin(mylambda, mylambda_err, lmin[ii], lmax[ii])
        for jj in range(nz):
            if verbose:
                print('Looping over redshift bin {:02d}/{:02d}, {:.3f}-{:.3f}'.format(
                    jj, nz, zmin[jj], zmax[jj]))
            if ii == 0:
                p_zbin[jj] = pzutils.p_in_zbin(cat['PZ'].data, cat['PZBINS'].data, zmin[jj], zmax[jj])
            nofz[ii, jj] =  np.sum(p_zbin[jj] * p_lbin[ii]) / vol[jj] # [Mpc**-3]

    # Estimate the variance using the bootstrap samples.
    nofz_err = np.zeros_like(nofz)
    if nboot is not None:
        nofz_boot = np.zeros([nboot, nlambda, nz])
        for ii in range(nboot):
            for jj in range(nlambda):
                for kk in range(nz):
                    # Total the probabilities for each bootstrap sample.
                    nofz_boot[ii, jj, kk] = np.sum( p_zbin[kk, bootindx[ii]] * p_lbin[jj, bootindx[ii]] ) / vol[kk]

        for ii in range(nlambda):
            for jj in range(nz):
                nofz_err[ii, jj] = np.sum( (nofz[ii, jj] - nofz_boot[:, ii, jj])**2 ) / (nboot - 1)

    # Write out.
    outfile = 'junk.txt'
    #outfile = outdir + "nz_lm_"+str(lmin[i])+"_"+str(lmax[i])+".dat"
    #if descale:
    #    outfile = outdir + "nz_desc_lm_"+str(lmin[i])+"_"+str(lmax[i])+".dat"

    with open(outfile, 'w') as out:
        for ii in range(nlambda):
            for jj in range(nz):
                out.write( '{:06.3f} {:06.3f} {:06.3f} {:06.3f} {:.5e} {:.3e}\n'.format(
                    lmin[ii], lmax[ii], zmin[jj], zmax[jj], nofz[ii, jj], np.sqrt(nofz_err[ii, jj]) ) )
