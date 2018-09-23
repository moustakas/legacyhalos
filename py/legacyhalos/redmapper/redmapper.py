"""
legacyhalos.redmapper.redmapper
===============================

Utility code specific to redMaPPer

"""
import numpy as np

import legacyhalos.io
import legacyhalos.misc
from legacyhalos.redmapper import pzutils

#import sys
#import scipy.special

#import cosmo


def compute_nz(cat=None, nboot=None, seed=None, area=None,
               dz=0.025, descale=False, verbose=True):
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

    # Initialize the output arrays.
    n_of_z = np.zeros([nlambda, nz])
    p_zbin = np.zeros([nz, ngal])
    p_lbin = np.zeros([nlambda, ngal])

    for i in range(nlambda):
        if verbose:
            print('Get probabilities for being in lambda bin {:02d}/{:02d}, {:.1f}-{:.1f}'.format(
                i, nlambda, lmin[i], lmax[i]))
        p_lbin[i] = pzutils.p_in_lmbin(mylambda, mylambda_err, lmin[i], lmax[i])
        for j in range(nz):
            if verbose:
                print('Looping over redshift bin {:02d}/{:02d}, {:.3f}-{:.3f}'.format(
                    j, nz, zmin[j], zmax[j]))
            if i == 0:
                p_zbin[j] = pzutils.p_in_zbin(cat['PZ'].data, cat['PZBINS'].data, zmin[j], zmax[j])
            n_of_z[i, j] =  np.sum(p_zbin[j] * p_lbin[i])

    # Normalize by area
    #n_of_z = n_of_z / area(zmid) / dz
    n_of_z = n_of_z / area / dz

    import pdb ; pdb.set_trace()

    #print >> sys.stderr, "Ready for a big loop"

    #Make bootstrap error estimate
    #Requires input bootstrap data
    n_of_z_err = np.zeros([nlambda,nz])
    nboot = len(bootlist)
    if nboot > 0:
        n_of_z_boot = np.zeros([nboot,nlambda,nz])
        for i in range(nboot):
            for j in range(nlambda):
                for k in range(nz):
                    #Total up probabilities for each bootstrap sample
                    n_of_z_boot[i,j,k] = np.sum(p_zbin[k,bootlist[i]]*p_lbin[j,bootlist[i]])

        #Normalize
        n_of_z_boot = n_of_z_boot/area(zmid)/dz

        #Estimate errors
        for i in range(nlambda):
            for j in range(nz):
                n_of_z_err[i,j] = np.sum( (n_of_z[i,j] - n_of_z_boot[:,i,j])**2 )/(nboot-1)

    #Print out results
    for i in range(nlambda):
        outfile = outdir + "nz_lm_"+str(lmin[i])+"_"+str(lmax[i])+".dat"
        if descale:
            outfile = outdir + "nz_desc_lm_"+str(lmin[i])+"_"+str(lmax[i])+".dat"
        f = open(outfile,'w')
        for j in range(nz):
            print >> f, zmin[j],zmax[j],n_of_z[i,j],np.sqrt(n_of_z_err[i,j])
        f.close()
