"""
legacyhalos.redmapper.redmapper
===============================

Utility code specific to redMaPPer

"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.special

import cosmo
import legacyhalos.misc
from legacyhalos.redmapper import pzutils

def redm_nz_calc(cat, area, outdir, bootlist, descale=False, area=None):
    '''Calculate n(z) for several thresholds in lambda.

    Outputs the results to a file.  Incorporates P(z) and error estimation
    Setting descale=True 


    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Input catalog.
    descale : :class:`bool`, optional, default False
        Apply scaleval correction to lambda_chisq (see Sec 5.1 in Rykoff+14).
    area : :class:`float`, optional, default None
        Area of the input catalog [deg**2].

    Returns
    -------

    Raises
    ------

    '''
    if area is None:
        area = legacyhalos.misc.area() # [deg2]


    #Binning currently determined by area and maximum redshift
    cat_zmax = np.max(cat['z_lambda'])
    if area(0.1) > 500:
        dz = 0.01
    else:
        dz = 0.025
    nz = int(np.ceil(cat_zmax/dz))
    nclusters = len(cat)
    zmin = np.array(range(nz))*dz
    zmax = zmin+(zmin[1]-zmin[0])
    zmid=(zmin+zmax)/2.0

    #Set lambda (richness) binning
    lmin = [5, 10, 20, 40, 60]
    lmax = [10, 20, 200, 200, 200]

    nlambda = len(lmin)
    n_of_z = np.zeros([nlambda,nz])

    if descale:
        mylambda = cat['lambda_chisq']/cat['scaleval']
    else:
        mylambda = cat['lambda_chisq']

    #Make arrays to save weights for later use -- reduce number of calcs
    #Set up p_zbin array
    p_zbin = np.zeros([nz,nclusters])
    #Set up p_lbin array
    p_lbin = np.zeros([nlambda,nclusters])

    #print >> sys.stderr, "First loop..."

    for i in range(nlambda):
        #print >> sys.stderr, i, lmin[i]
        #Get probabilities for being in each lambda bin
        p_lbin[i] = pz_utils.p_in_lmbin(cat['lambda_chisq'],cat['lambda_chisq_e'],lmin[i],lmax[i])
        for j in range(nz):
            #Total up probabilities
            if i == 0:
                p_zbin[j] = pz_utils.p_in_zbin(cat['pz'],cat['pzbins'],zmin[j],zmax[j])
            #print >> sys.stderr, i, j, nz, " Counting..."
            n_of_z[i,j] =  np.sum(p_zbin[j]*p_lbin[i])

    #Normalize by area
    n_of_z = n_of_z/area(zmid)/dz

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

    return
