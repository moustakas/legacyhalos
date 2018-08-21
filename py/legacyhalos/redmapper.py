"""
legacyhalos.misc
================

Miscellaneous utility code used by various scripts.

"""
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.special

import cosmo
import pz_utils


#Contains various P(z) manipulation utilities
#current treats pzbins as bin midpoints, uniform binning

#Probability that a cluster is in a specified bin in redshift
#For an array of clusters

def p_in_zbin(pz,pzbins,zmin,zmax):
    if zmin >= zmax:
        print >> sys.stderr, "ERROR:  zmin >= zmax"
        sys.exit(1)

    dz = np.copy(pzbins[:,1]-pzbins[:,0])
    p = np.zeros(len(pz))
    #Case where PDF is entirely in the pzbins range
    alist = np.where( (zmin<np.min(pzbins,1)) & (zmax>np.max(pzbins,1)) )[0]
    if len(alist) != 0:
        p[alist]=0*alist+1.

    #zmax cutoff in the pzbins range
    blist = np.where( (zmin<np.min(pzbins,1)) & (zmax<np.max(pzbins,1)) & (zmax>np.min(pzbins,1)) )[0]
    if len(blist) > 0:
        #print "b1"
        for i in range(len(blist)):
            bmax = np.max(np.where(pzbins[blist[i],:] < zmax)[0])
            slope = (pz[blist[i],bmax+1]-pz[blist[i],bmax])/dz[blist[i]]
            p[blist[i]] = np.sum(pz[blist[i],:bmax+1])*dz[blist[i]] - pz[blist[i],bmax]*dz[blist[i]]/2. + (pz[blist[i],bmax]*2 + slope*(zmax-pzbins[blist[i],bmax]) )*(zmax-pzbins[blist[i],bmax])/2.
            #print p[blist[i]],slope,pzbins[blist[i],bmax],pzbins[blist[i],bmax+1]

    #zmin cutoff in the pzbins range
    blist = np.where( (zmax>np.max(pzbins,1)) & (zmin>np.min(pzbins,1)) & (zmin<np.max(pzbins,1)) )[0]
    if len(blist) > 0:
        #print "b2"
        for i in range(len(blist)):
            bmin = np.min(np.where(pzbins[blist[i],:] > zmin)[0])
            slope = (pz[blist[i],bmin]-pz[blist[i],bmin-1])/dz[blist[i]]
            p[blist[i]] = np.sum(pz[blist[i],bmin:])*dz[blist[i]] - pz[blist[i],bmin]*dz[blist[i]]/2. + (pz[blist[i],bmin]*2 - slope*(pzbins[blist[i],bmin]-zmin) )*(pzbins[blist[i],bmin]-zmin)/2.

    #zmax and zmin cutoff in the pzbins range
    clist = np.where( (zmax<np.max(pzbins,1)) & (zmin>np.min(pzbins,1)) )[0]
    if len(clist) > 0:
        #print "c"
        for i in range(len(clist)):
            cmin = np.min(np.where(pzbins[clist[i],:] > zmin)[0])
            cmax = np.max(np.where(pzbins[clist[i],:] < zmax)[0])
            slope_min = (pz[clist[i],cmin]-pz[clist[i],cmin-1])/dz[clist[i]]
            slope_max = (pz[clist[i],cmax+1]-pz[clist[i],cmax])/dz[clist[i]]
            #print clist,cmin,cmax
            p[clist[i]] = np.sum(dz[clist[i]]*pz[clist[i],cmin:cmax+1]) - pz[clist[i],cmax]*dz[clist[i]]/2. + (pz[clist[i],cmax]*2 + slope_max*(zmax-pzbins[clist[i],cmax]) )*(zmax-pzbins[clist[i],cmax])/2. - pz[clist[i],cmin]*dz[clist[i]]/2. + (pz[clist[i],cmin]*2 - slope_min*(pzbins[clist[i],cmin]-zmin) )*(pzbins[clist[i],cmin]-zmin)/2.

    #Automatic cutoff to avoid issues with p>1 (numerical)
    plist = np.where(p > 1)[0]
    if len(plist) > 0:
        p[plist] = np.zeros(len(plist))+1
        print >> sys.stderr, "WARNING:  Some clusters have P>1, correcting"

    return p


#Similar, but this time does P(lm_min<lambda<lm_max)
#Note this assumes gaussian distribution
def p_in_lmbin(lambda_val,lambda_err,lm_min,lm_max):
    alist = np.where(lambda_err>0)[0]
    blist = np.where(lambda_err==0)[0]
    p = np.zeros_like(lambda_val).astype(float)
    if len(alist) > 0:
        p[alist] = 0.5*(scipy.special.erf((lm_max-lambda_val[alist])/lambda_err[alist]/np.sqrt(2)) - scipy.special.erf((lm_min-lambda_val[alist])/lambda_err[alist]/np.sqrt(2)))
    if len(blist) > 0:
        clist = np.where( (lambda_err == 0) & (lambda_val >= lm_min) & (lambda_val < lm_max) )[0]
        p[clist] = 0.*clist+1.

    return p


#Single random selection from P(z) distribution
def select_rand_z(pz,pzbins):
    z = -1.
    val = 0.
    p = 1.
    pmax = np.max(pz)
    dz = pzbins[1]-pzbins[0]
    mybins = np.where(pz > 0)[0]
    zmin = np.min(pzbins[mybins])
    zmax = np.max(pzbins[mybins])
        
    count = 0
    while p > val and count < 100:
        z = np.random.random_sample()*(zmax-zmin) + zmin
        p = np.random.random_sample()*pmax
        if len(np.where(z > pzbins)[0]) == 0:
            
            print >> sys.stderr, pmax, p, val, zmin, zmax, z
            val = pz[0]
        else:
            bin = max(np.where(z > pzbins)[0])
            val = pz[bin] + (z-pzbins[bin])*(pz[bin+1]-pz[bin])/dz
        count = count+1
        
    return z


#Make selection for a set of clusters
def select_rand_z_set(pz,pzbins):
    nclusters = len(pz)
    z = np.zeros(nclusters)
    for i in range(nclusters):
        z[i] = select_rand_z(pz[i],pzbins[i])

    return z

#Make a set of nboot boostrap samples from an input catalog
#Also makes redshifts drawn from P(z)
def make_boot_samples(nboot,cat):
    nclusters = len(cat)
    bootlist = np.random.randint(nclusters,size=(nboot,nclusters))
    zboot = np.zeros([nboot,nclusters])

    for i in range(nboot):
        zboot[i] = select_rand_z_set(cat[bootlist[i]]['pz'],cat[bootlist[i]]['pzbins'])

    return bootlist, zboot

#Makes a random galaxy sample using given probabilities
def make_boot_samples_gal(nboot,p):
    ngals = len(p)

    gboot = np.zeros([nboot,ngals])
    for i in range(nboot):
        gboot[i,:] = np.random.uniform(size=ngals)

    return gboot


#Determines the bootstrapped "lambda" value given gboot and cluster list
#Includes options necessary for doing appropriate lambda in abundance-matched case
def make_boot_lambda(bootlist,c_mem_id,scaleval,g_mem_id,gboot,p,
                     lambda_tr=[],do_abm=False):
    nboot = len(bootlist)
    ncl = len(c_mem_id)
    ngals = len(p)
    lmboot = np.zeros([nboot,ncl])

    if do_abm:
        if len(lambda_tr) != ncl:
            print >> sys.stderr, "WARNING: Supplied lambda values wrong length or empty"
            print >> sys.stderr, "         Unable to include corrected lmboot for ABM"
        else:
            #Make sorted lambda_list
            slambda = lambda_tr[np.argsort(lambda_tr)]
    
    for i in range(nboot):
        #Note -- assumes galaxies list already sorted on mem_id
        csort = np.argsort(c_mem_id[bootlist[i]])

        count_lo = 0L
        count_hi = 0L

        for j in range(ncl):
            #First -- check that this isn't the same as the last cluster; if it is,
            #Just copy results from last time and continue
            if c_mem_id[bootlist[i,csort[j]]] == c_mem_id[bootlist[i,csort[j-1]]]:
                lmboot[i,csort[j]] = lmboot[i,csort[j-1]]
                continue

            #If some cluster has been skipped, skip its galaxies
            while g_mem_id[count_lo] < c_mem_id[bootlist[i,csort[j]]]:
                count_lo = count_lo+1
            count_hi = count_lo
            #Get list of galaxies
            while g_mem_id[count_hi] == c_mem_id[bootlist[i,csort[j]]]:
                count_hi = count_hi+1
                if count_hi >= ngals:
                    break


            if count_lo == count_hi:
                continue

            #print >> sys.stderr,i,j,count_lo,count_hi,csort[j],gboot[i,count_lo],p[count_lo],len(lmboot[0])
            #Count galaxies, essentially using Heavisde function
            lmboot[i,csort[j]] = np.sum(0.5*(np.sign(p[count_lo:count_hi] - gboot[i,count_lo:count_hi])+1))*scaleval[bootlist[i,csort[j]]]

            count_lo = count_hi
            if count_hi >= ngals:
                break
        
        #Resort on lambda if necessary and if data supplied for later ABM
        #lambda_tr should be the true cluster lambda values
        if len(lambda_tr) > 0 and do_abm:
            #Sort on lmboot first, then enter slambda values from true clusters
            lmboot[i,:] = slambda[np.argsort(lmboot[i,:])]
            

    return lmboot


#Script for adding the desired subset of "ubermem" galaxies, which always
#includes those dimmer than 0.1L*, and possibly including those with r>r_lambda
def dimmer_rlambda_p(use_p_ext,c_mem_id,lambda_chisq,r_lambda,
                     g_mem_id,p,p_ext,r):
    #Case where we want all the ubermem galaxies
    if use_p_ext == 1:
        return p_ext

    #Otherwise, we include dim ones, but not those outside the cluster r_lambda
    #Set up the index to connect galaxies to clusters
    index = np.zeros(np.max([np.max(c_mem_id),np.max(g_mem_id)])+10).astype(long)-1
    index[c_mem_id] = range(len(c_mem_id))
    my_p = np.copy(p)
    #Then go through all galaxies, keeping those with r<r_lambda and p>0
    for i in range(len(p)):
        #if (r[i] < r_lambda[index[g_mem_id[i]]]) & (p[i] > 0):
        if r[i] < r_lambda[index[g_mem_id[i]]]:
            my_p[i] = p_ext[i]    

    #Add a cutoff at p=0.2 if use_p_ext=3
    if use_p_ext==3:
        plist = np.where(my_p < 0.2)[0]
        my_p[plist] = 0*plist

    return my_p

#Script for adding the desired subset of "ubermem" galaxies, which always
#includes those dimmer than 0.1L*, and possibly including those with r>r_lambda
#Handles the format for redmapper v5.10 and later
def dimmer_rlambda_p_new(use_p_ext,c_mem_id,lambda_chisq,r_lambda,
                         g_mem_id,p,r):
    #e include dim galaxies, but not those outside the cluster r_lambda
    #Set up the index to connect galaxies to clusters
    index = np.zeros(np.max([np.max(c_mem_id),np.max(g_mem_id)])+10).astype(long)-1
    index[c_mem_id] = range(len(c_mem_id))
    my_p = 0.*p
    #Then go through all galaxies, keeping those with r<r_lambda and p>0
    for i in range(len(p)):
        #modified by chto
        if (r[i] < r_lambda[index[g_mem_id[i]]]) & (p[i] > 0):
        #if r[i] < r_lambda[index[g_mem_id[i]]]:
            my_p[i] = p[i]    

    #Add a cutoff at p=0.2 if use_p_ext=8
    if use_p_ext==8:
        plist = np.where(my_p < 0.2)[0]
        my_p[plist] = 0*plist

    return my_p

#New version of bootstrap selection -- galaxies are selected AFTER this function is run,
#Note that this doesn't randomly resample lambda, redshift
def make_boot_samples_simple(nboot, cat):
    nclusters = len(cat)
    bootlist = np.random.randint(nclusters,size=(nboot,nclusters))
        
    return bootlist

#Making a random galaxy sample; note that this depends on the clusters selected
#And is not in the same format as the other version
def make_boot_samples_gal_full(bootlist,c_mem_id,g_mem_id,p):
    '''
    Input: bootlist, c_mem_id, g_mem_id, p
    Output: gboot structure which is an nboot long list of lists of lists of galaxies (whew)
    Each galaxy list is the list of galaxies selected to be in the corresponding cluster at 
    that position
    '''
    nboot = len(bootlist)
    nclusters = len(c_mem_id)
    ngals = len(g_mem_id)

    #Make a pair of nifty hash-tables
    #List provides the FIRST galaxy with each mem_match_id
    #Loops will need to go until next index is reached
    #Note this relies on mem_match_id being arranged in ascending order
    ulist = np.zeros(nclusters).astype(long)
    place = 0
    for i in range(ngals):
        if place == 0:
            ulist[place]=0
            place = place+1
            continue
        if g_mem_id[i] == g_mem_id[i-1]:
            continue
        ulist[place] = i
        place = place+1
        if place >= nclusters:
            break
    match_index = np.zeros(np.max(c_mem_id)+1).astype(long)-1
    match_index[c_mem_id] = ulist
    max_id = np.max(c_mem_id)

    gboot = []
    for i in range(nboot):
        #For each bootstrap sample
        gboot_single = []
        for j in range(nclusters):
            #For each cluster in this sample
            #Find first galaxy in this cluster
            mygal = match_index[c_mem_id[bootlist[i][j]]]
            #Find the end of the range of interest
            if c_mem_id[bootlist[i][j]] == max_id:
                endgal = len(g_mem_id)
            else:
                endgal = match_index[c_mem_id[bootlist[i][j]+1]]
            #Do a random uniform selection for this get of galaxies
            if endgal < mygal:
                print >> sys.stderr, "Something has gone wrong, ",mygal,endgal,i,j
            pselect = np.random.uniform(size=endgal-mygal)
            glist = np.where(pselect < p[mygal:endgal])[0]
            #Append those galaxies that make the cut to the list
            if len(glist) > 0:
                gboot_single.append(glist+mygal)
            else:
            #All galaxies failed -- oops
                gboot_single.append([])

        gboot.append(gboot_single)

    return match_index, gboot


# here

#Note this function is very basic, and has not been updated
def redm_nlambda_calc(cat,area,outdir,zmin,zmax):
    '''
    Calculates n_clusters(lambda) for several preset bins in redshift,
    then outputs the results to  files
    Incorporates P(z), although the current redshift bins are fairly broad
    Output units are in [Mpc/h]^-3
    '''
    cat_zmax = np.max(cat['z_lambda'])
    
    nlbins = 91
    lmbins = 10+np.array(range(nlbins))

    zlist = np.where(zmin < cat_zmax)[0]
    if len(zlist) == 0:
        print >> sys.stderr, "ERROR:  Catalog zmax < 0.15 ?!"
        print >> sys.stderr, "        Unable to calculate n(lambda)"
        return

    zmin = zmin[zlist]
    zmax = zmax[zlist]
    
    n_of_lambda = np.zeros([len(zmin),len(lmbins)])

    for i in range(len(zmin)):
        p_in_bin = pz_utils.p_in_zbin(cat['pz'],cat['pzbins'],zmin[i],zmax[i])
        for j in range(nlbins):
            mylist = np.where( (cat['lambda_chisq'] >= lmbins[j]) & (cat['lambda_chisq'] < lmbins[j] + 1) )[0]
            n_of_lambda[i,j] = np.sum(p_in_bin[mylist])

        #Calculate volume for this redshift range, with default cosmology
        #assumes are given in square degrees
        vol = cosmo.comoving_volume(zmin[i],zmax[i])*area/41252.96

        n_of_lambda[i,:] = n_of_lambda[i,:]/vol

        #print output
        outfile = outdir+"nlambda_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat"
            
        f = open(outfile,'w')
        for j in range(nlbins):
            print >> f, lmbins[j], n_of_lambda[i,j]
        f.close()

    return

def redm_nlambda(lambda_chisq,lambda_err,z,zmin,zmax,area,p_zbin,use_pz=False):
    '''
    Basic n(lambda) calculation; incorporates P(z) and P(lambda) if requested
    Note that this takes the p_in_zbin vector as an input to speed matters up
    '''
    cat_zmax = np.max(z)
    
    nlbins = 91
    lmbins = 10+np.array(range(nlbins))

    zlist = np.where(zmin < cat_zmax)[0]
    if len(zlist) == 0:
        print >> sys.stderr, "ERROR:  Catalog zmax < ",zmin[0]," ?!"
        print >> sys.stderr, "        Unable to calculate n(lambda)"
        return -1

    #print >> sys.stderr, "TEST2: ",len(zmin),len(zlist),cat_zmax
    zmin = zmin[zlist]
    zmax = zmax[zlist]
    
    n_of_lambda = np.zeros([len(zmin),len(lmbins)])

    for i in range(len(zmin)):
        if use_pz:
            for j in range(nlbins):
                p_in_lbin = pz_utils.p_in_lmbin(lambda_chisq,lambda_err,lmbins[j],lmbins[j]+1)
                n_of_lambda[i,j] = np.sum(p_zbin[i]*p_in_lbin)
        else:
            for j in range(nlbins):
                mylist = np.where( (lambda_chisq >= lmbins[j]) & (lambda_chisq < lmbins[j] + 1) & (z >= zmin[i]) & (z < zmax[i]) )[0]
                n_of_lambda[i,j] = len(mylist)

        #Calculate volume for this redshift range, with default cosmology
        #assumes are given in square degrees
        vol = cosmo.comoving_volume(zmin[i],zmax[i])*area/41252.96

        n_of_lambda[i,:] = n_of_lambda[i,:]/vol

    return n_of_lambda

#Updated to include widths in lambda
def redm_nlambda_err(lambda_chisq,lambda_err,z,pz,pzbins,
                     bootlist,outdir,zmin,zmax,area):
    '''
    Calculates n_clusters(lambda) for several preset bins in redshift,
    then outputs the results to  files
    Incorporates P(z), although the current redshift bins are fairly broad
    Output units are in [Mpc/h]^-3
    Includes error calculation
    '''

    nlbins = 91
    lmbins = 10+np.array(range(nlbins))
    nz = len(zmin)
    nclusters = len(lambda_chisq)
    
    #Set up the p_zbin array to save time
    p_zbin = np.zeros([nz,nclusters])
    for i in range(nz):
        p_zbin[i] = pz_utils.p_in_zbin(pz,pzbins,zmin[i],zmax[i])

    #Get the initial results first
    n_of_lambda = redm_nlambda(lambda_chisq,lambda_err,z,zmin,zmax,area,p_zbin,use_pz=True)

    nboot = len(bootlist)
    nl_boot = np.zeros([nboot,len(zmin),nlbins])
    for i in range(nboot):
        nl_boot[i,:,:] = redm_nlambda(lambda_chisq[bootlist[i]],lambda_err[bootlist[i]],z[bootlist[i]],zmin,zmax,area,p_zbin[:,bootlist[i]],use_pz=True)
        
    #Calculate the estimated error and covariance
    nl_err = np.zeros([len(zmin),nlbins])

    for i in range(len(zmin)):
        for j in range(nlbins):
            nl_err[i,j] = np.sum( (n_of_lambda[i,j] - nl_boot[:,i,j])**2 )/(nboot-1.) 
    nl_err = np.sqrt(nl_err)

    #Make the covariance matrix
    nl_covar = np.zeros([len(zmin),nlbins,nlbins])
    for i in range(len(zmin)):
        for j in range(nlbins):
            for k in range(nlbins):
                nl_covar[i,j,k] = np.sum( (n_of_lambda[i,j] - nl_boot[:,i,j])*
                                          (n_of_lambda[i,k] - nl_boot[:,i,k]) )/(nboot-1.) 

    #And print out the results
    for i in range(len(zmin)):
        outfile = outdir+"nlambda_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat"
        f = open(outfile,'w')
        for j in range(nlbins):
            print >> f, lmbins[j],n_of_lambda[i,j],nl_err[i,j]
        f.close()

    #And print out the full covariance matrix
    for i in range(len(zmin)):
        outfile = outdir+"nlambda_covar_z_"+str(zmin[i])+"_"+str(zmax[i])+".dat"
        f = open(outfile,'w')
        for j in range(nlbins):
            for k in range(nlbins):
                f.write(str(nl_covar[i,j,k])+" ")
            f.write("\n")
        f.close()

    return
