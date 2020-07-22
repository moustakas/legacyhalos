#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import astropy.io.fits as fits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

ccddir = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/'
sgadir = '/global/cscratch1/sd/ioannis/SGA-data-dr9alpha/'
outroot= '/global/cscratch1/sd/raichoor/tmpdir/tmp'

# returns pixels on the edge of the footprint
def get_footedge(isfoot,nside,nest):
    # isfoot = hp.nside2npix(nside) boolean array
    npix    = hp.nside2npix(nside)
    if (len(isfoot)!=npix):
        print('isfoot doesn t have hp.nside2npix(nside) pixels!')
        exit()
    hpind  = np.arange(npix,dtype=int)[isfoot]
    neighb = [hp.get_all_neighbours(nside,i,nest=nest) for i in hpind]
    isedge = np.zeros(npix,dtype=bool)
    isedge_sub = np.array([len(np.where(isfoot[x])[0])<8 for x in neighb])
    isedge[hpind] = isedge_sub
    return isedge


# dr9 ccds
d0  = fits.open(ccddir+'survey-ccds-90prime-dr9.kd.fits')[1].data
d1  = fits.open(ccddir+'survey-ccds-mosaic-dr9.kd.fits')[1].data
d2  = fits.open(ccddir+'survey-ccds-decam-dr9.kd.fits')[1].data
ccd = {}
for key in ['ra','dec','filter']:
	ccd[key] = np.array(d0[key].tolist() + d1[key].tolist() + d2[key].tolist())

# 
nside,nest  = 64,True
npix        = hp.nside2npix(nside)
pixs        = np.arange(npix,dtype=int)
theta,phi   = hp.pix2ang(nside,pixs,nest=nest)
ra,dec      = 180./np.pi*phi,90.-180./np.pi*theta

# dr9 pixels with g+r+z
ccdpixs     = hp.ang2pix(nside,(90.-ccd['dec'])*np.pi/180.,ccd['ra']*np.pi/180.,nest=nest)
isgrz       = np.ones(npix,dtype=bool)
for band in ['g','r','z']:
	reject = ~np.in1d(pixs,ccdpixs[ccd['filter']!=band])
	isgrz[reject] = False

# identifying pixels on the edge of the footprint
edgepixs = pixs[get_footedge(isgrz,nside,nest)]
# "thickening the border"
for step in range(5):
	edgepixs = np.unique(edgepixs.tolist() + sum([hp.get_all_neighbours(nside,p,nest=nest).tolist() for p in edgepixs],[])) ## safe: adding pixels on the edge

# sga
sga = fits.open(sgadir+'SGA-dropped-v3.0.fits')[1].data
sgapixs = hp.ang2pix(nside,(90.-sga['dec'])*np.pi/180.,sga['ra']*np.pi/180.,nest=nest)
inside = ((sga['dropbit'] & 2**1)!=0) & (~np.in1d(sgapixs,edgepixs))
onedge = ((sga['dropbit'] & 2**1)!=0)  & (np.in1d(sgapixs,edgepixs))


#
fig,ax = plt.subplots()
ax.scatter(ra[edgepixs],dec[edgepixs],c='y',zorder=0,s=2,label='healpix edge pixels')
ax.scatter(sga['ra'][onedge],sga['dec'][onedge],c='r',s=5,alpha=0.5,label='sga on edge ('+str(onedge.sum())+')')
ax.scatter(sga['ra'][inside],sga['dec'][inside],c='g',s=5,alpha=0.5,label='sga inside  ('+str(inside.sum())+')')
ax.grid(True)
ax.legend()
plt.savefig(outroot+'.png',bbox_inches='tight')
plt.close()

#
h   = fits.open(sgadir+'SGA-dropped-v3.0.fits')
h[1].data = h[1].data[inside]
h.writeto(outroot+'.fits',overwrite=True)




