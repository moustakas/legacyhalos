;+
; NAME:
;   LEGACYHALOS_PROFILES1D_ISEDFIT
;
; PURPOSE:
;   Use iSEDfit to compute stellar masses from the measured 1D
;   surface-brightness profiles.
;
; INPUTS:
;
; OPTIONAL INPUTS:
;   thissfhgrid
;
; KEYWORD PARAMETERS:
;   lsphot_dr6_dr7
;   lhphot
;   sdssphot_dr14
;   write_paramfile
;   build_grids
;   model_photometry
;   isedfit
;   kcorrect
;   qaplot_sed
;   gather_results
;   clobber
;
; OUTPUTS:
;
; COMMENTS:
;   See https://github.com/moustakas/legacyhalos for more info about the
;   fitting. 
;
; MODIFICATION HISTORY:
;   J. Moustakas, 2019 Feb 20, Siena
;
; Copyright (C) 2019, John Moustakas
; 
; This program is free software; you can redistribute it and/or modify 
; it under the terms of the GNU General Public License as published by 
; the Free Software Foundation; either version 2 of the License, or
; (at your option) any later version. 
; 
; This program is distributed in the hope that it will be useful, but 
; WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
; General Public License for more details. 
;-

function legacyhalos_maggies, maggies=maggies, ivarmaggies=ivarmaggies, $
  ra=ra, dec=dec, legacyhalos_dir=legacyhalos_dir, sampleprefix=sampleprefix, $
  rows=rows
    
    catfile = legacyhalos_dir+'paper2-profiles1d-flux.fits'
    splog, 'Reading '+catfile
    if n_elements(rows) ne 0 then begin
       cat = mrdfits(catfile,1,rows=rows)
    endif else begin
       cat = mrdfits(catfile,1)
    endelse

    ra = cat.ra
    dec = cat.dec
    zobj = cat.z
    
    factor = 1D-9 / transpose([ [cat.mw_transmission_g], [cat.mw_transmission_r], $
      [cat.mw_transmission_z] ])

stop
    
    maggies = float(transpose([ [cat.flux_g], [cat.flux_r], [cat.flux_z] ]) * factor)
    ivarmaggies = float(transpose([ [cat.flux_ivar_g], [cat.flux_ivar_r], [cat.flux_ivar_z] ]) / factor^2)
    
; add minimum uncertainties to grzW1W2
    k_minerror, maggies, ivarmaggies, [0.01,0.01,0.02,0.02,0.02]
;   k_minerror, maggies, ivarmaggies, [0.02,0.02,0.02,0.005,0.02]

return, zobj
end

function get_pofm, prefix, outprefix, isedfit_rootdir, $
  thissfhgrid=thissfhgrid, kcorr=kcorr
; read the marginalized posterior on stellar mass, pack it into the iSEDfit
; structure, read the k-corrections and return

    isedfit_dir = isedfit_rootdir+'isedfit_'+prefix+'/'
    montegrids_dir = isedfit_rootdir+'montegrids_'+prefix+'/'
    isedfit_paramfile = isedfit_dir+prefix+'_paramfile.par'

    fp = isedfit_filepaths(read_isedfit_paramfile(isedfit_paramfile,$
      thissfhgrid=thissfhgrid),isedfit_dir=isedfit_dir,band_shift=0.1,$
      montegrids_dir=montegrids_dir,outprefix=outprefix)
    
    npofm = 21 ; number of posterior samplings on stellar mass
    ngal = sxpar(headfits(fp.isedfit_dir+fp.isedfit_outfile+'.gz',ext=1), 'NAXIS2')
    
    nperchunk = ngal            ; 50000
    nchunk = ceil(ngal/float(nperchunk))
    
    for ii = 0, nchunk-1 do begin
       print, format='("Working on chunk ",I0,"/",I0)', ii+1, nchunk
       these = lindgen(nperchunk)+ii*nperchunk
       these = these[where(these lt ngal)]
;      these = these[0:99] ; test!

       delvarx, post
       outphot1 = read_isedfit(isedfit_paramfile,isedfit_dir=isedfit_dir,$
         montegrids_dir=montegrids_dir,outprefix=outprefix,index=these,$
         isedfit_post=post,thissfhgrid=thissfhgrid)
       outphot1 = struct_trimtags(struct_trimtags(outphot1,except='*HB*'),except='*HA*')
             
       if ii eq 0 then begin
          outphot = replicate(outphot1[0], ngal)
          outphot = struct_addtags(outphot, replicate({pofm: fltarr(npofm),$
            pofm_bins: fltarr(npofm)},ngal))
       endif
       outphot[these] = im_struct_assign(outphot1, outphot[these], /nozero)

       for jj = 0, n_elements(these)-1 do begin
          mn = min(post[jj].mstar)
          mx = max(post[jj].mstar)
          dm = (mx - mn) / (npofm - 1)

          pofm = im_hist1d(post[jj].mstar,binsize=dm,$
            binedge=0,obin=pofm_bins)
          outphot[these[jj]].pofm = pofm / im_integral(pofm_bins, pofm) ; normalize
          outphot[these[jj]].pofm_bins = pofm_bins
       endfor
    endfor             

    kcorrfile = fp.isedfit_dir+fp.kcorr_outfile+'.gz'
    print, 'Reading '+kcorrfile
    kcorr = mrdfits(kcorrfile,1)
          
return, outphot
end

pro legacyhalos_profiles1d_isedfit, isedfit=isedfit, kcorrect=kcorrect, qaplot_sed=qaplot_sed, $
  thissfhgrid=thissfhgrid, clobber=clobber

    legacyhalos_dir = getenv('LEGACYHALOS_DIR')+'/science/paper2/data/'
    isedfit_rootdir = getenv('LEGACYHALOS_ISEDFIT_DIR')+'/'

    if n_elements(thissfhgrid) eq 0 then begin
       splog, 'THISSFHGRID is a required input!'
       return
    endif

    sfhgridstring = 'sfhgrid'+string(thissfhgrid,format='(I2.2)')
    if keyword_set(maxold) then sfhgridstring = sfhgridstring+'-maxold'
    
; directories and prefixes for each dataset
    prefix = 'lsphot'
    outprefix = 'profiles1d'
    outsuffix = sfhgridstring+'-lsphot-dr6-dr7'
    extname = 'LSPHOT'
    
    isedfit_dir = isedfit_rootdir+'isedfit_'+prefix+'/'
    montegrids_dir = isedfit_rootdir+'montegrids_'+prefix+'/'
    isedfit_paramfile = isedfit_dir+prefix+'_paramfile.par'

    catfile = legacyhalos_dir+'paper2-profiles1d-flux.fits'
    splog, 'Reading '+catfile
    cat = mrdfits(catfile,1)
    ngal = n_elements(cat)

    zobj = cat.z
    factor = 1D-9 / transpose([ [cat.mw_transmission_g], [cat.mw_transmission_r], [cat.mw_transmission_z] ])
    wmaggies = transpose([ [fltarr(ngal)], [fltarr(ngal)] ]) ; placeholder for WISE photometry

    radfactor = reform(transpose(cmreplicate(factor,nrad)),3,ngal*nrad)
    radzobj = reform(transpose(cmreplicate(zobj,nrad)),ngal*nrad)
    
    absmag_filterlist = [sdss_filterlist(), legacysurvey_filterlist()]
    band_shift = 0.1

; initialize the output data structure
    nrad = n_elements(cat[0].rad)
    
    result = replicate({$
;     chi2:                   0.0,$
      mstar10:                0.0,$ ; r<10kpc
      mstar10_err:            0.0,$
      mstar30:                0.0,$ ; r<30kpc
      mstar30_err:            0.0,$
      mstar100:               0.0,$ ; r<100kpc
      mstar100_err:           0.0,$
      mstarrmax:              0.0,$ ; r<rmax kpc
      mstarrmax_err:          0.0,$
      mstarrad:      fltarr(nrad),$ ; stellar mass profile [Msun]
      mstarrad_err:  fltarr(nrad),$
      murad:         fltarr(nrad),$ ; stellar mass density profile [Msun/kpc2]
      murad_err:     fltarr(nrad)},ngal)
    result = struct_addtags(im_struct_trimtags(cat,except=['FLUX*', 'MW_*']),result)

; do the fit for various apertures
    if keyword_set(isedfit) then begin
       t0 = systime(1)

; radial profile
       maggies = reform(transpose([[[cat.fluxrad_g]], [[cat.fluxrad_r]], [[cat.fluxrad_z]]]),3,ngal*nrad) * radfactor
       ivarmaggies = reform(transpose([[[cat.fluxrad_ivar_g]], [[cat.fluxrad_ivar_r]], [[cat.fluxrad_ivar_z]]]),3,ngal*nrad) / radfactor^2
       
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post

stop       
       
       result.mstarrad[ii] = ised.mstar_50
       result.mstarrad_err[ii] = ised.mstar_err
       
       good = where(result.rad_area[ii] gt 0)
       result[good].murad[ii] = result[good].mstarrad[ii] - alog10(result[good].rad_area[ii])
       result[good].murad_err[ii] = result[good].mstarrad_err[ii]

          
       for ii = 0, nrad-1 do begin
          maggies = transpose([ [cat.fluxrad_g[ii]], [cat.fluxrad_r[ii]], [cat.fluxrad_z[ii]] ]) * factor
          ivarmaggies = transpose([ [cat.fluxrad_ivar_g[ii]], [cat.fluxrad_ivar_r[ii]], [cat.fluxrad_ivar_z[ii]] ]) / factor^2
          isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
            isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
            isedfit_results=ised, isedfit_post=ised_post
          result.mstarrad[ii] = ised.mstar_50
          result.mstarrad_err[ii] = ised.mstar_err

          good = where(result.rad_area[ii] gt 0)
          result[good].murad[ii] = result[good].mstarrad[ii] - alog10(result[good].rad_area[ii])
          result[good].murad_err[ii] = result[good].mstarrad_err[ii]
       endfor
stop       
       
; r<10kpc
       maggies = transpose([ [cat.flux10_g], [cat.flux10_r], [cat.flux10_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux10_ivar_g], [cat.flux10_ivar_r], [cat.flux10_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar10 = ised.mstar_50
       result.mstar10_err = ised.mstar_err

; r<30kpc
       maggies = transpose([ [cat.flux30_g], [cat.flux30_r], [cat.flux30_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux30_ivar_g], [cat.flux30_ivar_r], [cat.flux30_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar30 = ised.mstar_50
       result.mstar30_err = ised.mstar_err

; r<100kpc
       maggies = transpose([ [cat.flux100_g], [cat.flux100_r], [cat.flux100_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux100_ivar_g], [cat.flux100_ivar_r], [cat.flux100_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar100 = ised.mstar_50
       result.mstar100_err = ised.mstar_err
       
; r<rmax kpc
       maggies = transpose([ [cat.fluxrmax_g], [cat.fluxrmax_r], [cat.fluxrmax_z] ]) * factor
       ivarmaggies = transpose([ [cat.fluxrmax_ivar_g], [cat.fluxrmax_ivar_r], [cat.fluxrmax_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstarrmax = ised.mstar_50
       result.mstarrmax_err = ised.mstar_err

       splog, 'Total time (min) = '+strtrim((systime(1)-t0)/60.0,2)
    endif 
    
return
end
