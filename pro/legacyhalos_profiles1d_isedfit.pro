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

pro legacyhalos_profiles1d_isedfit, thissfhgrid=thissfhgrid, isedfit=isedfit, clobber=clobber

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

    outfile = legacyhalos_dir+'paper2-profiles1d-mstar.fits'
    if im_file_test(outfile,clobber=clobber) eq 1 then return

; read the catalog    
    catfile = legacyhalos_dir+'paper2-profiles1d-flux.fits'
    splog, 'Reading '+catfile
    cat = mrdfits(catfile,1)
    ngal = n_elements(cat)
    nrad = n_elements(cat[0].rad)
    
    zobj = cat.z
    factor = 1D-9 / transpose([ [cat.mw_transmission_g], [cat.mw_transmission_r], [cat.mw_transmission_z] ])
    wmaggies = transpose([ [fltarr(ngal)], [fltarr(ngal)] ]) ; placeholder for WISE photometry

    radfactor = reform(transpose(cmreplicate(factor,nrad)),3,ngal*nrad) ; [3,ngal*nrad] array
    radzobj = reform(transpose(cmreplicate(zobj,nrad)),ngal*nrad)       ; [ngal*nrad] array
    radwmaggies = reform(transpose(cmreplicate(wmaggies,nrad)),2,ngal*nrad) ; [2,ngal*nrad] array
    
    absmag_filterlist = [sdss_filterlist(), legacysurvey_filterlist()]
    band_shift = 0.1

; initialize the output data structure
    result = replicate({$
;     chi2:                   0.0,$
      mstar10:                0.0,$ ; r<10kpc
      mstar10_err:            0.0,$
      mstar30:                0.0,$ ; r<30kpc
      mstar30_err:            0.0,$
      mstar100:               0.0,$ ; r<100kpc
      mstar100_err:           0.0,$
      mstarrmax:              0.0,$ ; r<rmax
      mstarrmax_err:          0.0,$
      mstarrad:      fltarr(nrad),$ ; stellar mass profile [Msun]
      mstarrad_err:  fltarr(nrad),$
      murad:         fltarr(nrad),$ ; stellar mass density profile [Msun/kpc2]
      murad_err:     fltarr(nrad)},ngal)
    result = struct_addtags(im_struct_trimtags(cat,except=['FLUX*', 'MW_*']),result)

; do it!    
    if keyword_set(isedfit) then begin
       tall = systime(1)

; radial profile
       maggies = reform(transpose([[[cat.fluxrad_g]], [[cat.fluxrad_r]], $
         [[cat.fluxrad_z]]],[2,0,1]),3,ngal*nrad) * radfactor
       ivarmaggies = reform(transpose([[[cat.fluxrad_ivar_g]], [[cat.fluxrad_ivar_r]], $
         [[cat.fluxrad_ivar_z]]],[2,0,1]),3,ngal*nrad) / radfactor^2
       
       t0 = systime(1)
       isedfit, isedfit_paramfile, [maggies,radwmaggies], [ivarmaggies,radwmaggies], radzobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       ised_post = 0 ; save memory
       splog, 'Time for radial profile fitting (min) = '+strtrim((systime(1)-t0)/60.0,2)

       ised = reform(ised,nrad,ngal)
       ised_post = reform(ised_post,nrad,ngal)
       
       result.mstarrad = ised.mstar_50
       result.mstarrad_err = ised.mstar_err

; compute the stellar mass density (Msun/kpc2) profile       
       good = where(result.rad_area gt 0)
       murad = fltarr(nrad,ngal)
       (murad)[good] = (result.mstarrad)[good] - alog10((result.rad_area)[good])
       
       result.murad = murad
       result.murad_err = result.mstarrad_err

; r<10kpc
       t0 = systime(1)
       maggies = transpose([ [cat.flux10_g], [cat.flux10_r], [cat.flux10_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux10_ivar_g], [cat.flux10_ivar_r], [cat.flux10_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar10 = ised.mstar_50
       result.mstar10_err = ised.mstar_err
       splog, 'Time for r<10kpc fitting (min) = '+strtrim((systime(1)-t0)/60.0,2)

; r<30kpc
       t0 = systime(1)
       maggies = transpose([ [cat.flux30_g], [cat.flux30_r], [cat.flux30_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux30_ivar_g], [cat.flux30_ivar_r], [cat.flux30_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar30 = ised.mstar_50
       result.mstar30_err = ised.mstar_err
       splog, 'Time for r<30kpc fitting (min) = '+strtrim((systime(1)-t0)/60.0,2)

; r<100kpc
       t0 = systime(1)
       maggies = transpose([ [cat.flux100_g], [cat.flux100_r], [cat.flux100_z] ]) * factor
       ivarmaggies = transpose([ [cat.flux100_ivar_g], [cat.flux100_ivar_r], [cat.flux100_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstar100 = ised.mstar_50
       result.mstar100_err = ised.mstar_err
       splog, 'Time for r<100kpc fitting (min) = '+strtrim((systime(1)-t0)/60.0,2)
       
; r<rmax
       t0 = systime(1)
       maggies = transpose([ [cat.fluxrmax_g], [cat.fluxrmax_r], [cat.fluxrmax_z] ]) * factor
       ivarmaggies = transpose([ [cat.fluxrmax_ivar_g], [cat.fluxrmax_ivar_r], [cat.fluxrmax_ivar_z] ]) / factor^2
       isedfit, isedfit_paramfile, [maggies,wmaggies], [ivarmaggies,wmaggies], zobj, $
         isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, index=index, /nowrite, $
         isedfit_results=ised, isedfit_post=ised_post
       result.mstarrmax = ised.mstar_50
       result.mstarrmax_err = ised.mstar_err
       splog, 'Time for r<rmax fitting (min) = '+strtrim((systime(1)-t0)/60.0,2)
       splog, 'Time for all fitting (min) = '+strtrim((systime(1)-tall)/60.0,2)

       im_mwrfits, result, outfile, /clobber
    endif 
    
return
end
