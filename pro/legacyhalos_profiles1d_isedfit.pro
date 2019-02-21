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
  lsphot_dr6_dr7=lsphot_dr6_dr7, sdssphot_dr14=sdssphot_dr14, lhphot=lhphot, $
  rows=rows

; DR6/DR7 LegacySurvey (grz) + unWISE W1 & W2
    if keyword_set(lsphot_dr6_dr7) then begin
       catfile = legacyhalos_dir+'/sample/legacyhalos-'+sampleprefix+'-dr6-dr7.fits'
       splog, 'Reading '+catfile
       if n_elements(rows) ne 0 then begin
          cat = mrdfits(catfile,1,rows=rows)
       endif else begin
          cat = mrdfits(catfile,1)
       endelse
       ngal = n_elements(cat)

       ra = cat.ra
       dec = cat.dec
       zobj = cat.z
       
       factor = 1D-9 / transpose([ [cat.mw_transmission_g], [cat.mw_transmission_r], $
         [cat.mw_transmission_z] ])
       dmaggies = float(transpose([ [cat.flux_g], [cat.flux_r], [cat.flux_z] ]) * factor)
       divarmaggies = float(transpose([ [cat.flux_ivar_g], [cat.flux_ivar_r], $
         [cat.flux_ivar_z] ]) / factor^2)
       
       factor = 1D-9 / transpose([ [cat.mw_transmission_w1], [cat.mw_transmission_w2] ])
       wmaggies = float(transpose([ [cat.flux_w1], [cat.flux_w2] ]) * factor)
       wivarmaggies = float(transpose([ [cat.flux_ivar_w1], [cat.flux_ivar_w2] ]) / factor^2)

       maggies = [dmaggies, wmaggies]
       ivarmaggies = [divarmaggies, wivarmaggies]

; add minimum calibration uncertainties (in quadrature) to grzW1W2; see
; [desi-targets 2084]
       k_minerror, maggies, ivarmaggies, [0.01,0.01,0.02,0.02,0.02]
;      k_minerror, maggies, ivarmaggies, [0.003,0.003,0.006,0.005,0.02]
    endif

; custom LegacySurvey (grz) + unWISE W1 & W2    
    if keyword_set(lhphot) then begin
       catfile = legacyhalos_dir+'/sample/legacyhalos-results.fits'
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
       dmaggies = float(transpose([ [cat.flux_g], [cat.flux_r], [cat.flux_z] ]) * factor)
       divarmaggies = float(transpose([ [cat.flux_ivar_g], [cat.flux_ivar_r], $
         [cat.flux_ivar_z] ]) / factor^2)
       
       factor = 1D-9 / transpose([ [cat.mw_transmission_w1], [cat.mw_transmission_w2] ])
       wmaggies = float(transpose([ [cat.flux_w1], [cat.flux_w2] ]) * factor)
       wivarmaggies = float(transpose([ [cat.flux_ivar_w1], [cat.flux_ivar_w2] ]) / factor^2)

       maggies = [dmaggies, wmaggies]
       ivarmaggies = [divarmaggies, wivarmaggies]

; add minimum uncertainties to grzW1W2
       k_minerror, maggies, ivarmaggies, [0.01,0.01,0.02,0.02,0.02]
;      k_minerror, maggies, ivarmaggies, [0.02,0.02,0.02,0.005,0.02]
    endif

; SDSS ugriz + forced WISE photometry from Lang & Schlegel    
    if keyword_set(sdssphot_dr14) then begin
       catfile = legacyhalos_dir+'/sample/legacyhalos-'+sampleprefix+'-dr6-dr7.fits'
       splog, 'Reading '+catfile
       if n_elements(rows) ne 0 then begin
          cat = mrdfits(catfile,1,rows=rows)
       endif else begin
          cat = mrdfits(catfile,1)
       endelse
       ngal = n_elements(cat)

       ra = cat.ra
       dec = cat.dec
       zobj = cat.z

; protect against no photometry in the following SDSS_OBJID
; [1237654949982044274, 1237659146707141110, 1237654383056651070].  as far as I
; can tell these are real sources---with photometry in SkyServer, but for
; whatever reason they didn't match in my CasJobs query.  not tracking down, so
; we just won't get stellar masses for these...

       smaggies = fltarr(5, ngal)
       sivarmaggies = fltarr(5, ngal)
       notzero = where(cat.modelmaggies_ivar[2,*] gt 0,nnotzero)
       
       ratio = cat[notzero].cmodelmaggies[2,*]/cat[notzero].modelmaggies[2,*]
       factor = 1D-9 * rebin(ratio, 5, nnotzero) * 10D^(0.4*cat[notzero].extinction)
       smaggies[*,notzero] = cat[notzero].modelmaggies * factor
       sivarmaggies[*,notzero] = cat[notzero].modelmaggies_ivar / factor^2

       vega2ab = rebin([2.699,3.339],2,ngal) ; Vega-->AB from http://legacysurvey.org/dr5/description/#photometry
       glactc, cat.ra, cat.dec, 2000.0, gl, gb, 1, /deg
       ebv = rebin(reform(dust_getval(gl,gb,/interp,/noloop),1,ngal),2,ngal)
       coeff = rebin(reform([0.184,0.113],2,1),2,ngal) ; Galactic extinction coefficients from http://legacysurvey.org/dr5/catalogs

       factor = 1D-9 * 10^(0.4*coeff*ebv)*10^(-0.4*vega2ab)
       wmaggies = float(cat.wise_nanomaggies * factor)
       wivarmaggies = float(cat.wise_nanomaggies_ivar / factor^2)
       
       maggies = [smaggies, wmaggies]
       ivarmaggies = [sivarmaggies, wivarmaggies]

; add minimum uncertainties to ugrizW1W2
       k_minerror, maggies, ivarmaggies, [0.05,0.02,0.02,0.02,0.03,0.02,0.02]
    endif

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

pro legacyhalos_profiles1d_isedfit, isedfit=isedfit, kcorrect=kcorrect, qaplot_sed=qaplot_sed,
    thissfhgrid=thissfhgrid, clobber=clobber

    legacyhalos_dir = getenv('LEGACYHALOS_DIR')+'science/paper2/data/'
    isedfit_rootdir = getenv('LEGACYHALOS_ISEDFIT_DIR')

    if n_elements(thissfhgrid) eq 0 then begin
       splog, 'THISSFHGRID is a required input!'
       return
    endif

    sfhgridstring = 'sfhgrid'+string(thissfhgrid,format='(I2.2)')
    if keyword_set(maxold) then sfhgridstring = sfhgridstring+'-maxold'
    
; directories and prefixes for each dataset
    if keyword_set(lsphot_dr6_dr7) then begin
       prefix = 'lsphot'
       outprefix = 'profiles1d'
       outsuffix = sfhgridstring+'-lsphot-dr6-dr7'
       extname = 'LSPHOT'
    endif
    
    isedfit_dir = isedfit_rootdir+'isedfit_'+prefix+'/'
    montegrids_dir = isedfit_rootdir+'montegrids_'+prefix+'/'
    isedfit_paramfile = isedfit_dir+prefix+'_paramfile.par'

    spawn, ['mkdir -p '+isedfit_dir], /sh
    spawn, ['mkdir -p '+montegrids_dir], /sh

    absmag_filterlist = [sdss_filterlist(), legacysurvey_filterlist()]
    band_shift = 0.1

; --------------------------------------------------
; fit!
    if keyword_set(isedfit) then begin
       zobj = legacyhalos_maggies(maggies=maggies,ivarmaggies=ivarmaggies,$
         ra=ra,dec=dec,legacyhalos_dir=legacyhalos_dir,sampleprefix=sampleprefix, $
         lsphot_dr6_dr7=lsphot_dr6_dr7,sdssphot_dr14=sdssphot_dr14,lhphot=lhphot)
       t0 = systime(1)
       isedfit, isedfit_paramfile, maggies, ivarmaggies, zobj, ra=ra, $
         dec=dec, isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, $
         clobber=clobber, index=index, outprefix=outprefix, maxold=maxold
       splog, 'Total time for centrals (min) = '+strtrim((systime(1)-t0)/60.0,2)
    endif 

; --------------------------------------------------
; compute K-corrections
    if keyword_set(kcorrect) then begin
       if keyword_set(candidate_centrals) then begin
          if keyword_set(lhphot) then $
            catfile = legacyhalos_dir+'/sample/legacyhalos-results.fits' else $
              catfile = legacyhalos_dir+'/sample/legacyhalos-'+sampleprefix+'-dr6-dr7.fits'
          ngal = sxpar(headfits(catfile,ext=1), 'NAXIS2')
;         splog, 'Ridiculously hard-coding ngal here to speed things up!'
;         ngal = 6682618L 
          chunksize = ceil(ngal/float(ncandchunk))

          for ii = firstchunk, lastchunk do begin
             splog, 'Working on CHUNK '+strtrim(ii,2)+', '+strtrim(lastchunk,2)
             splog, im_today()
             t0 = systime(1)
             chunkprefix = outprefix+'_sat_chunk'+string(ii,format='(I2.2)')
             these = lindgen(chunksize)+ii*chunksize
             these = these[where(these lt ngal)]
;            these = these[0:99] ; test!

             zobj = legacyhalos_maggies(maggies=maggies,ivarmaggies=ivarmaggies,$
               ra=ra,dec=dec,legacyhalos_dir=legacyhalos_dir,sampleprefix=sampleprefix, $
               lsphot_dr6_dr7=lsphot_dr6_dr7,sdssphot_dr14=sdssphot_dr14,lhphot=lhphot,$
               rows=these)
             isedfit_kcorrect, isedfit_paramfile, isedfit_dir=isedfit_dir, $
               montegrids_dir=montegrids_dir, thissfhgrid=thissfhgrid, $
               absmag_filterlist=absmag_filterlist, band_shift=band_shift, $
               clobber=clobber, index=index, outprefix=chunkprefix
             splog, 'Total time candidate_centrals (min) = '+strtrim((systime(1)-t0)/60.0,2)
          endfor
       endif else begin
          zobj = legacyhalos_maggies(maggies=maggies,ivarmaggies=ivarmaggies,$
            ra=ra,dec=dec,legacyhalos_dir=legacyhalos_dir,sampleprefix=sampleprefix, $
            lsphot_dr6_dr7=lsphot_dr6_dr7,sdssphot_dr14=sdssphot_dr14,lhphot=lhphot)
          t0 = systime(1)
          isedfit_kcorrect, isedfit_paramfile, isedfit_dir=isedfit_dir, $
            montegrids_dir=montegrids_dir, thissfhgrid=thissfhgrid, $
            absmag_filterlist=absmag_filterlist, band_shift=band_shift, $
            clobber=clobber, index=index, outprefix=outprefix
          splog, 'Total time for centrals (min) = '+strtrim((systime(1)-t0)/60.0,2)
       endelse
    endif 

; --------------------------------------------------
; generate spectral energy distribution (SED) QAplots
    if keyword_set(qaplot_sed) then begin
       these = lindgen(50)
       isedfit_qaplot_sed, isedfit_paramfile, isedfit_dir=isedfit_dir, $
         montegrids_dir=montegrids_dir, outprefix=outprefix, thissfhgrid=thissfhgrid, $
         clobber=clobber, index=these, /xlog
    endif
    
return
end
