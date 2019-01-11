;+
; NAME:
;   LEGACYHALOS_ISEDFIT
;
; PURPOSE:
;   Use iSEDfit to compute stellar masses for the various legacyhalos samples.
;
; INPUTS:
;   At least one of /LSPHOT_DR6_DR7, /LHPHOT, /SDSSPHOT_DR14 must be set.
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
;   J. Moustakas, 2017 Dec 27, Siena
;   jm18jul16siena - fit first-pass legacyhalos photometry 
;   jm18aug17siena - update to latest sample and data model 
;
; Copyright (C) 2017-2018, John Moustakas
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

function lhphot_version
; v1.0 - original effort    
    ver = 'v1.0'
return, ver
end    

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

; mask out wonky unWISE photometry
       snr = maggies*sqrt(ivarmaggies)
       ww = where(abs(snr[3,*]) gt 1e3) & ivarmaggies[3,ww] = 0
       ww = where(abs(snr[4,*]) gt 1e3) & ivarmaggies[4,ww] = 0

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

pro legacyhalos_isedfit, lsphot_dr6_dr7=lsphot_dr6_dr7, sdssphot_dr14=sdssphot_dr14, lhphot=lhphot, $
  write_paramfile=write_paramfile, build_grids=build_grids, model_photometry=model_photometry, $
  isedfit=isedfit, kcorrect=kcorrect, qaplot_sed=qaplot_sed, thissfhgrid=thissfhgrid, $
  gather_results=gather_results, clobber=clobber, candidate_centrals=candidate_centrals, maxold=maxold, $
  firstchunk=firstchunk, lastchunk=lastchunk

; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/lsphot-dr6-dr7-1.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=2, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/lsphot-dr6-dr7-2.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=2, /maxold, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/lsphot-dr6-dr7-3.log 2>&1 &

; echo "legacyhalos_isedfit, /sdssphot_dr14, thissfhgrid=1, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/sdssphot-dr14-1.log 2>&1 &
; echo "legacyhalos_isedfit, /sdssphot_dr14, thissfhgrid=2, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/sdssphot-dr14-2.log 2>&1 &
; echo "legacyhalos_isedfit, /sdssphot_dr14, thissfhgrid=2, /maxold, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/sdssphot-dr14-3.log 2>&1 &

; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /candidate_centrals, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/lsphot-cand-dr6-dr7-1.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=2, /candidate_centrals, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/lsphot-cand-dr6-dr7-2.log 2>&1 &
; echo "legacyhalos_isedfit, /sdssphot_dr14, thissfhgrid=1, /candidate_centrals, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/sdssphot-cand-dr6-dr7-1.log 2>&1 &
; echo "legacyhalos_isedfit, /sdssphot_dr14, thissfhgrid=2, /candidate_centrals, /isedfit, /kcorrect, /qaplot_sed, /cl" | /usr/bin/nohup idl > logs/sdssphot-cand-dr6-dr7-2.log 2>&1 &
    
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /kcorrect, /candidate_centrals, firstchunk=0, lastchunk=4, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sat-9.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /kcorrect, /candidate_centrals, firstchunk=5, lastchunk=9, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sat-0.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /kcorrect, /candidate_centrals, firstchunk=10, lastchunk=14, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sat-1.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /kcorrect, /candidate_centrals, firstchunk=15, lastchunk=19, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sat-2.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, thissfhgrid=1, /kcorrect, /candidate_centrals, firstchunk=20, lastchunk=24, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sat-3.log 2>&1 &

; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, /build_grids, /model_phot, thissfhgrid=1, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sfhgrid01.log 2>&1 &
; echo "legacyhalos_isedfit, /lsphot_dr6_dr7, /build_grids, /model_phot, thissfhgrid=2, /cl" | /usr/bin/nohup idl > lsphot-dr6-dr7-sfhgrid02.log 2>&1 &

; echo "legacyhalos_isedfit, /sdssphot_dr14, /build_grids, /model_phot, thissfhgrid=1, /cl" | /usr/bin/nohup idl > sdssphot-dr14-sfhgrid01.log 2>&1 &
; echo "legacyhalos_isedfit, /sdssphot_dr14, /build_grids, /model_phot, thissfhgrid=2, /cl" | /usr/bin/nohup idl > sdssphot-dr14-sfhgrid02.log 2>&1 &
    
    legacyhalos_dir = getenv('LEGACYHALOS_DIR')

    if keyword_set(lsphot_dr6_dr7) eq 0 and keyword_set(lhphot) eq 0 and $
      keyword_set(sdssphot_dr14) eq 0 and keyword_set(gather_results) eq 0 then begin
       splog, 'Choose one of /LSPHOT_DR6_DR7, /LHPHOT, or /SDSSPHOT_DR14'
       return       
    endif

    if n_elements(thissfhgrid) eq 0 then begin
       splog, 'THISSFHGRID is a required input!'
       return
    endif

    sfhgridstring = 'sfhgrid'+string(thissfhgrid,format='(I2.2)')
    if keyword_set(maxold) then sfhgridstring = sfhgridstring+'-maxold'
    
; directories and prefixes for each dataset
    if keyword_set(lsphot_dr6_dr7) then begin
       prefix = 'lsphot'
       outprefix = 'lsphot_dr6_dr7'
       outsuffix = sfhgridstring+'-lsphot-dr6-dr7'
       extname = 'LSPHOT'
    endif
    if keyword_set(lhphot) then begin
       version = lhphot_version()
       prefix = 'lsphot'
       outprefix = 'lhphot_'+version
       outsuffix = sfhgridstring+'-lhphot-'+version
       extname = 'LHPHOT'
    endif
    if keyword_set(sdssphot_dr14) then begin
       prefix = 'sdssphot'
       outprefix = 'sdssphot_dr14'
       outsuffix = sfhgridstring+'-sdssphot-dr14'
       extname = 'SDSSPHOT'
    endif
    if keyword_set(maxold) then outprefix = outprefix+'_maxold'

    if keyword_set(candidate_centrals) then begin
       ncandchunk = 5
       sampleprefix = 'candidate-centrals'

       if n_elements(firstchunk) eq 0 then firstchunk = 0
       if n_elements(lastchunk) eq 0 then lastchunk = ncandchunk-1
    endif else begin
       sampleprefix = 'centrals'
    endelse

    isedfit_rootdir = getenv('IM_WORK_DIR')+'/projects/legacyhalos/isedfit/'
    
    isedfit_dir = isedfit_rootdir+'isedfit_'+prefix+'/'
    montegrids_dir = isedfit_rootdir+'montegrids_'+prefix+'/'
    isedfit_paramfile = isedfit_dir+prefix+'_paramfile.par'

    spawn, ['mkdir -p '+isedfit_dir], /sh
    spawn, ['mkdir -p '+montegrids_dir], /sh

; --------------------------------------------------
; gather the results and write out the final stellar mass catalog, including the
; posterior probability on stellar mass 
    if keyword_set(gather_results) then begin
       if keyword_set(candidate_centrals) then begin
          if keyword_set(lhphot) then $
            catfile = legacyhalos_dir+'/sample/legacyhalos-results.fits' else $
              catfile = legacyhalos_dir+'/sample/legacyhalos-'+sampleprefix+'-dr6-dr7.fits'
          ngal = sxpar(headfits(catfile,ext=1), 'NAXIS2')
          chunksize = ceil(ngal/float(ncandchunk))

          delvarx, outphot, outkcorr
          for ii = firstchunk, lastchunk do begin
             splog, 'Working on CHUNK '+strtrim(ii,2)+', '+strtrim(lastchunk,2)
             splog, im_today()
             t0 = systime(1)
             chunkprefix = outprefix+'_sat_chunk'+string(ii,format='(I2.2)')
             these = lindgen(chunksize)+ii*chunksize
             these = these[where(these lt ngal)]
;            these = these[0:99] ; test!

             outphot1 = get_pofm(prefix,chunkprefix,isedfit_rootdir,$
               thissfhgrid=thissfhgrid,kcorr=outkcorr1)
             if n_elements(outphot) eq 0 then begin
                outphot = im_empty_structure(outphot1[0], ncopies=ngal, empty_value=-999)
                outkcorr = im_empty_structure(outkcorr1[0], ncopies=ngal, empty_value=-999)
             endif
             outphot[these] = temporary(outphot1)
             outkcorr[these] = temporary(outkcorr1)
          endfor
       endif else begin
          outphot = get_pofm(prefix,outprefix,isedfit_rootdir,$
            thissfhgrid=thissfhgrid,kcorr=outkcorr)
       endelse
       outfile = legacyhalos_dir+'/sample/'+sampleprefix+'-'+outsuffix+'.fits'

       splog, 'Writing '+outfile
       mwrfits, outphot, outfile, /create
       mwrfits, outkcorr, outfile

       hdr = headfits(outfile,ext=1)
       sxaddpar, hdr, 'EXTNAME', extname+'-ISEDFIT'
       modfits, outfile, 0, hdr, exten_no=1

       hdr = headfits(outfile,ext=2)
       sxaddpar, hdr, 'EXTNAME', extname+'-KCORR'
       modfits, outfile, 0, hdr, exten_no=2
    endif
    
; --------------------------------------------------
; define the filters and the redshift ranges
    if keyword_set(lsphot_dr6_dr7) or keyword_set(lhphot) then begin
       filterlist = [legacysurvey_filterlist(), wise_filterlist(/short)]
       zminmax = [0.05,0.6]
       nzz = 61
    endif
    if keyword_set(sdssphot_dr14) then begin
       filterlist = [sdss_filterlist(), wise_filterlist(/short)]
       zminmax = [0.05,0.6]
       nzz = 61
    endif

    absmag_filterlist = [sdss_filterlist(), legacysurvey_filterlist()]
    band_shift = 0.1
    
; --------------------------------------------------
; write the parameter file
    if keyword_set(write_paramfile) then begin
; SFHGRID01 - general SFH + dust, emission lines
       write_isedfit_paramfile, params=params, isedfit_dir=isedfit_dir, $
         prefix=prefix, filterlist=filterlist, spsmodels='fsps_v2.4_miles', $
         imf='chab', redcurve='charlot', igm=0, zminmax=zminmax, nzz=nzz, $
         nmodel=50000L, age=[0.1,13.0], tau=[0.0,6], Zmetal=[0.004,0.03], $
         /delayed, /nebular, clobber=clobber, sfhgrid=1
; SFHGRID02 - no dust, no emission lines
       write_isedfit_paramfile, params=params, isedfit_dir=isedfit_dir, $
         prefix=prefix, filterlist=filterlist, spsmodels='fsps_v2.4_miles', $
         imf='chab', redcurve='none', igm=0, zminmax=zminmax, nzz=nzz, $
         nmodel=50000L, age=[0.1,13.0], tau=[0.0,6], Zmetal=[0.004,0.03], $
         AV=[0,0], /delayed, nebular=0, clobber=clobber, sfhgrid=2, /append
    endif

; --------------------------------------------------
; build the Monte Carlo grids    
    if keyword_set(build_grids) then begin
       isedfit_montegrids, isedfit_paramfile, isedfit_dir=isedfit_dir, $
         montegrids_dir=montegrids_dir, thissfhgrid=thissfhgrid, clobber=clobber
    endif

; --------------------------------------------------
; calculate the model photometry 
    if keyword_set(model_photometry) then begin
       isedfit_models, isedfit_paramfile, isedfit_dir=isedfit_dir, $
         montegrids_dir=montegrids_dir, thissfhgrid=thissfhgrid, $
         clobber=clobber
    endif

; --------------------------------------------------
; fit!
    if keyword_set(isedfit) then begin
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
               lsphot_dr6_dr7=lsphot_dr6_dr7,sdssphot_dr14=sdssphot_dr14,lhphot=lhphot, $
               rows=these)
             isedfit, isedfit_paramfile, maggies, ivarmaggies, zobj, ra=ra, $
               dec=dec, isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, $
               clobber=clobber, index=index, outprefix=chunkprefix, maxold=maxold
             splog, 'Total time candidate_centrals (min) = '+strtrim((systime(1)-t0)/60.0,2)
          endfor
       endif else begin
          zobj = legacyhalos_maggies(maggies=maggies,ivarmaggies=ivarmaggies,$
            ra=ra,dec=dec,legacyhalos_dir=legacyhalos_dir,sampleprefix=sampleprefix, $
            lsphot_dr6_dr7=lsphot_dr6_dr7,sdssphot_dr14=sdssphot_dr14,lhphot=lhphot)
          t0 = systime(1)
          isedfit, isedfit_paramfile, maggies, ivarmaggies, zobj, ra=ra, $
            dec=dec, isedfit_dir=isedfit_dir, thissfhgrid=thissfhgrid, $
            clobber=clobber, index=index, outprefix=outprefix, maxold=maxold
          splog, 'Total time for centrals (min) = '+strtrim((systime(1)-t0)/60.0,2)
       endelse
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
