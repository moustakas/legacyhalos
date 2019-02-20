"""
legacyhalos.integrate
=====================

Code to integrate the surface brightness profiles, including extrapolation.

"""
import os, warnings, pdb
import numpy as np
from astropy.table import Table, Column, hstack, vstack, join

import legacyhalos.io
import legacyhalos.misc

def _init_phot(nrad_uniform=50, ngal=1, band=('g', 'r', 'z')):
    """Initialize the output photometry table for a single object.

    """
    phot = Table()

    #if withid:
    #    phot.add_column(Column(name='mem_match_id', data=np.zeros(ngal).astype(np.int32)))
    
    [phot.add_column(Column(name='rmax_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    
    [phot.add_column(Column(name='flux10_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='flux30_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='flux100_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='fluxrmax_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    
    [phot.add_column(Column(name='flux10_ivar_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='flux30_ivar_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='flux100_ivar_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]
    [phot.add_column(Column(name='fluxrmax_ivar_{}'.format(bb), dtype='f4', length=ngal)) for bb in band]

    nr = nrad_uniform
    phot.add_column(Column(name='rad', dtype='f4', length=ngal, shape=(nr,)))
    phot.add_column(Column(name='rad_area', dtype='f4', length=ngal, shape=(nr,)))
    [phot.add_column(Column(name='fluxrad_{}'.format(bb), dtype='f4', length=ngal, shape=(nr,))) for bb in band]
    [phot.add_column(Column(name='fluxrad_ivar_{}'.format(bb), dtype='f4', length=ngal, shape=(nr,))) for bb in band]

    return phot

def _dointegrate(radius, sb, sberr, rmin=None, rmax=None, band='r'):
    """Do the actual profile integration.

    """
    from scipy import integrate
    from scipy.interpolate import interp1d

    # Evaluate the profile at r=rmin
    if rmin is None:
        rmin = 0.0
        sberr_rmin = sberr[0]
    else:
        sberr_rmin = interp1d(radius, sberr, kind='linear', fill_value='extrapolate')(rmin)
        
    sb_rmin = interp1d(radius, sb, kind='quadratic', fill_value='extrapolate')(rmin)

    if rmax is None:
        rmax = radius.max() # [kpc]

    if rmax > radius.max():
        return -1, -1, -1 # do not extrapolate outward
    else:
        # Interpolate the last point to the desired rmax
        sb_rmax = interp1d(radius, sb, kind='linear')(rmax)
        sberr_rmax = np.sqrt(interp1d(radius, sberr**2, kind='linear')(rmax))

        keep = np.where((radius > rmin) * (radius < rmax))[0]
        nkeep = len(keep)

        _radius = np.insert(radius[keep], [0, nkeep], [rmin, rmax])
        _sb = np.insert(sb[keep], [0, nkeep], [sb_rmin, sb_rmax])
        _sberr = np.insert(sberr[keep], [0, nkeep], [sberr_rmin, sberr_rmax])

        # Integrate!
        flux = 2 * np.pi * integrate.simps(x=_radius, y=_radius*_sb)      # [nanomaggies]
        var = 2 * np.pi * integrate.simps(x=_radius, y=_radius*_sberr**2) # [nanomaggies_ivar]

        if band == 'r':
            area = 2 * np.pi * integrate.simps(x=_radius, y=_radius) # [kpc2]
        else:
            area = -1
        
        if flux < 0 or var < 0 or np.isnan(flux) or np.isnan(var):
            print('Negative or infinite flux or variance in band {}'.format(band))
            return -1, -1, -1
        else:
            return flux, 1/var, area

def integrate(ellipsefit, nrad_uniform=50, minerr=0.01, debug=False):
    """Integrate the data and the model to get the final photometry.

    flux_obs_[grz] : observed integrated flux
    flux_int_[grz] : integrated (extrapolated) flux
    deltamag_in_[grz] : flux extrapolated inward
    deltamag_out_[grz] : flux extrapolated outward
    deltamag_[grz] : magnitude correction between flux_obs_[grz] and flux_int_[grz] or
      deltamag_in_[grz] + deltamag_out_[grz]

    """
    from astropy.table import Table, Column

    allband, pixscale = ellipsefit['band'], ellipsefit['pixscale']
    arcsec2kpc = legacyhalos.misc.arcsec2kpc(ellipsefit['redshift']) # [kpc/arcsec]
    
    def _get_sbprofile(ellipsefit, band, minerr=0.01):
        radius = ellipsefit[band].sma * np.sqrt(1 - ellipsefit[band].eps) * pixscale * arcsec2kpc # [kpc]
        sb = ellipsefit[band].intens / arcsec2kpc**2 # [nanomaggies/kpc**2]
        sberr = np.sqrt( (ellipsefit[band].int_err*arcsec2kpc**2)**2 + (0.4 * np.log(10) * sb * minerr)**2 )
        return radius, sb, sberr

    phot = _init_phot(nrad_uniform=nrad_uniform)

    # First integrate to r=10, 30, 100, and max kpc.
    min_r, max_r = [], []
    for band in allband:
        radius, sb, sberr = _get_sbprofile(ellipsefit, band, minerr=minerr)

        min_r.append(radius.min())
        max_r.append(radius.max())

        for rmax in (10, 30, 100, None):
            obsflux, obsivar, _ = _dointegrate(radius, sb, sberr, rmax=rmax, band=band)

            if rmax is not None:
                fkey = 'flux{}_{}'.format(rmax, band)
                ikey = 'flux{}_ivar_{}'.format(rmax, band)
            else:
                fkey = 'fluxrmax_{}'.format(band)
                ikey = 'fluxrmax_ivar_{}'.format(band)

            phot[fkey] = obsflux
            phot[ikey] = obsivar
        phot['rmax_{}'.format(band)] = radius.max()

    # Now integrate over fixed apertures to get the differential flux. 
    min_r, max_r = np.min(min_r), np.max(max_r)
    rad_uniform = 10**np.linspace(np.log10(min_r), np.log10(max_r), nrad_uniform+1)
    rmin_uniform, rmax_uniform = rad_uniform[:-1], rad_uniform[1:]
    phot['rad'] = (rmax_uniform - rmin_uniform) / 2 + rmin_uniform
    
    for band in allband:
        radius, sb, sberr = _get_sbprofile(ellipsefit, band, minerr=minerr)
        
        for ii, (rmin, rmax) in enumerate(zip(rmin_uniform, rmax_uniform)):
            obsflux, obsivar, obsarea = _dointegrate(radius, sb, sberr, rmin=rmin, rmax=rmax, band=band)
            #print(band, ii, rmin, rmax, 22.5-2.5*np.log10(obsflux), obsarea)

            if band == 'r':
                phot['rad_area'][0][ii] = obsarea

            phot['fluxrad_{}'.format(band)][0][ii] = obsflux
            phot['fluxrad_ivar_{}'.format(band)][0][ii] = obsivar
            
    pdb.set_trace()

    return phot

def get_results(sample, verbose=False, debug=False):

    ngal = len(sample)

    # Initialize the top-level output table
    results = sample.copy()[['mem_match_id']]
    #results.add_column(Column(name='ellipse', data=np.repeat(0, ngal).astype(np.bool_)))
    for wavepower in ('-nowavepower', ''):
        for modeltype in ('single', 'exponential', 'double'):
            colname = 'chi2_sersic_{}{}'.format(modeltype, wavepower).replace('-', '_')
            results.add_column(Column(name=colname, data=np.repeat(chi2fail, ngal).astype('f4')))
    
    sersic_single = _init_sersic(ngal, mem_match_id=sample['mem_match_id'].data, modeltype='single')
    sersic_double = _init_sersic(ngal, mem_match_id=sample['mem_match_id'].data, modeltype='double')
    sersic_exponential = _init_sersic(ngal, mem_match_id=sample['mem_match_id'].data, modeltype='exponential')

    sersic_single_nowavepower = _init_sersic(ngal, modeltype='single')
    sersic_double_nowavepower = _init_sersic(ngal, modeltype='double')
    sersic_exponential_nowavepower = _init_sersic(ngal, modeltype='exponential')

    allobjid, allobjdir = legacyhalos.io.get_objid(sample)

    for ii, (objid, objdir) in enumerate( zip(np.atleast_1d(allobjid), np.atleast_1d(allobjdir)) ):
        if verbose:
            print(ii, objid)
        else:
            if (ii % 100) == 0:
                print('Processing {} / {}'.format(ii, ngal))

        #ellipsefit = legacyhalos.io.read_ellipsefit(objid, objdir, verbose=verbose)
        #if bool(ellipsefit):
        #    if ellipsefit['success']:
        #        results['ellipse'][ii] = 1

        # Sersic modeling
        for wavepower in ('-nowavepower', ''):
            for modeltype in ('single', 'exponential', 'double'):
                modelname = '{}{}'.format(modeltype, wavepower)
                colname = 'sersic_{}'.format(modelname).replace('-', '_')
            
                sersic = legacyhalos.io.read_sersic(objid, objdir, modeltype=modelname, verbose=True)
                if bool(sersic):
                    results['chi2_{}'.format(colname)][ii] = sersic['chi2']

                    data = locals()[colname]
                    data['success'][ii] = sersic['success']
                    data['converged'][ii] = sersic['converged']

                    # best-fitting parameters
                    for param in sersic['params']:
                        data[param][ii] = sersic[param]
                        data['{}_err'.format(param)][ii] = sersic['{}_err'.format(param)]

                    # integrate to get the total photometry
                    if sersic['success']:
                        phot = sersic_integrate(sersic, debug=debug)
                        for col in phot.colnames:
                            data[col][ii] = phot[col]
                    locals()[colname] = data

                    if debug:
                        display_sersic(sersic)

    return (results, sersic_single, sersic_double, sersic_exponential,
            sersic_single_nowavepower, sersic_double_nowavepower,
            sersic_exponential_nowavepower)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', default=4, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--debug', action='store_true', help='Render a debuggin plot.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite an existing file.')
    parser.add_argument('--verbose', action='store_true', help='Be verbose.')
    args = parser.parse_args()

    # Read the sample
    sample = legacyhalos.io.read_sample(first=args.first, last=args.last, verbose=args.verbose)
    ngal = len(sample)

    # Divide the sample
    if args.ncpu > 1:
        subsample = np.array_split(sample, args.ncpu)
        resultargs = list()
        for ii in range(args.ncpu):
            resultargs.append( (Table(subsample[ii]), args.verbose, args.debug) )

        pool = multiprocessing.Pool(args.ncpu)
        out = pool.map(_get_results, resultargs)
        out = list(zip(*out))

        results = vstack(out[0])
        sersic_single = vstack(out[1])
        sersic_double = vstack(out[2])
        sersic_exponential = vstack(out[3])        
        sersic_single_nowavepower = vstack(out[4])
        sersic_double_nowavepower = vstack(out[5])
        sersic_exponential_nowavepower = vstack(out[6])
    else:
        (results, sersic_single, sersic_double, sersic_exponential,
         sersic_single_nowavepower, sersic_double_nowavepower,
         sersic_exponential_nowavepower) = get_results(sample, verbose=args.verbose, debug=args.debug)
    
    # Now assemble a final set of LegacyHalos (LH) photometry using the
    # best-fitting model.
    lhphot = _init_phot(ngal, withid=True)
    lhphot['mem_match_id'] = sample['mem_match_id']
    #extname = [model.upper() for model in ['sersic_single_nowavepower', 'sersic_exponential_nowavepower',
    #           'sersic_double_nowavepower', 'sersic_single', 'sersic_exponential',
    #           'sersic_double']]
    
    allmodels = (sersic_single, sersic_double, sersic_exponential,
                 sersic_single_nowavepower, sersic_double_nowavepower,
                 sersic_exponential_nowavepower)
        
    allchi2 = np.vstack( (results['chi2_sersic_single_nowavepower'].data, 
                          results['chi2_sersic_exponential_nowavepower'].data,
                          results['chi2_sersic_double_nowavepower'].data, 
                          results['chi2_sersic_single'].data, 
                          results['chi2_sersic_exponential'].data,
                          results['chi2_sersic_double'].data ) )
    best = np.argmin(allchi2, axis=0)
    for ibest in sorted(set(best)):
        rows = np.where( ibest == best )[0]
        #phot = legacyhalos.io.read_results(extname=extname[ibest], rows=rows, verbose=True)
        phot = allmodels[ibest][rows]
        for col in lhphot.colnames:
            if col != 'mem_match_id':
                lhphot[col][rows] = phot[col]

    # Write out summary statistics
    if args.verbose:
        print('Summary statistics:')
        print('  Number of galaxies: {} '.format(ngal))
        #print('  Ellipse-fitting: {}/{} ({:.1f}%) '.format(
        #    np.sum(results['ellipse']), ngal, 100*np.sum(results['ellipse'])/ngal))
        for wavepower in ('-nowavepower', ''):
            for modeltype in ('single', 'exponential', 'double'):
                modelname = '{}{}'.format(modeltype, wavepower)
                colname = 'chi2_sersic_{}'.format(modelname).replace('-', '_')
                ndone = np.sum(results[colname] < chi2fail)
                fracdone = 100 * ndone / ngal
                print('  {}: {}/{} ({:.1f}%) '.format(modelname, ndone, ngal, fracdone))
    print(results)

    print('HACK!!!!!!!!!!!!!!1')
    lsphot = legacyhalos.io.read_parent(extname='LSPHOT', upenn=True, verbose=args.verbose)
    rm = legacyhalos.io.read_parent(extname='REDMAPPER', upenn=True, verbose=args.verbose)

    cols = [
        'flux_w1', 'flux_ivar_w1', 'flux_w2', 'flux_ivar_w2',
        'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 'mw_transmission_w1', 'mw_transmission_w2'
        ]
            
    lhphot = hstack( (lhphot, lsphot[cols][args.first:args.last]) )
    lhphot = hstack( (lhphot, rm[ ['ra', 'dec', 'z'] ][args.first:args.last]) )

    # Write out
    legacyhalos.io.write_results(lhphot, results=results, sersic_single=sersic_single,
                                 sersic_double=sersic_double, sersic_exponential=sersic_exponential,
                                 sersic_single_nowavepower=sersic_single_nowavepower,
                                 sersic_double_nowavepower=sersic_double_nowavepower,
                                 sersic_exponential_nowavepower=sersic_exponential_nowavepower,
                                 verbose=args.verbose, clobber=args.clobber)

def legacyhalos_integrate(onegal, galaxy=None, galaxydir=None, verbose=False,
                          debug=False, hsc=False):
    """Top-level wrapper script to integrate the surface-brightness profile of a
    single galaxy.

    """
    if galaxydir is None or galaxy is None:
        if hsc:
            galaxy, galaxydir = legacyhalos.hsc.get_galaxy_galaxydir(onegal)
        else:
            galaxy, galaxydir = legacyhalos.io.get_galaxy_galaxydir(onegal)

    # Read the ellipse-fitting results and 
    ellipsefit = legacyhalos.io.read_ellipsefit(galaxy, galaxydir)
    if bool(ellipsefit):
        if ellipsefit['success']:

            res = integrate(ellipsefit, debug=debug)

            return 1
        else:
            return 0
    else:
        return 0        
