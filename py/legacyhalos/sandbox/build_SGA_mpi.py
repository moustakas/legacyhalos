    # Build the SGA and return--
    if args.build_SGA:
        if rank == 0:
            tall = time.time()
            from legacyhalos.SGA import _init_ellipse_SGA, _write_ellipse_SGA
            buildit, filenames, refcat = _init_ellipse_SGA(clobber=args.clobber)
        else:
            buildit, filenames, refcat = None, None, None
            
        if comm:
            buildit = comm.bcast(buildit, root=0)
            refcat = comm.bcast(refcat, root=0)

        if buildit:
            from legacyhalos.SGA import _build_ellipse_SGA_one
            from astrometry.util.multiproc import multiproc
            
            ranksample = sample[groups[rank]]
            rankfullsample = fullsample[np.isin(fullsample['GROUP_ID'], ranksample['GROUP_ID'])]

            mp = multiproc(nthreads=args.nproc)
            buildargs = []
            for onegal in ranksample:
                buildargs.append((onegal, rankfullsample[rankfullsample['GROUP_ID'] == onegal['GROUP_ID']], refcat, args.verbose))
            results = mp.map(_build_ellipse_SGA_one, buildargs)
            results = list(zip(*results))

            # Unpack the array of lists into a list of arrays--
            cat = list(filter(None, results[0]))
            notractor = list(filter(None, results[1]))
            noellipse = list(filter(None, results[2]))
            nogrz = list(filter(None, results[3]))
            
            print('Rank {}: N(sample)={}, N(fullsample)={}, N(cat)={}, N(notractor)={}, N(noellipse)={}, N(nogrz)={}'.format(
                rank, len(ranksample), len(rankfullsample), len(cat), len(notractor), len(noellipse), len(nogrz)))

            if comm is not None:
                cat = comm.gather(cat, root=0)
                notractor = comm.gather(notractor, root=0)
                noellipse = comm.gather(noellipse, root=0)
                nogrz = comm.gather(nogrz, root=0)

            if rank == 0:
                # We need to unpack the lists we get from comm.gather--
                if args.mpi:
                    from astropy.table import vstack
                    if len(cat) > 0:
                        newcat = []
                        for cc in cat:
                            if len(cc) > 0:
                                newcat.append(vstack(cc))
                        cat = newcat
                    if len(notractor) > 0:
                        newnotractor = []
                        for cc in notractor:
                            if len(cc) > 0:
                                newnotractor.append(vstack(cc))
                        notractor = newnotractor
                    if len(noellipse) > 0:
                        newnoellipse = []
                        for cc in noellipse:
                            if len(cc) > 0:
                                newnoellipse.append(vstack(cc))
                        noellipse = newnoellipse
                    if len(nogrz) > 0:
                        newnogrz = []
                        for cc in nogrz:
                            if len(cc) > 0:
                                newnogrz.append(vstack(cc))
                        nogrz = newnogrz
                    
                print('Rank {}: N(sample)={}, N(fullsample)={}, N(cat)={}, N(notractor)={}, N(noellipse)={}, N(nogrz)={}'.format(
                    rank, len(sample), len(fullsample), len(cat), len(notractor), len(noellipse), len(nogrz)))
                outfile, notractorfile, noellipsefile, nogrzfile = filenames
                _write_ellipse_SGA(cat, notractor, noellipse, nogrz, outfile, notractorfile,
                                   noellipsefile, nogrzfile, refcat, skipfull=False)
                print('Finished {} after {:.3f} minutes'.format(suffix.upper(), (time.time() - tall) / 60))
