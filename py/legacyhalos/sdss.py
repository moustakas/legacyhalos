"""
legacyhalos.sdss
================

Code to handle the SDSS coadds.

"""
import os, time, copy, pdb
import subprocess
import numpy as np
import fitsio

import legacyhalos.io
import legacyhalos.misc

from legacyhalos.misc import RADIUS_CLUSTER_KPC

def download(sample, pixscale=0.396, bands='gri', clobber=False):
    """Note that the cutout server has a maximum cutout size of 3000 pixels.
    
    montage -bordercolor white -borderwidth 1 -tile 2x2 -geometry +0+0 -resize 512 \
      NGC0628-SDSS.jpg NGC3184-SDSS.jpg NGC5194-SDSS.jpg NGC5457-SDSS.jpg chaos-montage.png

    """
    for onegal in sample:
        gal, galdir = legacyhalos.io.get_galaxy_galaxydir(onegal)
    
        size_mosaic = 2 * legacyhalos.misc.cutout_radius_kpc(pixscale=pixscale, # [pixel]
            redshift=onegal['Z'], radius_kpc=RADIUS_CLUSTER_KPC)
        print(gal, size_mosaic)

        # Individual FITS files--
        outfile = os.path.join(galdir, '{}-sdss-image-gri.fits'.format(gal))
        if os.path.exists(outfile) and clobber is False:
            print('Already downloaded {}'.format(outfile))
        else:
            cmd = 'wget -c -O {outfile} '
            cmd += 'http://legacysurvey.org/viewer-dev/fits-cutout?ra={ra}&dec={dec}&pixscale={pixscale}&size={size}&layer=sdss'
            cmd = cmd.format(outfile=outfile, ra=onegal['RA'], dec=onegal['DEC'],
                             pixscale=pixscale, size=size_mosaic)
            print(cmd)
            err = subprocess.call(cmd.split())
            time.sleep(1)

            # Unpack into individual bandpasses and compress.
            imgs, hdrs = fitsio.read(outfile, header=True)
            [hdrs.delete(key) for key in ('BANDS', 'BAND0', 'BAND1', 'BAND2')]
            for ii, band in enumerate(bands):
                hdr = copy.deepcopy(hdrs)
                hdr.add_record(dict(name='BAND', value=band, comment='SDSS bandpass'))
                bandfile = os.path.join(galdir, '{}-sdss-image-{}.fits.fz'.format(gal, band))
                if os.path.isfile(bandfile):
                    os.remove(bandfile)
                print('Writing {}'.format(bandfile))
                fitsio.write(bandfile, imgs[ii, :, :], header=hdr)

            pdb.set_trace()

            print('Removing {}'.format(outfile))
            os.remove(outfile)

        # Color mosaic--
        outfile = os.path.join(galdir, '{}-sdss-image-gri.jpg'.format(gal))
        if os.path.exists(outfile) and clobber is False:
            print('Already downloaded {}'.format(outfile))
        else:
            if os.path.exists(outfile) and clobber:
                os.remove(outfile) # otherwise wget will complain
            cmd = 'wget -c -O {outfile} '
            cmd += 'http://legacysurvey.org/viewer-dev/jpeg-cutout?ra={ra}&dec={dec}&pixscale={pixscale}&size={size}&layer=sdss'
            cmd = cmd.format(outfile=outfile, ra=onegal['RA'], dec=onegal['DEC'],
                             pixscale=pixscale, size=size_mosaic)
            print(cmd)
            err = subprocess.call(cmd.split())
            time.sleep(1)

