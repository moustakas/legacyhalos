#!/usr/bin/env python

"""Build a movie from a subset of the DR6 gallery thumbnails of galaxies.

See http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr6/gallery

"""

import os
import numpy as np
from glob import glob

topdir = os.path.join(os.getenv('LEGACYHALOS_DIR'), 'talks', 'dr6-gallery')

allthumb = np.array(list(glob(os.path.join(topdir, 'png', 'thumb-*.png'))))
nthumb = len(allthumb)

nperframe = 6
nframe = np.ceil(nthumb / nperframe).astype('int')

def make_movie(these, count):
    montagefile = os.path.join(topdir, '{:03d}.png'.format(count))
    cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x2 -geometry 512x512 '
    cmd = cmd+' '.join([name for name in these])
    cmd = cmd+' {}'.format(montagefile)
    print(cmd)
    os.system(cmd)

count = 0
indx = np.arange(nperframe)

# Make the individual frames
for ii in range(nframe-1):
    if ii == 0:
        these = allthumb[indx]
        make_movie(these, count)

    for jj in range(nperframe):
        newindx = (ii+1) * nperframe + jj
        if newindx < nthumb:
            indx[jj] = newindx

            these = allthumb[indx]
            make_movie(these, count)

            count = count + 1
        else:
            break

outfile = os.path.join(topdir, 'dr6-gallery.mp4')
cmd = 'ffmpeg -framerate 1 -i {}/%03d.png -vf "fps=1,format=yuv420p" -vcodec libx264 -y {}'.format(
    topdir, outfile)
print(cmd)
os.system(cmd)

# Clean up
for ii in np.arange(len(splitthumb)):
    os.remove(os.path.join(topdir, '{:03d}.png').format(ii))

# Montage of all the objects.
montagefile = os.path.join(topdir, 'dr6-gallery-all.png')
cmd = 'montage -tile 11x7 -geometry 512x512 '
cmd = cmd+' '.join([name for name in allthumb])
cmd = cmd+' {}'.format(montagefile)
print(cmd)
os.system(cmd)
