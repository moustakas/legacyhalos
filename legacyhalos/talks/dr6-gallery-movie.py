#!/usr/bin/env python

"""
Build a movie from some of the DR6 gallery thumbnails.
"""

import os
import numpy as np
from glob import glob

topdir = os.path.join(os.getenv('LEGACYHALOS_DIR'), 'talks', 'dr6-gallery')

allthumb = list(glob(os.path.join(topdir, 'png', 'thumb-*.png')))
nthumb = len(allthumb)
splitthumb = np.array_split(allthumb, np.ceil(nthumb / 6).astype('int'))

for ii, sff in enumerate(splitthumb):
    montagefile = os.path.join(topdir, '{:02d}.png'.format(ii))
    cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x2 -geometry 512x512 '
    cmd = cmd+' '.join([name for name in sff])
    cmd = cmd+' {}'.format(montagefile)
    print(cmd)
    os.system(cmd)

outfile = os.path.join(topdir, 'dr6-gallery.mp4')
cmd = 'ffmpeg -framerate 1/3 -i {}/%02d.png -vf "fps=1,format=yuv420p" -vcodec libx264 -y {}'.format(
    topdir, outfile)
print(cmd)
os.system(cmd)

for ii in np.arange(len(splitthumb)):
    os.remove(os.path.join(topdir, '{:02d}.png').format(ii))
