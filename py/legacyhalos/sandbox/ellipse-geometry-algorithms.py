#!/usr/bin/env python
"""Testing out various shape measurements.

from legacyhalos.io import read_multiband
import numpy.ma as ma
dd = read_multiband('41113651750406589', '8/410/41113651750406589/')
fitsio.write('image.fits', ma.filled(dd['z_masked'], fill_value=0), clobber=True)

from legacyhalos.mge import find_galaxy
import numpy as np
gg = find_galaxy(img, nblob=1, binning=3, quiet=False)
print(gg.xpeak, gg.ypeak, gg.eps, gg.majoraxis, np.radians(gg.pa-90))
"""

import fitsio, pdb
import numpy as np
import numpy.ma as ma
from photutils import data_properties
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture

import matplotlib.pyplot as plt

from astropy.visualization import AsinhStretch as Stretch
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval as Interval

from legacyhalos.mge import find_galaxy

img = fitsio.read('image.fits')
img = ma.masked_array(img, img==0)

props = data_properties(img, mask=img==0)
props2 = find_galaxy(img, nblob=1, fraction=0.05, binning=5, quiet=False, plot=True)

geo = EllipseGeometry(x0=props2.xpeak, y0=props2.ypeak, sma=props2.majoraxis,
                      eps=props2.eps, pa=np.radians(props2.pa-90))
ell = Ellipse(img, geometry=geo)

# Fit at least 10 isophotes between a semi-major axis of 1 and (75% times?) the
# majoraxis and take the mean value.
nsma = 10
smamin, smamax = 0.05*props2.majoraxis, props2.majoraxis # /3
step = np.int((smamax - smamin) / nsma)
smagrid = np.linspace(smamin, smamax, nsma).astype(int)
sma0 = (smamax-smamin)/2
#sma0 = smamin * 5
print(smamin, smamax, sma0, step, smagrid)

print('Fitting image')
iso = ell.fit_image(sma0, minsma=smamin, maxsma=smamax,
                    integrmode='median', sclip=3, nclip=2,
                    maxrit=None, step=0.5, linear=False,
                    maxgerr=0.5)
print(iso.sma)
print(iso.stop_code)
#print(iso.intens)

good = np.where(iso.stop_code < 4)[0]
nn = len(good)
print('Starting values: x0={}, y0={}, eps={}, pa={}'.format(geo.x0, geo.y0, geo.eps, props2.pa))
#print('Final values: x0={}, y0={}, eps={}, pa={}'.format(geo.x0, geo.y0, geo.eps, props2.pa))

xx0 = iso.x0[good]
yy0 = iso.y0[good]
eeps = iso.eps[good]
ppa = np.degrees(iso.pa[good])+90

print('x0 = {}+/-{}'.format(np.mean(xx0), np.std(xx0)/np.sqrt(nn)))
print('y0 = {}+/-{}'.format(np.mean(yy0), np.std(yy0)/np.sqrt(nn)))
print('eps = {}+/-{}'.format(np.mean(eeps), np.std(eeps)/np.sqrt(nn)))
print('pa = {}+/-{}'.format(np.mean(ppa), np.std(ppa)/np.sqrt(nn)))
#print(iso.x0[good], iso.y0[good], iso.eps[good], iso.pa

cmap = 'viridis'
interval = Interval(contrast=0.9)
norm = ImageNormalize(img, interval=interval, stretch=Stretch(a=0.95))

im = plt.imshow(img, origin='lower', norm=norm, cmap=cmap, #cmap=cmap[filt],
                interpolation='nearest')

for ss in iso.sma[good]:
    efit = iso.get_closest(ss)
    x, y = efit.sampled_coordinates()
    plt.plot(x, y, color='k', lw=1, alpha=0.75)

EllipticalAperture((props.xcentroid.value, props.ycentroid.value),
                   props.semimajor_axis_sigma.value,
                   props.semiminor_axis_sigma.value,
                   props.orientation.value).plot(color='blue', lw=1, alpha=0.75)
EllipticalAperture((props2.xpeak, props2.ypeak), props2.majoraxis, 
                   props2.majoraxis*(1-props2.eps), 
                   np.radians(props2.pa-90)).plot(color='red', lw=1, alpha=0.75)

plt.savefig('junk.png')

#g = EllipseGeometry(x0=266, y0=266, eps=0.240, sma=110, pa=1.0558)
#ell = Ellipse(img, geometry=g)
#iso = ell.fit_image(3, integrmode='median', sclip=3, nclip=2, linear=False, step=0.1)

pdb.set_trace()
