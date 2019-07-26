# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

# Some code was copied on Nov/20/2016 from https://github.com/kbarbary/sfdmap/ commit: bacdbbd
# which was originally Licensed under an MIT "Expat" license:

# Copyright (c) 2016 Kyle Barbary

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
=============
desiutil.dust
=============

Get :math:`E(B-V)` values from the `Schlegel, Finkbeiner & Davis (1998; SFD98)`_ dust map.

.. _`Schlegel, Finkbeiner & Davis (1998; SFD98)`: http://adsabs.harvard.edu/abs/1998ApJ...500..525S.
"""

import os
import numpy as np
from astropy.io.fits import getdata
from astropy.coordinates import SkyCoord
from astropy import units as u


def _bilinear_interpolate(data, y, x):
    """Map a two-dimensional integer pixel-array at float coordinates.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Pixelized array of values.
    y : :class:`float` or :class:`~numpy.ndarray`
        y coordinates (each integer y is a row) of
        location in pixel-space at which to interpolate.
    x : :class:`float` or :class:`~numpy.ndarray`
        x coordinates (each integer x is a column) of
        location in pixel-space at which to interpolate.

    Returns
    -------
    :class:`float` or :class:`~numpy.ndarray`
        Interpolated data values at the passed locations.

    Notes
    -----
    Taken in full from https://github.com/kbarbary/sfdmap/
    """
    yfloor = np.floor(y)
    xfloor = np.floor(x)
    yw = y - yfloor
    xw = x - xfloor

    # pixel locations
    y0 = yfloor.astype(np.int)
    y1 = y0 + 1
    x0 = xfloor.astype(np.int)
    x1 = x0 + 1

    # clip locations out of range
    ny, nx = data.shape
    y0 = np.maximum(y0, 0)
    y1 = np.minimum(y1, ny-1)
    x0 = np.maximum(x0, 0)
    x1 = np.minimum(x1, nx-1)

    return ((1.0-xw) * (1.0-yw) * data[y0, x0] +
            xw       * (1.0-yw) * data[y0, x1] +
            (1.0-xw) * yw       * data[y1, x0] +
            xw       * yw       * data[y1, x1])


class _Hemisphere(object):
    """Represents one of the hemispheres (in a single file).

    Parameters
    ----------
    fname : :class:`str`
        File name containing one hemisphere of the dust map.
    scaling : :class:`float`
        Multiplicative factor by which to scale the dust map.

    Attributes
    ----------
    data : :class:`~numpy.ndarray`
        Pixelated array of dust map values.
    crpix1, crpix2 : :class:`float`
        World Coordinate System: Represent the 1-indexed
        X and Y pixel numbers of the poles.
    lam_scal : :class:`int`
        Number of pixels from b=0 to b=90 deg.
    lam_nsgp : :class:`int`
        +1 for the northern hemisphere, -1 for the south.

    Notes
    -----
    Taken in full from https://github.com/kbarbary/sfdmap/
    """
    def __init__(self, fname, scaling):
        self.data, header = getdata(fname, header=True)
        self.data *= scaling
        self.crpix1 = header['CRPIX1']
        self.crpix2 = header['CRPIX2']
        self.lam_scal = header['LAM_SCAL']
        self.sign = header['LAM_NSGP']  # north = 1, south = -1

    def ebv(self, l, b, interpolate):
        """Project Galactic longitude/latitude to lambert pixels (See SFD98).

        Parameters
        ----------
        l, b : :class:`numpy.ndarray`
            Galactic longitude and latitude.
        interpolate : :class:`bool`
            If ``True`` use bilinear interpolation to obtain values.

        Returns
        -------
        :class:`~numpy.ndarray`
            Reddening values.
        """
        x = (self.crpix1 - 1.0 +
             self.lam_scal * np.cos(l) *
             np.sqrt(1.0 - self.sign * np.sin(b)))
        y = (self.crpix2 - 1.0 -
             self.sign * self.lam_scal * np.sin(l) *
             np.sqrt(1.0 - self.sign * np.sin(b)))

        # Get map values at these pixel coordinates.
        if interpolate:
            return _bilinear_interpolate(self.data, y, x)
        else:
            x = np.round(x).astype(np.int)
            y = np.round(y).astype(np.int)

            # some valid coordinates are right on the border (e.g., x/y = 4096)
            x = np.clip(x, 0, self.data.shape[1]-1)
            y = np.clip(y, 0, self.data.shape[0]-1)
            return self.data[y, x]


class SFDMap(object):
    """Map of E(B-V) from Schlegel, Finkbeiner and Davis (1998).

    Use this class for repeated retrieval of E(B-V) values when
    there is no way to retrieve all the values at the same time: It keeps
    a reference to the FITS data from the maps so that each FITS image
    is read only once.

    Parameters
    ----------
    mapdir : :class:`str`, optional, defaults to :envvar:`DUST_DIR`.
        Directory in which to find dust map FITS images, named
        ``SFD_dust_4096_ngp.fits`` and ``SFD_dust_4096_sgp.fits``.
        If not specified, the value of the :envvar:`DUST_DIR` environment
        variable is used, otherwise an empty string is used.
    north, south : :class:`str`, optional
        Names of north and south galactic pole FITS files. Defaults are
        ``SFD_dust_4096_ngp.fits`` and ``SFD_dust_4096_sgp.fits``
        respectively.
    scaling : :class:`float`, optional, defaults to 1
        Scale all E(B-V) map values by this multiplicative factor.
        Pass scaling=0.86 for the recalibration from
        `Schlafly & Finkbeiner (2011) <http://adsabs.harvard.edu/abs/2011ApJ...737..103S)>`_.

    Notes
    -----
    Modified from https://github.com/kbarbary/sfdmap/
    """
    def __init__(self, mapdir=None, north="SFD_dust_4096_ngp.fits",
                 south="SFD_dust_4096_sgp.fits", scaling=1.):

        if mapdir is None:
            mapdir = os.environ.get('DUST_DIR', '')
        self.mapdir = mapdir

        # don't load maps initially
        self.fnames = {'north': north, 'south': south}
        self.hemispheres = {'north': None, 'south': None}

        self.scaling = scaling

    def ebv(self, *args, **kwargs):
        """Get E(B-V) value(s) at given coordinate(s).

        Parameters
        ----------
        coordinates : :class:`~astropy.coordinates.SkyCoord` or :class:`~numpy.ndarray`
            If one argument is passed, assumed to be an :class:`~astropy.coordinates.SkyCoord`
            instance, in which case the ``frame`` and ``unit`` keyword arguments are
            ignored. If two arguments are passed, they are treated as
            ``latitute, longitude`` (can be scalars or arrays or a tuple), in which
            case the frame and unit are taken from the passed keywords.
        frame : :class:`str`, optional, defaults to ``'icrs'``
            Coordinate frame, if two arguments are passed. Allowed values are any
            :class:`~astropy.coordinates.SkyCoord` frame, and ``'fk5j2000'`` and ``'j2000'``.
        unit : :class:`str`, optional, defaults to ``'degree'``
            Any :class:`~astropy.coordinates.SkyCoord` unit.
        interpolate : :class:`bool`, optional, defaults to ``True``
            Interpolate between the map values using bilinear interpolation.

        Returns
        -------
        :class:`~numpy.ndarray`
            Specific extinction E(B-V) at the given locations.

        Notes
        -----
        Modified from https://github.com/kbarbary/sfdmap/
        """
        # collect kwargs
        frame = kwargs.get('frame', 'icrs')
        unit = kwargs.get('unit', 'degree')
        interpolate = kwargs.get('interpolate', True)

        # ADM convert to a frame understood by SkyCoords
        # ADM (for backwards-compatibility)
        if frame in ('fk5j2000', 'j2000'):
            frame = 'fk5'

        # compatibility: treat single argument 2-tuple as (RA, Dec)
        if (
                (len(args) == 1) and (type(args[0]) is tuple)
                and (len(args[0]) == 2)
        ):
            args = args[0]

        if len(args) == 1:
            # treat object as already an astropy.coordinates.SkyCoords
            try:
                c = args[0]
            except AttributeError:
                raise ValueError("single argument must be "
                                 "astropy.coordinates.SkyCoord")

        elif len(args) == 2:
            lat, lon = args
            c = SkyCoord(lat, lon, unit=unit, frame=frame)

        else:
            raise ValueError("too many arguments")

        # ADM extract Galactic coordinates from astropy
        l, b = c.galactic.l.radian, c.galactic.b.radian

        # Check if l, b are scalar. If so, convert to 1-d arrays.
        # ADM use numpy.atleast_1d. Store whether the
        # ADM passed values were scalars or not
        return_scalar = not np.atleast_1d(l) is l
        l, b = np.atleast_1d(l), np.atleast_1d(b)

        # Initialize return array
        values = np.empty_like(l)

        # Treat north (b>0) separately from south (b<0).
        for pole, mask in (('north', b >= 0), ('south', b < 0)):
            if not np.any(mask):
                continue

            # Initialize hemisphere if it hasn't already been done.
            if self.hemispheres[pole] is None:
                fname = os.path.join(self.mapdir, self.fnames[pole])
                self.hemispheres[pole] = _Hemisphere(fname, self.scaling)

            values[mask] = self.hemispheres[pole].ebv(l[mask], b[mask],
                                                      interpolate)

        if return_scalar:
            return values[0]
        else:
            return values

    def __repr__(self):
        return ("SFDMap(mapdir={!r}, north={!r}, south={!r}, scaling={!r})"
                .format(self.mapdir, self.fnames['north'],
                        self.fnames['south'], self.scaling))


def ebv(*args, **kwargs):
    """Convenience function, equivalent to ``SFDMap().ebv(*args)``.
    """

    m = SFDMap(mapdir=kwargs.get('mapdir', None),
               north=kwargs.get('north', "SFD_dust_4096_ngp.fits"),
               south=kwargs.get('south', "SFD_dust_4096_sgp.fits"),
               scaling=kwargs.get('scaling', 1.))
    return m.ebv(*args, **kwargs)
