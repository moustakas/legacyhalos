"""
legacyhalos.sersic
==================

Code to do Sersic on the surface brightness profiles.

"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import legacyhalos.io

