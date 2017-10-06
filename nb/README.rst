Guide to legacyhalos Notebooks
==============================

This README briefly describes the purpose of each notebook, and the logical
order in which they should be reviewed.

Preparatory Work
----------------

* `redmapper-casjobs.ipynb`_ -- The redMaPPer/v6.3.1 catalog of central galaxies
  we adopt for our analysis (*r8_run_redmapper_v6.3.1_lgt5_catalog.fit*)
  contains optical photometry from SDSS/DR8 and no WISE photometry.  In this
  notebook we document how we use `SDSS/CasJobs`_ to assemble updated SDSS/DR14
  *ugriz* and unWISE *W1-W4* (forced) photometry (and, in some cases,
  coordinates) for this catalog.  The final output catalog is called
  *redmapper-v6.3.1-sdssWISEphot.fits*.

* `match-legacysurvey-redmapper.ipynb`_ -- This notebook documents how we match
  the updated *redmapper-v6.3.1-sdssWISEphot.fits* catalog to the Legacy Survey
  DR5 photometric (Tractor) catalogs at NERSC.  

Analysis
--------
* `legacysurvey-redmapper-parent.ipynb`_ -- Build a parent sample of central
  galaxies (as defined by being in the redMaPPer/v6.3.1 catalog) with full-depth
  Legacy Survey grzW1W2 imaging as part of the fifth data release (DR5).  We
  also identify the subset of the sample with UPenn-PhotDec photometry.

Obsolete
--------
* `match-upenn-redmapper.ipynb`_ -- The code in this notebook is slated to be
  incorporated into the `legacysurvey-redmapper-parent.ipynb`_ notebook.


.. _`SDSS/CasJobs`: http://skyserver.sdss.org/CasJobs

.. _`redmapper-casjobs.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/redmapper-casjobs.ipynb 

.. _`match-legacysurvey-redmapper.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/match-legacysurvey-redmapper.ipynb

.. _`legacysurvey-redmapper-parent.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/legacysurvey-redmapper-parent.ipynb

.. _`match-upenn-redmapper.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/match-upenn-redmapper.ipynb
