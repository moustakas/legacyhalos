Guide to legacyhalos Notebooks
==============================

This README briefly describes the purpose of each notebook, and the logical
order in which they should be reviewed.


Preparatory Work
----------------

* `redmapper-casjobs.ipynb`_ -- The redMaPPer/v6.3.1 catalog of central galaxies
  we use provides optical photometry from SDSS/DR8.  In this notebook we
  document how we use `SDSS/CasJobs`_ to assemble updated SDSS/DR14 *ugriz* and
  unWISE *W1-W4* (forced) photometry (and, in some cases, coordinates) for this
  catalog.  

* `match-legacysurvey-redmapper.ipynb`_ -- This notebook documents how we match
  the updated redMaPPer/v6.3.1 catalog to the Legacy Survey DR3 photometric
  catalogs.

Analysis
--------
* `legacysurvey-redmapper-parent.ipynb`_ -- Build a parent sample of galaxies
  satisfying the following criteria: (1) in the redMaPPer cluster catalog; (2)
  has full-depth Legacy Survey grzW1W2 imaging as part of the third data release
  (DR3); and (3) has UPenn-PhotDec photometry.

* ...

.. _`SDSS/CasJobs`: http://skyserver.sdss.org/CasJobs

.. _`redmapper-casjobs.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/redmapper-casjobs.ipynb 

.. _`match-legacysurvey-redmapper.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/match-legacysurvey-redmapper.ipynb

.. _`legacysurvey-redmapper-parent.ipynb`: https://github.com/moustakas/legacyhalos/blob/master/nb/legacysurvey-redmapper-parent.ipynb
