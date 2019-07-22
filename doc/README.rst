Guide to *legacyhalos* Analysis
===============================

This README briefly describes our analysis procedure, and the purpose and
content of each notebook.

Input Data and Sample Selection
-------------------------------

1. *Assemble SDSS+unWISE photometry.*

   First, we use `SDSS/CasJobs`_ to assemble SDSS/DR14 *ugriz* and unWISE
   *W1-W4* (forced) photometry for the redMaPPer/v6.3.1 catalog of central
   galaxies (**dr8_run_redmapper_v6.3.1_lgt5_catalog.fit**) and satellite
   galaxies (**dr8_run_redmapper_v6.3.1_lgt5_catalog_members.fit**).  (Note that
   these original catalogs have optical photometry from SDSS/DR8 and no WISE
   photometry.)  This step is documented in the `redmapper-sdssWISEphot.ipynb`_
   notebook.

2. *Assemble Legacy Surveys photometry.*
   
   Next, we assemble *grz* and *W1-W4* photometry from the `Legacy Surveys/DR6`_
   and `Legacy Surveys/DR7`_ sweep files for the full catalog of centrals and
   satellites using the `match-legacysurvey-redmapper.slurm`_ SLURM script.

3. *Select the parent sample.*

   Finally, we build the parent sample of central and candidate central galaxies
   as the set of galaxies in the *redMaPPer/v6.3.1* catalogs with full-depth
   *grzW1W2* photometry from the Legacy Surveys.  This step is documented in the
   `legacyhalos-sample-selection.ipynb`_ Jupyter notebook, which also derives
   the area of the joint sample.

Analysis for Paper 1
--------------------

1. Build the sample by running the Python script `build-paper1-sample`_.

2. Build the n(z) and n(lambda) relations:
   `legacyhalos-paper1-smf --nofz --noflambda --dr dr6-dr7 --clobber --verbose`

2. Build the stellar mass functions:
   `legacyhalos-paper1-smf --smf --sfhgrid 1 --lsphot --dr dr6-dr7 --clobber --verbose`
   `legacyhalos-paper1-smf --smf --sfhgrid 2 --lsphot --dr dr6-dr7 --clobber --verbose`
   `legacyhalos-paper1-smf --smf --sfhgrid 1 --sdssphot --clobber --verbose`
   `legacyhalos-paper1-smf --smf --sfhgrid 2 --sdssphot --clobber --verbose`

2. Stellar masses.

3. Stellar mass functions.

Analysis for Paper 2
--------------------

1. Coadds.

2. Ellipse-fitting.

3. Surface brightness profile modeling.

4. Stellar masses and mass densities.

Future Work
-----------

What's next?

References
----------

**Relevant redMaPPer papers**

* `Rykoff et al. 2012, Robust Optical Richness Estimation with Reduced Scatter`_
* `Saro et al. 2015, Constraints on the richness-mass relation and the optical-SZE positional offset distribution for SZE-selected clusters`_
* `Simet et al. 2017, Weak lensing measurement of the mass-richness relation of SDSS redMaPPer clusters`_
* `Melchior et al. 2017, Weak-lensing mass calibration of redMaPPer galaxy clusters in Dark Energy Survey Science Verification data`_

* `Rykoff et al. 2014, redMaPPer. I. Algorithm and SDSS DR8 Catalog`_
* `Rozo et al. 2014, redMaPPer II: X-Ray and SZ Performance Benchmarks for the SDSS Catalog`_
* `Rozo et al. 2015a, redMaPPer - III. A detailed comparison of the Planck 2013 and SDSS DR8 redMaPPer cluster catalogues`_
* `Rozo et al. 2015b, redMaPPer - IV. Photometric membership identification of red cluster galaxies with 1 per cent precision`_
* `Rykoff et al. 2016, The RedMaPPer Galaxy Cluster Catalog From DES Science Verification Data`_

**Relevant Upenn-PhotDec papers**

* `Vikram et al. 2010, PyMorph: Automated Galaxy Structural Parameter Estimation using Python`_
* `Meert et al. 2013, Simulations of single- and two-component galaxy decompositions for spectroscopically selected galaxies from the SDSS`_
* `Meert et al. 2015, A catalogue of 2D photometric decompositions in the SDSS-DR7 spectroscopic main galaxy sample: preferred models and systematics`_
* `Meert et al. 2016, A catalogue of 2D photometric decompositions in the SDSS-DR7 spectroscopic main galaxy sample: extension to g and i bands`_
  
* `Bernardi et al. 2013, The massive end of the luminosity and stellar mass functions: dependence on the fit to the light profile`_
* `Bernardi et al. 2014, Systematic effects on the size-luminosity relations of early- and late-type galaxies: dependence on model fitting and morphology`_
* `Bernardi et al. 2016, The massive end of the luminosity and stellar mass functions and clustering from CMASS to SDSS: evidence for and against passive evolution`_
* `Bernardi et al. 2017a, The high mass end of the stellar mass function: Dependence on stellar population models and agreement between fits to the light profile`_

* `Fischer et al. 2017, Comparing pymorph and SDSS photometry - I. Background sky and model fitting effects`_
* `Bernardi et al. 2017b, Comparing pymorph and SDSS photometry - II. The differences are more than semantics and are not dominated by intracluster light`_

* `Mendel et al. 2014, A Catalog of Bulge, Disk, and Total Stellar Mass Estimates for the Sloan Digital Sky Survey`_


.. _`SDSS/CasJobs`: http://skyserver.sdss.org/CasJobs

.. _`redmapper-sdssWISEphot.ipynb`: https://nbviewer.jupyter.org/github/moustakas/legacyhalos/blob/master/doc/redmapper-sdssWISEphot.ipynb

.. _`match-legacysurvey-redmapper.slurm`: https://github.com/moustakas/legacyhalos/blob/master/bin/match-legacysurvey-redmapper.slurm

.. _`Legacy Surveys/DR6`: http://legacysurvey.org/dr6/files/#sweep-catalogs

.. _`Legacy Surveys/DR7`: http://legacysurvey.org/dr7/files/#sweep-catalogs

.. _`legacyhalos-sample-selection.ipynb`: https://nbviewer.jupyter.org/github/moustakas/legacyhalos/blob/master/doc/legacyhalos-sample-selection.ipynb

.. _`build-paper1-sample`: https://github.com/moustakas/legacyhalos/blob/paper1-sample/science/paper1/build-paper1-sample



.. _`Rykoff et al. 2012, Robust Optical Richness Estimation with Reduced Scatter`: http://adsabs.harvard.edu/abs/2012ApJ...746..178R

.. _`Saro et al. 2015, Constraints on the richness-mass relation and the optical-SZE positional offset distribution for SZE-selected clusters`: http://adsabs.harvard.edu/abs/2015MNRAS.454.2305S

.. _`Simet et al. 2017, Weak lensing measurement of the mass-richness relation of SDSS redMaPPer clusters`: http://adsabs.harvard.edu/abs/2017MNRAS.466.3103S

.. _`Melchior et al. 2017, Weak-lensing mass calibration of redMaPPer galaxy clusters in Dark Energy Survey Science Verification data`: http://adsabs.harvard.edu/abs/2017MNRAS.469.4899M

.. _`Rykoff et al. 2014, redMaPPer. I. Algorithm and SDSS DR8 Catalog`: http://adsabs.harvard.edu/abs/2014ApJ...785..104R

.. _`Rozo et al. 2014, redMaPPer II: X-Ray and SZ Performance Benchmarks for the SDSS Catalog`: http://adsabs.harvard.edu/abs/2014ApJ...783...80R

.. _`Rozo et al. 2015a, redMaPPer - III. A detailed comparison of the Planck 2013 and SDSS DR8 redMaPPer cluster catalogues`: http://adsabs.harvard.edu/abs/2015MNRAS.450..592R

.. _`Rozo et al. 2015b, redMaPPer - IV. Photometric membership identification of red cluster galaxies with 1 per cent precision`: http://adsabs.harvard.edu/abs/2015MNRAS.453...38R

.. _`Rykoff et al. 2016, The RedMaPPer Galaxy Cluster Catalog From DES Science
  Verification Data`: http://adsabs.harvard.edu/abs/2016ApJS..224....1R

.. _`Vikram et al. 2010, PyMorph: Automated Galaxy Structural Parameter Estimation using Python`: https://arxiv.org/abs/1007.4965

.. _`Meert et al. 2013, Simulations of single- and two-component galaxy decompositions for spectroscopically selected galaxies from the SDSS`: http://adsabs.harvard.edu/abs/2013MNRAS.433.1344M

.. _`Meert et al. 2015, A catalogue of 2D photometric decompositions in the SDSS-DR7 spectroscopic main galaxy sample: preferred models and systematics`: http://adsabs.harvard.edu/abs/2015MNRAS.446.3943M

.. _`Meert et al. 2016, A catalogue of 2D photometric decompositions in the SDSS-DR7 spectroscopic main galaxy sample: extension to g and i bands`: http://adsabs.harvard.edu/abs/2016MNRAS.455.2440M  

.. _`Bernardi et al. 2013, The massive end of the luminosity and stellar mass functions: dependence on the fit to the light profile`: http://adsabs.harvard.edu/abs/2013MNRAS.436..697B

.. _`Bernardi et al. 2014, Systematic effects on the size-luminosity relations of early- and late-type galaxies: dependence on model fitting and morphology`: http://adsabs.harvard.edu/abs/2014MNRAS.443..874B

.. _`Bernardi et al. 2016, The massive end of the luminosity and stellar mass functions and clustering from CMASS to SDSS: evidence for and against passive evolution`: http://adsabs.harvard.edu/abs/2016MNRAS.455.4122B

.. _`Bernardi et al. 2017a, The high mass end of the stellar mass function: Dependence on stellar population models and agreement between fits to the light profile`: http://adsabs.harvard.edu/abs/2017MNRAS.467.2217B

.. _`Fischer et al. 2017, Comparing pymorph and SDSS photometry - I. Background sky and model fitting effects`: http://adsabs.harvard.edu/abs/2017MNRAS.467..490F

.. _`Bernardi et al. 2017b, Comparing pymorph and SDSS photometry - II. The differences are more than semantics and are not dominated by intracluster light`: http://adsabs.harvard.edu/abs/2017MNRAS.468.2569B

.. _`Mendel et al. 2014, A Catalog of Bulge, Disk, and Total Stellar Mass Estimates for the Sloan Digital Sky Survey`: http://adsabs.harvard.edu/abs/2014ApJS..210....3M
