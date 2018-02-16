#!/bin/bash

# Set up the software environment we need in order to run legacyhalos scripts at
# NERSC.

# conda create --prefix $CSCRATCH/conda-envs/legacyhalos --file $DESICONDA/pkg_list.txt
# source activate $CSCRATCH/conda-envs/legacyhalos
# conda install astropy
# conda install photutils -c astropy

#cd $CSCRATCH/
#git clone https://github.com/dstndstn/astrometry.net.git
#cd astrometry.net/
#make
#make py
#make extra
#make install INSTALL_DIR=/scratch1/scratchdirs/desiproc/DRcode/build/
#export PYTHONPATH=$SCRATCH/DRcode/build/lib/python:$PYTHONPATH
#export PATH=$SCRATCH/DRcode/build/bin:$PATH
#cd ..
#git clone https://github.com/dstndstn/tractor.git
#cd tractor/
#make
#python setup.py install --prefix=/scratch1/scratchdirs/desiproc/DRcode/build/
#cd ..

dr=dr5-new
desiconda_version=20180103-1.2.3-img
#desiconda_version=20170818-1.1.12-img

echo '$desiconda='$desiconda_version
module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
#module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/$desiconda_version/modulefiles
module load desiconda

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/$dr
export LEGACYPIPE_DIR=${CSCRATCH}/repos/legacypipe
export LEGACYHALOS_DIR=${CSCRATCH}/legacyhalos
export LEGACYHALOS_CODE_DIR=${CSCRATCH}/repos/legacyhalos

echo '$LEGACYHALOS_DIR='$LEGACYHALOS_DIR
echo '$LEGACYHALOS_CODE_DIR='$LEGACYHALOS_CODE_DIR
echo '$LEGACY_SURVEY_DIR='$LEGACY_SURVEY_DIR

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PATH=$LEGACYHALOS_CODE_DIR/bin:$PATH

export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYHALOS_CODE_DIR:$PYTHONPATH

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

if [ "$NERSC_HOST" == "cori" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/cori
fi
if [ "$NERSC_HOST" == "edison" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/edison
fi  

export GAIA_CAT_DIR=/project/projectdirs/cosmo/work/gaia/chunks-gaia_rel1

module load unwise_coadds
module load dust

echo '$GAIA_CAT_DIR='$GAIA_CAT_DIR
