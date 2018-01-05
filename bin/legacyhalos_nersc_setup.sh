#!/bin/bash

# Set up the software environment we need in order to run legacyhalos scripts at
# NERSC.

dr=dr5-new
#desiconda_version=20180103-1.2.3-img
desiconda_version=20170818-1.1.12-img

echo '$desiconda='$desiconda_version
#module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/$desiconda_version/modulefiles
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

export OMP_NUM_THREADS=1
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

if [ "$NERSC_HOST" == "cori" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/cori
fi
if [ "$NERSC_HOST" == "edison" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/edison
fi  

module load unwise_coadds
module load dust
