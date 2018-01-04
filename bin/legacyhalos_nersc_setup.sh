#!/bin/bash

# Set up the software environment we need in order to run legacyhalos scripts at
# NERSC.

dr=dr5-new
desiconda_version=20180103-1.2.3-img

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/$dr
export LEGACYPIPE_DIR=${CSCRATCH}/repos/legacypipe
export LEGACYHALOS_CODE_DIR=${CSCRATCH}/repos/legacyhalos

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PATH=$LEGACYHALOS_CODE_DIR/bin:$PATH

export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYHALOS_CODE_DIR:$PYTHONPATH

export OMP_NUM_THREADS=1
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

echo 'Loading software version desiconda/'$desiconda_version
module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
module load desiconda

if [ "$NERSC_HOST" == "cori" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/cori
fi
if [ "$NERSC_HOST" == "edison" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/edison
fi  

module load unwise_coadds
module load dust
