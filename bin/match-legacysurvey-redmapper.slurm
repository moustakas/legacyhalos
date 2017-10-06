#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L cscratch1,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o match-legacysurvey-redmapper.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

module load python/3.5-anaconda
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}

time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc 24 \
    -f fits -F $TRACTOR_FILELIST \
    ${RMOUTDIR}/redmapper-${RMVERSION}-sdssWISEphot.fits \
    $TRACTOR_INDIR \
    ${RMOUTDIR}/legacysurvey-${DR}-redmapper-${RMVERSION}.fits
