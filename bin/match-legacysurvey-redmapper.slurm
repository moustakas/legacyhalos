#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o match-legacysurvey-redmapper.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

module load python/3.6-anaconda
# faster python start-up
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# add the legacypipe code to PYTHONPATH
export PYTHONPATH=$PWD/../py:$PYTHONPATH

drversion=dr3
rmversion=v6.3.1

# generate filelist (faster than iterating in python)
ROOT=/project/projectdirs/cosmo/data/legacysurvey/dr3
TRACTOR=$ROOT/tractor

FILELIST=`mktemp`
find $ROOT/tractor -name 'tractor-*.fits' > $FILELIST

time srun -u --cpu_bind=no -N 1 python-mpi match-external-catalogue.py -v \
--numproc 64 \
-I \
-f fits \
-F $FILELIST \
$SCRATCH/redmapper/$rmversion/dr8_run_redmapper_$rmversion_lgt5_catalog.fit \
$TRACTOR \
$SCRATCH/redmapper/legacysurvey-$drversion-redmapper-$rmversion.fits
