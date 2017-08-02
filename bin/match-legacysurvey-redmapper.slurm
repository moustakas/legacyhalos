#! /bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L cscratch1,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o match-legacysurvey-dr3-redmapper-v6.3.1.%j
#SBATCH --profile=all

# Submit with sbatch match-legacysurvey-redmapper.slurm

set -x
export ATP_ENABLED=0

module load python/3.5-anaconda
# faster python start-up
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# add the legacypipe code to PATH and PYTHONPATH
export LEGACYPIPE_DIR=$CSCRATCH/legacypipe
export PATH=$LEGACYPIPE_DIR/bin:${PATH}
export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}

drversion=dr3
sweepversion=3.1
rmversion=v6.3.1

# generate filelist (faster than iterating in python)
ROOT=/project/projectdirs/cosmo/data/legacysurvey/$drversion
TRACTOR=$ROOT/tractor
FILELIST=$CSCRATCH/redmapper/$rmversion/legacysurvey-$drversion-filelist
find -L $ROOT/tractor -name 'tractor-*.fits' > $FILELIST

# match to the centrals
infile=$CSCRATCH/redmapper/$rmversion/dr8_run_redmapper_${rmversion}_lgt5_catalog.fit
outfile=$CSCRATCH/redmapper/$rmversion/legacysurvey-$drversion-redmapper-${rmversion}.fits
time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v \
--numproc 64 --tolerance 1.0 -I -f fits -F $FILELIST $infile $TRACTOR $outfile

# match to the satellites
infile=$CSCRATCH/redmapper/$rmversion/dr8_run_redmapper_${rmversion}_lgt5_catalog_members.fit
outfile=$CSCRATCH/redmapper/$rmversion/legacysurvey-$drversion-redmapper-${rmversion}-members.fits
time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v \
--numproc 64 --tolerance 1.0 -I -f fits -F $FILELIST $infile $TRACTOR $outfile
