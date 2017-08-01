#! /bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o match-legacysurvey-redmapper.%j
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

drversion=dr3.1
rmversion=v6.3.1

# generate filelist (faster than iterating in python)
ROOT=/project/projectdirs/cosmo/data/legacysurvey/$drversion
TRACTOR=$ROOT/tractor

find $ROOT/tractor -name 'tractor-*.fits' > filelist

# match to the centrals
infile=$CSCRATCH/redmapper/$rmversion/dr8_run_redmapper_$rmversion_lgt5_catalog.fit
outfile=$CSCRATCH/redmapper/$rmversion/legacysurvey-$drversion-redmapper-$rmversion.fits
time srun -u --cpu_bind=no -N 1 python-mpi match-external-catalogue.py -v \
--numproc 64 -I -f fits -F filelist $infile $TRACTOR $outfile

# match to the satellites
infile=$CSCRATCH/redmapper/$rmversion/dr8_run_redmapper_$rmversion_lgt5_catalog_members.fit
outfile=$CSCRATCH/redmapper/$rmversion/legacysurvey-$drversion-redmapper-$rmversion-members.fits
time srun -u --cpu_bind=no -N 1 python-mpi match-external-catalogue.py -v \
--numproc 64 -I -f fits -F filelist $infile $TRACTOR $outfile
