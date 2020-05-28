#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o match-legacysurvey-redmapper.slurm.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:60:00

# In this SLURM script we match the redMaPPer/v6.3.1 central and satellite
# galaxy catalogs to the DR6+DR7 Legacy Surveys sweeps files using NERSC/cori.
# 
# To run the script from an interactive node do:
#   salloc -N 1 -C haswell -t 00:30:00 --qos interactive -L SCRATCH,project -A cosmo

module use /global/common/software/desi/cori/desiconda/20190311-1.2.7-img/modulefiles
module load desiconda

export LEGACYPIPE_DIR=$HOME/repos/git/legacypipe
export LEGACYHALOS_CODE_DIR=$HOME/repos/git/legacyhalos
export REDMAPPER_DIR=/global/project/projectdirs/desi/users/ioannis/redmapper

export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH

export NORTH_FILELIST=$HOME/tmp/north_filelist
export SOUTH_FILELIST=$HOME/tmp/south_filelist

find /global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0 -name 'sweep-*.fits' > $NORTH_FILELIST
find /global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0 -name 'sweep-*.fits' > $SOUTH_FILELIST

time srun -u --cpu-bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-centrals-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr8-north-centrals-v6.3.1-lgt5.fits \
    -v --numproc 32 -f fits -F $NORTH_FILELIST

time srun -u --cpu-bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-centrals-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr8-south-centrals-v6.3.1-lgt5.fits \
    -v --numproc 32 -f fits -F $SOUTH_FILELIST

time srun -u --cpu-bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-members-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr8-north-members-v6.3.1-lgt5.fits \
    -v --numproc 32 -f fits -F $NORTH_FILELIST

time srun -u --cpu-bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-members-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr8-south-members-v6.3.1-lgt5.fits \
    -v --numproc 32 -f fits -F $SOUTH_FILELIST
