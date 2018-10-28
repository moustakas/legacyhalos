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

module use /global/common/software/desi/$NERSC_HOST/desiconda/20180512-1.2.5-img/modulefiles
module load desiconda

export LEGACYPIPE_DIR=$HOME/repos/git/legacypipe
export LEGACYHALOS_CODE_DIR=$HOME/repos/git/legacyhalos

export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH

export DR6_FILELIST=$SCRATCH/tmp/dr6_filelist
export DR7_FILELIST=$SCRATCH/tmp/dr7_filelist
export LEGACYSURVEY_FILELIST=$SCRATCH/tmp/legacysurvey_filelist

find /project/projectdirs/cosmo/data/legacysurvey/dr6/sweep/6.0 -name 'sweep-*.fits' > $DR6_FILELIST
# find /project/projectdirs/desi/target/gaia_dr2_match_dr6/dr6/sweep/6.0 -name 'sweep-*.fits' > $DR6_FILELIST
find /global/projecta/projectdirs/cosmo/data/legacysurvey/dr7/sweep/7.1 -name 'sweep-*.fits' > $DR7_FILELIST
cat $DR6_FILELIST $DR7_FILELIST > $LEGACYSURVEY_FILELIST

time srun -u --cpu_bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-centrals-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr6-dr7-centrals-v6.3.1-lgt5.fits \
    -v --numproc 8 -f fits -F $LEGACYSURVEY_FILELIST

time srun -u --cpu_bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper \
    $REDMAPPER_DIR/v6.3.1/redmapper-v6.3.1-lgt5-members-sdssWISEphot-dr14.fits $PWD \
    $REDMAPPER_DIR/v6.3.1/legacysurvey-dr6-dr7-members-v6.3.1-lgt5.fits \
    -v --numproc 8 -f fits -F $LEGACYSURVEY_FILELIST
