#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o match-legacysurvey-redmapper.slurm.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:30:00

# To run the script from an interactive node do:
# salloc -N 1 -C haswell -t 00:60:00 --qos interactive -L SCRATCH,project -A cosmo

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

export DR6_FILELIST=$SCRATCH/tmp/dr6_filelist
export DR7_FILELIST=$SCRATCH/tmp/dr7_filelist
export LEGACYSURVEY_FILELIST=$SCRATCH/tmp/legacysurvey_filelist

find /global/cscratch1/sd/adamyers/gaiamatch/dr6/sweep/6.0 -name 'sweep-*.fits' > $DR6_FILELIST
find /global/projecta/projectdirs/cosmo/data/legacysurvey/dr7/sweep/7.1 -name 'sweep-*.fits' > $DR7_FILELIST
cat $DR6_FILELIST $DR7_FILELIST > $LEGACYSURVEY_FILELIST

time srun -u --cpu_bind=no python $LEGACYHALOS_CODE_DIR/bin/match-legacysurvey-redmapper -v \
     --numproc 8 -f fits -F $LEGACYSURVEY_FILELIST --copycols MEM_MATCH_ID \
     $REDMAPPER_DIR/redmapper-v6.3.1-lgt5-sdssWISEphot-dr14.fits \
     . \
     $REDMAPPER_DIR/v6.3.1/legacysurvey-dr6-dr7-redmapper-v6.3.1-lgt5.fits
