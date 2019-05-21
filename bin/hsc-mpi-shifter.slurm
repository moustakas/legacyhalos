#! /bin/bash

#SBATCH -A desi
#SBATCH -C haswell # knl
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

# #SBATCH -p regular
# #SBATCH -N 1
# #SBATCH -n 3
# #SBATCH -t 02:00:00

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00

# Set up the software and dependencies
export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/project/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth:/global/project/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4
export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_DIR=/global/project/projectdirs/cosmo/staging/largegalaxies/v2.0
export PS1CAT_DIR=/global/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3
export GALEX_DIR=/global/project/projectdirs/cosmo/data/galex/images
export PYTHONNOUSERSITE=1

# Custom variables
export LEGACYPIPE_DIR=/global/homes/i/ioannis/repos/git/legacypipe
export LEGACYHALOS_DIR=/global/project/projectdirs/desi/users/ioannis/legacyhalos
export LEGACYHALOS_DATA_DIR=/global/project/projectdirs/desi/users/ioannis/legacyhalos-data
export LEGACYHALOS_HTML_DIR=/global/project/projectdirs/cosmo/www/temp/ioannis/legacyhalos-html
export LEGACYHALOS_CODE_DIR=/global/homes/i/ioannis/repos/git/legacyhalos
export HSC_DIR=${LEGACYHALOS_DIR}/hsc # temporary
export HSC_DATA_DIR=/global/project/projectdirs/desi/users/ioannis/hsc-data
export HSC_HTML_DIR=/global/project/projectdirs/cosmo/www/temp/ioannis/hsc-html
export REDMAPPER_DIR=/global/project/projectdirs/desi/users/ioannis/redmapper
# export SPS_HOME=$SCRATCH/fsps
export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8

# Use local check-outs
export PATH=$LEGACYHALOS_CODE_DIR/bin:$PATH
export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PYTHONPATH=$LEGACYHALOS_CODE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH

# Some NERSC-specific options to get MPI working properly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

# pipeline coadds
time srun -N 1 -n 4 -c 8 shifter $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --coadds --nproc 8 --mpi --hsc --first 1 --last 5 --clobber
#time srun -N 1 -n 3 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --coadds --nproc 8 --mpi --hsc

# custom coadds
#time srun -N 8 -n 24 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --custom-coadds --nproc 8 --mpi --hsc
#time srun -N 8 -n 24 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --custom-coadds --nproc 8 --mpi --hsc --clobber

# ellipse
#time srun -N 1 -n 12 --cpu-bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --ellipse --mpi --hsc
#time srun -N 1 -n 24 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --ellipse --mpi --hsc --clobber

# html
#time srun -N 1 -n 3 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --htmlplots --nproc 8 --mpi --hsc
#time srun -N 2 -n 6 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --html --nproc 8 --mpi --hsc --clobber
