#! /bin/bash
# Script for running the legacyhalos code in parallel at NERSC.
# srun -N 1 -n 4 -c 8 shifter --image=docker:flagnarg/legacyhalos:latest ./hsc-mpi-shifter.sh

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

export PYTHONNOUSERSITE=1 # Don't add ~/.local/ to Python's sys.path

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

ncores=8

maxmem=134217728 # Cori/Haswell = 128 GB
grep -q "Xeon Phi" /proc/cpuinfo && maxmem=100663296 # Cori/KNL = 98 GB
let usemem=${maxmem}*${ncores}/32

#time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc-mpi --htmlplots --mpi --verbose
time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc-mpi --ellipse --nproc $ncores --mpi --verbose
#time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc-mpi --custom-coadds --nproc $ncores --mpi --verbose
#time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc-mpi --coadds --nproc $ncores --mpi --verbose
