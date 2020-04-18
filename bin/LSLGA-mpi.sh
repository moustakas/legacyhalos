#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [largegalaxy-coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the largegalaxy-coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

#salloc -N 4 -C haswell -A desi -L cfs -t 04:00:00 --qos interactive --image=docker:flagnarg/legacyhalos:latest
#srun -n 16 -c 8 shifter --module=mpich-cle6 /global/u2/i/ioannis/repos/git/legacyhalos/bin/LSLGA-mpi.sh largegalaxy-coadds 8 > LSLGA-largegalaxy-coadds.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

if ! [[ $ncores =~ $'^[0-9]+$' ]]; then
   echo "ncores must be a positive integer" >&2; exit 1
fi

echo 'Working on stage '$stage' with '$ncores' cores.'

# Set up the needed environment variables dependencies--
export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo5/fulldepth:/global/cfs/cdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo5
export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr2/mod
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v6.0/LSLGA-v6.0.kd.fits
export PS1CAT_DIR=/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3
export GALEX_DIR=/global/cfs/cdirs/cosmo/data/galex/images

# custom variables
export PYTHONNOUSERSITE=1 # Don't add ~/.local/ to Python's sys.path
export LEGACYPIPE_DIR=/global/homes/i/ioannis/repos/git/legacypipe

export LEGACYHALOS_DIR=/global/cfs/cdirs/desi/users/ioannis/legacyhalos
export LEGACYHALOS_DATA_DIR=/global/cfs/cdirs/desi/users/ioannis/legacyhalos-data
export LEGACYHALOS_HTML_DIR=/global/cfs/cdirs/cosmo/www/temp/ioannis/legacyhalos-html
export LEGACYHALOS_CODE_DIR=/global/homes/i/ioannis/repos/git/legacyhalos

export LSLGA_DIR=/global/cfs/cdirs/desi/users/ioannis/LSLGA
export LSLGA_DATA_DIR=/global/cfs/cdirs/desi/users/ioannis/LSLGA-data-DR9fg
export LSLGA_HTML_DIR=/global/cfs/cdirs/cosmo/www/temp/ioannis/LSLGA-html-DR9fg

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9

# Use local check-outs of the code--
export PATH=$LEGACYHALOS_CODE_DIR/bin:$PATH
export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PYTHONPATH=$LEGACYHALOS_CODE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH

# Some NERSC-specific options to get MPI working properly--
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

# Limit the memory to ensure we don't kill the whole job--
maxmem=134217728 # Cori/Haswell = 128 GB
grep -q "Xeon Phi" /proc/cpuinfo && maxmem=100663296 # Cori/KNL = 98 GB
let usemem=${maxmem}*${ncores}/32

if [ $stage = "largegalaxy-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/LSLGA-mpi --largegalaxy-coadds --nproc $ncores --mpi --verbose
elif [ $stage = "pipeline-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/LSLGA-mpi --pipeline-coadds --nproc $ncores --mpi --verbose
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/LSLGA-mpi --ellipse --nproc $ncores --mpi --verbose
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/LSLGA-mpi --htmlplots --nproc $ncores --mpi --verbose
else
    echo "Unrecognized stage "$stage
fi
