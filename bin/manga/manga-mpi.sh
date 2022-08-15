#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

# perlmutter
#salloc -N 4 -C cpu -A desi -L cfs -t 04:00:00 --qos interactive --image=legacysurvey/legacyhalos:v1.1
#srun -n 32 -c 16 shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh coadds 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-coadds.log.1 2>&1 &
#srun -n 32 -c 16 shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh ellipse 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-ellipse.log.1 2>&1 &
#srun -n 32 -c 16 shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh resampled 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-resampled.log.1 2>&1 &
#srun -n 16 -c 1 shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh htmlplots 1 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-htmlplots.log.1 2>&1 &
#srun -n 16 -c 1 shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh resampled_htmlplots 1 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/resampled-htmlplots.log.1 2>&1 &

# cori
#salloc -N 16 -C haswell -A desi -L cfs -t 04:00:00 --qos interactive --image=legacysurvey/legacyhalos:v1.1
#srun -n 32 -c 16 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh coadds 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-coadds.log.1 2>&1 &
#srun -n 32 -c 16 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh ellipse 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-ellipse.log.1 2>&1 &
#srun -n 32 -c 16 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh resampled 16 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-resampled.log.1 2>&1 &
#srun -n 16 -c 1 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh htmlplots 1 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/manga-htmlplots.log.1 2>&1 &
#srun -n 16 -c 1 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi.sh resampled_htmlplots 1 > /global/cfs/cdirs/desi/users/ioannis/manga-data/logs/resampled-htmlplots.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

source $LEGACYHALOS_CODE_DIR/bin/manga/manga-env

#maxmem=134217728 # Cori/Haswell = 128 GB (limit the memory per job).
#grep -q "Xeon Phi" /proc/cpuinfo && maxmem=100663296 # Cori/KNL = 98 GB
#let usemem=${maxmem}*${ncores}/32

if [ $stage = "test" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --help
elif [ $stage = "coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --coadds --nproc $ncores --mpi --verbose
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --ellipse --nproc $ncores --mpi --verbose
elif [ $stage = "resampled" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --resampled-phot --nproc $ncores --mpi --verbose
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --htmlplots --nproc $ncores --mpi --verbose --clobber
elif [ $stage = "resampled_htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/manga/manga-mpi --htmlplots --resampled-phot --nproc $ncores --mpi --verbose --clobber
else
    echo "Unrecognized stage "$stage
fi
