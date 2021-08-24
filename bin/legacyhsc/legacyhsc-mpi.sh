#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

#salloc -N 8 -C haswell -A desi -L cfs,SCRATCH -t 04:00:00 --qos interactive --image=legacysurvey/legacyhalos:v0.1
#srun -n 64 -c 4 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi.sh coadds 4 > legacyhsc-coadds.log.1 2>&1 &
#srun -n 64 -c 4 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi.sh ellipse 4 > legacyhsc-ellipse.log.1 2>&1 &
#srun -n 64 -c 1 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi.sh htmlplots 1 > legacyhsc-htmlplots.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

source $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-env

#maxmem=134217728 # Cori/Haswell = 128 GB (limit the memory per job).
#grep -q "Xeon Phi" /proc/cpuinfo && maxmem=100663296 # Cori/KNL = 98 GB
#let usemem=${maxmem}*${ncores}/32

if [ $stage = "test" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi --help
elif [ $stage = "coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi --coadds --nproc $ncores --mpi --verbose --last 3
elif [ $stage = "pipeline-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi --pipeline-coadds --nproc $ncores --mpi --verbose
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi --ellipse --nproc $ncores --mpi --verbose --sky-tests
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/legacyhsc/legacyhsc-mpi --htmlplots --nproc $ncores --mpi --verbose
else
    echo "Unrecognized stage "$stage
fi
