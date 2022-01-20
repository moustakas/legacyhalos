#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

#salloc -N 4 -C haswell -A desi -L cfs,SCRATCH -t 04:00:00 --qos interactive --image=legacysurvey/legacyhalos:v1.0.1
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh coadds 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-coadds.log.1 2>&1 &
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh ellipse 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-ellipse.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

source $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-env

if [ $stage = "test" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --help
elif [ $stage = "coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --coadds --nproc $ncores --mpi --d25min 4 --d25max 10
elif [ $stage = "pipeline-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --pipeline-coadds --nproc $ncores --mpi
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --ellipse --nproc $ncores --mpi --d25min 4 --d25max 7.5
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --htmlplots --nproc $ncores --mpi
else
    echo "Unrecognized stage "$stage
fi
