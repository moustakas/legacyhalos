#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

# perlmutter
#salloc -N 4 -C cpu -A desi -L cfs -t 02:00:00 --qos interactive --image=legacysurvey/legacyhalos:v1.1
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh coadds 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-coadds.log.1 2>&1 &
#srun -n 128 -c 4 --kill-on-bad-exit=0 --no-kill shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh rebuild-unwise 4 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-rebuild-unwise.log.1 2>&1 &
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh ellipse 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-ellipse.log.1 2>&1 &
#srun -n 256 -c 1 --kill-on-bad-exit=0 --no-kill shifter --module=mpich $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh htmlplots 32 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-htmlplots.log.1 2>&1 &

# cori
#salloc -N 4 -C haswell -A desi -L cfs,SCRATCH -t 04:00:00 --qos interactive --image=legacysurvey/legacyhalos:v1.1
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh coadds 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-coadds.log.1 2>&1 &
#srun -n 128 -c 4 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh rebuild-unwise 4 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-rebuild-unwise.log.1 2>&1 &
#srun -n 8 -c 16 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh ellipse 16 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-ellipse.log.1 2>&1 &
#srun -n 256 -c 1 --kill-on-bad-exit=0 --no-kill shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi.sh htmlplots 32 > /global/cfs/cdirs/desi/users/ioannis/virgofilaments-data/logs/virgofilaments-htmlplots.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

source $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-env

if [ $stage = "test" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --help
elif [ $stage = "coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --coadds --nproc $ncores --mpi
elif [ $stage = "pipeline-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --pipeline-coadds --nproc $ncores --mpi
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --ellipse --nproc $ncores --mpi
elif [ $stage = "rebuild-unwise" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --rebuild-unwise --nproc $ncores --mpi
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/virgofilaments/virgofilaments-mpi --htmlplots --nproc $ncores --mpi
else
    echo "Unrecognized stage "$stage
fi
