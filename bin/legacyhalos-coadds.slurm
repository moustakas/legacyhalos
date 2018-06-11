#! /bin/bash

#SBATCH -A cosmo
#SBATCH -p regular
#SBATCH -N 3
#SBATCH -n 24
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 02:00:00
#SBATCH -o legacyhalos-coadds.log.%j

# sbatch ~/scratch/repos/legacyhalos/bin/legacyhalos-coadds.slurm

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

time srun -N 1 -n 4 -c 4 --cpu_bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 19 \
    --custom-coadds --ncpu 4 --mpi

#time srun -N 3 -n 20 -c 4 --cpu_bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 19 \
#    --custom-coadds --ncpu 4 --mpi

#time srun -N 3 -n 24 -c 4 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 99 \
#    --custom-coadds --ncpu 4 --mpi
