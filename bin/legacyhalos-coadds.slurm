#! /bin/bash

#SBATCH -A cosmo
#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -N 3
#SBATCH -n 20
#SBATCH -t 01:00:00
#SBATCH -L SCRATCH,project
#SBATCH -o legacyhalos-coadds.log.%j

# #SBATCH -t 02:00:00
# #SBATCH -p debug
# #SBATCH -t 00:30:00

# sbatch ~/scratch/repos/legacyhalos/bin/legacyhalos-coadds.slurm

# On Cori:   20 tasks: -c 4 -n 20 -N 20*4/32=2.5
# On Edison: 20 tasks: -c 4 -n 20 -N 20*4/24=3.3

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

time srun -N 3 -n 20 -c 4 --cpu_bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 19 \
    --custom-coadds --ncpu 4 --mpi --clobber

#time srun -N 3 -n 20 -c 4 --cpu_bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 19 \
#    --custom-coadds --ncpu 4 --mpi

#time srun -N 3 -n 24 -c 4 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 99 \
#    --custom-coadds --ncpu 4 --mpi
