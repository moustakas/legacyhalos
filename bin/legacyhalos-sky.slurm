#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o sky.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

#SBATCH -p regular
#SBATCH -N 32
#SBATCH -n 48
#SBATCH -t 02:00:00

# #SBATCH -p debug
# #SBATCH -N 6
# #SBATCH -n 16
# #SBATCH -t 00:30:00

# for cori only
#SBATCH -C haswell

# sbatch ~/scratch/repos/legacyhalos/bin/legacyhalos-sky.slurm

# Example calculation of how many nodes I need, given a number of MPI tasks:
  # On Cori:   20 tasks: -c 4 -n 20 -N 20*4/32=2.5
  # On Edison: 20 tasks: -c 4 -n 20 -N 20*4/24=3.3

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

time srun -N 32 -n 48 -c 4 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --sky --ncpu 4 --mpi --seed 1

# time srun -N 6 -n 16 -c 12 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --sky --first 0 --last 40 --ncpu 12 --mpi --clobber
