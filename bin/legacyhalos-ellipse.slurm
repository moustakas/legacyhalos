#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o ellipse.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

# #SBATCH -p regular
# #SBATCH -N 32
# #SBATCH -n 48
# #SBATCH -t 02:00:00

#SBATCH -p debug
#SBATCH -n 48
#SBATCH -t 00:30:00

# for cori only
# #SBATCH -C haswell

# sbatch ~/scratch/repos/legacyhalos/bin/legacyhalos-ellipse.slurm

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

# do the whole sample!
time srun -n 48 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --ellipse --mpi

