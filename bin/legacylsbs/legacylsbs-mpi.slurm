#! /bin/bash

#SBATCH -A desi
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# #SBATCH -p debug
# #SBATCH -N 4
# #SBATCH -n 12
# #SBATCH -t 00:30:00

# for cori only
#SBATCH -C haswell

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

# pipeline coadds
time srun -N 1 -n 1 -c 32 $LEGACYHALOS_CODE_DIR/bin/lsbs-mpi --coadds --nproc 32 --mpi

# custom coadds
# time srun -N 4 -n 12 -c 8 $LEGACYHALOS_CODE_DIR/bin/lsbs-mpi --first 0 --last 49 --custom-coadds --nproc 8 --mpi --hsc

# ellipse
# time srun -N 1 -n 32 $LEGACYHALOS_CODE_DIR/bin/lsbs-mpi --first 0 --last 49 --ellipse --mpi --hsc 

# html
# time srun -N 6 -n 18 -c 8 $LEGACYHALOS_CODE_DIR/bin/lsbs-mpi --last 99 --html --nproc 8 --mpi --hsc --clobber
