#! /bin/bash

#SBATCH -A desi
#SBATCH -L project
#SBATCH -o hsc.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

# #SBATCH -p regular
# #SBATCH -N 1
# #SBATCH -n 3
# #SBATCH -t 02:00:00

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -n 12
#SBATCH -t 00:30:00

# for cori only
# #SBATCH -C haswell

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

# do the whole sample!
# time srun -N 32 -n 48 -c 4 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --coadds --nproc 4 --mpi

# pipeline coadds
#time srun -N 1 -n 3 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --coadds --nproc 8 --mpi --hsc
#time srun -N 4 -n 12 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 1 --last 49 --coadds --nproc 8 --mpi --hsc

# custom coadds
# time srun -N 4 -n 12 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --custom-coadds --nproc 8 --mpi --hsc

# ellipse
time srun -N 1 -n 32 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --ellipse --mpi --hsc 

# html
# time srun -N 6 -n 18 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --last 99 --html --nproc 8 --mpi --hsc --clobber
