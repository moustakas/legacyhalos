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
# #SBATCH -C haswell # for cori only

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 00:30:00

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/legacyhalos-env-nersc

# pipeline coadds
#time srun -N 1 -n 4 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --coadds --nproc 8 --mpi --hsc
#time srun -N 1 -n 3 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --coadds --nproc 8 --mpi --hsc

# custom coadds
#time srun -N 8 -n 24 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --custom-coadds --nproc 8 --mpi --hsc
#time srun -N 8 -n 24 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --custom-coadds --nproc 8 --mpi --hsc --clobber

# ellipse
time srun -N 1 -n 12 --cpu-bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --ellipse --mpi --hsc --clobber
#time srun -N 1 -n 24 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --ellipse --mpi --hsc --clobber

# html
#time srun -N 1 -n 3 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --htmlplots --nproc 8 --mpi --hsc
#time srun -N 2 -n 6 -c 8 $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 49 --html --nproc 8 --mpi --hsc --clobber
