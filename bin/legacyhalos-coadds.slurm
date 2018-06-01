#! /bin/bash

#SBATCH -A cosmo
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -n 12
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o legacyhalos-coadds.log.%j

source $HOME/software/develop-legacysurveys-code

time srun -N 2 -n 12 -c 4 --cpu_bind=cores $LEGACYHALOS_CODE_DIR/bin/legacyhalos-mpi --first 0 --last 100 --custom-coadds --ncpu 4 --mpi
