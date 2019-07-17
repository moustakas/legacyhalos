#! /bin/bash
#SBATCH -A desi
# #SBATCH -C knl
#SBATCH -C haswell
#SBATCH -L project
#SBATCH -o legacystreams.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --image=docker:flagnarg/legacyhalos:latest

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 00:30:00

# coadds, custom coadds
time srun -N 2 -n 4 -c 16 shifter --image=docker:flagnarg/legacyhalos:latest ./legacystreams-mpi-shifter.sh
